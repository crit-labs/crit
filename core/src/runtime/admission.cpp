// Concurrent admission: FIFO with bounded lookahead, or reuse-aware selection, into free generation lanes.
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace crit::core {
namespace {
// Model callbacks run without the scheduler lock: submitters, cancellation and the notifications of in-flight
// operations take it, and a notification may come from the engine's own device queue (a host function), which a
// callback such as open() can wait for (an allocation or copy that synchronizes). Only the worker erases from
// `waiting` and submit() only appends to it, so its indices stay valid while the lock is released.
template<class Call> auto unlocked(std::unique_lock<std::mutex>& lock, Call&& call) {
    struct Relock { std::unique_lock<std::mutex>& held; ~Relock() { held.lock(); } };
    lock.unlock();
    const Relock relock{lock};
    return call();
}
}
void Engine::Impl::set_step_limit(Lane& lane) {
    if(profile.max_request_steps==1) return;
    const auto depth=concurrent->execution_depth(continuation(lane).request);
    if(!depth || depth>declaration.max_request_steps) throw std::runtime_error("invalid request execution depth");
    lane.step_limit=std::min(depth,profile.max_request_steps);
}
bool Engine::Impl::admit_reuse(std::size_t lane_index, Clock::time_point now, std::unique_lock<std::mutex>& lock) {
    auto& lane=lanes[lane_index];
    if(generations[lane_index]==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("lane generation exhausted");
    std::optional<std::size_t> selected;
    SchedulingHint selected_hint;
    long double best=-1;
    for(std::size_t j=0; j<waiting.size() && j<=profile.admission_lookahead; ++j) {
        const auto r=waiting[j];
        if(r->scoring) break;
        if(!r->output) { complete(r,Finish::length); waiting.erase(waiting.begin()+std::ptrdiff_t(j)); return true; }
        RequestView view{{r->id,generations[lane_index]+1,std::uint32_t(lane_index)},r->prompt,r->output,&r->options};
        const auto hint=unlocked(lock,[&]{return concurrent->scheduling_hint(view);});
        if(hint.reusable_tokens>r->prompt.size()) throw std::runtime_error("invalid reusable token estimate");
        const bool forced=r->admission_bypasses>=profile.max_admission_bypasses ||
            now-r->submitted>=std::chrono::milliseconds(profile.admission_starvation_ms);
        const auto admission=unlocked(lock,[&]{return concurrent->admission(view);});
        if(admission==Admission::infeasible) {
            complete(r,Finish::failed,runtime_failure("request memory entitlement infeasible"));
            waiting.erase(waiting.begin()+std::ptrdiff_t(j)); return true;
        }
        if(admission!=Admission::ready && admission!=Admission::deferred)
            throw std::runtime_error("invalid admission decision");
        const bool shared=hint.prefix_pending || (hint.prefix_affinity && std::any_of(lanes.begin(),lanes.end(),[&](const auto& active) {
            return active.request && !active.retiring && !active.prefilled && active.prefix_affinity==hint.prefix_affinity;
        }));
        if(admission==Admission::ready && (!shared || forced)) {
            const auto cost=hint.remaining_ns ? hint.remaining_ns : std::numeric_limits<std::uint64_t>::max();
            const auto priority=(static_cast<long double>(detail::elapsed_ns(r->submitted,now))+1)/static_cast<long double>(cost);
            if(forced || priority>best) { selected=j; selected_hint=hint; best=priority; }
        }
        if(forced) break; // Never bypass an aged/capped request to reach a follower.
    }
    if(!selected) return false;
    auto r=waiting[*selected];
    for(std::size_t j=0;j<*selected;++j) ++waiting[j]->admission_bypasses;
    RequestKey key{r->id,++generations[lane_index],std::uint32_t(lane_index)};
    lane.key=key; lane.request=r; lane.prefix_affinity=selected_hint.prefix_affinity;
    r->trace_key=key; waiting.erase(waiting.begin()+std::ptrdiff_t(*selected));
    // A hint was only a prediction: open must revalidate and own actual state.
    r->phases.admit(); detail::TraceScope span(trace_buffer,TracePhase::open,key);
    unlocked(lock,[&]{
        lane.session=concurrent->open({key,r->prompt,r->output,&r->options});
        if(!lane.session) throw std::runtime_error("model opened null concurrent session");
        set_step_limit(lane);
    });
    return true;
}
void Engine::Impl::admit() {
    std::unique_lock lock(wakeup->mutex);
    const auto now = std::chrono::steady_clock::now();
    for(auto it=waiting.begin();it!=waiting.end();) {
        auto r = *it;
        if(*r->cancelled || stopping) {
            complete(r,Finish::cancelled); it = waiting.erase(it);
        } else if(now-r->submitted >= std::chrono::milliseconds(profile.pending_timeout_ms)) {
            complete(r,Finish::failed,runtime_failure("pending request deadline expired"));
            it = waiting.erase(it);
        } else ++it;
    }
    if(pressure_owner || !suspended.empty()) return;
    for(std::size_t i=0;i<lanes.size() && !waiting.empty();++i) {
        auto& lane = lanes[i];
        if(lane.request) continue;
        if(profile.admission_policy==AdmissionPolicy::reuse_aware) {
            while(!lane.request && !waiting.empty()) if(!admit_reuse(i,now,lock)) return;
            continue;
        }
        std::vector<std::shared_ptr<Request>> skipped;
        std::size_t index=0;
        for(std::uint32_t inspected=0; index<waiting.size() && inspected<=profile.admission_lookahead; ++inspected) {
            auto r=waiting[index];
            if(r->scoring) break; // Backfill never crosses an exclusive barrier.
            const auto erase=[&]{waiting.erase(waiting.begin()+std::ptrdiff_t(index));};
            if(!r->output) { erase(); complete(r,Finish::length); continue; }
            if(generations[i] == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("lane generation exhausted");
            RequestKey key{r->id,generations[i]+1,std::uint32_t(i)};
            RequestView view{key,r->prompt,r->output,&r->options};
            const auto admission = [&]{
                detail::TraceScope span(trace_buffer,TracePhase::admission,key);
                return unlocked(lock,[&]{return concurrent->admission(view);});
            }();
            if(admission == Admission::deferred) {
                if(r->admission_bypasses >= profile.max_admission_bypasses || inspected==profile.admission_lookahead) break;
                skipped.push_back(r); ++index; continue;
            }
            if(admission == Admission::infeasible) {
                erase();
                complete(r,Finish::failed,runtime_failure("request memory entitlement infeasible"));
                continue;
            }
            if(admission != Admission::ready) throw std::runtime_error("invalid admission decision");
            for(auto& earlier:skipped) ++earlier->admission_bypasses;
            // Retain before open: a throwing open may leave live readers.
            lane.key=key; lane.request=r; generations[i]=key.generation;
            r->trace_key=key; erase();
            detail::TraceScope span(trace_buffer,TracePhase::open,key);
            r->phases.admit();
            unlocked(lock,[&]{
                lane.session=concurrent->open(view);
                if(!lane.session) throw std::runtime_error("model opened null concurrent session");
                set_step_limit(lane);
            });
            break;
        }
        // A deferred head/barrier applies equally to all free lanes. Retry
        // on the next worker turn, after active work/retirement can progress.
        if(!lane.request) break;
    }
}
}
