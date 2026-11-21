// Concurrent admission into free generation lanes: by priority, the oldest first among equals, scores as barriers.
#include "engine.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

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
// The index in `waiting` of the generation admitted next: of the requests queued before the first score (an
// exclusive barrier: no request overtakes it, and it overtakes none), the lowest priority value, the oldest among
// equals; `waiting.size()` where the queue is empty or a score is its head.
std::size_t next_waiting(const std::deque<std::shared_ptr<detail::Request>>& waiting) {
    std::size_t best = waiting.size();
    for(std::size_t i=0;i<waiting.size() && !waiting[i]->scoring;++i)
        if(best == waiting.size() || waiting[i]->options.priority < waiting[best]->options.priority) best = i;
    return best;
}
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
    for(std::size_t i=0;i<lanes.size() && !waiting.empty();++i) {
        auto& lane = lanes[i];
        if(lane.request) continue;
        while(!lane.request && !waiting.empty()) {
            // An index, not an iterator: submit() appends while the lock is released around model callbacks.
            const std::size_t index = next_waiting(waiting);
            if(index == waiting.size()) break; // A score heads the queue: an exclusive barrier generation waits behind.
            auto r=waiting[index];
            const auto taken = [&] { waiting.erase(waiting.begin()+std::ptrdiff_t(index)); };
            if(!r->output) { taken(); complete(r,Finish::length); continue; }
            if(generations[i] == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("lane generation exhausted");
            RequestKey key{r->id,generations[i]+1,std::uint32_t(i)};
            RequestView view{key,r->prompt,r->output,&r->options};
            const auto admission = unlocked(lock,[&]{return model->admission(view);});
            if(admission == Admission::deferred) break;
            if(admission == Admission::infeasible) {
                taken();
                complete(r,Finish::failed,runtime_failure("request memory entitlement infeasible"));
                continue;
            }
            if(admission != Admission::ready) throw std::runtime_error("invalid admission decision");
            // Retain before open: a throwing open may leave live readers.
            lane.key=key; lane.request=r; generations[i]=key.generation;
            taken();
            r->phases.admit();
            unlocked(lock,[&]{
                lane.session=model->open(view);
                if(!lane.session) throw std::runtime_error("model opened null session");
            });
        }
        // A deferred next request (no lower priority overtakes it) or a barrier applies equally to all free lanes. Retry
        // on the next worker turn, after active work/retirement can progress.
        if(!lane.request) break;
    }
}
}
