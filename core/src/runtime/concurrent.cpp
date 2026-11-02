// The concurrent worker loop, lane retirement, memory-pressure suspension and recovery, and engine failure.
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace crit::core {
void Engine::Impl::begin_retirement(Lane& lane, Finish finish) {
    lane.finish = finish; lane.terminal = true;
    if(lane.executing) return; // Every submitted descendant must settle first.
    lane.tail.reset();
    lane.retirement_started = std::chrono::steady_clock::now();
    if(trace_buffer.enabled()) {
        lane.trace_retirement=trace_buffer.operation(); lane.trace_started=trace_buffer.now();
        trace_buffer.write({0,lane.trace_retirement,lane.trace_started,lane.key,TracePhase::retirement,TraceMark::begin});
    }
    lane.receipt = concurrent->retire(lane.key,*lane.session,finish);
    if(!lane.receipt) throw std::runtime_error("null retirement receipt");
    lane.retiring = true;
}
void Engine::Impl::poll_retirements() {
    for(auto& lane:lanes) if(lane.request) {
        if(!lane.retiring && !lane.executing && !lane.suspension && !lane.restoration && (lane.terminal || *lane.request->cancelled))
            begin_retirement(lane,lane.terminal ? lane.finish : Finish::cancelled);
        if(!lane.retiring) continue;
        if(lane.receipt->ready()) {
            // Receipt destruction can reference the session; release it first.
            lane.receipt.reset(); lane.session->close(lane.finish); lane.session.reset();
            if(trace_buffer.enabled()) {
                const auto now=trace_buffer.now();
                trace_buffer.write({0,lane.trace_retirement,now,lane.key,TracePhase::retirement,TraceMark::end,now-lane.trace_started});
            }
            const auto request=lane.request; const auto finish=lane.finish;
            if(pressure_owner == request->id || pressure_release == request->id) pressure_owner = pressure_release = 0;
            ++capacity_epoch;
            lane = Lane{};
            {
                std::lock_guard lock(wakeup->mutex);
                busy=!waiting.empty() || !suspended.empty() || std::any_of(lanes.begin(),lanes.end(),[](const auto& value){return bool(value.request);});
            }
            complete(request,finish,std::move(request->failure));
        } else if(std::chrono::steady_clock::now()-lane.retirement_started >= std::chrono::milliseconds(profile.retirement_timeout_ms))
            throw std::runtime_error("retirement deadline expired without settlement");
    }
}
ContinuationView Engine::Impl::continuation(const Lane& lane) const {
    return {{lane.key,lane.request->prompt,lane.request->output,&lane.request->options},
            lane.request->result.tokens,lane.frontier,lane.prefilled};
}
void Engine::Impl::poll_recovery() {
    for(auto& lane:lanes) if(lane.request) {
        if(lane.suspension && lane.suspension->ready()) {
            lane.image = lane.suspension->take();
            if(!lane.image) throw std::runtime_error("suspension returned no recovery image");
            lane.suspension.reset(); lane.session.reset();
            ++lane.request->result.timing.preemptions;
            // Allocate the destination before transferring ownership.
            suspended.emplace_back();
            suspended.back() = std::move(lane); lane = Lane{};
            ++capacity_epoch;
        } else if(lane.restoration && lane.restoration->ready()) {
            lane.session = lane.restoration->take();
            if(!lane.session) throw std::runtime_error("restoration returned no session");
            auto& timing = lane.request->result.timing;
            timing.recomputed_tokens += lane.restoration->recomputed_tokens();
            timing.recovery_ns += detail::elapsed_ns(lane.recovery_started,Clock::now());
            lane.restoration.reset(); lane.image.reset();
            ++capacity_epoch;
        } else if((lane.suspension || lane.restoration) &&
                  Clock::now()-lane.recovery_started >= std::chrono::milliseconds(profile.batch_timeout_ms))
            throw std::runtime_error("recovery deadline expired without settlement");
    }
    for(auto it=suspended.begin();it!=suspended.end();) {
        if(*it->request->cancelled) {
            const auto request=it->request;
            it=suspended.erase(it); ++capacity_epoch;
            complete(request,Finish::cancelled);
        } else ++it;
    }
    // A resumed request owns the solo recovery route through its first new
    // step; younger admissions cannot consume that route or cause thrashing.
    if(suspended.empty() || std::any_of(lanes.begin(),lanes.end(),[](const auto& l){return bool(l.request);})) return;
    auto oldest=std::min_element(suspended.begin(),suspended.end(),[](const auto& a,const auto& b){return a.request->id<b.request->id;});
    auto& lane=lanes.front();
    if(generations.front()==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("lane generation exhausted");
    lane=std::move(*oldest); suspended.erase(oldest);
    lane.key={lane.request->id,++generations.front(),0}; lane.frozen=false;
    lane.request->trace_key=lane.key; lane.blocked_epoch=std::numeric_limits<std::uint64_t>::max();
    pressure_owner=pressure_release=lane.request->id; lane.recovery_started=Clock::now();
    lane.restoration=concurrent->restore(continuation(lane),*lane.image);
    if(!lane.restoration) throw std::runtime_error("null restoration operation");
    lane.restoration->submit([signal=wakeup]{signal->notify();});
}
void Engine::Impl::memory_blocked(const BatchPreparation& preparation) {
    if(preparation.execution || preparation.blocked.lane>=lanes.size() ||
       lanes[preparation.blocked.lane].key!=preparation.blocked)
        throw std::runtime_error("invalid incremental preparation");
    auto& blocked=lanes[preparation.blocked.lane];
    if(preparation.status==Admission::infeasible) {
        blocked.request->failure=runtime_failure("request cannot fit execution resources alone");
        begin_retirement(blocked,Finish::failed); return;
    }
    if(preparation.status!=Admission::deferred || !preparation.pressure)
        throw std::runtime_error("blocked preparation omitted memory pressure");
    blocked.blocked_epoch=capacity_epoch;
    pressure_owner=pressure_release=blocked.request->id;
    for(const auto& lane:lanes) if(lane.request && !lane.retiring)
        pressure_owner=std::min(pressure_owner,lane.request->id);
    Lane* victim=nullptr;
    for(auto& lane:lanes) if(lane.request && lane.request->id>pressure_owner)
        lane.frozen=bool(lane.executing);
    for(auto& lane:lanes) if(lane.request && lane.request->id>pressure_owner && !lane.retiring && !lane.terminal &&
        !lane.executing && !lane.suspension && !lane.restoration &&
        concurrent->reclaimable(*preparation.pressure,continuation(lane),*lane.session) &&
        (!victim || lane.request->id>victim->request->id)) victim=&lane;
    if(victim) {
        victim->recovery_started=Clock::now();
        victim->suspension=concurrent->suspend(continuation(*victim),*victim->session);
        if(!victim->suspension) throw std::runtime_error("null suspension operation");
        victim->suspension->submit([signal=wakeup]{signal->notify();});
    } else if(std::count_if(lanes.begin(),lanes.end(),[](const auto& l){return bool(l.request);})==1 && !blocked.executing) {
        blocked.request->failure=runtime_failure("memory pressure cannot make progress with a solo request");
        begin_retirement(blocked,Finish::failed);
    } else if(std::all_of(lanes.begin(),lanes.end(),[&](const auto& lane) {
        return !lane.request || (!lane.executing && !lane.suspension && !lane.restoration &&
            !lane.retiring && lane.blocked_epoch==capacity_epoch);
    })) {
        // Non-recoverable requests may hold mutually needed pools. With no
        // runnable work or pending settlement, another retry cannot free them.
        blocked.request->failure=runtime_failure("memory pressure has no reclaimable progress route");
        begin_retirement(blocked,Finish::failed);
    }
}
void Engine::Impl::fail_concurrent(std::exception_ptr error) noexcept {
    // Every failed request shares `error`; the last completion takes the worker's own reference. Submission
    // stops with the poison, so the queue cannot grow after it is counted.
    std::size_t failing=0;
    { std::lock_guard lock(wakeup->mutex); poisoned=true; failing=waiting.size(); }
    failing+=suspended.size()+std::size_t(bool(active_score))+std::size_t(std::count_if(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}));
    const auto fail=[&](const std::shared_ptr<Request>& request, bool retained) {
        if(--failing) complete(request,Finish::failed,error,retained);
        else complete(request,Finish::failed,std::move(error),retained);
    };
    // Failure closes observation spans even if cleanup cannot prove that
    // physical owners are safe to release and must quarantine them instead.
    for(auto& batch:batches)batch.step.finish(true);
    bool settled=false;
    try { quiesce_model(); drain_backend(); settled=true; } catch(...) {}
    if(settled)batches.clear();
    if(active_score) {
        const auto request=active_score;
        if(settled) {
            std::vector<float>().swap(request->score_output);
            request->query_output = {};
            active_score.reset();
        }
        fail(request,!settled);
    }
    for(auto& lane:lanes) if(lane.request) {
        const auto request=lane.request;
        if(settled) {
            lane.suspension.reset(); lane.restoration.reset(); lane.receipt.reset(); if(lane.session) lane.session->close(Finish::failed);
            lane = Lane{};
        }
        // Match normal retirement: a completed handle must not race
        // session close or the release of settled session reservations.
        // Unknown work still retains the complete lane for quarantine.
        fail(request,!settled);
    }
    for(auto& lane:suspended) fail(lane.request,!settled);
    if(settled)suspended.clear();
    std::lock_guard lock(wakeup->mutex);
    for(auto& r:waiting) fail(r,!settled);
    if(settled) waiting.clear();
    busy=false;
}
void Engine::Impl::run_concurrent() noexcept {
    std::exception_ptr failure;
    try {
        backend->bind_thread();
        for(;;) {
            // Clear before inspection, not before sleeping: a completion
            // arriving during inspection must prevent the subsequent wait.
            { std::lock_guard lock(wakeup->mutex); wakeup->changed=false;
                if(stopping) {
                    for(auto& lane:lanes) if(lane.request) *lane.request->cancelled=true;
                    for(auto& lane:suspended) *lane.request->cancelled=true;
                }
            }
            const bool advanced=poll_batches();
            poll_retirements(); poll_recovery(); admit();
            if(score_concurrent()) continue;
            if(step_concurrent()) continue;
            if(advanced)continue;
            std::unique_lock lock(wakeup->mutex);
            const bool occupied=std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);});
            busy=occupied || !waiting.empty() || !suspended.empty();
            if(stopping && !busy) return;
            if(!busy) wakeup->cv.wait(lock,[&]{return stopping || !waiting.empty();});
            else {
                auto deadline=Clock::time_point::max();
                for(const auto& batch:batches)
                    deadline=std::min(deadline,batch.started+std::chrono::milliseconds(profile.batch_timeout_ms));
                // Admission and retirement still expose polling interfaces.
                // Pending batches instead wake us on completion or deadline.
                if(batches.empty() || !waiting.empty() || !suspended.empty() || std::any_of(lanes.begin(),lanes.end(),
                   [](const auto& lane){return lane.request && !lane.executing;}))
                    deadline=std::min(deadline,Clock::now()+std::chrono::milliseconds(1));
                wakeup->cv.wait_until(lock,deadline,[&]{return wakeup->changed;});
            }
        }
    } catch(...) { failure=std::current_exception(); }
    // Outside the handler, so the worker's only reference to the error is the one fail_concurrent hands on.
    fail_concurrent(std::move(failure));
}
}
