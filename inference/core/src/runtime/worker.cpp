// The worker loop, lane retirement and engine failure.
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace crit::core {
void Engine::Impl::retire(Lane& lane, Finish finish) {
    model->retire(lane.key,*lane.session,finish);
    const auto request=lane.request;
    lane = Lane{};   // the session ends before its request's waiter can see the result
    {
        std::lock_guard lock(wakeup->mutex);
        busy=!waiting.empty() || std::any_of(lanes.begin(),lanes.end(),[](const auto& value){return bool(value.request);});
    }
    complete(request,finish);
}
void Engine::Impl::retire_ended() {
    for(auto& lane:lanes) if(lane.request && (lane.terminal || *lane.request->cancelled))
        retire(lane,lane.terminal ? lane.finish : Finish::cancelled);
}
void Engine::Impl::fail(std::exception_ptr error) noexcept {
    // Every failed request shares `error`; the last completion takes the worker's own reference. Submission
    // stops with the poison, so the queue cannot grow after it is counted.
    std::size_t failing=0;
    { std::lock_guard lock(wakeup->mutex); poisoned=true; failing=waiting.size(); }
    failing+=std::size_t(bool(active_score))+std::size_t(std::count_if(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}));
    const auto end=[&](const std::shared_ptr<Request>& request) {
        if(--failing) complete(request,Finish::failed,error);
        else complete(request,Finish::failed,std::move(error));
    };
    if(active_score) {
        const auto request=std::move(active_score);
        std::vector<float>().swap(request->score_output);
        request->query_output = {};
        end(request);
    }
    for(auto& lane:lanes) if(lane.request) {
        // As in retirement: the session ends before its request's waiter can see the failure.
        const auto request=lane.request;
        lane = Lane{};
        end(request);
    }
    std::lock_guard lock(wakeup->mutex);
    for(auto& r:waiting) end(r);
    waiting.clear();
    busy=false;
}
void Engine::Impl::run() noexcept {
    std::exception_ptr failure;
    try {
        for(;;) {
            // Clear before inspection, not before sleeping: a submission or cancellation
            // arriving during inspection must prevent the subsequent wait.
            { std::lock_guard lock(wakeup->mutex); wakeup->changed=false;
                if(stopping) for(auto& lane:lanes) if(lane.request) *lane.request->cancelled=true;
            }
            retire_ended(); admit();
            if(score_next()) continue;
            if(step()) continue;
            std::unique_lock lock(wakeup->mutex);
            const bool occupied=std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);});
            busy=occupied || !waiting.empty();
            if(stopping && !busy) return;
            if(!busy) wakeup->cv.wait(lock,[&]{return stopping || !waiting.empty();});
            // A deferred admission is asked again; a cancellation wakes the worker.
            else wakeup->cv.wait_until(lock,Clock::now()+std::chrono::milliseconds(1),[&]{return wakeup->changed;});
        }
    } catch(...) { failure=std::current_exception(); }
    // Outside the handler, so the worker's only reference to the error is the one fail() hands on.
    fail(std::move(failure));
}
}
