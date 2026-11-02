// The serial worker: one request at a time through Model/Session.
#include "engine.h"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crit::core {
void Engine::Impl::execute(const std::shared_ptr<Request>& request) {
    request->trace_key={request->id,request->scoring ? 0u : 1u,0};
    const auto key=request->trace_key;
    trace_buffer.point(TracePhase::admission,key);
    request->phases.admit();
    std::unique_ptr<Session> session;
    Finish finish = Finish::failed;
    std::exception_ptr error;
    bool retained = false;
    try {
        backend->bind_thread();
        if(*request->cancelled) finish = Finish::cancelled;
        else if(request->scoring) finish = score(request);
        else if(!request->output) finish = Finish::length;
        else {
            {
                detail::TraceScope span(trace_buffer,TracePhase::open,key);
                session = model->open_request(request->prompt,request->options);
            }
            if(!session) throw std::runtime_error("model opened null session");
            std::uint32_t frontier = 0;
            bool prefill_complete = false;
            finish = Finish::length;
            while(request->result.tokens.size() < request->output) {
                detail::TraceScope step(trace_buffer,TracePhase::step,key);
                if(*request->cancelled) {finish = Finish::cancelled; break;}
                const auto remaining = request->output-std::uint32_t(request->result.tokens.size());
                auto chunk = std::min(profile.prefill_chunk,std::uint32_t(request->prompt.size())-frontier);
                if(profile.prefill_target_ns && !prefill_complete && chunk) {
                    const auto advised=session->prefill_limit({frontier,std::uint32_t(request->prompt.size())-frontier,
                        chunk,0,profile.prefill_target_ns});
                    if(advised>chunk)throw std::runtime_error("prefill advice exceeds granted work");
                    if(advised)chunk=advised;
                }
                const WorkBudget budget{chunk,remaining,prefill_complete};
                auto progress = [&]{
                    detail::TraceScope span(trace_buffer,prefill_complete ? TracePhase::decode : TracePhase::prefill,key);
                    return session->advance(budget);
                }();
                detail::validate_deferred_progress(progress,"invalid deferred model progress");
                if(progress.readiness) {
                    detail::TraceScope span(trace_buffer,TracePhase::readiness,key);
                    progress.readiness->wait();
                }
                if(progress.deferred_tokens) progress.tokens = session->ready_tokens();
                if(trace_buffer.enabled()) {
                    try { for(const auto& timing:session->device_timings()) trace_buffer.device(step.operation(),key,timing); }
                    catch(...) { trace_buffer.drop(); }
                }
                if(*request->cancelled) {finish = Finish::cancelled; break;}
                detail::validate_progress(progress,budget,frontier,*request,"invalid or stalled model progress");
                frontier += progress.prompt_tokens;
                const bool was_prefilled = prefill_complete;
                prefill_complete = progress.prefill_complete;
                detail::validate_tokens(progress.tokens,detail::valid_vocabulary(declaration),"generated token outside the valid vocabulary");
                const auto rows = detail::logprob_rows(request->options,progress,detail::valid_vocabulary(declaration));
                // A text decoding failure fails this request before its commit.
                detail::StepText text;
                const auto d = detail::decide(progress.tokens,stops(*request),remaining,request->text_stream.get(),text);
                if(d.finish == Finish::stop) finish = Finish::stop;
                {
                    std::lock_guard lock(request->mutex);
                    const auto before = detail::extents(request->result);
                    // Commit may invalidate the borrowed batch. Stage its accepted
                    // values first, withholding publication under the request lock.
                    try {
                        detail::stage_output(*request,resources,rows,d.accepted,text);
                        if(d.accepted) request->result.tokens.insert(request->result.tokens.end(),progress.tokens.begin(),progress.tokens.begin()+d.accepted);
                        detail::TraceScope span(trace_buffer,TracePhase::commit,key);
                        session->commit(d.accepted,d.terminal);
                    }
                    catch(...) {detail::restore(request->result,before); throw;}
                }
                request->phases.step(progress.prompt_tokens,progress.reused_prompt_tokens,was_prefilled,prefill_complete,d.accepted);
                trace_buffer.point(TracePhase::publication,key);
                request->cv.notify_all();
                if(d.terminal) break;
            }
        }
    } catch(...) { error = std::current_exception(); finish = Finish::failed; }
    try {
        drain_backend(key);
        std::vector<float>().swap(request->score_output);
        request->query_output = {};
    }
    catch(...) {
        if(!error) error = std::current_exception();
        finish = Finish::failed;
        std::lock_guard lock(wakeup->mutex); poisoned = true;
        retained_session = std::move(session);
        retained_request = request;
        retained = true;
    }
    if(session) {
        detail::TraceScope span(trace_buffer,TracePhase::retirement,key);
        session->close(finish); session.reset();
    }
    {
        std::lock_guard lock(wakeup->mutex);
        busy = false;
    }
    complete(request,finish,std::move(error),retained);
}
void Engine::Impl::run() noexcept {
    const trace::SinkBinding operator_sink(operator_sink_here());
    if(concurrent) { run_concurrent(); return; }
    for(;;) {
        std::shared_ptr<Request> request;
        {
            std::unique_lock lock(wakeup->mutex);
            wakeup->cv.wait(lock,[&]{return stopping || pending;});
            if(stopping && !pending) return;
            request = std::move(pending);
        }
        execute(request);
    }
}
}
