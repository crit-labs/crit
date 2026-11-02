// Engine construction and model declaration checks, shutdown and quarantine, request completion and the
// Engine accessors.
#include "engine.h"
#include "crit/native_engine.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace crit::core {
namespace {
std::atomic<std::size_t> quarantined_count{0};
}
std::size_t quarantined_engines() noexcept { return quarantined_count.load(); }
Engine::Impl::Impl(Profile p, std::shared_ptr<Backend> b):profile(p),resources(p.host_limit,p.device_limit),
    trace_storage(resources.reserve(Memory::host,p.trace.capacity <= 65536 ? p.trace.capacity*sizeof(TraceEvent) : 0)),
    trace_buffer(p.trace),
    operator_storage(resources.reserve(Memory::host,p.trace.operators <= 65536 ? p.trace.operators*sizeof(trace::Range) : 0)),
    operator_ranges(make_operator_ranges(p.trace,trace_buffer)),backend(std::move(b)) {
    if(!p.context || !p.prefill_chunk || p.prefill_chunk > p.context || !p.host_limit || !p.device_limit || !backend ||
       !p.max_concurrency || p.max_concurrency > max_request_concurrency || p.max_pending > 1024 ||
       !p.pending_timeout_ms || !p.retirement_timeout_ms || !p.prefill_starvation_steps ||
       !p.max_request_steps || p.max_request_steps>64 || !p.max_step_output_tokens ||
       !p.max_admission_bypasses || !p.admission_starvation_ms || !p.batch_timeout_ms || p.max_inflight_batches>64 ||
       (p.max_inflight_batches && !p.max_batch_tokens) || p.admission_lookahead > max_request_concurrency + 1024 ||
       (p.admission_policy != AdmissionPolicy::fifo && p.admission_policy != AdmissionPolicy::reuse_aware) ||
       (p.batch_policy != BatchPolicy::fair && p.batch_policy != BatchPolicy::decode_first) ||
       (p.batch_policy == BatchPolicy::decode_first && !p.max_batch_tokens))
        throw std::invalid_argument("invalid core profile/backend/factory");
}
std::unique_ptr<trace::Recorder> Engine::Impl::make_operator_ranges(const TraceConfig& config,const detail::TraceBuffer& lifecycle) {
    if(config.operators > 65536) throw std::invalid_argument("invalid trace operator capacity");
    return config.operators ? std::make_unique<trace::Recorder>(config.operators,lifecycle.epoch()) : nullptr;
}
void Engine::Impl::drain_backend(RequestKey key) {
    detail::TraceScope span(trace_buffer,TracePhase::drain,key);
    backend->drain();
}
void Engine::Impl::quiesce_model() {
    if(concurrent) {
        detail::TraceScope span(trace_buffer,TracePhase::quiesce);
        std::exception_ptr failure;
        for(auto& batch:batches) if(batch.execution)try { batch.execution->quiesce(); } catch(...) { if(!failure)failure=std::current_exception(); }
        for(auto& lane:lanes) {
            try { if(lane.suspension) lane.suspension->quiesce(); } catch(...) { if(!failure) failure=std::current_exception(); }
            try { if(lane.restoration) lane.restoration->quiesce(); } catch(...) { if(!failure) failure=std::current_exception(); }
        }
        try { concurrent->quiesce(); } catch(...) { if(!failure)failure=std::current_exception(); }
        if(failure)std::rethrow_exception(failure);
    }
}
void Engine::Impl::initialize(Factory factory) {
    if(!factory) throw std::invalid_argument("null model factory");
    const trace::SinkBinding operator_sink(operator_sink_here());
    backend->initialize(resources);
    backend->bind_thread();
    model = factory(profile,{resources,*backend});
    if(!model) throw std::invalid_argument("model factory returned null");
    concurrent = dynamic_cast<ConcurrentModel*>(model.get());
    drain_backend();
    declaration = model->info();
    if(declaration.implementation.empty() || !declaration.vocabulary || declaration.context < profile.context || declaration.prefill_chunk < profile.prefill_chunk)
        throw std::invalid_argument("model declaration does not cover core profile");
    if(declaration.query_scoring && !declaration.scoring)
        throw std::invalid_argument("query scoring requires the scoring capability");
    if(declaration.valid_vocabulary > declaration.vocabulary)
        throw std::invalid_argument("valid_vocabulary exceeds vocabulary");
    if(declaration.max_top_logprobs > top_logprobs_limit || declaration.max_top_logprobs > detail::valid_vocabulary(declaration))
        throw std::invalid_argument("max_top_logprobs exceeds top_logprobs_limit or the valid vocabulary");
    if(declaration.token_logprobs && !declaration.generation)
        throw std::invalid_argument("token_logprobs requires generation");
    if(declaration.max_top_logprobs && !declaration.token_logprobs && !declaration.query_scoring)
        throw std::invalid_argument("max_top_logprobs requires token_logprobs or query_scoring");
    auto paths = declaration.score_paths;
    std::sort(paths.begin(),paths.end());
    if(std::any_of(paths.begin(),paths.end(),[](const std::string& path){return path.empty();}) ||
       std::adjacent_find(paths.begin(),paths.end()) != paths.end())
        throw std::invalid_argument("score paths must be nonempty and unique");
    if((declaration.mixed_batches && (!concurrent || !declaration.decode_token_work)) ||
       (profile.max_batch_tokens && !declaration.mixed_batches))
        throw std::invalid_argument("mixed token work requires a capable concurrent model");
    if(!declaration.prefill_work_quantum ||
       (profile.max_batch_tokens && declaration.prefill_work_quantum>profile.max_batch_tokens))
        throw std::invalid_argument("prefill work quantum must fit the mixed batch budget");
    detail::validate_tokens(declaration.stop_tokens,declaration.vocabulary);
    if(profile.admission_policy == AdmissionPolicy::reuse_aware && (!concurrent || !declaration.scheduling_hints))
        throw std::invalid_argument("reuse-aware admission requires scheduling hints");
    if(profile.prefill_target_ns && !declaration.prefill_advice)
        throw std::invalid_argument("latency chunk sizing requires prefill advice");
    if(profile.max_inflight_batches && (!concurrent || !declaration.async_batches || !declaration.mixed_batches))
        throw std::invalid_argument("execution pipeline requires async mixed batches");
    if(profile.memory_preemption && (!profile.max_inflight_batches || !declaration.memory_preemption))
        throw std::invalid_argument("memory preemption requires incremental async execution");
    if(!declaration.max_request_steps || profile.max_request_steps>declaration.max_request_steps ||
       (profile.max_request_steps>1 && !profile.max_inflight_batches))
        throw std::invalid_argument("dependent steps require a capable async model");
    frontend = model->text_frontend();
    if(!declaration.max_concurrency || declaration.max_concurrency > max_request_concurrency || declaration.max_concurrency < profile.max_concurrency ||
       (!concurrent && (profile.max_concurrency != 1 || profile.max_pending)))
        throw std::invalid_argument("model concurrency declaration does not cover core profile");
    if(concurrent) {
        lanes.resize(profile.max_concurrency); generations.resize(profile.max_concurrency);
        row_lookup.resize(profile.max_concurrency);
    }
    worker = std::thread([this]{run();});
}
bool Engine::Impl::shutdown() noexcept {
    {
        std::lock_guard lock(wakeup->mutex);
        stopping = true;
        wakeup->changed = true;
        if(pending) *pending->cancelled = true;
    }
    wakeup->cv.notify_all();
    if(worker.joinable()) worker.join();
    const trace::SinkBinding operator_sink(operator_sink_here());
    try { backend->bind_thread(); quiesce_model(); drain_backend(); } catch(...) {return false;}
    batches.clear(); // Release operation owners before their borrowed sessions.
    for(auto& lane:lanes) {
        lane.suspension.reset(); lane.restoration.reset(); lane.receipt.reset();
        if(lane.session) lane.session->close(Finish::failed);
        lane.session.reset();
        lane = Lane{};
    }
    suspended.clear();
    waiting.clear();
    active_score.reset();
    if(retained_session) {retained_session->close(Finish::failed); retained_session.reset();}
    retained_request.reset();
    model.reset();
    return true;
}
// Publishes completion, exactly once per request; this is the worker's last access to the request's owners. From
// here they belong to the consumer, who releases them before wait() returns or the handle is destroyed
// (Handle::Impl::settle), unless `retained` marks a request the quarantined runtime keeps for unsettled readers.
// Callers move their reference to `error` in, so the consumer, not the worker, normally destroys the exception.
void Engine::Impl::complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error, bool retained) noexcept {
    trace_buffer.point(TracePhase::completion,r->trace_key,finish);
    r->text_stream.reset(); // Worker-private; the published text stays in the result.
    {
        std::lock_guard lock(r->mutex);
        // Staging is worker-private and no longer needed after retirement.
        std::vector<Token>().swap(r->staging);
        r->staging_storage = {};
        r->result.finish = finish; r->error = std::move(error); r->retained = retained; r->done = true;
        const auto now = Clock::now();
        auto& p = r->phases;
        auto& t = r->result.timing;
        t.queued_ns = detail::elapsed_ns(r->submitted, p.has_admitted ? p.admitted : now);
        if(p.has_admitted) {
            t.total_ns = detail::elapsed_ns(p.admitted, now);
            if(p.has_prefilled) t.prefill_ns = detail::elapsed_ns(p.admitted, p.prefilled);
            if(p.has_first) t.first_token_ns = detail::elapsed_ns(p.admitted, p.first);
            if(p.has_prefilled && p.has_first) t.decode_ns = detail::elapsed_ns(p.prefilled, p.last);
        }
        t.prefill_tokens = p.prefill_tokens; t.reused_prompt_tokens = p.reused_tokens; t.decode_steps = p.decode_steps;
    }
    r->cv.notify_all();
}
Engine::Engine(Profile p,std::shared_ptr<Backend> b,Factory f) {
    impl_ = std::shared_ptr<Impl>(new Impl(p,std::move(b)),[](Impl* owner) noexcept {
        if(owner->shutdown()) {delete owner; return;}
        // Keep the services facades, model, prompt and storage owners at their
        // original addresses after an unprovable final drain. Never call their
        // destructors on this path. The process must recover its failed backend.
        static Impl* retained = nullptr;
        static std::mutex retention_mutex;
        try {std::lock_guard lock(retention_mutex); owner->retained_next = retained; retained = owner;}
        catch(...) { /* Never free unknown live work even if diagnostics fail. */ }
        ++quarantined_count;
        std::fprintf(stderr,"crit core: failed backend drain; complete runtime retained for process recovery\n");
    });
    impl_->initialize(std::move(f));
}
Engine::~Engine() = default;
void NativeEngine::retain_owner() const noexcept {
    if(engine && engine->impl_) engine->impl_->native_owner = owner;
}
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
ModelInfo Engine::info() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->declaration; }
Profile Engine::profile() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->profile; }
Usage Engine::usage(Memory memory) const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->resources.usage(memory); }
TraceSnapshot Engine::trace() const {
    if(!impl_) throw std::logic_error("engine is moved from");
    auto result = impl_->trace_buffer.snapshot();
    if(impl_->operator_ranges) result.operators = impl_->operator_ranges->snapshot();
    return result;
}
}
