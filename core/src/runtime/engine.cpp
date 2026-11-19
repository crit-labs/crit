// Engine construction and model declaration checks, shutdown, request completion and the Engine accessors.
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
Engine::Impl::Impl(Profile p):profile(p),resources(p.host_limit,p.device_limit) {
    if(!p.context || !p.prefill_chunk || p.prefill_chunk > p.context || !p.host_limit || !p.device_limit ||
       !p.max_concurrency || p.max_concurrency > max_request_concurrency || p.max_pending > 1024 ||
       !p.pending_timeout_ms || !p.prefill_starvation_steps ||
       (p.batch_policy != BatchPolicy::fair && p.batch_policy != BatchPolicy::decode_first) ||
       (p.batch_policy == BatchPolicy::decode_first && !p.max_batch_tokens))
        throw std::invalid_argument("invalid core profile/factory");
}
void Engine::Impl::initialize(Factory factory) {
    if(!factory) throw std::invalid_argument("null model factory");
    model = factory(profile,resources);
    if(!model) throw std::invalid_argument("model factory returned null");
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
    if((declaration.mixed_batches && !declaration.decode_token_work) ||
       (profile.max_batch_tokens && !declaration.mixed_batches))
        throw std::invalid_argument("mixed token work requires a capable concurrent model");
    if(!declaration.prefill_work_quantum ||
       (profile.max_batch_tokens && declaration.prefill_work_quantum>profile.max_batch_tokens))
        throw std::invalid_argument("prefill work quantum must fit the mixed batch budget");
    if(bool(declaration.prefill_pass_tokens) != bool(declaration.prefill_pass_steps) ||
       declaration.prefill_pass_tokens > declaration.context)
        throw std::invalid_argument("prefill passes need prefill_pass_tokens within the context and prefill_pass_steps");
    detail::validate_tokens(declaration.stop_tokens,declaration.vocabulary);
    if(!declaration.max_concurrency || declaration.max_concurrency > max_request_concurrency || declaration.max_concurrency < profile.max_concurrency)
        throw std::invalid_argument("model concurrency declaration does not cover core profile");
    lanes.resize(profile.max_concurrency); generations.resize(profile.max_concurrency);
    row_lookup.resize(profile.max_concurrency);
    worker = std::thread([this]{run();});
}
void Engine::Impl::shutdown() noexcept {
    {
        std::lock_guard lock(wakeup->mutex);
        stopping = true;
        wakeup->changed = true;
    }
    wakeup->cv.notify_all();
    if(worker.joinable()) worker.join();
    for(auto& lane:lanes) lane = Lane{};
    waiting.clear();
    active_score.reset();
    model.reset();
}
// Publishes completion, exactly once per request; this is the worker's last access to the request's owners. From
// here they belong to the consumer, who releases them before wait() returns or the handle is destroyed
// (Handle::Impl::settle).
// Callers move their reference to `error` in, so the consumer, not the worker, normally destroys the exception.
void Engine::Impl::complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error) noexcept {
    {
        std::lock_guard lock(r->mutex);
        // Staging is worker-private and no longer needed after retirement.
        std::vector<Token>().swap(r->staging);
        r->staging_storage = {};
        r->result.finish = finish; r->error = std::move(error); r->done = true;
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
Engine::Engine(Profile p,Factory f) {
    impl_ = std::shared_ptr<Impl>(new Impl(p),[](Impl* owner) noexcept { owner->shutdown(); delete owner; });
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
}
