// Request handles: cancellation, waiting, streaming sinks and release of a consumed request's charges.
#include "engine.h"
#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crit::core {
Handle::Impl::~Impl() {
    if(request) {
        const auto held = request; // Outlives the lock on its mutex.
        *held->cancelled = true;
        try { engine->wakeup->notify(); } catch(...) {}
        std::unique_lock lock(held->mutex);
        held->cv.wait(lock,[&]{return held->done;});
        settle(lock);
    }
    if(outstanding) --*outstanding;
}
// Ends this handle's use of its done request, with the request lock held; returns with it released. The caller
// keeps its own reference. Completion was the worker's last access to the request's owners, so they are freed
// and their charges returned here, on the consuming thread: usage() read on this thread after wait() returns or
// the handle is destroyed no longer includes them. A retained request keeps its charges with the quarantined
// runtime. The worker may still drop its own reference afterwards; that releases nothing.
void Handle::Impl::settle(std::unique_lock<std::mutex>& lock) noexcept {
    const auto held = std::move(request);
    const bool retained = held->retained;
    // An unconsumed error ends on this thread too, after the lock: its destructor may be user code.
    const auto unconsumed = std::move(held->error);
    lock.unlock();
    if(!retained) held->release();
}
// Waits for completion and moves the result out. With `streaming`, the published tokens are copied under the request
// lock by `copy(result, index)`, which takes the tokens from `index` on (one, or every token published so far) and
// returns how many it took, and delivered outside it by `deliver()`, in order on this thread. A throwing delivery
// cancels the request, waits for it to finish, settles and rethrows; the handle stays consumed.
template<class Copy, class Deliver> Result Handle::Impl::consume(bool streaming, Copy copy, Deliver deliver) {
    const auto held = request; // Outlives the lock on its mutex.
    std::size_t consumed = 0;
    std::unique_lock lock(held->mutex);
    held->streaming = streaming;
    for(;;) {
        held->cv.wait(lock,[&]{return held->done || (streaming && consumed < held->result.tokens.size());});
        if(streaming && consumed < held->result.tokens.size()) {
            consumed += copy(std::as_const(held->result),consumed);
            lock.unlock();
            try { deliver(); } catch(...) {
                // Rethrown from this handler: std::current_exception() is empty for a foreign exception, and
                // forced unwinding (pthread_exit, thread cancellation) must not end inside a handler.
                *held->cancelled = true;
                try { engine->wakeup->notify(); } catch(...) {}
                lock.lock();
                held->cv.wait(lock,[&]{return held->done;});
                settle(lock);
                throw;
            }
            lock.lock();
            continue;
        }
        break;
    }
    const auto error = std::move(held->error);
    auto result = std::move(held->result);
    settle(lock);
    if(error) std::rethrow_exception(error);
    return result;
}
Handle::Handle() noexcept = default;
Handle::Handle(std::unique_ptr<Impl> p):impl_(std::move(p)),cancellation_(impl_ ? impl_->request->cancelled : nullptr),
    wake_cancel_(impl_ ? impl_->engine->wakeup : nullptr),
    id_(impl_ ? impl_->request->id : 0) {}
Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;
void Handle::cancel() noexcept {
    if(cancellation_) *cancellation_ = true;
    try { if(wake_cancel_) wake_cancel_->notify(); } catch(...) {}
}
std::uint64_t Handle::id() const noexcept { return cancellation_ ? id_ : 0; }
Result Handle::wait(TokenSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    Token token = 0;
    return state->consume(bool(sink),[&](const Result& r,std::size_t i){token = r.tokens[i]; return std::size_t(1);},
                          [&]{sink(token);});
}
Result Handle::wait_tokens(TokensSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    std::vector<Token> tokens;
    return state->consume(bool(sink),[&](const Result& r,std::size_t i) {
        tokens.assign(r.tokens.begin()+std::ptrdiff_t(i),r.tokens.end());
        return tokens.size();
    },[&]{sink(std::span<const Token>(tokens));});
}
Result Handle::wait_output(OutputSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    // Options and the text flag are immutable after submission. The row and the text piece are copied under the
    // lock: publication may grow the result while the sink runs.
    const auto& request = *state->request;
    const bool logprobs = request.options.logprobs.has_value(), text = request.text;
    const std::size_t width = request.options.logprobs.value_or(0);
    OutputEvent event;
    std::vector<TokenLogprob> top;
    std::string piece;
    top.reserve(width);
    return state->consume(bool(sink),[&](const Result& r,std::size_t i) {
        event.index = i; event.token = r.tokens[i];
        if(logprobs) {
            event.logprob = r.logprobs[i];
            top.assign(r.top_logprobs.begin()+std::ptrdiff_t(i*width),r.top_logprobs.begin()+std::ptrdiff_t((i+1)*width));
        }
        if(text) {
            const std::size_t begin = i ? r.text_ends[i-1] : 0;
            piece.assign(r.text,begin,r.text_ends[i]-begin);
        }
        return std::size_t(1);
    },[&]{
        event.top_logprobs = top; event.text = piece;
        sink(std::as_const(event));
    });
}
}
