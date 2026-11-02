// Submission: request option checks, storage accounting and the Engine::submit entry points.
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crit::core {
namespace {
constexpr std::size_t logit_bias_limit = 1024;
void validate_penalty(const std::optional<double>& value, const char* message) {
    if(value && (!std::isfinite(*value) || *value < -2 || *value > 2)) throw std::invalid_argument(message);
}
void validate_options(const RequestOptions& options, const ModelInfo& info, bool scoring, bool frontend) {
    const auto& s = options.sampling;
    const auto& support = info.sampling;
    if(scoring && detail::has_generation_options(options))
        throw std::invalid_argument("scoring does not accept generation options");
    const std::pair<bool,const char*> unsupported[]{
        {s.temperature && !support.temperature,"temperature"},{s.top_p && !support.top_p,"top_p"},
        {s.top_k && !support.top_k,"top_k"},{s.seed && !support.seed,"seed"},{s.min_p && !support.min_p,"min_p"},
        {s.repetition_penalty && !support.penalties,"repetition_penalty"},
        {s.frequency_penalty && !support.penalties,"frequency_penalty"},
        {s.presence_penalty && !support.penalties,"presence_penalty"},
        {!s.logit_bias.empty() && !support.logit_bias,"logit_bias"},{s.min_tokens && !support.min_tokens,"min_tokens"}};
    for(const auto& [missing,field]:unsupported)
        if(missing) throw std::invalid_argument(std::string("model does not support a requested sampling option: ")+field);
    if(s.temperature && (!std::isfinite(*s.temperature) || *s.temperature < 0))
        throw std::invalid_argument("temperature must be finite and nonnegative");
    if(s.top_p && (!std::isfinite(*s.top_p) || *s.top_p <= 0 || *s.top_p > 1))
        throw std::invalid_argument("top_p must be finite and in (0, 1]");
    if(s.top_k && *s.top_k > info.vocabulary)
        throw std::invalid_argument("top_k exceeds vocabulary");
    if(s.min_p && (!std::isfinite(*s.min_p) || *s.min_p < 0 || *s.min_p > 1))
        throw std::invalid_argument("min_p must be finite and in [0, 1]");
    if(s.repetition_penalty && (!std::isfinite(*s.repetition_penalty) || *s.repetition_penalty <= 0))
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    validate_penalty(s.frequency_penalty,"frequency_penalty must be finite and in [-2, 2]");
    validate_penalty(s.presence_penalty,"presence_penalty must be finite and in [-2, 2]");
    if(s.logit_bias.size() > logit_bias_limit) throw std::invalid_argument("logit_bias exceeds 1024 entries");
    if(!s.logit_bias.empty()) {
        // Padding ids are never tokens, so a bias could only promote a non-token.
        std::vector<Token> ids;
        ids.reserve(s.logit_bias.size());
        for(const auto& entry:s.logit_bias) {
            if(entry.token < 0 || std::uint32_t(entry.token) >= detail::valid_vocabulary(info))
                throw std::invalid_argument("logit_bias token outside the valid vocabulary");
            if(std::isnan(entry.bias) || entry.bias == std::numeric_limits<float>::infinity())
                throw std::invalid_argument("logit_bias must be finite or negative infinity");
            ids.push_back(entry.token);
        }
        std::sort(ids.begin(),ids.end());
        if(std::adjacent_find(ids.begin(),ids.end()) != ids.end()) throw std::invalid_argument("logit_bias tokens must be unique");
    }
    if(options.stop_tokens) detail::validate_tokens(*options.stop_tokens,info.vocabulary);
    if(options.context && !info.request_context)
        throw std::invalid_argument("model does not support request context");
    if(options.logprobs && !info.token_logprobs) throw std::invalid_argument("model does not support generation logprobs");
    if(options.logprobs && *options.logprobs > info.max_top_logprobs) throw std::invalid_argument("logprobs exceeds max_top_logprobs");
    if(options.stop_strings.size() > stop_strings_limit) throw std::invalid_argument("stop_strings exceeds 16 entries");
    for(const auto& stop:options.stop_strings) {
        if(stop.empty()) throw std::invalid_argument("stop_strings entries must be nonempty");
        if(stop.size() > stop_string_bytes_limit) throw std::invalid_argument("stop_strings entry exceeds 256 bytes");
        if(!valid_utf8(stop)) throw std::invalid_argument("stop_strings entry is not valid utf-8");
    }
    if(!options.stop_strings.empty() && !frontend) throw std::invalid_argument("stop strings require an independent text frontend");
    if(options.text && !frontend) throw std::invalid_argument("text output requires an independent text frontend");
}
void charge_bytes(std::size_t& total, std::size_t count, std::size_t width = 1) {
    if(width && count > (std::numeric_limits<std::size_t>::max()-total)/width)
        throw std::overflow_error("request storage accounting overflow");
    total += count*width;
}
}
Handle Engine::submit(std::vector<Token> prompt,std::uint32_t output,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.generation) throw std::logic_error("model lacks generation capability");
    return impl_->submit(std::move(prompt),output,false,0,std::move(options));
}
Handle Engine::submit_score(std::vector<Token> prompt,std::uint32_t first,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.scoring) throw std::logic_error("model lacks scoring capability");
    if(prompt.size() < 2 || !first || first >= prompt.size()) throw std::invalid_argument("invalid scoring target domain");
    return impl_->submit(std::move(prompt),0,true,first,std::move(options));
}
Handle Engine::submit_score(std::vector<Token> prompt,std::uint32_t first,ScoreQueries queries,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    const auto& info = impl_->declaration;
    if(!info.scoring || !info.query_scoring) throw std::logic_error("model lacks query scoring capability");
    if(prompt.size() < 2 || !first || first >= prompt.size()) throw std::invalid_argument("invalid scoring target domain");
    const std::size_t targets = prompt.size()-first;
    // per_target zero with no ids scores the targets alone.
    if(queries.per_target ? queries.ids.size() / queries.per_target != targets || queries.ids.size() % queries.per_target
                          : !queries.ids.empty())
        throw std::invalid_argument("query ids must hold per_target ids for every scoring target");
    if(queries.top_k > info.max_top_logprobs) throw std::invalid_argument("top_k exceeds max_top_logprobs");
    if(!queries.path.empty() && std::find(info.score_paths.begin(),info.score_paths.end(),queries.path) == info.score_paths.end())
        throw std::invalid_argument("unknown score path");
    detail::validate_tokens(queries.ids,detail::valid_vocabulary(info),"query id outside the valid vocabulary");
    // A target at or past the valid vocabulary has no log-probability; reject it before any reservation.
    detail::validate_tokens(std::span<const Token>(prompt).subspan(first),detail::valid_vocabulary(info),
                    "scoring target outside the valid vocabulary");
    return impl_->submit(std::move(prompt),0,true,first,std::move(options),std::move(queries));
}
Handle Engine::Impl::submit(std::vector<Token> prompt,std::uint32_t output,bool scoring,std::uint32_t first,RequestOptions options,
                            std::optional<ScoreQueries> queries) {
    if(prompt.empty() || prompt.size() > profile.context || output > profile.context-prompt.size())
        throw std::invalid_argument("prompt plus output exceeds context");
    detail::validate_tokens(prompt,declaration.vocabulary);
    validate_options(options,declaration,scoring,bool(frontend));
    // The text stream exists before admission so decoder creation failures reject the submission. The frontend
    // is thread-safe; its decoder is used serially, by the worker after publication below.
    const bool text = options.text || !options.stop_strings.empty();
    std::unique_ptr<TextStream> text_stream;
    if(text) text_stream = detail::make_text_stream(frontend,prompt,options.stop_strings,declaration.vocabulary);
    // Reject overload before charging request memory. Keep the allowance check
    // and publication atomic with other submitters; failed construction leaves
    // neither an outstanding slot nor a retained reservation.
    std::lock_guard lock(wakeup->mutex);
    if(stopping || poisoned) throw std::runtime_error("engine unavailable");
    if(!concurrent && busy) throw std::runtime_error("single-request core is busy");
    if(concurrent && outstanding->load() >= std::size_t(profile.max_concurrency)+profile.max_pending)
        throw std::runtime_error("outstanding request allowance exhausted");
    if(next_request == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("request identity exhausted");
    auto request = std::make_shared<Request>();
    // Account retained prompt and maximum result bytes, including handle lifetime.
    const auto result_items = scoring ? prompt.size()-first : output;
    std::size_t bytes = 0;
    charge_bytes(bytes,prompt.capacity(),sizeof(Token));
    charge_bytes(bytes,result_items,scoring ? sizeof(float) : sizeof(Token));
    if(queries) {
        charge_bytes(bytes,queries->ids.capacity(),sizeof(Token));           // retained query ids
        charge_bytes(bytes,queries->ids.size(),sizeof(float));               // query log-probabilities
        charge_bytes(bytes,result_items,sizeof(Token));                      // argmax per target
        charge_bytes(bytes,result_items,queries->top_k*sizeof(TokenLogprob)); // ranked alternatives per target
        charge_bytes(bytes,queries->path.capacity());                        // retained path name
    }
    if(options.stop_tokens) charge_bytes(bytes,options.stop_tokens->capacity(),sizeof(Token));
    if(options.context) charge_bytes(bytes,options.context->host_bytes());
    charge_bytes(bytes,options.sampling.logit_bias.capacity(),sizeof(LogitBias));
    charge_bytes(bytes,options.stop_strings.capacity(),sizeof(std::string));
    for(const auto& stop:options.stop_strings) charge_bytes(bytes,stop.capacity());
    if(options.logprobs) {
        charge_bytes(bytes,output,sizeof(float));                              // chosen log-probabilities
        charge_bytes(bytes,output,*options.logprobs*sizeof(TokenLogprob));     // ranked alternatives
    }
    if(text) {
        charge_bytes(bytes,output,sizeof(std::uint32_t));                      // text_ends
        // The stream's matcher copy: each stop byte with its border byte, offsets, states and the holdback.
        // Released text is charged as it grows (reserve_text).
        for(const auto& stop:options.stop_strings) charge_bytes(bytes,stop.size(),2);
        charge_bytes(bytes,options.stop_strings.size()+1,sizeof(std::uint32_t)+sizeof(std::uint16_t));
        charge_bytes(bytes,stop_string_bytes_limit);
    }
    request->storage = resources.reserve(Memory::host,bytes);
    if(concurrent) {
        request->staging_storage = resources.reserve(Memory::host,std::size_t(output)*sizeof(Token));
        request->staging.reserve(output);
    }
    request->prompt = std::move(prompt);
    request->options = std::move(options);
    request->output = output; request->scoring = scoring; request->first_target = first;
    if(scoring) request->result.scores.reserve(result_items);
    else request->result.tokens.reserve(output);
    if(request->options.logprobs) {
        request->result.logprobs.reserve(output);
        request->result.top_logprobs.reserve(std::size_t(output)**request->options.logprobs);
    }
    if(text) {
        request->result.text_ends.reserve(output);
        request->text = true;
        request->text_stream = std::move(text_stream);
    }
    if(queries) {
        request->result.query_scores.reserve(queries->ids.size());
        request->result.argmax.reserve(result_items);
        request->result.top_logprobs.reserve(result_items*queries->top_k);
        request->queries = std::move(*queries);
        request->has_queries = true;
    }
    // Retain the runtime before the worker can observe the request.
    auto handle = std::make_unique<Handle::Impl>();
    handle->engine = shared_from_this();
    request->id = ++next_request; request->result.request_id=request->id;
    request->trace_key={request->id,0,0};
    request->submitted=std::chrono::steady_clock::now();
    if(concurrent) {
        waiting.push_back(request);
        ++*outstanding; handle->outstanding=outstanding;
    } else pending=request;
    // No throwing operation may follow attaching an uncompleted request to a
    // handle: its destructor waits. In particular queue growth precedes this.
    handle->request = request;
    busy = true;
    trace_buffer.point(TracePhase::submitted,request->trace_key);
    wakeup->changed=true;
    wakeup->cv.notify_one();
    return Handle(std::move(handle));
}
}
