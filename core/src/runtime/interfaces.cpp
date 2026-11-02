// Defaults of the public Model, Session and ConcurrentModel interfaces.
#include "request.h"
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
std::unique_ptr<Session> Model::open_request(std::span<const Token> prompt, const RequestOptions& options) {
    if(detail::has_sampling(options.sampling) || options.context || options.logprobs || !options.stop_strings.empty() || options.text)
        throw std::logic_error("model must implement open_request for adapter options");
    return open(prompt);
}
std::vector<float> Model::score(std::span<const Token>, std::uint32_t) {
    throw std::logic_error("model does not implement scoring");
}
std::vector<float> Model::score_request(std::span<const Token> prompt, std::uint32_t first, const RequestOptions& options) {
    if(detail::has_generation_options(options) || options.context)
        throw std::logic_error("model must implement score_request for adapter options");
    return score(prompt,first);
}
QueryScores Model::score_queries(std::span<const Token>, std::uint32_t, const ScoreQueries&, const RequestOptions&) {
    throw std::logic_error("model declares query_scoring but does not implement score_queries");
}
std::vector<Token> Model::tokenize(std::string_view) {
    throw std::logic_error("model lacks legacy tokenizer callbacks");
}
std::string Model::detokenize(std::span<const Token>) {
    throw std::logic_error("model lacks legacy tokenizer callbacks");
}
std::unique_ptr<Session> ConcurrentModel::open(std::span<const Token>) {
    throw std::logic_error("concurrent model requires identified request admission");
}
std::span<const Token> Session::ready_tokens() {
    throw std::logic_error("session returned deferred_tokens without ready_tokens");
}
std::span<const Token> ConcurrentModel::ready_tokens(const RequestKey&) {
    throw std::logic_error("concurrent model returned deferred_tokens without ready_tokens");
}
std::vector<RowProgress> ConcurrentModel::execute(std::span<const WorkItem>) {
    throw std::logic_error("mixed_batches model must implement execute");
}
std::unique_ptr<BatchExecution> ConcurrentModel::prepare_batch(std::span<const WorkItem>) {
    throw std::logic_error("async_batches model must implement prepare_batch");
}
BatchPreparation ConcurrentModel::prepare_incremental(std::span<const WorkItem>) {
    throw std::logic_error("memory_preemption model must implement prepare_incremental");
}
std::unique_ptr<Suspension> ConcurrentModel::suspend(const ContinuationView&, ConcurrentSession&) {
    throw std::logic_error("memory_preemption model must implement suspend");
}
std::unique_ptr<Restoration> ConcurrentModel::restore(const ContinuationView&, const ResumeImage&) {
    throw std::logic_error("memory_preemption model must implement restore");
}
}
