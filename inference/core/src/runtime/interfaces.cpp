// Defaults of the public Model interface.
#include "request.h"
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
std::vector<float> Model::score_request(std::span<const Token>, std::uint32_t, const RequestOptions&) {
    throw std::logic_error("model does not implement scoring");
}
QueryScores Model::score_queries(std::span<const Token>, std::uint32_t, const ScoreQueries&, const RequestOptions&) {
    throw std::logic_error("model declares query_scoring but does not implement score_queries");
}
std::vector<RowProgress> Model::execute(std::span<const WorkItem>) {
    throw std::logic_error("mixed_batches model must implement execute");
}
}
