// Scoring and query scoring, and the exclusive scoring barrier.
#include "engine.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace crit::core {
Finish Engine::Impl::score(const std::shared_ptr<Request>& request) {
    if(*request->cancelled) return Finish::cancelled;
    if(request->has_queries) return score_queries(request);
    request->score_output = model->score_request(request->prompt,request->first_target,request->options);
    const auto& scores = request->score_output;
    if(scores.size() != request->prompt.size()-request->first_target ||
       !std::all_of(scores.begin(),scores.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("scoring output count/finiteness mismatch");
    if(*request->cancelled) {
        std::vector<float>().swap(request->score_output);
        return Finish::cancelled;
    }
    // Keep Core's pre-reserved extent, not an arbitrary backend capacity.
    request->result.scores.assign(scores.begin(),scores.end());
    std::vector<float>().swap(request->score_output);
    return Finish::length;
}
Finish Engine::Impl::score_queries(const std::shared_ptr<Request>& request) {
    request->query_output = model->score_queries(request->prompt,request->first_target,request->queries,request->options);
    const auto& out = request->query_output;
    const std::size_t targets = request->prompt.size()-request->first_target;
    const auto& queries = request->queries;
    if(out.target.size() != targets || out.queries.size() != targets*queries.per_target ||
       out.top.size() != targets*queries.top_k || (!out.argmax.empty() && out.argmax.size() != targets) ||
       !std::all_of(out.target.begin(),out.target.end(),[](float x){return std::isfinite(x);}) ||
       !std::all_of(out.queries.begin(),out.queries.end(),[](float x){return !std::isnan(x) && x != std::numeric_limits<float>::infinity();}))
        throw std::runtime_error("query scoring output count/finiteness mismatch");
    if(!std::all_of(out.target.begin(),out.target.end(),detail::logprob_value) || !std::all_of(out.queries.begin(),out.queries.end(),detail::logprob_value))
        throw std::runtime_error("query scoring log-probability above zero");
    detail::validate_ranked(out.top,queries.top_k,detail::valid_vocabulary(declaration));
    detail::validate_tokens(out.argmax,detail::valid_vocabulary(declaration),"argmax token outside the valid vocabulary");
    if(*request->cancelled) {
        request->query_output = {};
        return Finish::cancelled;
    }
    // Keep Core's pre-reserved extents, not arbitrary backend capacities.
    request->result.scores.assign(out.target.begin(),out.target.end());
    request->result.query_scores.assign(out.queries.begin(),out.queries.end());
    request->result.argmax.assign(out.argmax.begin(),out.argmax.end());
    request->result.top_logprobs.assign(out.top.begin(),out.top.end());
    request->query_output = {};
    return Finish::length;
}
bool Engine::Impl::score_next() {
    {
        std::lock_guard lock(wakeup->mutex);
        if(waiting.empty() || !waiting.front()->scoring ||
           std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}))
            return false;
        active_score = waiting.front(); waiting.pop_front();
    }
    active_score->phases.admit();
    const auto finish = score(active_score);
    {
        std::lock_guard lock(wakeup->mutex);
        busy = !waiting.empty();
    }
    complete(active_score,finish);
    active_score.reset();
    return true;
}
}
