// Concurrent step groups: decode, prefill and mixed token-work groups, and group publication.
#include "engine.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crit::core {
WorkItem Engine::Impl::work(const Lane& lane) const {
    const auto output=lane.request->output-std::uint32_t(lane.request->result.tokens.size());
    return {lane.key,lane.session.get(),{
        std::min(profile.prefill_chunk,std::uint32_t(lane.request->prompt.size())-lane.frontier),output,lane.prefilled}};
}
void Engine::Impl::publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows) {
    if(items.size()!=rows.size()) throw std::runtime_error("decode row membership count mismatch");
    struct Staged {
        Lane* lane; std::span<const Token> tokens;
        std::uint32_t frontier, reused; bool prefilled; Finish finish;
        detail::Extents before;
        detail::Pass pass;
    };
    std::vector<Staged> staged;
    std::vector<CommitDecision> decisions;
    staged.reserve(items.size()); decisions.reserve(items.size());
    // Reuse the lane-indexed table, clearing it even on validation failures.
    struct ClearRows {
        std::vector<RowProgress*>& lookup; std::vector<RowProgress>& rows;
        ~ClearRows() { for(const auto& row:rows) if(row.key.lane<lookup.size()) lookup[row.key.lane]=nullptr; }
    } clear{row_lookup,rows};
    for(auto& row:rows) {
        if(row.key.lane>=lanes.size() || lanes[row.key.lane].key!=row.key)
            throw std::runtime_error("stale row identity");
        auto& slot=row_lookup[row.key.lane];
        if(slot) throw std::runtime_error("duplicate row identity");
        slot=&row;
    }
    for(const auto& item:items) if(!row_lookup[item.key.lane] || row_lookup[item.key.lane]->key!=item.key)
        throw std::runtime_error("missing row identity");
    // Validate every row first, so a contract violation stages nothing.
    std::vector<detail::LogprobRows> logprobs;
    logprobs.reserve(items.size());
    for(const auto& item:items) {
        const auto& p = row_lookup[item.key.lane]->progress;
        auto& lane = lanes[item.key.lane];
        if(lane.terminal) { logprobs.emplace_back(); continue; }
        if(item.budget.token_work &&
           (p.token_work > item.budget.token_work ||
            p.reused_prompt_tokens > p.prompt_tokens ||
            p.prompt_tokens - p.reused_prompt_tokens > std::uint64_t(p.token_work) + lane.pass.tokens ||
            (lane.prefilled && !p.token_work)))
            throw std::runtime_error("mixed token work exceeds grant or omits computation");
        detail::validate_progress(p,item.budget,lane.frontier,*lane.request,declaration,lane.pass,
                                  "invalid or stalled concurrent progress");
        detail::validate_tokens(p.tokens,detail::valid_vocabulary(declaration),"generated token outside the valid vocabulary");
        logprobs.push_back(detail::logprob_rows(lane.request->options,p,detail::valid_vocabulary(declaration)));
    }
    // Every borrowed row is materialized before commit invalidates any row. Logprob rows go straight
    // into the pre-reserved result, invisible until publication extends result.tokens.
    for(std::size_t i=0;i<items.size();++i) {
        const auto& item = items[i];
        const auto& p = row_lookup[item.key.lane]->progress;
        auto& lane = lanes[item.key.lane];
        auto& request = *lane.request;
        const bool cancelled = *request.cancelled;
        detail::Decision d{0,Finish::cancelled,true};
        detail::Extents before;
        {
            std::lock_guard lock(request.mutex);
            before = detail::extents(request.result);
            if(!cancelled && !lane.terminal) {
                d = detail::decide(p.tokens,stops(request),request.output-std::uint32_t(request.result.tokens.size()));
                detail::stage_output(request,logprobs[i],d.accepted);   // into capacity reserved at submission
            }
        }
        // Admission has reserved the largest possible accepted row. Reuse
        // it so a later submission cannot consume this group's headroom.
        auto& copied=request.staging;
        copied.clear();
        if(d.accepted) copied.assign(p.tokens.begin(),p.tokens.begin()+d.accepted);
        // The pass the row holds after this step, and its steps in a row without a consumed prompt token.
        const detail::Pass pass{p.pass_tokens,p.prompt_tokens || !p.pass_tokens ? 0 : lane.pass.steps+1};
        staged.push_back({&lane,copied,lane.frontier+p.prompt_tokens,p.reused_prompt_tokens,p.prefill_complete,d.finish,before,pass});
        decisions.push_back({item.key,item.session,d.accepted,d.terminal,cancelled || lane.terminal});
    }
    try { model->commit(decisions); }
    catch(...) {
        for(const auto& value:staged) {
            std::lock_guard lock(value.lane->request->mutex);
            detail::restore(value.lane->request->result,value.before);
        }
        throw;
    }
    for(std::size_t i=0;i<staged.size();++i) {
        auto& value = staged[i]; auto& lane = *value.lane;
        lane.request->phases.step(value.frontier-lane.frontier,value.reused,lane.prefilled,value.prefilled,value.tokens.size());
        lane.frontier=value.frontier; lane.prefilled=value.prefilled; lane.pass=value.pass;
        bool wake;
        {
            std::lock_guard lock(lane.request->mutex);
            lane.request->result.tokens.insert(lane.request->result.tokens.end(),value.tokens.begin(),value.tokens.end());
            wake = lane.request->streaming;
        }
        // A waiter that takes its tokens only at completion is not woken for each publication.
        if(wake) lane.request->cv.notify_all();
    }
    for(std::size_t i=0;i<staged.size();++i) if(decisions[i].terminal) retire(*staged[i].lane,staged[i].finish);
}
bool Engine::Impl::step() {
    if(profile.max_batch_tokens) return step_mixed();
    std::vector<WorkItem> decode;
    std::size_t prefill = lanes.size();
    for(std::size_t n=0;n<lanes.size();++n) {
        const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
        if(!lane.request) continue;
        if(lane.prefilled) decode.push_back(work(lane));
        else if(prefill==lanes.size()) prefill=i;
    }
    if(!decode.empty() && (prefer_decode || prefill==lanes.size())) {
        publish_group(decode,model->decode(decode)); prefer_decode=false; return true;
    }
    if(prefill!=lanes.size()) {
        auto item=work(lanes[prefill]);
        publish_group(std::span<const WorkItem>(&item,1),{model->prefill(item)});
        prefill_cursor=(prefill+1)%lanes.size(); prefer_decode=true; return true;
    }
    return false;
}
bool Engine::Impl::step_mixed() {
    std::vector<WorkItem> items;
    auto remaining=profile.max_batch_tokens;
    const auto eligible=[&](const Lane& lane){return lane.request && !lane.terminal &&
        !*lane.request->cancelled && lane.request->result.tokens.size()<lane.request->output;};
    const auto add=[&](std::size_t i) {
        auto item=work(lanes[i]);
        const auto minimum=item.budget.prefill_complete ? 1u : declaration.prefill_work_quantum;
        if(remaining<minimum)return;
        item.budget.token_work=minimum;
        items.push_back(item); remaining-=minimum;
    };
    const auto expand=[&](WorkItem& item) {
        const auto quantum=item.budget.prefill_complete ? 1u : declaration.prefill_work_quantum;
        const auto tokens=item.budget.prefill_complete ? declaration.decode_token_work : item.budget.prefill_tokens;
        const auto desired=(std::uint64_t(tokens)+quantum-1)/quantum*quantum;
        const auto extra=std::uint32_t(std::min<std::uint64_t>(remaining,desired-item.budget.token_work)/quantum*quantum);
        item.budget.token_work+=extra; remaining-=extra;
    };
    if(profile.batch_policy==BatchPolicy::fair) {
        // Minimum grants before widening; rotate spare capacity fairly.
        for(std::size_t n=0;n<lanes.size() && remaining;++n) {
            const auto i=(prefill_cursor+n)%lanes.size();
            if(eligible(lanes[i])) add(i);
        }
        if(!items.empty()) prefill_cursor=(items.front().key.lane+1)%lanes.size();
        for(auto& item:items) expand(item);
    } else {
        // One oldest prefill receives a minimum grant before decoders once
        // aged. This still makes progress with a one-token group budget.
        auto urgent=lanes.size();
        for(std::size_t n=0;n<lanes.size();++n) {
            const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
            if(eligible(lane) && !lane.prefilled && lane.prefill_wait_steps>=profile.prefill_starvation_steps &&
               (urgent==lanes.size() || lane.prefill_wait_steps>lanes[urgent].prefill_wait_steps)) urgent=i;
        }
        // When one prefill consumes the whole budget, aged prefills must not
        // starve decoders in turn. Alternate forced prefill and decode.
        if(remaining<=declaration.prefill_work_quantum && reserved_prefill_last && std::any_of(lanes.begin(),lanes.end(),[&](const auto& lane) {
            return eligible(lane) && lane.prefilled;
        })) urgent=lanes.size();
        reserved_prefill_last=urgent!=lanes.size();
        if(urgent!=lanes.size()) add(urgent);
        const auto decode_start=items.size();
        for(std::size_t n=0;n<lanes.size() && remaining;++n) {
            const auto i=(decode_cursor+n)%lanes.size();
            if(eligible(lanes[i]) && lanes[i].prefilled) add(i);
        }
        if(items.size()>decode_start) decode_cursor=(items[decode_start].key.lane+1)%lanes.size();
        for(std::size_t j=decode_start;j<items.size();++j) expand(items[j]);
        for(std::size_t n=0;n<lanes.size() && remaining;++n) {
            const auto i=(prefill_cursor+n)%lanes.size();
            if(i!=urgent && eligible(lanes[i]) && !lanes[i].prefilled) add(i);
        }
        bool moved_cursor=false;
        for(auto& item:items) if(!item.budget.prefill_complete) {
            if(!moved_cursor) { prefill_cursor=(item.key.lane+1)%lanes.size(); moved_cursor=true; }
            expand(item);
        }
        for(auto& lane:lanes) if(eligible(lane) && !lane.prefilled &&
            lane.prefill_wait_steps<std::numeric_limits<std::uint32_t>::max()) ++lane.prefill_wait_steps;
    }
    if(items.empty()) return false;
    for(auto& item:items) if(!item.budget.prefill_complete) {
        item.budget.prefill_tokens=std::min(item.budget.prefill_tokens,item.budget.token_work);
        lanes[item.key.lane].prefill_wait_steps=0;
    }
    publish_group(items,model->execute(items));
    return true;
}
}
