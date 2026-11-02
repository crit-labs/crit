// Concurrent step groups: decode, prefill and mixed token-work groups, the asynchronous pipeline, and group
// publication.
#include "engine.h"
#include <algorithm>
#include <chrono>
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
    auto output=lane.request->output-std::uint32_t(lane.request->result.tokens.size())-lane.reserved_output;
    if(profile.max_request_steps>1) output=std::min(output,profile.max_step_output_tokens);
    WorkItem item{lane.key,lane.session.get(),{
        std::min(profile.prefill_chunk,std::uint32_t(lane.request->prompt.size())-lane.frontier),output,lane.prefilled}};
    if(profile.max_request_steps>1) { item.step=lane.next_step; item.predecessor=lane.tail; }
    return item;
}
void Engine::Impl::advise_prefill(WorkItem& item, std::uint32_t decode_work) {
    if(!profile.prefill_target_ns || item.budget.prefill_complete || !item.budget.prefill_tokens)return;
    const auto& lane=lanes[item.key.lane];
    const auto limit=concurrent->prefill_limit(item,{lane.frontier,
        std::uint32_t(lane.request->prompt.size())-lane.frontier,item.budget.prefill_tokens,decode_work,profile.prefill_target_ns});
    if(limit>item.budget.prefill_tokens)throw std::runtime_error("prefill advice exceeds granted work");
    if(limit) {
        item.budget.prefill_tokens=limit;
        if(item.budget.token_work) {
            const auto quantum=declaration.prefill_work_quantum;
            item.budget.token_work=std::uint32_t((std::uint64_t(limit)+quantum-1)/quantum*quantum);
        }
    }
}
void Engine::Impl::publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows, std::uint64_t operation, BatchExecution* execution) {
    if(items.size()!=rows.size()) throw std::runtime_error("decode row membership count mismatch");
    struct Staged {
        Lane* lane; std::span<const Token> tokens;
        std::uint32_t frontier, reused; bool prefilled; Finish finish;
        detail::Extents before; std::exception_ptr failure;
    };
    std::vector<Staged> staged;
    std::vector<CommitDecision> decisions;
    staged.reserve(items.size()); decisions.reserve(items.size());
    // Reuse the lane-indexed table, clearing it even on validation/fence
    // failures. Validate the whole group before waiting on borrowed rows.
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
    for(const auto& item:items) if(!row_lookup[item.key.lane] || row_lookup[item.key.lane]->key!=item.key || row_lookup[item.key.lane]->step!=item.step)
        throw std::runtime_error("missing row identity");
    for(const auto& item:items) {
        const auto* row=row_lookup[item.key.lane];
        if(row->progress.readiness) {
            detail::TraceScope span(trace_buffer,TracePhase::readiness,item.key);
            row->progress.readiness->wait();
        }
    }
    for(auto& row:rows) if(row.progress.deferred_tokens) {
        detail::validate_deferred_progress(row.progress,"invalid deferred concurrent progress");
        row.progress.tokens = execution ? execution->ready_tokens(row.key) : concurrent->ready_tokens(row.key);
    }
    if(trace_buffer.enabled()) {
        try {
            for(const auto& timing:execution ? execution->device_timings() : concurrent->device_timings())
                for(const auto& item:items) trace_buffer.device(operation,item.key,timing);
        } catch(...) { trace_buffer.drop(); }
    }
    // Validate every row before any text decoding, so a contract violation stages nothing.
    std::vector<detail::LogprobRows> logprobs;
    logprobs.reserve(items.size());
    for(const auto& item:items) {
        const auto& p = row_lookup[item.key.lane]->progress;
        auto& lane = lanes[item.key.lane];
        if(lane.terminal) { logprobs.emplace_back(); continue; }
        if(item.budget.token_work &&
           (p.token_work > item.budget.token_work ||
            p.reused_prompt_tokens > p.prompt_tokens ||
            p.prompt_tokens - p.reused_prompt_tokens > p.token_work ||
            (lane.prefilled && !p.token_work)))
            throw std::runtime_error("mixed token work exceeds grant or omits computation");
        detail::validate_progress(p,item.budget,lane.frontier,*lane.request,"invalid or stalled concurrent progress");
        detail::validate_tokens(p.tokens,detail::valid_vocabulary(declaration),"generated token outside the valid vocabulary");
        logprobs.push_back(detail::logprob_rows(lane.request->options,p,detail::valid_vocabulary(declaration)));
    }
    // Every borrowed row is materialized before commit invalidates any row. Logprob rows and text go straight
    // into the pre-reserved result, invisible until publication extends result.tokens.
    for(std::size_t i=0;i<items.size();++i) {
        const auto& item = items[i];
        const auto& p = row_lookup[item.key.lane]->progress;
        auto& lane = lanes[item.key.lane];
        auto& request = *lane.request;
        const bool cancelled = *request.cancelled;
        detail::Decision d{0,Finish::cancelled,true};
        detail::Extents before;
        std::exception_ptr failure;
        {
            std::lock_guard lock(request.mutex);
            before = detail::extents(request.result);
        }
        // A text decoding or text storage failure ends only this row: it commits like a cancellation and
        // retires with Finish::failed, reporting the error from wait().
        if(!cancelled && !lane.terminal) try {
            detail::StepText text;
            d = detail::decide(p.tokens,stops(request),request.output-std::uint32_t(request.result.tokens.size()),request.text_stream.get(),text);
            std::lock_guard lock(request.mutex);
            try { detail::stage_output(request,resources,logprobs[i],d.accepted,text); }
            catch(...) { detail::restore(request.result,before); throw; }
        } catch(...) { failure = std::current_exception(); d = {0,Finish::failed,true}; }
        // Admission has reserved the largest possible accepted row. Reuse
        // it so a later submission cannot consume this group's headroom.
        auto& copied=request.staging;
        copied.clear();
        if(d.accepted) copied.assign(p.tokens.begin(),p.tokens.begin()+d.accepted);
        staged.push_back({&lane,copied,lane.frontier+p.prompt_tokens,p.reused_prompt_tokens,p.prefill_complete,d.finish,before,failure});
        decisions.push_back({item.key,item.session,d.accepted,d.terminal,cancelled || bool(failure) || lane.terminal,item.step,lane.terminal});
    }
    try {
        detail::TraceScope span(trace_buffer,TracePhase::commit,items);
        if(execution)execution->commit(decisions); else concurrent->commit(decisions);
    } catch(...) {
        for(const auto& value:staged) {
            std::lock_guard lock(value.lane->request->mutex);
            detail::restore(value.lane->request->result,value.before);
        }
        throw;
    }
    for(std::size_t i=0;i<staged.size();++i) {
        auto& value = staged[i]; auto& lane = *value.lane;
        if(decisions[i].discard) continue;
        lane.request->phases.step(value.frontier-lane.frontier,value.reused,lane.prefilled,value.prefilled,value.tokens.size());
        lane.frontier=value.frontier; lane.prefilled=value.prefilled;
        if(value.failure) lane.request->failure = value.failure;
        {
            std::lock_guard lock(lane.request->mutex);
            lane.request->result.tokens.insert(lane.request->result.tokens.end(),value.tokens.begin(),value.tokens.end());
        }
        trace_buffer.point(TracePhase::publication,lane.key);
        lane.request->cv.notify_all();
    }
    for(std::size_t i=0;i<staged.size();++i) if(decisions[i].terminal && !decisions[i].discard) begin_retirement(*staged[i].lane,staged[i].finish);
}
bool Engine::Impl::step_concurrent() {
    if(profile.max_inflight_batches && batches.size()>=profile.max_inflight_batches)return false;
    if(profile.max_batch_tokens) return step_mixed();
    std::vector<WorkItem> decode;
    std::size_t prefill = lanes.size();
    for(std::size_t n=0;n<lanes.size();++n) {
        const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
        if(!lane.request || lane.retiring) continue;
        if(lane.prefilled) decode.push_back(work(lane));
        else if(prefill==lanes.size()) prefill=i;
    }
    if(!decode.empty() && (prefer_decode || prefill==lanes.size())) {
        detail::TraceScope step(trace_buffer,TracePhase::step,std::span<const WorkItem>(decode));
        auto rows=[&]{
            detail::TraceScope span(trace_buffer,TracePhase::decode,std::span<const WorkItem>(decode));
            return concurrent->decode(decode);
        }();
        publish_group(decode,std::move(rows),step.operation()); prefer_decode=false; return true;
    }
    if(prefill!=lanes.size()) {
        auto item=work(lanes[prefill]); advise_prefill(item);
        detail::TraceScope step(trace_buffer,TracePhase::step,item.key);
        auto row=[&]{
            detail::TraceScope span(trace_buffer,TracePhase::prefill,item.key);
            return concurrent->prefill(item);
        }();
        publish_group(std::span<const WorkItem>(&item,1),{std::move(row)},step.operation());
        prefill_cursor=(prefill+1)%lanes.size(); prefer_decode=true; return true;
    }
    return false;
}
bool Engine::Impl::step_mixed() {
    std::vector<WorkItem> items;
    auto remaining=profile.max_batch_tokens;
    const auto eligible=[&](const Lane& lane){return lane.request && !lane.retiring && !lane.terminal && !lane.frozen &&
        !*lane.request->cancelled && !lane.suspension && !lane.restoration && lane.blocked_epoch!=capacity_epoch &&
        lane.executing<(lane.prefilled ? lane.step_limit : 1) &&
        lane.request->result.tokens.size()+lane.reserved_output<lane.request->output;};
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
    std::uint32_t decode_work=0;
    for(const auto& item:items) if(item.budget.prefill_complete)decode_work+=item.budget.token_work;
    for(auto& item:items) if(!item.budget.prefill_complete) {
        item.budget.prefill_tokens=std::min(item.budget.prefill_tokens,item.budget.token_work);
        advise_prefill(item,decode_work);
        lanes[item.key.lane].prefill_wait_steps=0;
    }
    if(profile.max_inflight_batches) {
        // Retain the batch before prepare/submit; neither a throwing submit
        // nor a lost completion can release its sessions or storage early.
        batches.emplace_back(std::move(items),trace_buffer);
        auto& batch=batches.back();
        if(profile.memory_preemption) {
            auto preparation=concurrent->prepare_incremental(batch.items);
            if(preparation.status!=Admission::ready) {
                // No operation was submitted; partial reservations must be gone.
                const bool member=std::any_of(batch.items.begin(),batch.items.end(),[&](const auto& item){return item.key==preparation.blocked;});
                if(!member) throw std::runtime_error("blocked preparation identifies a nonmember");
                batches.pop_back(); memory_blocked(preparation); return true;
            }
            batch.execution=std::move(preparation.execution);
            // The blocked (or restored) request got its step's memory: the pressure is over and admissions resume.
            if(pressure_release && std::any_of(batch.items.begin(),batch.items.end(),[&](const auto& item){return item.key.request==pressure_release;}))
                pressure_owner=pressure_release=0;
        } else batch.execution=concurrent->prepare_batch(batch.items);
        if(!batch.execution)throw std::runtime_error("null batch execution");
        for(const auto& item:batch.items) {
            auto& lane=lanes[item.key.lane]; ++lane.executing;
            lane.reserved_output+=item.budget.output_tokens;
            if(item.step) {
                if(lane.next_step==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("step sequence exhausted");
                ++lane.next_step;
            }
        }
        batch.execution->submit([signal=wakeup]{signal->notify();});
        if(profile.max_request_steps>1) for(const auto& item:batch.items) {
            auto& lane=lanes[item.key.lane];
            if(lane.step_limit>1) {
                lane.tail=batch.execution->continuation({item.key,item.step});
                if(!lane.tail) throw std::runtime_error("dependent execution omitted continuation");
            }
        }
        return true;
    }
    detail::TraceScope step(trace_buffer,TracePhase::step,std::span<const WorkItem>(items));
    auto rows=[&]{
        detail::TraceScope span(trace_buffer,TracePhase::mixed,std::span<const WorkItem>(items));
        return concurrent->execute(items);
    }();
    publish_group(items,std::move(rows),step.operation());
    return true;
}
bool Engine::Impl::poll_batches() {
    bool progress=false;
    for(auto it=batches.begin();it!=batches.end();) {
        const bool ordered=std::all_of(it->items.begin(),it->items.end(),[&](const auto& item) {
            return !item.step || item.step==lanes[item.key.lane].publish_step;
        });
        if(it->execution->ready() && ordered) {
            auto rows=it->execution->progress();
            publish_group(it->items,std::move(rows),it->step.operation(),it->execution.get());
            for(const auto& item:it->items) {
                auto& lane=lanes[item.key.lane]; --lane.executing;
                lane.reserved_output-=item.budget.output_tokens;
                if(item.step) ++lane.publish_step;
                if(!lane.executing) { lane.tail.reset(); lane.frozen=false; }
            }
            it=batches.erase(it); ++capacity_epoch; progress=true;
        } else {
            if(Clock::now()-it->started>=std::chrono::milliseconds(profile.batch_timeout_ms))
                throw std::runtime_error("batch execution deadline expired without settlement");
            ++it;
        }
    }
    return progress;
}
}
