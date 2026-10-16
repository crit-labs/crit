#include "crit/core.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, std::string_view text) {
    try { action(); } catch(const std::exception& e) {
        require(std::string_view(e.what()).find(text)!=std::string_view::npos,"unexpected rejection"); return;
    }
    throw std::runtime_error("expected rejection missing");
}
// Release contract: once wait()/wait_output() returns, or a handle's destructor returns, the request's host charges
// are already back in the ledger as seen from that thread.
bool settled(const Engine& e) { return e.usage(Memory::host).live==0; }
struct State {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> admit_enabled{false}, release_work{false}, release_retirement{false};
    std::atomic<bool> fail_drain{false}, fail_quiesce{false}, quiet{false}, drained{false};
    std::atomic<int> initialized{0}, opened{0}, closed{0}, destroyed{0}, unsafe{0}, quiesced{0}, drains{0};
    std::optional<Resources> ledger; // the backend's retained copy, which outlives the engine
    std::atomic<int> retirement_checks{0}, score_calls{0};
    std::atomic<bool> release_score{false};
    bool hold_work=false, hold_retirement=false, hold_score=false, scoring=true, hold_close=false;
    bool mixed=false;
    bool hints=false;
    bool advice=false;
    std::vector<PrefillWindow> windows;
    std::map<Token,SchedulingHint> hint_table;
    std::vector<std::size_t> prefills_at_open;
    std::vector<std::vector<WorkBudget>> mixed_groups;
    std::vector<std::vector<RequestKey>> mixed_keys;
    std::atomic_bool release_blocked{false};
    std::atomic<int> blocked_checks{0};
    bool close_entered=false, release_close=false;
    std::string fault;
    std::uint32_t declared_concurrency=8;
    bool work_entered=false, retirement_entered=false, score_entered=false;
    int opened_at_score=0;
    // Fake queued backend writes into the allocation returned by score().
    std::span<float> score_destination;
    float score_value=-0.25F;
    std::vector<RequestKey> opened_keys;
    std::vector<std::uint64_t> prefill_order;
    std::vector<std::vector<std::uint64_t>> groups;
    std::map<std::uint64_t,std::vector<Token>> committed;
    std::vector<DeviceTiming> timings;
    std::atomic<int> timing_calls{0};
    void wait_for(const std::function<bool()>& predicate) {
        std::unique_lock lock(mutex);
        require(cv.wait_for(lock,std::chrono::seconds(3),predicate),"fixture boundary wait expired");
    }
};
struct FixtureBackend final:Backend {
    std::shared_ptr<State> state;
    explicit FixtureBackend(std::shared_ptr<State> s):state(std::move(s)) {}
    void initialize(Resources& resources) override { state->ledger=resources; ++state->initialized; }
    void bind_thread() override {}
    void drain() override {
        ++state->drains;
        if(state->fail_drain) throw std::runtime_error("injected drain failure");
        std::fill(state->score_destination.begin(),state->score_destination.end(),state->score_value);
        state->score_destination={};
        state->drained=true;
    }
};
struct FixtureSession final:ConcurrentSession {
    std::shared_ptr<State> state;
    RequestView view;
    Reservation storage;
    std::uint32_t frontier=0, staged_frontier=0, generated=0;
    std::vector<Token> candidates;
    bool settled=false, was_closed=false;
    FixtureSession(std::shared_ptr<State> s, RequestView v, Resources& resources):
        state(std::move(s)),view(v),storage(resources.reserve(Memory::device,4)) {
        ++state->opened;
        std::lock_guard lock(state->mutex); state->opened_keys.push_back(v.key);
        state->prefills_at_open.push_back(state->prefill_order.size()); state->cv.notify_all();
    }
    ~FixtureSession() override { ++state->destroyed; if(!was_closed) ++state->unsafe; }
    void close(Finish) noexcept override {
        std::unique_lock lock(state->mutex);
        if(state->hold_close) {
            state->close_entered=true; state->cv.notify_all();
            if(!state->cv.wait_for(lock,std::chrono::seconds(3),[&]{return state->release_close;})) ++state->unsafe;
        }
        if(was_closed || (!settled && !(state->quiet && state->drained))) ++state->unsafe;
        was_closed=true; ++state->closed;
        state->cv.notify_all();
    }
};
struct FixtureReceipt final:Retirement {
    FixtureSession& session;
    explicit FixtureReceipt(FixtureSession& s):session(s) {}
    bool ready() override {
        auto& s=*session.state;
        if(s.fault=="receipt_throw") throw std::runtime_error("injected receipt failure");
        if(s.hold_retirement && session.view.key.request==1 && !s.release_retirement) {
            std::lock_guard lock(s.mutex); ++s.retirement_checks; s.retirement_entered=true; s.cv.notify_all(); return false;
        }
        session.settled=true; return true;
    }
};
struct FixtureReadiness final:Readiness {
    std::shared_ptr<State> state;
    explicit FixtureReadiness(std::shared_ptr<State> s):state(std::move(s)) {}
    void wait() override {
        std::unique_lock lock(state->mutex); state->work_entered=true; state->cv.notify_all();
        require(state->cv.wait_for(lock,std::chrono::seconds(3),[&]{return state->release_work.load();}),"fixture readiness wait expired");
    }
};
struct ScoreContext final:RequestContext {
    float value=-0.75F;
    std::size_t host_bytes() const noexcept override { return sizeof(value); }
};
struct FixtureModel final:ConcurrentModel {
    std::shared_ptr<State> state;
    Resources& resources;
    Reservation weights;
    Reservation score_storage;
    FixtureReadiness readiness;
    FixtureModel(std::shared_ptr<State> s, Resources& r):state(std::move(s)),resources(r),
        weights(r.reserve(Memory::device,12)),readiness(state) {}
    ModelInfo info() const override {
        if(state->fault=="info_throw") throw std::runtime_error("injected declaration failure");
        ModelInfo result{"independent.concurrent.contract.fixture",128,64,8,{63},true,state->scoring,state->declared_concurrency,{},true};
        result.mixed_batches=state->mixed;
        result.scheduling_hints=state->hints;
        result.prefill_advice=state->advice;
        result.decode_token_work=3;
        return result;
    }
    SchedulingHint scheduling_hint(const RequestView& v) override {
        auto found=state->hint_table.find(v.prompt.front());
        return found==state->hint_table.end() ? SchedulingHint{} : found->second;
    }
    std::uint32_t prefill_limit(const WorkItem&,const PrefillWindow& w) override {
        state->windows.push_back(w);
        return state->fault=="advice" ? w.token_limit+1 : std::min(2u,w.token_limit);
    }
    Admission admission(const RequestView& v) override {
        if(v.prompt.front()==90) return Admission::infeasible;
        if(v.prompt.front()==89 && !state->release_blocked) {
            std::lock_guard lock(state->mutex); ++state->blocked_checks; state->cv.notify_all();
            return Admission::deferred;
        }
        return state->admit_enabled ? Admission::ready : Admission::deferred;
    }
    std::unique_ptr<ConcurrentSession> open(const RequestView& view) override {
        score_storage={};
        if(state->fault=="open_throw") throw std::runtime_error("injected open failure");
        return std::make_unique<FixtureSession>(state,view,resources);
    }
    RowProgress prefill(const WorkItem& item) override {
        auto& s=static_cast<FixtureSession&>(*item.session);
        require(s.view.key==item.key,"prefill crossed request identity");
        s.staged_frontier=s.frontier+item.budget.prefill_tokens;
        s.candidates.clear();
        {
            std::lock_guard lock(state->mutex); state->prefill_order.push_back(item.key.request); state->cv.notify_all();
        }
        auto consumed=item.budget.prefill_tokens;
        if(state->fault=="prefill_shape") ++consumed;
        return {item.key,{consumed,s.staged_frontier==s.view.prompt.size(),s.candidates,
            state->hold_work && item.key.request==1 ? &readiness : nullptr}};
    }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        std::vector<std::uint64_t> group;
        for(const auto& item:items) {
            auto& s=static_cast<FixtureSession&>(*item.session);
            require(s.view.key==item.key && item.budget.prefill_complete && !item.budget.prefill_tokens,"decode membership/state mismatch");
            const auto count=std::min(item.budget.output_tokens,std::uint32_t(s.view.prompt.front()%3+1));
            s.candidates.resize(count);
            for(std::uint32_t i=0;i<count;++i) s.candidates[i]=s.view.prompt.front()+1+Token(s.generated+i);
            if(state->fault=="token_range") s.candidates.front()=128;
            if(state->fault=="output_budget") s.candidates.resize(item.budget.output_tokens+1,1);
            if(state->fault=="stalled") s.candidates.clear();
            rows.push_back({item.key,{0,true,s.candidates,nullptr}}); group.push_back(item.key.request);
        }
        {
            std::lock_guard lock(state->mutex); state->groups.push_back(group); state->cv.notify_all();
        }
        std::reverse(rows.begin(),rows.end()); // Returned order must not define identity.
        if(state->fault=="stale") ++rows.front().key.generation;
        if(state->fault=="duplicate" && rows.size()>1) rows.back().key=rows.front().key;
        if(state->fault=="missing") rows.pop_back();
        return rows;
    }
    std::span<const DeviceTiming> device_timings() override {
        ++state->timing_calls;
        if(state->fault=="timings") throw std::runtime_error("injected timing failure");
        return state->timings;
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        std::vector<WorkBudget> budgets;
        for(const auto& item:items) {
            budgets.push_back(item.budget);
            auto bounded=item;
            bounded.budget.output_tokens=std::min(item.budget.output_tokens,item.budget.token_work);
            auto row=item.budget.prefill_complete ? decode(std::span<const WorkItem>(&bounded,1)).front() : prefill(item);
            row.progress.token_work=item.budget.prefill_complete ? std::uint32_t(row.progress.tokens.size()) : row.progress.prompt_tokens;
            if(state->fault=="mixed_work") row.progress.token_work=item.budget.token_work+1;
            if(state->fault=="mixed_zero") row.progress.token_work=0;
            rows.push_back(row);
        }
        {
            std::lock_guard lock(state->mutex); state->mixed_groups.push_back(budgets);
            std::vector<RequestKey> keys;
            for(const auto& item:items) keys.push_back(item.key);
            state->mixed_keys.push_back(std::move(keys));
        }
        std::reverse(rows.begin(),rows.end());
        if(state->fault=="mixed_stale") ++rows.front().key.generation;
        if(state->fault=="mixed_duplicate" && rows.size()>1) rows.back().key=rows.front().key;
        return rows;
    }
    void commit(std::span<const CommitDecision> group) override {
        // Failure may mutate one session. Core must publish none of this group.
        if(state->fault=="commit_throw" && !static_cast<FixtureSession&>(*group.front().session).candidates.empty()) {
            ++static_cast<FixtureSession&>(*group.front().session).generated;
            throw std::runtime_error("injected grouped commit failure");
        }
        std::lock_guard lock(state->mutex);
        for(const auto& decision:group) {
            auto& s=static_cast<FixtureSession&>(*decision.session);
            require(s.view.key==decision.key && decision.accepted<=s.candidates.size(),"invalid commit mapping");
            require(!decision.cancelled || (decision.terminal && !decision.accepted),"cancelled commit not terminal/empty");
            auto& history=state->committed[decision.key.request];
            history.insert(history.end(),s.candidates.begin(),s.candidates.begin()+decision.accepted);
            s.generated+=decision.accepted; s.frontier=s.staged_frontier;
            std::fill(s.candidates.begin(),s.candidates.end(),-1); // Invalidate every borrowed row.
        }
    }
    std::unique_ptr<Retirement> retire(RequestKey key, ConcurrentSession& session, Finish) override {
        auto& s=static_cast<FixtureSession&>(session);
        require(key==s.view.key,"retirement crossed generation");
        if(state->fault=="null_receipt") return {};
        return std::make_unique<FixtureReceipt>(s);
    }
    void quiesce() override {
        ++state->quiesced;
        if(state->fail_quiesce) throw std::runtime_error("injected quiesce failure");
        state->quiet=true;
    }
    std::vector<float> score(std::span<const Token> prompt,std::uint32_t first) override {
        // Reserve the deliberately oversized returned scratch. The next normal
        // request callback can release this charge after Core's successful drain.
        score_storage={};
        score_storage=resources.reserve(Memory::host,64*sizeof(float));
        std::vector<float> scores; scores.reserve(64); scores.resize(prompt.size()-first,0.0F);
        {
            std::unique_lock lock(state->mutex);
            require(state->opened==state->closed,"scoring overlapped live generation state");
            ++state->score_calls; state->score_entered=true; state->opened_at_score=state->opened.load(); state->cv.notify_all();
            if(state->hold_score) require(state->cv.wait_for(lock,std::chrono::seconds(3),[&]{return state->release_score.load();}),"fixture scoring wait expired");
        }
        if(state->fault=="score_throw") throw std::runtime_error("injected scoring failure");
        if(state->fault=="score_count") scores.push_back(0.0F);
        if(state->fault=="score_nan") state->score_value=std::numeric_limits<float>::quiet_NaN();
        if(state->fault=="score_drain") state->fail_drain=true;
        state->score_destination=scores;
        return scores;
    }
    std::vector<float> score_request(std::span<const Token> prompt,std::uint32_t first,const RequestOptions& options) override {
        if(options.context) {
            const auto* context=dynamic_cast<const ScoreContext*>(options.context.get());
            require(context!=nullptr,"wrong score context");
            state->score_value=context->value;
            return score(prompt,first);
        }
        return Model::score_request(prompt,first,options); // Exercise the legacy hook too.
    }
    std::vector<Token> tokenize(std::string_view) override { return {1}; }
    std::string detokenize(std::span<const Token>) override { return "fixture"; }
};
Profile profile(std::uint32_t concurrency=3, std::uint32_t pending=3) { return {64,2,8192,8192,concurrency,pending,3000,3000}; }
std::unique_ptr<Engine> engine(const std::shared_ptr<State>& state, Profile p=profile()) {
    return std::make_unique<Engine>(p,std::make_shared<FixtureBackend>(state),[state](const Profile&,Services services){
        return std::make_unique<FixtureModel>(state,services.resources);
    });
}
std::vector<Token> expected(Token start, std::uint32_t count) {
    std::vector<Token> v; for(std::uint32_t i=0;i<count;++i) v.push_back(start+1+Token(i)); return v;
}
void mixed_batches() {
    for(auto cap:{1U,3U,7U}) {
        auto s=std::make_shared<State>(); s->mixed=true;
        auto p=profile(); p.max_batch_tokens=cap; p.trace.capacity=512; p.host_limit=131072;
        auto e=engine(s,p);
        auto a=e->submit({1},7), b=e->submit(std::vector<Token>(12,10),5), c=e->submit({20,20,20},8);
        s->admit_enabled=true;
        const auto ra=a.wait(), rb=b.wait(), rc=c.wait();
        require(ra.tokens==expected(1,7) && rb.tokens==expected(10,5) && rc.tokens==expected(20,8),"mixed output differs from separate scheduling");
        bool mixed=false;
        for(const auto& group:s->mixed_groups) {
            std::uint32_t used=0; bool prefill=false, decode=false;
            for(const auto& budget:group) {
                used+=budget.token_work;
                prefill|=!budget.prefill_complete; decode|=budget.prefill_complete;
                require(budget.token_work && budget.prefill_tokens<=budget.token_work,"invalid row work grant");
            }
            require(used<=cap,"aggregate mixed token quota exceeded");
            mixed|=prefill && decode;
        }
        if(cap>=3) require(mixed,"prefill and decode never shared a group");
        require(ra.timing.prefill_tokens==1 && rb.timing.prefill_tokens==12,"mixed prompt ledger changed");
        const auto snapshot=e->trace();
        require(std::any_of(snapshot.events.begin(),snapshot.events.end(),[](const auto& x){return x.phase==TracePhase::mixed;}),"missing mixed trace");
        e.reset(); require(s->closed==3 && !s->unsafe,"mixed retirement failed");
    }
    auto unsupported=std::make_shared<State>(); auto p=profile(); p.max_batch_tokens=4;
    rejects([&]{engine(unsupported,p);},"mixed token work");
    for(const auto* fault:{"mixed_work","mixed_zero","mixed_stale","mixed_duplicate"}) {
        auto s=std::make_shared<State>(); s->mixed=true; s->fault=fault;
        auto e=engine(s,p); auto a=e->submit({1,1},3), b=e->submit({10,10},3);
        s->admit_enabled=true;
        rejects([&]{a.wait();}, fault==std::string_view("mixed_work") || fault==std::string_view("mixed_zero") ? "token work" : "identity");
        rejects([&]{b.wait();}, fault==std::string_view("mixed_work") || fault==std::string_view("mixed_zero") ? "token work" : "identity");
        require(s->committed.empty(),"invalid mixed group published");
        e.reset(); require(!s->unsafe,"failed mixed group released live state");
    }
    {
        auto s=std::make_shared<State>(); s->mixed=true; s->hold_work=true;
        auto e=engine(s,p); auto a=e->submit({1,1},3), b=e->submit({10,10},3);
        s->admit_enabled=true;
        s->wait_for([&]{return s->work_entered;});
        b.cancel();
        { std::lock_guard lock(s->mutex); s->release_work=true; s->cv.notify_all(); }
        const auto cancelled=b.wait();
        require(cancelled.finish==Finish::cancelled && cancelled.tokens.empty(),"mixed cancellation published tokens");
        require(a.wait().tokens==expected(1,3),"mixed cancellation changed another row");
        e.reset(); require(!s->unsafe,"mixed cancellation freed live state");
    }
    {
        auto s=std::make_shared<State>(); s->mixed=true; s->fault="commit_throw";
        auto e=engine(s,p); auto a=e->submit({1},3), b=e->submit({10},3);
        s->admit_enabled=true;
        std::vector<Token> published;
        rejects([&]{a.wait([&](Token t){published.push_back(t);});},"grouped commit");
        rejects([&]{b.wait();},"grouped commit");
        require(published.empty(),"failed mixed commit published tokens");
        e.reset(); require(!s->unsafe,"mixed commit failure freed live state");
    }
}
void identities_and_fairness() {
    auto s=std::make_shared<State>(); auto e=engine(s);
    auto a=e->submit({1,1,1,1,1,1},7), b=e->submit({10,10,10,10,10,10},5), c=e->submit({20,20,20,20,20,20},8);
    require(a.id()==1 && b.id()==2 && c.id()==3,"request identities not monotonic");
    s->admit_enabled=true;
    const auto ra=a.wait(), rb=b.wait(), rc=c.wait();
    require(ra.request_id==1 && ra.tokens==expected(1,7) && rb.tokens==expected(10,5) && rc.tokens==expected(20,8),"independent histories/budgets crossed");
    for(const auto* r:{&ra,&rb,&rc})
        require(r->timing.prefill_tokens==6 && r->timing.first_token_ns<=r->timing.total_ns && r->timing.decode_steps>0,
                "concurrent phase timing wrong");
    {
        std::lock_guard lock(s->mutex);
        require(s->prefill_order==std::vector<std::uint64_t>({1,2,3,1,2,3,1,2,3}),"multi-prefill rotation failed");
        require(std::any_of(s->groups.begin(),s->groups.end(),[](const auto& v){return v.size()==3;}),"compact multi-row execution never occurred");
        require(s->committed[1]==ra.tokens && s->committed[2]==rb.tokens && s->committed[3]==rc.tokens,"publication did not preserve commit copies");
    }
    require(s->initialized==1 && s->opened==3 && s->closed==3 && s->unsafe==0,"shared backend/request ownership failed");
    require(e->usage(Memory::device).live==12 && settled(*e),"completed handles left request storage");
    auto next=e->submit({30},2); const auto id=next.id(); require(next.wait().tokens==expected(30,2) && id==4,"engine not reusable");
    { std::lock_guard lock(s->mutex); require(s->opened_keys.back().generation==2,"reused lane lost generation"); }
    e.reset(); require(s->destroyed==4 && s->unsafe==0,"final session ownership failed");
}
void prefill_decode_interleave() {
    auto s=std::make_shared<State>(); auto e=engine(s,profile(2,1));
    auto a=e->submit({1},8), b=e->submit(std::vector<Token>(12,20),3); s->admit_enabled=true;
    require(a.wait().tokens==expected(1,8) && b.wait().tokens==expected(20,3),"mixed prompt lengths failed");
    // The short request completed while long prefill still rotated through chunks.
    std::lock_guard lock(s->mutex);
    require(s->groups.size()>2 && s->groups.front()==std::vector<std::uint64_t>({1}),"decode did not interleave with staged prefill");
}
void queued_cancel_and_limits() {
    auto s=std::make_shared<State>(); auto e=engine(s,profile(1,1));
    auto a=e->submit({1},2), b=e->submit({10},2);
    rejects([&]{ e->submit({20},2); },"allowance");
    b.cancel(); require(b.wait().finish==Finish::cancelled && s->opened==0,"queued cancellation opened model state");
    auto c=e->submit({20},2); s->admit_enabled=true;
    require(a.wait().tokens==expected(1,2) && c.wait().tokens==expected(20,2),"FIFO after queued cancellation failed");
}
void pending_timeout() {
    auto s=std::make_shared<State>(); auto p=profile(1,0); p.pending_timeout_ms=20; auto e=engine(s,p);
    auto a=e->submit({1},2); rejects([&]{a.wait();},"pending request deadline");
    require(s->opened==0 && s->quiesced==0,"queue expiry touched physical state");
    s->admit_enabled=true; require(e->submit({10},2).wait().tokens==expected(10,2),"queue expiry poisoned healthy engine");
}
void retained_handles() {
    auto s=std::make_shared<State>(); s->admit_enabled=true; auto e=engine(s,profile(1,0));
    auto a=e->submit({1},2);
    s->wait_for([&]{return s->closed.load()==1;});
    rejects([&]{e->submit({10},2);},"allowance");
    require(e->usage(Memory::host).live>0,"unconsumed result was not charged");
    require(a.wait().tokens==expected(1,2),"retained result lost");
    require(settled(*e),"consumed result stayed charged");
    // A handle that kept the runtime alive shuts it down before its wait returns.
    auto b=e->submit({10},2); e.reset(); require(b.wait().tokens==expected(10,2) && s->unsafe==0,"handle did not retain runtime");
    require(s->ledger->usage(Memory::host).live==0 && s->ledger->usage(Memory::device).live==0,"runtime shutdown left charges");
}
void isolated_retirement() {
    auto s=std::make_shared<State>(); s->hold_retirement=true; auto e=engine(s,profile(2,2));
    auto a=e->submit({1},2), b=e->submit({10},9); s->admit_enabled=true;
    s->wait_for([&]{return s->retirement_entered;});
    require(b.wait().tokens==expected(10,9),"retiring neighbor stopped decode");
    require(s->closed==1,"unsettled request closed early");
    auto c=e->submit({20},2); require(c.wait().tokens==expected(20,2),"settled lane was not reusable");
    { std::lock_guard lock(s->mutex); require(s->opened_keys.back().lane==1 && s->opened_keys.back().generation==2,"occupied retiring lane was reused"); }
    s->release_retirement=true; require(a.wait().tokens==expected(1,2) && s->unsafe==0,"delayed settlement lost state");
    require(settled(*e),"delayed retirement left request storage after wait");
}
void cancel_active_and_abandon() {
    auto s=std::make_shared<State>(); s->hold_work=true; s->hold_retirement=true; s->admit_enabled=true; auto e=engine(s,profile(2,2));
    auto a=e->submit({1},6); s->wait_for([&]{return s->work_entered;});
    auto b=e->submit({10},7); a.cancel(); s->release_work=true; s->cv.notify_all();
    s->wait_for([&]{return s->retirement_entered;});
    require(b.wait().tokens==expected(10,7) && s->closed==1,"cancellation disrupted neighbor or closed live state");
    s->release_retirement=true; const auto result=a.wait();
    require(result.finish==Finish::cancelled && result.tokens.empty(),"cancelled pending work was published");
    s->admit_enabled=false;
    { auto abandoned=e->submit({20},2); }
    require(s->opened==2 && s->unsafe==0,"abandoned queued handle opened state");
    require(settled(*e),"abandoned or cancelled handles retained request storage");
}
void streaming_cancel() {
    auto s=std::make_shared<State>(); s->admit_enabled=true; auto e=engine(s,profile(1,0));
    auto a=e->submit({1},30); bool signalled=false;
    const auto result=a.wait([&](Token){ if(!signalled) {signalled=true; a.cancel();} });
    require(signalled && !result.tokens.empty() && s->unsafe==0,"consumed streaming handle cancellation failed");
    // Commit races can finish before the callback. No promise of preempting a
    // committed window; final status may be length or cancelled.
}
void failed_completion_waits_for_close() {
    auto s=std::make_shared<State>(); s->fault="commit_throw"; s->hold_close=true;
    auto e=engine(s,profile(1,0));
    auto handle=e->submit({1},3); s->admit_enabled=true;
    auto waiter=std::async(std::launch::async,[request=std::move(handle)]() mutable {
        rejects([&]{request.wait();},"grouped commit");
    });
    s->wait_for([&]{return s->close_entered;});
    const auto before=waiter.wait_for(std::chrono::milliseconds(25));
    {
        std::lock_guard lock(s->mutex); s->release_close=true; s->cv.notify_all();
    }
    waiter.get();
    require(before==std::future_status::timeout,"failed handle completed before session close");
    require(s->closed==1 && s->destroyed==1 && s->unsafe==0 && e->usage(Memory::device).live==12,
            "failed completion retained settled session resources");
    require(settled(*e),"failed completion retained request storage");
}
void failed_group(std::string fault, std::string_view message) {
    auto s=std::make_shared<State>(); s->fault=std::move(fault); auto e=engine(s,profile(2,0));
    auto a=e->submit({1},3), b=e->submit({10},3); s->admit_enabled=true;
    std::vector<Token> seen;
    rejects([&]{a.wait([&](Token t){seen.push_back(t);});},message);
    rejects([&]{b.wait();},message);
    {
        std::lock_guard lock(s->mutex);
        // Earlier successful solo/batch groups remain published. Nothing from
        // the failed group may appear; commit failure need not undo model state.
        require(seen==s->committed[a.id()] && s->quiesced==1 && s->unsafe==0,"failed group published or skipped settlement");
    }
    require(e->usage(Memory::device).live==12,"failed group retained settled sessions");
    require(settled(*e),"settled failure retained request storage after wait");
    rejects([&]{e->submit({20},1);},"unavailable");
}
void retirement_timeout(bool quiesce_failure, bool drain_failure) {
    auto s=std::make_shared<State>(); s->hold_retirement=true;
    auto p=profile(1,0); p.retirement_timeout_ms=20; auto e=engine(s,p);
    s->fail_quiesce=quiesce_failure; s->fail_drain=drain_failure; s->drained=false;
    auto a=e->submit({1},2); s->admit_enabled=true;
    rejects([&]{a.wait();},"retirement deadline");
    const auto before=quarantined_engines();
    // Unproved settlement keeps the request and its charges in the quarantined runtime; otherwise wait() returns
    // after they were released.
    if(quiesce_failure || drain_failure)
        require(s->closed==0 && s->destroyed==0 && e->usage(Memory::host).live>0,"unknown readers were released");
    else require(s->closed==1 && s->unsafe==0 && settled(*e),"timeout did not quiesce/drain before close");
    e.reset();
    if(quiesce_failure || drain_failure) require(quarantined_engines()==before+1 && s->closed==0,"failed settlement owner not quarantined");
    require(s->unsafe==0,"timeout freed live owner");
}
void admission_and_stop() {
    auto s=std::make_shared<State>(); s->admit_enabled=true; auto e=engine(s,profile(1,1));
    rejects([&]{e->submit({90},2).wait();},"infeasible");
    auto a=e->submit({60},7); const auto result=a.wait();
    require(result.tokens==expected(60,3) && result.finish==Finish::stop,"per-row stop trimming failed");
    require(e->submit({1},0).wait().tokens.empty() && s->opened==1,"zero output opened state");
    rejects([&]{e->submit(std::vector<Token>(64,1),1);},"context");
    rejects([&]{e->submit({128},1);},"vocabulary");
    require(e->submit_score({1,2},1).wait().scores==std::vector<float>({-0.25F}),"legacy scoring hook not used");
}
void scoring_barrier() {
    auto s=std::make_shared<State>(); s->admit_enabled=true; s->hold_retirement=true; s->hold_score=true;
    auto p=profile(2,2); p.host_limit=1<<20; p.trace={2048,0}; auto e=engine(s,p);
    auto a=e->submit({1},2); s->wait_for([&]{return s->retirement_entered;});
    auto context=std::make_shared<ScoreContext>(); std::weak_ptr<ScoreContext> weak=context;
    RequestOptions options; options.context=context;
    auto b=e->submit_score({10,11,12},1,std::move(options)); context.reset();
    auto c=e->submit({20},2);
    const auto scans=s->retirement_checks.load();
    s->wait_for([&]{return s->retirement_checks>=scans+2;});
    require(s->score_calls==0 && s->opened==1 && !weak.expired(),"score barrier passed unfinished retirement");
    s->release_retirement=true;
    s->wait_for([&]{return s->score_entered;});
    require(s->opened_at_score==1 && s->opened==1 && s->closed==1,"later generation passed score barrier");
    s->release_score=true; s->cv.notify_all();
    const auto ra=a.wait(), rb=b.wait(), rc=c.wait();
    require(ra.tokens==expected(1,2) && rc.tokens==expected(20,2),"scoring changed generation output");
    require(rb.scores==std::vector<float>({-0.75F,-0.75F}) && rb.tokens.empty() && rb.finish==Finish::length,"scores published before queued writes settled");
    require(rb.scores.capacity()<64,"backend scratch capacity escaped into public result");
    require(weak.expired() && s->unsafe==0,"score context retained after later request completed");
    {
        std::lock_guard lock(s->mutex);
        require(s->opened_keys.back().lane==0 && s->opened_keys.back().generation==2,"scoring consumed a generation lane identity");
    }
    const auto snapshot=e->trace();
    require(std::any_of(snapshot.events.begin(),snapshot.events.end(),[&](const auto& event) {
        return event.key.request==b.id() && !event.key.generation && event.phase==TracePhase::score && event.mark==TraceMark::end;
    }),"score trace confused scoring with a generation lane");
}
void scoring_pending(bool timeout) {
    auto s=std::make_shared<State>(); s->admit_enabled=true; s->hold_retirement=true;
    auto p=profile(2,1); if(timeout) p.pending_timeout_ms=30; auto e=engine(s,p);
    auto a=e->submit({1},2); s->wait_for([&]{return s->retirement_entered;});
    auto b=e->submit_score({1,2},1);
    if(timeout) rejects([&]{b.wait();},"pending request deadline");
    else {b.cancel(); require(b.wait().finish==Finish::cancelled,"queued score cancellation ignored");}
    require(s->score_calls==0,"cancelled or expired score invoked model");
    require(e->submit({20},2).wait().tokens==expected(20,2),"removed score still blocked later generation");
    s->release_retirement=true; a.wait();
    require(e->submit_score({1,2},1).wait().scores==std::vector<float>({-0.25F}) && s->quiesced==0,"queued score failure poisoned engine");
}
void scoring_active_cancel() {
    auto s=std::make_shared<State>(); s->admit_enabled=true; s->hold_score=true; auto e=engine(s,profile(1,1));
    auto a=e->submit_score({1,2,3},1); s->wait_for([&]{return s->score_entered;});
    auto b=e->submit({20},2); a.cancel(); s->release_score=true; s->cv.notify_all();
    const auto result=a.wait();
    require(result.finish==Finish::cancelled && result.scores.empty(),"active cancelled score published values");
    require(b.wait().tokens==expected(20,2) && s->score_destination.empty() && s->quiesced==0,"score cancellation skipped drain or poisoned generation");
    require(settled(*e),"cancelled score retained request storage");
}
void scoring_failure(std::string fault,std::string_view message,bool failed_quiesce=false) {
    auto s=std::make_shared<State>(); s->admit_enabled=true; s->hold_score=true; s->fault=std::move(fault);
    auto e=engine(s,profile(1,1)); s->fail_quiesce=failed_quiesce;
    auto context=std::make_shared<ScoreContext>(); std::weak_ptr<ScoreContext> weak=context;
    RequestOptions options; options.context=context;
    auto a=e->submit_score({1,2,3},1,std::move(options)); context.reset();
    s->wait_for([&]{return s->score_entered;});
    auto b=e->submit({20},2); s->release_score=true; s->cv.notify_all();
    rejects([&]{a.wait();},message); rejects([&]{b.wait();},message);
    require(s->opened==0 && s->quiesced==1,"failed score admitted later generation or skipped quiesce");
    rejects([&]{e->submit({1},1);},"unavailable");
    const auto before=quarantined_engines(); const bool unsettled=s->fail_drain || failed_quiesce;
    if(unsettled) require(!weak.expired(),"unsettled scoring context freed");
    e.reset();
    require(s->unsafe==0,"score failure freed live generation state");
    if(unsettled) {
        require(quarantined_engines()==before+1 && !weak.expired(),"score failure did not retain complete owner");
        if(s->fail_drain) {
            // A late reader still has the same allocation, even after engine
            // destruction. ASan also detects premature release at this access.
            require(s->score_destination.size()==2 && s->score_destination.front()==0.0F,"failed drain lost returned score storage");
            s->score_destination.front()=7.0F;
        }
    } else require(weak.expired() && quarantined_engines()==before,"settled score context leaked");
}
void scoring_capability_and_budget() {
    auto s=std::make_shared<State>(); s->scoring=false; auto e=engine(s);
    rejects([&]{e->submit_score({1,2},1);},"scoring capability");
    auto limited=std::make_shared<State>(); limited->admit_enabled=true; limited->hold_retirement=true;
    auto p=profile(1,1); p.host_limit=20; auto tight=engine(limited,p);
    auto a=tight->submit({1},2); limited->wait_for([&]{return limited->retirement_entered;});
    rejects([&]{tight->submit_score({1,2,3},1);},"ceiling");
    limited->release_retirement=true;
    require(a.wait().tokens==expected(1,2),"score admission consumed generation entitlement");
}
void startup_declarations() {
    auto s=std::make_shared<State>(); s->declared_concurrency=1;
    rejects([&]{engine(s,profile(2,0));},"concurrency declaration");
    auto p=profile(); p.max_concurrency=max_request_concurrency+1;
    rejects([&]{engine(s,p);},"profile");
    p=profile(); p.batch_policy=BatchPolicy::decode_first;
    rejects([&]{engine(s,p);},"profile");
    p=profile(); p.max_admission_bypasses=0;
    rejects([&]{engine(s,p);},"profile");
    auto failing=std::make_shared<State>(); failing->fault="info_throw";
    rejects([&]{engine(failing);},"declaration failure");
    require(failing->quiesced==1 && failing->quiet && failing->unsafe==0,"failed startup skipped model quiescence");
}
void staging_ceiling() {
    auto s=std::make_shared<State>(); auto p=profile(2,0); p.host_limit=28;
    auto e=engine(s,p);
    auto a=e->submit({1},3);
    rejects([&]{e->submit({10},3);},"ceiling");
    s->admit_enabled=true;
    require(a.wait().tokens==expected(1,3),"rejected admission starved accepted work");
    require(settled(*e),"completed staging reservation was retained");
    require(e->submit({10},3).wait().tokens==expected(10,3),"failed admission consumed an allowance");
    require(s->quiesced==0 && s->unsafe==0 && e->usage(Memory::host).peak==28,"staging entitlement escaped the ceiling");
}
void overload_preserves_active_budget() {
    auto s=std::make_shared<State>(); s->hold_work=true; s->admit_enabled=true;
    auto p=profile(1,0); p.host_limit=220;
    auto e=engine(s,p);
    auto active=e->submit({1},3);
    s->wait_for([&]{return s->work_entered;});
    const auto before=e->usage(Memory::host);
    rejects([&]{e->submit({10},50);},"allowance");
    const auto after=e->usage(Memory::host);
    // Peak detects even a short-lived reservation by the rejected submission.
    require(before.live==after.live && before.peak==after.peak,"overload charged the active request's budget");
    s->release_work=true; s->cv.notify_all();
    require(active.wait().tokens==expected(1,3),"overload failed accepted work");
    require(e->submit({10},3).wait().tokens==expected(10,3),"overload poisoned the engine");
}
void scaled_batches() {
    auto s=std::make_shared<State>(); s->mixed=true; s->declared_concurrency=32;
    auto p=profile(32,0); p.host_limit=4*1024*1024; p.trace={8192,0}; p.max_batch_tokens=64;
    auto e=engine(s,p); std::vector<Handle> requests;
    for(int i=0;i<32;++i) requests.push_back(e->submit({Token(i%20+1)},3));
    s->admit_enabled=true;
    for(int i=0;i<32;++i) require(requests[i].wait().tokens==expected(Token(i%20+1),3),"large batch output mismatch");
    const auto trace=e->trace();
    std::map<std::uint64_t,std::set<std::uint32_t>> traced;
    for(const auto& event:trace.events) if(event.phase==TracePhase::mixed && event.mark==TraceMark::begin)
        traced[event.operation].insert(event.key.lane);
    require(std::any_of(traced.begin(),traced.end(),[](const auto& group){return group.second.size()==32;}),"trace truncated large batch");
    require(trace.overwritten==0 && trace.dropped==0 && s->unsafe==0,"large batch lifetime/trace failure");
}
void decode_priority() {
    auto s=std::make_shared<State>(); s->mixed=true;
    auto p=profile(2,0); p.max_batch_tokens=1; p.batch_policy=BatchPolicy::decode_first; p.prefill_starvation_steps=2;
    auto e=engine(s,p); auto a=e->submit({1},20), b=e->submit({10,11,12,13},4);
    s->admit_enabled=true;
    require(a.wait().tokens==expected(1,20) && b.wait().tokens==expected(10,4),"decode-first changed outputs");
    require(s->mixed_groups[1][0].prefill_complete && s->mixed_keys[1][0].request==a.id(),"decode did not receive priority");
    require(!s->mixed_groups[2][0].prefill_complete && s->mixed_keys[2][0].request==b.id(),"aged prefill did not receive its grant");
    std::size_t last=0, chunks=0;
    for(std::size_t i=0;i<s->mixed_groups.size();++i) {
        require(s->mixed_groups[i][0].token_work==1,"decode-first exceeded quota");
        if(s->mixed_keys[i][0].request==b.id() && !s->mixed_groups[i][0].prefill_complete) {
            require(i-last<=3,"prefill starved behind decoding"); last=i; ++chunks;
        }
    }
    require(chunks==4,"long prefill was not chunked");
}
void admission_backfill() {
    for(auto lookahead:{0u,4u}) {
        auto s=std::make_shared<State>(); auto p=profile(1,3);
        p.admission_lookahead=lookahead; p.max_admission_bypasses=2;
        auto e=engine(s,p);
        auto head=e->submit({89},1), a=e->submit({1},1), b=e->submit({10},1), c=e->submit({20},1);
        s->admit_enabled=true;
        if(lookahead) {
            require(a.wait().tokens==expected(1,1) && b.wait().tokens==expected(10,1),"ready backfill failed");
            s->wait_for([&]{return s->blocked_checks>=6;});
            require(s->opened==2,"capped head was bypassed again");
            head.cancel(); require(head.wait().finish==Finish::cancelled,"deferred head cancellation failed");
            require(c.wait().tokens==expected(20,1),"cancelling blocked head did not unblock queue");
        } else {
            s->wait_for([&]{return s->blocked_checks>=6;});
            require(s->opened==0,"default FIFO backfilled"); s->release_blocked=true;
            require(head.wait().tokens==expected(89,1),"head failed after readiness changed");
            a.wait(); b.wait(); c.wait();
            require(s->opened_keys.front().request==head.id(),"FIFO admission order changed");
        }
    }
    auto s=std::make_shared<State>(); auto p=profile(2,2); p.admission_lookahead=4;
    auto e=engine(s,p); auto head=e->submit({89},1), score=e->submit_score({1,2,3},1), tail=e->submit({1},2);
    s->admit_enabled=true; s->wait_for([&]{return s->blocked_checks>=6;});
    require(s->opened==0 && s->score_calls==0,"backfill crossed score barrier");
    head.cancel(); head.wait(); score.wait(); tail.wait();
    require(s->opened_at_score==0 && s->score_calls==1,"score barrier ordering changed");
}
void many_aged_prefills() {
    auto s=std::make_shared<State>(); s->mixed=true;
    auto p=profile(4,0); p.max_batch_tokens=1; p.batch_policy=BatchPolicy::decode_first; p.prefill_starvation_steps=1;
    auto e=engine(s,p); auto decoder=e->submit({1},30);
    std::vector<Handle> prefills;
    for(int i=0;i<3;++i) prefills.push_back(e->submit(std::vector<Token>(12,10+i),1));
    s->admit_enabled=true;
    require(decoder.wait().tokens==expected(1,30),"aged prefills changed decoder output");
    for(int i=0;i<3;++i) require(prefills[i].wait().tokens==expected(10+i,1),"multiple prefills failed");
    std::size_t last_decode=0, decodes=0;
    for(std::size_t i=0;i<s->mixed_groups.size();++i)
        if(s->mixed_keys[i][0].request==decoder.id() && s->mixed_groups[i][0].prefill_complete) {
            // Other prefills eventually become decoders too; the decoder cursor
            // grants each of at most four live decoders a turn.
            require(i-last_decode<=8,"aged prefills starved the decoder"); last_decode=i; ++decodes;
        }
    require(decodes==30,"decoder did not finish");
}
void latency_advice() {
    for(bool mixed:{false,true}) {
        auto s=std::make_shared<State>(); s->advice=true; s->mixed=mixed;
        auto p=profile(2,0); p.prefill_chunk=8; p.prefill_target_ns=5000;
        if(mixed)p.max_batch_tokens=10;
        auto e=engine(s,p); auto a=e->submit({1,2,3,4,5,6,7},3),b=e->submit({10},10);
        s->admit_enabled=true; a.wait(); b.wait();
        require(s->windows.size()>=5,"Prefill advice not used");
        bool reached_later=false;
        for(const auto& w:s->windows) {
            require(w.target_ns==5000 && w.token_limit<=8,"Wrong prefill advice bounds");
            reached_later|=w.occupied_tokens>=4;
        }
        require(reached_later,"Prefill advice lacks occupied context");
    }
    auto s=std::make_shared<State>();s->advice=true;s->fault="advice";
    auto p=profile();p.prefill_target_ns=1;
    auto e=engine(s,p);auto a=e->submit({1},1);s->admit_enabled=true;
    rejects([&]{a.wait();},"advice exceeds");
}
void reuse_admission() {
    for(bool capped:{false,true}) {
        auto s=std::make_shared<State>(); s->hints=true;
        s->hint_table={{10,{1000000,0,0,false}},{20,{1000,0,0,false}},{30,{1,0,0,false}}};
        auto p=profile(1,3); p.admission_policy=AdmissionPolicy::reuse_aware;
        p.admission_lookahead=3; p.admission_starvation_ms=30000;
        p.max_admission_bypasses=capped ? 1u : 8u;
        auto e=engine(s,p);
        auto a=e->submit({10},1),b=e->submit({20},1),c=e->submit({30},1);
        s->admit_enabled=true; a.wait(); b.wait(); c.wait();
        require(s->opened_keys[0].request==3,"Reuse cost did not affect admission");
        require(s->opened_keys[1].request==(capped ? 1u : 2u),"Reuse bypass bound failed");
    }
    {
        auto s=std::make_shared<State>(); s->hints=true;
        s->hint_table={{10,{10,0,7,false}},{20,{10,0,7,false}}};
        auto p=profile(2,1); p.admission_policy=AdmissionPolicy::reuse_aware; p.admission_lookahead=2;
        auto e=engine(s,p); auto a=e->submit({10},2),b=e->submit({20},2);
        s->admit_enabled=true; a.wait(); b.wait();
        require(s->prefills_at_open[1]>=1,"Duplicate pending prefix computed together");
    }
    {
        auto s=std::make_shared<State>(); s->hints=true;
        s->hint_table={{10,{10,0,7,true}}};
        auto p=profile(1,0); p.admission_policy=AdmissionPolicy::reuse_aware; p.admission_starvation_ms=5;
        auto e=engine(s,p); auto a=e->submit({10},1); s->admit_enabled=true;
        require(a.wait().tokens==expected(10,1),"Pending affinity starved past deadline");
    }
    {
        auto s=std::make_shared<State>(); s->hints=true;
        auto p=profile(2,2); p.admission_policy=AdmissionPolicy::reuse_aware; p.admission_lookahead=4;
        auto e=engine(s,p); auto a=e->submit({10},2),score=e->submit_score({1,2,3},1),b=e->submit({20},2);
        s->admit_enabled=true; a.wait(); score.wait(); b.wait();
        require(s->opened_at_score==1,"Reuse scheduling crossed score barrier");
    }
    {
        auto s=std::make_shared<State>(); auto p=profile(); p.admission_policy=AdmissionPolicy::reuse_aware;
        rejects([&]{engine(s,p);},"scheduling hints");
    }
}
void concurrent_tracing() {
    auto s=std::make_shared<State>();
    auto p=profile(2,0); p.host_limit=1<<20; p.trace={2048,0};
    auto e=engine(s,p);
    auto a=e->submit({1},3), b=e->submit({10},4);
    s->admit_enabled=true;
    a.wait(); b.wait();
    const auto snapshot=e->trace();
    std::map<std::uint64_t,std::vector<RequestKey>> groups;
    for(const auto& event:snapshot.events) if(event.phase==TracePhase::decode && event.mark==TraceMark::begin)
        groups[event.operation].push_back(event.key);
    bool grouped=false;
    for(const auto& [operation,keys]:groups) {
        require(operation>0,"group operation lacks identity");
        if(keys.size()==2) {
            grouped=true;
            require(keys[0].request!=keys[1].request && keys[0].lane!=keys[1].lane,
                    "grouped trace aliases request identities");
        }
    }
    require(grouped,"batched decode not correlated by operation");
    e->submit({20},2).wait();
    const auto reused=e->trace();
    require(std::any_of(reused.events.begin(),reused.events.end(),[](const auto& x) {
        return x.key.request==3 && x.key.generation==2 && x.phase==TracePhase::retirement && x.mark==TraceMark::end;
    }),"retired lane trace lost its generation");
    // Repeated snapshots contend only with the bounded recorder, never the model.
    std::jthread reader([&](std::stop_token stop){while(!stop.stop_requested()) {const auto copy=e->trace(); (void)copy;}});
    for(int i=0;i<8;++i) e->submit({1},3).wait();
    reader.request_stop(); reader.join();
    require(s->unsafe==0,"tracing changed concurrent retirement safety");
    auto untimed=std::make_shared<State>(); untimed->admit_enabled=true;
    untimed->timings={{DeviceStage::forward,0,10}};
    engine(untimed)->submit({1},3).wait();
    require(untimed->timing_calls==0,"device timings requested without tracing");
    auto timed=std::make_shared<State>();
    timed->timings={{DeviceStage::draft,0,700},{DeviceStage::accept,1,300}};
    auto te=engine(timed,p);
    auto c=te->submit({1},3), d=te->submit({10},4);
    timed->admit_enabled=true;
    c.wait(); d.wait();
    const auto devices=te->trace();
    std::map<std::uint64_t,std::set<std::uint64_t>> rows;
    std::size_t device_events=0;
    for(const auto& x:devices.events) if(x.phase==TracePhase::device) {
        ++device_events; rows[x.operation].insert(x.key.request);
        require(x.mark==TraceMark::point && ((x.stage==DeviceStage::draft && x.index==0 && x.duration_ns==700) ||
                (x.stage==DeviceStage::accept && x.index==1 && x.duration_ns==300)),"device timing payload");
        require(std::any_of(devices.events.begin(),devices.events.end(),[&](const auto& y) {
            return y.phase==TracePhase::step && y.mark==TraceMark::begin && y.operation==x.operation && y.key==x.key;
        }),"device timing outside its step operation");
    }
    require(device_events>=std::size_t(timed->timing_calls)*2 && !devices.dropped,"device timings not recorded per group");
    require(std::any_of(rows.begin(),rows.end(),[](const auto& x){return x.second.size()==2;}),"grouped device timings not keyed per row");
    auto failing=std::make_shared<State>(); failing->fault="timings";
    auto fe=engine(failing,p);
    auto f=fe->submit({1},3);
    failing->admit_enabled=true;
    const auto kept=f.wait();
    const auto lost=fe->trace();
    require(kept.tokens==expected(1,3) && failing->timing_calls>0,"throwing device timings failed the request");
    require(lost.dropped==std::uint64_t(failing->timing_calls.load()) &&
            std::none_of(lost.events.begin(),lost.events.end(),[](const auto& x){return x.phase==TracePhase::device;}),
            "throwing device timings not counted as dropped");
}
// Generation outputs on every concurrent path: decode groups, mixed execute groups, async batches and deferred
// rounds. Request prompt p generates p.front()+1+generated+i like FixtureModel, with recomputable log-probability
// rows; ids 120-127 are padding. The first decode row of a request whose prompt starts with 50 carries `fault`.
float own_logprob(Token t) { return -0.25F-0.001F*float(t); }
TokenLogprob alternative(Token t,std::uint32_t k) { return {Token((t+Token(k))%120),-0.25F*float(k+1)-0.001F*float(t)}; }
// Token t decodes to the letter 'A' + t % 26; token 99 fails and token 90 decodes to more text than a small ceiling.
struct Letters final:TextFrontend {
    std::vector<Token> tokenize(std::string_view) const override { return {}; }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string text;
        for(const auto t:tokens) {
            if(t==99) throw std::runtime_error("letter decoder failure");
            if(t==90) text.append(std::size_t(1)<<15,'w'); else text.push_back(char('A'+t%26));
        }
        return text;
    }
};
enum class Path { decode, mixed, async };
struct OutputState {
    std::mutex mutex;
    Path path=Path::decode;
    bool deferred=false;
    std::uint32_t per_step=2;
    std::string fault;
    std::atomic<bool> admit{true};
    std::shared_ptr<const TextFrontend> frontend;
    std::map<std::uint64_t,std::vector<Token>> committed;
    std::map<std::uint64_t,std::vector<std::uint32_t>> accepted;  // decode rows only
    std::set<std::uint64_t> cancelled;                             // requests with a cancelled commit row
    std::vector<std::size_t> open_live;
    std::vector<Finish> finishes;
    std::map<std::uint64_t,Finish> closed;                         // by request
    // Decode rows of the request whose prompt starts with `gated` wait, after its first published row, until
    // `sink_failed` is set and then pause 2 ms each, so its consumer's cancellation lands long before the budget ends.
    Token gated=-1;
    std::atomic<bool> sink_failed{false};
    int unsafe=0, batches=0;
};
struct OutputSession final:ConcurrentSession {
    std::shared_ptr<OutputState> state;
    RequestView view;
    std::uint32_t frontier=0, staged=0, generated=0, ready=0;
    bool decoding=false, faulted=false;
    std::vector<Token> tokens;
    std::vector<float> chosen;
    std::vector<TokenLogprob> top;
    OutputSession(std::shared_ptr<OutputState> s,const RequestView& v):state(std::move(s)),view(v) {}
    void close(Finish finish) noexcept override {
        std::lock_guard lock(state->mutex);
        state->finishes.push_back(finish); state->closed[view.key.request]=finish;
    }
};
struct NoWait final:Readiness { void wait() override {} };
struct Settled final:Retirement { bool ready() override { return true; } };
struct OutputHost final:Backend { void bind_thread() override {} void drain() override {} };
struct OutputModel final:ConcurrentModel {
    std::shared_ptr<OutputState> state;
    Resources& resources;
    NoWait readiness;
    std::map<std::uint32_t,OutputSession*> sessions; // by lane; worker-only
    OutputModel(std::shared_ptr<OutputState> s,Resources& r):state(std::move(s)),resources(r) {}
    ModelInfo info() const override {
        ModelInfo m{"output.concurrent.fixture",128,64,8,{100},true,false,8};
        m.valid_vocabulary=120; m.token_logprobs=true; m.max_top_logprobs=4;
        m.mixed_batches=state->path!=Path::decode; m.async_batches=state->path==Path::async; m.decode_token_work=4;
        m.request_context=true; m.scoring=true; m.query_scoring=true;
        return m;
    }
    std::shared_ptr<const TextFrontend> text_frontend() const override { return state->frontend; }
    Admission admission(const RequestView&) override { return state->admit ? Admission::ready : Admission::deferred; }
    std::unique_ptr<ConcurrentSession> open(const RequestView& view) override {
        { std::lock_guard lock(state->mutex); state->open_live.push_back(resources.usage(Memory::host).live); }
        auto session=std::make_unique<OutputSession>(state,view);
        sessions[view.key.lane]=session.get();
        return session;
    }
    RowProgress row(const WorkItem& item) {
        auto& s=static_cast<OutputSession&>(*item.session);
        Progress p;
        s.tokens.clear(); s.chosen.clear(); s.top.clear();
        s.decoding=item.budget.prefill_complete;
        if(!s.decoding) {
            s.staged=s.frontier+item.budget.prefill_tokens;
            p.prompt_tokens=item.budget.prefill_tokens; p.prefill_complete=s.staged==s.view.prompt.size();
            p.token_work=item.budget.token_work ? p.prompt_tokens : 0;
            return {item.key,p};
        }
        if(s.view.prompt.front()==state->gated && s.generated) {
            for(int i=0;i<5000 && !state->sink_failed;++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        auto count=std::min(item.budget.output_tokens,state->per_step);
        if(item.budget.token_work) count=std::min(count,item.budget.token_work);
        // A deferred round supplies rows for its whole grant and learns its count after readiness.
        const auto rows=state->deferred ? item.budget.output_tokens : count;
        const auto width=s.view.options->logprobs;
        for(std::uint32_t i=0;i<rows;++i) {
            const auto t=s.view.prompt.front()+1+Token(s.generated+i);
            s.tokens.push_back(t);
            if(!width) continue;
            s.chosen.push_back(own_logprob(t));
            for(std::uint32_t k=0;k<*width;++k) s.top.push_back(alternative(t,k));
        }
        if(s.view.prompt.front()==50 && !s.faulted) {
            s.faulted=true;
            const auto& f=state->fault;
            if(f=="lp_nan") s.chosen[0]=std::numeric_limits<float>::quiet_NaN();
            else if(f=="lp_count") s.chosen.resize(count-1);
            else if(f=="lp_unsorted") std::swap(s.top[0].logprob,s.top[1].logprob);
            else if(f=="lp_padding") s.top[1].token=121;
            else if(f=="token_padding") s.tokens[0]=121;
        }
        p.prefill_complete=true; p.token_work=item.budget.token_work ? count : 0;
        p.logprobs=s.chosen; p.top_logprobs=s.top;
        if(state->deferred) { s.ready=count; p.readiness=&readiness; p.deferred_tokens=true; }
        else p.tokens=std::span<const Token>(s.tokens).first(count);
        return {item.key,p};
    }
    RowProgress prefill(const WorkItem& item) override { return row(item); }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        for(const auto& item:items) rows.push_back(row(item));
        std::reverse(rows.begin(),rows.end()); // Returned order must not define identity.
        return rows;
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override { return decode(items); }
    std::span<const Token> ready_tokens(const RequestKey& key) override {
        auto& s=*sessions.at(key.lane);
        return std::span<const Token>(s.tokens).first(s.ready);
    }
    std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem> items) override;
    void commit(std::span<const CommitDecision> group) override {
        std::lock_guard lock(state->mutex);
        for(const auto& d:group) {
            auto& s=static_cast<OutputSession&>(*d.session);
            if(d.key!=s.view.key || d.accepted>s.tokens.size() || (d.cancelled && (d.accepted || !d.terminal))) ++state->unsafe;
            auto& history=state->committed[d.key.request];
            history.insert(history.end(),s.tokens.begin(),s.tokens.begin()+d.accepted);
            if(s.decoding) state->accepted[d.key.request].push_back(d.accepted);
            if(d.cancelled) state->cancelled.insert(d.key.request);
            s.generated+=d.accepted; s.frontier=s.staged;
            // A physical commit may invalidate every borrowed row.
            std::fill(s.tokens.begin(),s.tokens.end(),-1); std::fill(s.chosen.begin(),s.chosen.end(),1.0F);
            std::fill(s.top.begin(),s.top.end(),TokenLogprob{-1,1.0F});
        }
    }
    std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession&,Finish) override { return std::make_unique<Settled>(); }
    void quiesce() override {}
    std::vector<float> score_request(std::span<const Token> prompt,std::uint32_t first,const RequestOptions&) override {
        std::vector<float> out;
        for(std::size_t t=first;t<prompt.size();++t) out.push_back(own_logprob(prompt[t]));
        return out;
    }
    QueryScores score_queries(std::span<const Token> prompt,std::uint32_t first,const ScoreQueries& queries,const RequestOptions&) override {
        QueryScores out;
        for(std::size_t t=first;t<prompt.size();++t) {
            out.target.push_back(own_logprob(prompt[t]));
            for(std::uint32_t k=0;k<queries.per_target;++k) out.queries.push_back(-1.0F-0.125F*float(k));
            for(std::uint32_t k=0;k<queries.top_k;++k) out.top.push_back(alternative(prompt[t],k));
        }
        return out;
    }
};
// Completes during submit; Core polls it like any in-flight batch.
struct OutputBatch final:BatchExecution {
    OutputModel& model;
    std::vector<WorkItem> items;
    std::vector<RowProgress> rows;
    bool done=false;
    OutputBatch(OutputModel& m,std::span<const WorkItem> w):model(m),items(w.begin(),w.end()) {}
    void submit(Notify notify) override { rows=model.execute(items); done=true; notify(); }
    bool ready() override { return done; }
    std::vector<RowProgress> progress() override { return rows; }
    std::span<const Token> ready_tokens(const RequestKey& key) override { return model.ready_tokens(key); }
    void commit(std::span<const CommitDecision> group) override {
        model.commit(group);
        std::lock_guard lock(model.state->mutex); ++model.state->batches;
    }
    void quiesce() override {}
};
std::unique_ptr<BatchExecution> OutputModel::prepare_batch(std::span<const WorkItem> items) {
    return std::make_unique<OutputBatch>(*this,items);
}
std::unique_ptr<Engine> output_engine(const std::shared_ptr<OutputState>& state,std::size_t host=std::size_t(1)<<20) {
    Profile p{64,2,host,host,3,3,3000,3000};
    if(state->path!=Path::decode) p.max_batch_tokens=8;
    if(state->path==Path::async) p.max_inflight_batches=2;
    return std::make_unique<Engine>(p,std::make_shared<OutputHost>(),[state](const Profile&,Services services) {
        return std::make_unique<OutputModel>(state,services.resources);
    });
}
void check_output_rows(const Result& r,std::uint32_t width) {
    require(r.logprobs.size()==r.tokens.size() && r.top_logprobs.size()==r.tokens.size()*width,"concurrent logprob row counts");
    for(std::size_t i=0;i<r.tokens.size();++i) {
        require(r.logprobs[i]==own_logprob(r.tokens[i]),"concurrent chosen log-probability");
        for(std::uint32_t k=0;k<width;++k) {
            const auto e=alternative(r.tokens[i],k), a=r.top_logprobs[i*width+k];
            require(a.token==e.token && a.logprob==e.logprob,"concurrent ranked alternative");
        }
    }
}
void concurrent_logprobs() {
    for(const auto path:{Path::decode,Path::mixed,Path::async}) for(const bool deferred:{false,true}) {
        auto st=std::make_shared<OutputState>(); st->path=path; st->deferred=deferred;
        auto e=output_engine(st);
        RequestOptions two; two.logprobs=2;
        RequestOptions own; own.logprobs=0;
        auto a=e->submit({1},7,two), b=e->submit(std::vector<Token>(5,10),5,own), c=e->submit({20},6), d=e->submit({97},8,two);
        const auto ra=a.wait(), rb=b.wait(), rc=c.wait(), rd=d.wait();
        require(ra.tokens==expected(1,7) && rb.tokens==expected(10,5) && rc.tokens==expected(20,6),"concurrent logprob outputs");
        check_output_rows(ra,2);
        require(rb.logprobs.size()==5 && rb.top_logprobs.empty() && rc.logprobs.empty() && rc.top_logprobs.empty(),"requested row widths");
        // A stop token inside a multi-token row trims the rows with the tokens.
        require(rd.tokens==expected(97,3) && rd.finish==Finish::stop,"concurrent stop trim");
        check_output_rows(rd,2);
        {
            std::lock_guard lock(st->mutex);
            require(st->committed[a.id()]==ra.tokens && st->committed[d.id()]==rd.tokens && !st->unsafe,"concurrent commit copies");
            require(path!=Path::async || st->batches>0,"async batches not used");
        }
        require(settled(*e),"concurrent logprob storage not released");
    }
    // Admission charges the logprob rows, the text offsets and stream, and the staging row.
    auto st=std::make_shared<OutputState>(); st->frontend=std::make_shared<Letters>();
    auto e=output_engine(st);
    RequestOptions o; o.logprobs=3; o.text=true;
    require(e->submit({10},6,o).wait().text=="LMNOPQ","concurrent text");
    std::lock_guard lock(st->mutex);
    require(st->open_live.back()==sizeof(Token)+6*sizeof(Token)+6*sizeof(float)+6*3*sizeof(TokenLogprob)+6*sizeof(std::uint32_t)+6+256+
                                  6*sizeof(Token),"concurrent output accounting");
}
void concurrent_logprob_faults() {
    const std::vector<std::pair<const char*,const char*>> faults{{"lp_nan","nan, infinite or positive"},{"lp_count","count mismatch"},
        {"lp_unsorted","not ranked"},{"lp_padding","valid vocabulary"},{"token_padding","generated token outside the valid vocabulary"}};
    for(const auto& [fault,message]:faults) for(const auto path:{Path::decode,Path::async}) {
        auto st=std::make_shared<OutputState>(); st->fault=fault; st->path=path; st->admit=false;
        auto e=output_engine(st);
        RequestOptions o; o.logprobs=2;
        // Admit both together so the faulty row meets an unfinished neighbor.
        auto good=e->submit({10},12,o), bad=e->submit({50},6,o);
        st->admit=true;
        std::vector<Token> seen;
        rejects([&]{good.wait([&](Token t){seen.push_back(t);});},message);
        rejects([&]{bad.wait();},message);
        {
            // A row contract violation fails the engine; nothing from that group is committed or published.
            std::lock_guard lock(st->mutex);
            require(seen==st->committed[good.id()] && st->committed[bad.id()].empty() && !st->unsafe,"faulty logprob group published");
        }
        rejects([&]{e->submit({20},1);},"unavailable");
    }
}
void concurrent_text() {
    for(const auto path:{Path::decode,Path::mixed,Path::async}) {
        auto st=std::make_shared<OutputState>(); st->path=path; st->frontend=std::make_shared<Letters>(); st->per_step=1;
        auto e=output_engine(st,16384);
        RequestOptions t; t.text=true;
        auto good=e->submit({10},5,t), failing=e->submit({97},5,t), huge=e->submit({89},3,t), plain=e->submit({20},3);
        // A decoding or text storage failure ends only its own request; its failed row commits as cancelled.
        std::vector<Token> delivered;
        rejects([&]{failing.wait([&](Token token){delivered.push_back(token);});},"letter decoder failure");
        require(delivered==std::vector<Token>({98}),"failed text row published");
        rejects([&]{huge.wait();},"ceiling");
        const auto rg=good.wait(), rp=plain.wait();
        require(rg.tokens==expected(10,5) && rg.text=="LMNOP" && rg.text_ends==std::vector<std::uint32_t>({1,2,3,4,5}),"text neighbor");
        require(rp.tokens==expected(20,3) && rp.text.empty() && rp.text_ends.empty(),"plain neighbor");
        require(e->submit({30},2,t).wait().text=="FG","text failure poisoned the engine");
        {
            std::lock_guard lock(st->mutex);
            require(st->cancelled.count(failing.id()) && st->cancelled.count(huge.id()) && st->committed[huge.id()].empty() && !st->unsafe,
                    "failed text rows not committed as cancelled");
            require(std::count(st->finishes.begin(),st->finishes.end(),Finish::failed)==2,"failed text rows not retired as failed");
        }
        require(settled(*e),"text failures retained storage");
    }
    // Stop strings inside one multi-token row and across rows; the completing token is kept.
    for(const auto path:{Path::decode,Path::mixed,Path::async}) {
        auto st=std::make_shared<OutputState>(); st->path=path; st->frontend=std::make_shared<Letters>(); st->per_step=3;
        auto e=output_engine(st);
        RequestOptions inside; inside.stop_strings={"M"};
        auto a=e->submit({10},8,inside);
        RequestOptions across; across.stop_strings={"NO"}; across.logprobs=1;
        auto b=e->submit({10},8,across);
        const auto ra=a.wait(), rb=b.wait();
        require(ra.tokens==expected(10,2) && ra.text=="L" && ra.finish==Finish::stop,"stop inside a row");
        require(rb.tokens==expected(10,4) && rb.text=="LM" && rb.finish==Finish::stop && rb.text_ends==std::vector<std::uint32_t>({1,2,2,2}),
                "stop across rows");
        check_output_rows(rb,1);
        std::lock_guard lock(st->mutex);
        require(st->accepted[a.id()]==std::vector<std::uint32_t>({2}) && st->accepted[b.id()]==std::vector<std::uint32_t>({3,1}),
                "stop string commits");
    }
}
void concurrent_wait_output() {
    struct Seen { std::size_t index; Token token; std::optional<float> logprob; std::vector<TokenLogprob> top; std::string text; };
    auto st=std::make_shared<OutputState>(); st->frontend=std::make_shared<Letters>(); st->path=Path::async;
    auto e=output_engine(st);
    RequestOptions o; o.logprobs=2; o.text=true;
    auto other=e->submit({20},6,o);
    std::vector<Seen> seen;
    const auto result=e->submit({10},5,o).wait_output([&](const OutputEvent& event) {
        seen.push_back({event.index,event.token,event.logprob,{event.top_logprobs.begin(),event.top_logprobs.end()},std::string(event.text)});
    });
    require(seen.size()==5 && result.text=="LMNOP","concurrent output events");
    for(std::size_t i=0;i<seen.size();++i) {
        require(seen[i].index==i && seen[i].token==result.tokens[i] && seen[i].logprob==result.logprobs[i] &&
                seen[i].text==std::string(1,char('L'+i)),"concurrent event fields");
        require(seen[i].top.size()==2 && seen[i].top[0].token==result.top_logprobs[2*i].token,"concurrent event alternatives");
    }
    // A throwing sink cancels its own request (retired as cancelled, well before its 50-token budget); the neighbor
    // completes.
    st->gated=30;
    auto cancelled=e->submit({30},50,o);
    const auto id=cancelled.id();
    std::size_t delivered=0;
    rejects([&]{cancelled.wait_output([&](const OutputEvent&){++delivered; st->sink_failed=true; throw std::runtime_error("sink failure");});},
            "sink failure");
    require(other.wait().tokens==expected(20,6),"sink failure disturbed a neighbor");
    require(e->submit({40},2,o).wait().text=="PQ" && settled(*e),"sink failure poisoned the engine");
    std::lock_guard lock(st->mutex);
    require(delivered==1 && st->closed.count(id) && st->closed[id]==Finish::cancelled && st->committed[id].size()<50,
            "throwing output sink did not cancel its request");
}
}
// A request context whose destructor can be slowed. `destroyed` is set when the destructor finishes, so it shows the
// request's options were destroyed, not merely unreferenced. A slow destructor keeps a release by the worker, after
// the waiter returned, observable: the request's storage charge is released only after its options.
struct Probe final:RequestContext {
    std::shared_ptr<std::atomic<bool>> destroyed;
    std::size_t bytes;
    std::chrono::microseconds slow;
    Probe(std::shared_ptr<std::atomic<bool>> d,std::size_t b,std::chrono::microseconds s):destroyed(std::move(d)),bytes(b),slow(s) {}
    ~Probe() override {
        const auto until=std::chrono::steady_clock::now()+slow;
        while(std::chrono::steady_clock::now()<until) {}
        *destroyed=true;
    }
    std::size_t host_bytes() const noexcept override { return bytes; }
};
RequestOptions probed(const std::shared_ptr<std::atomic<bool>>& destroyed,bool slow) {
    RequestOptions o; o.context=std::make_shared<Probe>(destroyed,256,std::chrono::microseconds(slow ? 100 : 0));
    return o;
}
// Many short requests on decode, mixed and asynchronous batches, with and without deferred rounds. Waiters on their own
// threads race the worker, which completes their requests back to back; each checks its own request right after
// wait() returns, and the ledger is checked once they all returned. Then single requests, cancelled and abandoned
// handles are checked one at a time.
void concurrent_release_race() {
    for(const auto path:{Path::decode,Path::mixed,Path::async}) for(const bool deferred:{false,true}) {
        auto st=std::make_shared<OutputState>(); st->path=path; st->deferred=deferred; st->frontend=std::make_shared<Letters>();
        auto e=output_engine(st);
        for(int round=0;round<40;++round) {
            constexpr int width=3;
            std::vector<std::shared_ptr<std::atomic<bool>>> destroyed;
            std::vector<Handle> handles;
            for(int k=0;k<width;++k) {
                destroyed.push_back(std::make_shared<std::atomic<bool>>(false));
                auto o=probed(destroyed.back(),(round+k)%3==0);
                switch((round+k)%4) {
                case 0: o.logprobs=2; o.text=true; handles.push_back(e->submit({Token(10+10*k)},5,std::move(o))); break;
                case 1: o.stop_strings={"N"}; handles.push_back(e->submit({10},6,std::move(o))); break;
                case 2: handles.push_back(e->submit_score({1,2,3},1,std::move(o))); break;
                default: handles.push_back(e->submit_score({1,2,3,4},1,ScoreQueries{{5,6,7},1,2},std::move(o)));
                }
            }
            std::atomic<int> retained{0}, failed{0};
            std::vector<std::thread> waiters;
            for(int k=0;k<width;++k) waiters.emplace_back([&,k]{
                try { handles[k].wait(); } catch(...) { ++failed; }
                if(!*destroyed[k]) ++retained;
            });
            for(auto& waiter:waiters) waiter.join();
            require(!failed,"race request failed");
            require(!retained,"request context outlived its returned wait");
            require(settled(*e),"request charges outlived their returned waits");
        }
        for(int i=0;i<60;++i) {
            auto destroyed=std::make_shared<std::atomic<bool>>(false);
            auto o=probed(destroyed,i%3==0); o.logprobs=1;
            switch(i%3) {
            case 0: require(e->submit({20},4,std::move(o)).wait().tokens==expected(20,4),"race generation"); break;
            case 1: { auto h=e->submit({20},40,std::move(o)); h.cancel(); h.wait(); break; }
            default: { auto h=e->submit({20},40,std::move(o)); } // destroyed without wait
            }
            require(*destroyed && settled(*e),"request charges outlived its settled handle");
        }
        std::lock_guard lock(st->mutex);
        require(!st->unsafe && (path!=Path::async || st->batches>0),"race fixture lifetime or path");
    }
}
// The decoder of a request blocks its destruction, when that request completes, until the test opens the gate
// (bounded, so a failing test still finishes).
struct DecoderGate { std::mutex mutex; std::condition_variable cv; bool entered=false, open=false; };
struct GatedDecoder final:TextDecoder {
    std::shared_ptr<DecoderGate> gate;
    explicit GatedDecoder(std::shared_ptr<DecoderGate> g):gate(std::move(g)) {}
    ~GatedDecoder() override {
        std::unique_lock lock(gate->mutex);
        gate->entered=true; gate->cv.notify_all();
        gate->cv.wait_for(lock,std::chrono::seconds(3),[&]{return gate->open;});
    }
    std::string push(std::span<const Token> tokens) override { return std::string(tokens.size(),'g'); }
    std::string finish() override { return {}; }
};
struct GatedLetters final:TextFrontend {
    std::shared_ptr<DecoderGate> gate;
    explicit GatedLetters(std::shared_ptr<DecoderGate> g):gate(std::move(g)) {}
    std::vector<Token> tokenize(std::string_view) const override { return {}; }
    std::string detokenize(std::span<const Token> tokens) const override { return std::string(tokens.size(),'g'); }
    std::unique_ptr<TextDecoder> create_decoder() const override { return std::make_unique<GatedDecoder>(gate); }
};
// A settled engine failure completes its queue one request after another. The second queued request's decoder holds
// the worker inside that loop, while the queue still lists the first one; that first request's failed wait must
// already have released its charges.
void queued_release_on_failure() {
    auto gate=std::make_shared<DecoderGate>();
    auto st=std::make_shared<OutputState>(); st->fault="lp_nan"; st->admit=false; st->frontend=std::make_shared<GatedLetters>(gate);
    const Profile p{64,2,1<<20,1<<20,1,3,3000,3000};
    auto e=std::make_unique<Engine>(p,std::make_shared<OutputHost>(),[st](const Profile&,Services services) {
        return std::make_unique<OutputModel>(st,services.resources);
    });
    RequestOptions faulty; faulty.logprobs=2;
    auto failing=e->submit({50},6,std::move(faulty));
    auto destroyed=std::make_shared<std::atomic<bool>>(false);
    RequestOptions first; first.context=std::make_shared<Probe>(destroyed,4096,std::chrono::microseconds(0));
    const auto before=e->usage(Memory::host).live;
    auto queued=e->submit({10},6,std::move(first));
    const auto charge=e->usage(Memory::host).live-before;
    RequestOptions text; text.text=true;
    auto gated=e->submit({20},6,std::move(text));
    const auto all=e->usage(Memory::host).live;
    st->admit=true;
    rejects([&]{queued.wait();},"nan");
    {
        // A fixture boundary, not release polling: the worker is now completing the second queued request.
        std::unique_lock lock(gate->mutex);
        require(gate->cv.wait_for(lock,std::chrono::seconds(3),[&]{return gate->entered;}),"worker never completed the gated request");
    }
    // Charges only decrease while the worker completes the queue, so this bound is exact for the first request.
    const bool released=*destroyed && e->usage(Memory::host).live<=all-charge;
    { std::lock_guard lock(gate->mutex); gate->open=true; gate->cv.notify_all(); }
    require(released,"queued request kept its charges after its failed wait returned");
    rejects([&]{gated.wait();},"nan");
    rejects([&]{failing.wait();},"nan");
    require(settled(*e),"failed requests kept their charges after their waits");
    rejects([&]{e->submit({20},1);},"unavailable");
}
int main(int argc,char** argv) {
    try {
        const std::string only=argc>1 ? argv[1] : "";
        const std::vector<std::pair<std::string,std::function<void()>>> tests{
            {"latency_advice",latency_advice}, {"reuse_admission",reuse_admission}, {"scaled_batches",scaled_batches}, {"decode_priority",decode_priority}, {"admission_backfill",admission_backfill},
            {"many_aged_prefills",many_aged_prefills},
            {"mixed_batches",mixed_batches},
            {"identities_fairness_variable_rows",identities_and_fairness}, {"prefill_decode_interleave",prefill_decode_interleave},
            {"queued_cancel_overload",queued_cancel_and_limits}, {"pending_timeout",pending_timeout}, {"retained_handles",retained_handles},
            {"independent_retirement_reuse",isolated_retirement}, {"active_cancel_abandon",cancel_active_and_abandon},
            {"streaming_cancel",streaming_cancel}, {"admission_stop_context",admission_and_stop}, {"startup_declarations",startup_declarations},
            {"commit_failure",[]{failed_group("commit_throw","grouped commit");}}, {"stale_generation",[]{failed_group("stale","identity");}},
            {"failed_completion_waits_for_close",failed_completion_waits_for_close},
            {"duplicate_membership",[]{failed_group("duplicate","identity");}}, {"missing_membership",[]{failed_group("missing","membership");}},
            {"token_range",[]{failed_group("token_range","vocabulary");}}, {"output_budget",[]{failed_group("output_budget","progress");}},
            {"stalled_progress",[]{failed_group("stalled","progress");}}, {"prefill_shape",[]{failed_group("prefill_shape","progress");}},
            {"open_failure",[]{failed_group("open_throw","open failure");}}, {"null_retirement",[]{failed_group("null_receipt","retirement receipt");}},
            {"retirement_exception",[]{failed_group("receipt_throw","receipt failure");}}, {"staging_ceiling",staging_ceiling},
            {"overload_preserves_active_budget",overload_preserves_active_budget},
            {"scoring_barrier",scoring_barrier}, {"scoring_queued_cancel",[]{scoring_pending(false);}},
            {"scoring_pending_timeout",[]{scoring_pending(true);}}, {"scoring_active_cancel",scoring_active_cancel},
            {"scoring_throw",[]{scoring_failure("score_throw","scoring failure");}},
            {"scoring_count",[]{scoring_failure("score_count","finiteness");}},
            {"scoring_nan",[]{scoring_failure("score_nan","finiteness");}},
            {"scoring_drain_quarantine",[]{scoring_failure("score_drain","drain failure");}},
            {"scoring_quiesce_quarantine",[]{scoring_failure("score_throw","scoring failure",true);}},
            {"scoring_capability_and_budget",scoring_capability_and_budget},
            {"retirement_timeout",[]{retirement_timeout(false,false);}}, {"quiesce_quarantine",[]{retirement_timeout(true,false);}},
            {"drain_quarantine",[]{retirement_timeout(false,true);}},{"tracing",concurrent_tracing},
            {"logprobs",concurrent_logprobs},{"logprob_faults",concurrent_logprob_faults},{"text",concurrent_text},
            {"wait_output",concurrent_wait_output},{"release_race",concurrent_release_race},
            {"queued_release_on_failure",queued_release_on_failure}};
        bool found=false;
        for(const auto& [name,run]:tests) if(only.empty() || only==name) {
            run(); found=true; std::cout<<"crit concurrent contract passed: "<<name<<std::endl;
        }
        require(found,"unknown test");
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
