#include "crit/core.h"
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
// Forced unwinding and foreign exceptions as glibc and the Itanium unwinder raise them.
#if defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__)) && __has_include(<pthread.h>) && __has_include(<unwind.h>)
#define CRIT_TEST_SINK_UNWIND 1
#include <cstring>
#include <pthread.h>
#include <unwind.h>
#endif

using namespace crit::core;
namespace {
void require(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F f, const char* text) {
    try { f(); } catch(const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos,"unexpected rejection diagnostic");
        return;
    }
    throw std::runtime_error("expected rejection did not occur");
}
// Release contract: once wait()/wait_output() returns, or a handle's destructor returns, the request's host charges
// are already back in the ledger as seen from that thread.
bool settled(const Engine& e, std::size_t live) { return e.usage(Memory::host).live == live; }
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    void wait() {
        std::unique_lock lock(mutex);
        entered = true; cv.notify_all();
        cv.wait(lock,[&]{return released;});
    }
    void observed() { std::unique_lock lock(mutex); cv.wait(lock,[&]{return entered;}); }
    void release() { std::lock_guard lock(mutex); released = true; cv.notify_all(); }
};
struct State {
    std::shared_ptr<Gate> gate;
    std::size_t gate_after = 0;
    std::atomic<bool> pending{false};
    std::atomic<int> drain_failures{0};
    std::atomic<int> closed{0}, destroyed{0}, models_destroyed{0}, commits{0}, unsafe{0};
    std::atomic<int> initialized{0};
    std::optional<Resources> ledger;  // the backend's retained copy, which outlives the engine
    std::atomic<Finish> reason{Finish::failed};
    std::vector<std::uint32_t> chunks;
    std::uint32_t reuse = 0;  // prompt tokens the first advance reports as reused
    bool advice = false;
    std::string fault;
    std::span<float> score_destination;
    std::vector<Token> outputs{2,3,4,5};
    std::vector<DeviceTiming> timings;
    std::atomic<int> timing_calls{0};
};
struct FixtureBackend : Backend {
    std::shared_ptr<State> state;
    explicit FixtureBackend(std::shared_ptr<State> s):state(std::move(s)) {}
    void initialize(Resources& resources) override {
        state->ledger = resources;
        ++state->initialized;
        if(state->fault=="backend_init") throw std::runtime_error("backend initialization failure");
    }
    void bind_thread() override {require(state->initialized>0,"thread bound before backend initialization");}
    void drain() override {
        if(state->drain_failures != 0) {
            if(state->drain_failures > 0) --state->drain_failures;
            throw std::runtime_error("backend drain failure");
        }
        if(state->pending) {
            if(state->gate) state->gate->wait();
            std::fill(state->score_destination.begin(),state->score_destination.end(),-0.25F);
            state->score_destination={};
            state->pending = false;
        }
    }
};
struct FixtureSession : Session, Readiness {
    std::shared_ptr<State> state;
    Reservation memory;
    std::span<const Token> prompt;
    std::uint32_t frontier = 0;
    std::size_t next = 0;
    Token output = -999;
    bool awaited = false;
    FixtureSession(std::shared_ptr<State> s,Resources& r,std::span<const Token> p):state(std::move(s)),memory(r.reserve(Memory::device,7)),prompt(p) {}
    ~FixtureSession() override { if(state->pending) ++state->unsafe; ++state->destroyed; }
    std::uint32_t prefill_limit(const PrefillWindow& w) override {
        require(w.occupied_tokens==frontier && w.remaining_tokens==prompt.size()-frontier,"Serial advice frontier");
        return 1;
    }
    Progress advance(WorkBudget budget) override {
        if(state->fault == "advance") {state->pending = true; throw std::runtime_error("advance failure");}
        if(state->fault == "stall") return {};
        if(state->reuse && frontier == 0) {
            frontier = state->reuse;
            return {state->reuse,false,{},nullptr,false,state->reuse};
        }
        if(state->fault == "reuse_late" && frontier && frontier < prompt.size()) {
            ++frontier;
            return {1,false,{},nullptr,false,1};
        }
        const auto count = std::min(budget.prefill_tokens,std::uint32_t(prompt.size())-frontier);
        frontier += count;
        if(count) state->chunks.push_back(count);
        if(state->fault == "shape") return {budget.prefill_tokens+1,false,{}};
        if(frontier != prompt.size()) return {count,false,{}};
        if(state->fault == "token") {output = -1; return {count,true,{&output,1}};}
        if(state->gate && !awaited && next >= state->gate_after) {
            output = -999; state->pending = true;
            return {count,true,{&output,1},this};
        }
        output = state->outputs.at(next++%state->outputs.size());
        return {count,true,{&output,1}};
    }
    void wait() override {
        state->gate->wait();
        require(!prompt.empty() && prompt[0] == 1,"borrowed prompt did not remain live");
        output = state->outputs.at(next++%state->outputs.size());
        state->pending = false; awaited = true;
    }
    std::span<const DeviceTiming> device_timings() override {
        ++state->timing_calls;
        if(state->fault == "timings") throw std::runtime_error("timing failure");
        return state->timings;
    }
    void commit(std::uint32_t, bool) override {
        if(state->fault == "commit") {state->pending = true; throw std::runtime_error("commit failure");}
        ++state->commits;
        output = 63; // A physical commit may invalidate the borrowed pending batch.
        state->pending = true;
    }
    void close(Finish reason) noexcept override {
        if(state->pending) ++state->unsafe;
        state->reason = reason; ++state->closed;
    }
};
struct FixtureModel : Model {
    std::shared_ptr<State> state;
    Resources& resources;
    Reservation weights;
    Reservation score_storage;
    FixtureModel(std::shared_ptr<State> s,Services services):state(std::move(s)),resources(services.resources),weights(resources.reserve(Memory::device,11)) {}
    ~FixtureModel() override { if(state->pending) ++state->unsafe; ++state->models_destroyed; }
    ModelInfo info() const override {ModelInfo m{"independent.contract.fixture",64,32,4,{5},true,true};m.prefill_advice=state->advice;return m;}
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        score_storage={};
        if(state->fault == "open") {state->pending = true; throw std::runtime_error("open failure");}
        return std::make_unique<FixtureSession>(state,resources,prompt);
    }
    std::vector<Token> tokenize(std::string_view) override {return {1,2};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
    std::vector<float> score(std::span<const Token> prompt,std::uint32_t first) override {
        if(state->fault == "score") {state->pending = true; throw std::runtime_error("score failure");}
        score_storage={}; score_storage=resources.reserve(Memory::host,(prompt.size()-first)*sizeof(float));
        if(state->fault == "nan") return {std::numeric_limits<float>::quiet_NaN()};
        std::vector<float> scores(prompt.size()-first,-0.25F);
        if(state->fault=="score_retry" || state->fault=="score_fatal") {
            state->drain_failures=state->fault=="score_retry" ? 1 : -1;
            state->score_destination=scores; state->pending=true;
        }
        return scores;
    }
};
// Query scoring: per target a log-probability, per_target query log-probabilities and an argmax.
struct QueryModel : FixtureModel {
    using FixtureModel::FixtureModel;
    ModelInfo info() const override {
        ModelInfo info = FixtureModel::info();
        info.query_scoring = true;
        return info;
    }
    QueryScores score_queries(std::span<const Token> prompt,std::uint32_t first,const ScoreQueries& queries,
                              const RequestOptions&) override {
        const std::size_t targets = prompt.size()-first;
        QueryScores out;
        out.target.assign(targets,-0.5F);
        for(std::size_t i = 0; i < queries.ids.size(); ++i) out.queries.push_back(-1.0F-0.125F*float(i));
        if(state->fault == "query_inf") out.queries.back() = -std::numeric_limits<float>::infinity();
        if(state->fault == "query_nan") out.queries.front() = std::numeric_limits<float>::quiet_NaN();
        for(std::size_t t = 0; t < targets; ++t) out.argmax.push_back(prompt[first+t]);
        if(state->fault == "query_argmax") out.argmax.front() = 99;
        if(state->fault == "query_short") out.target.pop_back();
        return out;
    }
};
Profile profile() {return {32,2,1024,1024};}
std::unique_ptr<Engine> engine(const std::shared_ptr<State>& s) {
    return std::make_unique<Engine>(profile(),std::make_shared<FixtureBackend>(s),[s](const Profile&,Services services) {
        return std::make_unique<FixtureModel>(s,services);
    });
}
void resources() {
    Resources r(100,50);
    {
        auto a = r.reserve(Memory::device,40);
        auto alias = a;
        a = {};
        require(r.usage(Memory::device).live == 40 && alias.bytes() == 40,"aliased reservation double counted/released");
        rejects([&]{r.reserve(Memory::device,11);},"ceiling");
        auto host = r.reserve(Memory::host,100);
        rejects([&]{r.reserve(Memory::host,1);},"ceiling");
        rejects([&]{r.reserve(static_cast<Memory>(99),1);},"domain");
    }
    require(r.usage(Memory::device).live == 0 && r.usage(Memory::device).peak == 40,"reservation lifetime/peak wrong");
    auto retained = [&]{Resources temporary(2,2);return temporary.reserve(Memory::host,2);}();
    require(retained.bytes() == 2,"reservation outlives ledger facade");
    std::array<std::byte,1024> buffer{};
    Arena arena(buffer.data()+1,511);
    auto range = arena.allocate(16,128);
    require(reinterpret_cast<std::uintptr_t>(range.data())%128 == 0,"unaligned arena output");
    const auto mark = arena.mark();
    auto second = arena.allocate(20,1);
    arena.rewind(mark);
    require(arena.allocate(20,1).data() == second.data(),"rewind failed");
    rejects([&]{arena.allocate(1,3);},"power of two");
    rejects([&]{arena.allocate(9999);},"capacity");
    rejects([&]{arena.rewind(900);},"future");
    rejects([&]{Arena invalid(reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-1),8);},"overflow");
}
void ordinary() {
    auto s = std::make_shared<State>();
    auto e = engine(s);
    require(e->tokenize("a") == std::vector<Token>({1,2}),"frontend output");
    std::vector<Token> delivered;
    auto result = e->submit({1,2,3,4,6},3).wait([&](Token token){delivered.push_back(token);});
    require(result.tokens == std::vector<Token>({2,3,4}) && delivered == result.tokens,"generation/streaming mismatch");
    require(s->chunks == std::vector<std::uint32_t>({2,2,1}),"prefill budget not respected");
    require(result.finish == Finish::length && s->closed == 1 && s->unsafe == 0,"normal close/lifetime wrong");
    require(e->usage(Memory::device).live == 11 && e->usage(Memory::device).peak == 18,"session reservation not released");
    require(settled(*e,0),"request storage not released after consume");
    auto stopped = e->submit({1},8).wait();
    require(stopped.tokens == std::vector<Token>({2,3,4,5}) && stopped.finish == Finish::stop,"EOS not honored");
    auto scored = e->submit_score({1,2,3},1).wait();
    require(scored.scores == std::vector<float>({-0.25F,-0.25F}),"scores mismatch");
    auto handle = e->submit({1},0);
    require(handle.wait().tokens.empty(),"zero output request");
    rejects([&]{handle.wait();},"consumed");
    rejects([&]{e->submit({},1);},"context");
    rejects([&]{e->submit({1},32);},"context");
    rejects([&]{e->submit({-1},1);},"vocabulary");
    rejects([&]{e->submit_score({1},0);},"target");
    auto advised=std::make_shared<State>(); advised->advice=true;
    auto p=profile();p.prefill_target_ns=100;
    Engine adaptive(p,std::make_shared<FixtureBackend>(advised),[advised](const Profile&,Services svc){return std::make_unique<FixtureModel>(advised,svc);});
    adaptive.submit({1,2,3,4,6},1).wait();
    require(advised->chunks==std::vector<std::uint32_t>({1,1,1,1,1}),"Serial advice not applied");
}
void query_scoring() {
    auto engine_with = [](const std::shared_ptr<State>& s) {
        return std::make_unique<Engine>(profile(),std::make_shared<FixtureBackend>(s),[s](const Profile&,Services services) {
            return std::make_unique<QueryModel>(s,services);
        });
    };
    auto s = std::make_shared<State>();
    auto e = engine_with(s);
    const auto scored = e->submit_score({1,2,3,4},1,ScoreQueries{{5,6,7,8,9,10},2}).wait();
    require(scored.scores == std::vector<float>({-0.5F,-0.5F,-0.5F}),"query target scores");
    require(scored.query_scores.size() == 6 && scored.query_scores[5] == -1.0F-0.125F*5.0F,"query log-probabilities");
    require(scored.argmax == std::vector<Token>({2,3,4}),"query argmax");
    require(settled(*e,0),"query result storage not released after consume");
    require(e->submit_score({1,2,3},1).wait().scores.size() == 2,"plain scoring still works with a query model");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{5,6,7},2});},"per_target");
    // per_target zero with no ids scores the targets alone; ids then need a nonzero per_target.
    const auto alone = e->submit_score({1,2,3},1,ScoreQueries{{},0}).wait();
    require(alone.scores.size() == 2 && alone.query_scores.empty() && alone.argmax == std::vector<Token>({2,3}),"targets-only query scoring");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{5},0});},"per_target");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{5,99},1});},"vocabulary");
    for(const auto& [fault,text] : std::initializer_list<std::pair<const char*,const char*>>{
            {"query_nan","finiteness"},{"query_short","count"},{"query_argmax","vocabulary"}}) {
        auto f = std::make_shared<State>(); f->fault = fault;
        auto fe = engine_with(f);
        rejects([&]{fe->submit_score({1,2,3},1,ScoreQueries{{5,6},1}).wait();},text);
    }
    auto inf = std::make_shared<State>(); inf->fault = "query_inf";
    auto ie = engine_with(inf);
    const auto unreachable = ie->submit_score({1,2,3},1,ScoreQueries{{5,6},1}).wait();
    require(std::isinf(unreachable.query_scores.back()),"a zero-probability query may be negative infinity");
    auto plain = std::make_shared<State>();
    auto pe = engine(plain);
    rejects([&]{pe->submit_score({1,2,3},1,ScoreQueries{{5,6},1});},"query scoring");
}
void timing() {
    auto s = std::make_shared<State>();
    auto e = engine(s);
    // Prefill chunks 2, 2, 1; the last chunk completes prefill and emits the first
    // token, then two decode steps emit one token each.
    const auto r = e->submit({1,2,3,4,6},3).wait();
    const auto& t = r.timing;
    require(r.tokens.size() == 3 && t.prefill_tokens == 5 && t.decode_steps == 2,"phase counts wrong");
    require(t.prefill_ns <= t.first_token_ns && t.first_token_ns <= t.total_ns,"phase order wrong");
    require(t.decode_ns <= t.total_ns && t.total_ns > 0,"decode/total duration wrong");
    const auto scored = e->submit_score({1,2,3},1).wait();
    require(scored.timing.prefill_tokens == 0 && scored.timing.first_token_ns == 0 && scored.timing.decode_ns == 0,
            "scoring reported generation phases");
}
void reuse() {
    // A retained prefix larger than one prefill chunk is reported without computation,
    // then the rest is computed in ordinary chunks.
    auto s = std::make_shared<State>(); s->reuse = 3;
    auto e = engine(s);
    const auto r = e->submit({1,2,3,4,6},2).wait();
    require(r.tokens == std::vector<Token>({2,3}),"reused prefix changed output");
    require(s->chunks == std::vector<std::uint32_t>({2}),"reused prefix was computed");
    require(r.timing.prefill_tokens == 5 && r.timing.reused_prompt_tokens == 3,"reused prompt tokens not reported");
    s->reuse = 0;
    const auto fresh = e->submit({1,2,3,4,6},2).wait();
    require(fresh.timing.reused_prompt_tokens == 0,"reuse leaked into the next request");
    // Reuse after a computed prompt token is not a prefix.
    auto late = std::make_shared<State>(); late->fault = "reuse_late";
    auto e2 = engine(late);
    rejects([&]{e2->submit({1,2,3,4,6},2).wait();},"invalid");
}
void asynchronous(bool cancel, bool drop_engine) {
    auto s = std::make_shared<State>(); s->gate = std::make_shared<Gate>();
    auto e = engine(s);
    auto handle = e->submit({1,2,3},2);
    s->gate->observed();
    rejects([&]{e->submit({1},1);},"busy");
    if(drop_engine) { e.reset(); require(s->models_destroyed == 0,"handle did not retain runtime"); }
    if(cancel) handle.cancel();
    s->gate->release();
    const auto result = handle.wait();
    require(result.tokens == (cancel ? std::vector<Token>{} : std::vector<Token>{2,3}),"readiness/cancel ordering");
    require(result.finish == (cancel ? Finish::cancelled : Finish::length),"finish reason");
    require(s->unsafe == 0 && s->closed == 1 && s->destroyed == 1,"asynchronous storage released early");
    // A handle that kept the runtime alive shuts it down before its wait returns.
    if(drop_engine) require(s->models_destroyed == 1 && s->ledger->usage(Memory::host).live == 0 && s->ledger->usage(Memory::device).live == 0,
                            "retained runtime did not release after handle");
    else require(settled(*e,0),"completed request storage not released after wait");
}
void errors() {
    for(const auto& [fault,diagnostic]:std::vector<std::pair<std::string,std::string>>{
        {"open","open failure"},{"advance","advance failure"},{"commit","commit failure"},
        {"stall","stalled"},{"shape","invalid"},{"token","vocabulary"},{"score","score failure"},{"nan","finiteness"}}) {
        auto s = std::make_shared<State>(); s->fault = fault;
        auto e = engine(s);
        auto handle = (fault == "score" || fault == "nan") ? e->submit_score({1,2},1) : e->submit({1},2);
        int published = 0;
        rejects([&]{handle.wait([&](Token){++published;});},diagnostic.c_str());
        require(published == 0,"failed step published an uncommitted token");
        require(!s->pending && s->unsafe == 0 && e->usage(Memory::device).live == 11,"failure cleanup did not drain/release");
        // A failed score's returned allocation stays charged to the model until its next request callback.
        require(settled(*e,fault == "nan" ? sizeof(float) : 0),"failed request storage not released after wait");
        if(fault != "open" && fault != "score" && fault != "nan") require(s->reason == Finish::failed,"failure close reason");
        s->fault.clear();
        require(e->submit({1},1).wait().tokens == std::vector<Token>({2}),"engine could not recover after request failure");
    }
    auto s = std::make_shared<State>();
    auto e = engine(s);
    rejects([&]{e->submit({1},8).wait([](Token){throw std::runtime_error("consumer failure");});},"consumer failure");
    require(s->unsafe == 0 && s->closed == 1,"throwing sink did not drain");
}
void score_ownership() {
    for(const bool fatal:{false,true}) {
        auto s=std::make_shared<State>(); s->fault=fatal ? "score_fatal" : "score_retry";
        auto e=engine(s);
        rejects([&]{e->submit_score({1,2,3},1).wait();},"backend drain failure");
        const auto before=quarantined_engines();
        if(!fatal) {
            require(s->score_destination.empty() && !s->pending,"score retry did not settle returned allocation");
            s->fault.clear();
            require(e->submit({1},1).wait().tokens==std::vector<Token>({2}),"settled serial score error poisoned engine");
        }
        e.reset();
        if(fatal) {
            require(quarantined_engines()==before+1 && s->models_destroyed==0,"serial score owner not quarantined");
            require(s->score_destination.size()==2 && s->score_destination.front()==-0.25F,"serial score allocation freed before drain");
            s->score_destination.front()=7.0F;
        } else require(s->models_destroyed==1 && s->unsafe==0,"settled serial score owner leaked");
    }
}
void abandoned() {
    auto s = std::make_shared<State>(); s->gate = std::make_shared<Gate>();
    auto e = engine(s);
    auto handle = e->submit({1},3);
    s->gate->observed();
    auto dropping = std::async(std::launch::async,[handle=std::move(handle)]() mutable {handle = {};});
    s->gate->release();
    dropping.get();
    require(s->unsafe == 0 && s->closed == 1,"abandoned handle did not drain");
    require(settled(*e,0),"abandoned handle retained request storage");
}
void cancel_while_waiting() {
    auto s = std::make_shared<State>(); s->gate = std::make_shared<Gate>();
    auto e = engine(s);
    auto handle = e->submit({1},3);
    s->gate->observed();
    auto waiting = std::async(std::launch::async,[&]{return handle.wait();});
    handle.cancel(); // Must remain effective even after wait consumes its private state.
    s->gate->release();
    const auto result = waiting.get();
    require(result.finish == Finish::cancelled && result.tokens.empty(),"concurrent wait/cancel lost cancellation");
    require(s->unsafe == 0 && s->closed == 1,"concurrent cancellation released live views");
    require(settled(*e,0),"cancelled request storage not released after wait");
    auto streamed = std::make_shared<State>();
    streamed->gate = std::make_shared<Gate>(); streamed->gate_after = 1;
    auto streaming_engine = engine(streamed);
    auto streaming_handle = streaming_engine->submit({1},3);
    const auto partial = streaming_handle.wait([&](Token) {
        // Callback execution establishes that wait already consumed private state.
        streaming_handle.cancel(); streamed->gate->release();
    });
    require(partial.tokens == std::vector<Token>({2}) && partial.finish == Finish::cancelled,"stream consumer could not cancel its consumed handle");
    require(streamed->unsafe == 0 && streamed->closed == 1,"stream cancellation failed drained cleanup");
}
void startup() {
    auto s = std::make_shared<State>();
    auto backend = std::make_shared<FixtureBackend>(s);
    rejects([&]{Engine e(profile(),backend,[](const Profile&,Services)->std::unique_ptr<Model>{return {};});},"null");
    rejects([&]{Engine e(profile(),backend,[s](const Profile&,Services)->std::unique_ptr<Model>{s->pending=true;throw std::runtime_error("factory failure");});},"factory failure");
    require(!s->pending,"failed startup did not drain");
    auto p = profile(); p.context = 0;
    rejects([&]{Engine e(p,backend,[](const Profile&,Services)->std::unique_ptr<Model>{return {};});},"profile");
    auto failing=std::make_shared<State>(); failing->fault="backend_init";
    bool constructed=false;
    rejects([&]{Engine e(profile(),std::make_shared<FixtureBackend>(failing),[&](const Profile&,Services)->std::unique_ptr<Model>{constructed=true;return {};});},"backend initialization failure");
    require(!constructed && failing->initialized==1,"backend initialization failure reached model factory");
    auto working=engine(std::make_shared<State>());
    const auto applied=working->profile();
    require(applied.context==32 && applied.prefill_chunk==2 && applied.host_limit==1024 && applied.device_limit==1024,"applied profile not exposed");
    std::vector<Token> excess_capacity{1}; excess_capacity.reserve(1024);
    rejects([&]{working->submit(std::move(excess_capacity),1);},"ceiling");
}
void borrowed_startup() {
    // Model construction may throw before its destructor can unregister a
    // borrowed drain target. A weak lifetime probe detects that error without
    // ever dereferencing freed storage in this fixture.
    struct BorrowingBackend : FixtureBackend {
        std::weak_ptr<Token> target;
        bool registered = false;
        int empty_drains = 0, target_drains = 0, expired_drains = 0;
        using FixtureBackend::FixtureBackend;
        void drain() override {
            if(registered) {
                const auto live = target.lock();
                if(!live) {
                    ++expired_drains;
                    throw std::runtime_error("borrowed drain target expired");
                }
                require(*live == 23,"borrowed target changed before drain");
                ++target_drains;
            } else {
                ++empty_drains;
            }
            FixtureBackend::drain();
        }
    };
    struct BorrowingModel : FixtureModel {
        BorrowingBackend& backend;
        std::shared_ptr<Token> target = std::make_shared<Token>(23);
        bool declaration_failure;
        BorrowingModel(std::shared_ptr<State> s,Services services,bool validation_failure,bool info_failure)
            :FixtureModel(std::move(s),services),
             backend(dynamic_cast<BorrowingBackend&>(services.backend)),declaration_failure(info_failure) {
            if(validation_failure) throw std::runtime_error("late construction validation");
            // All throwing construction work precedes non-owning publication.
            backend.target = target;
            backend.registered = true;
        }
        ~BorrowingModel() override {
            backend.registered = false;
            backend.target.reset();
        }
        ModelInfo info() const override {
            if(declaration_failure) throw std::runtime_error("declaration failure");
            return FixtureModel::info();
        }
    };
    for(const auto stage : {0,1,2}) {
        auto s = std::make_shared<State>();
        auto backend = std::make_shared<BorrowingBackend>(s);
        Resources observed(1,1);
        auto factory = [s,&observed,stage](const Profile&,Services services) -> std::unique_ptr<Model> {
            observed = services.resources;
            return std::make_unique<BorrowingModel>(s,services,stage==0,stage==1);
        };
        if(stage==0) {
            rejects([&]{Engine e(profile(),backend,factory);},"late construction validation");
            require(backend->empty_drains>0 && backend->target_drains==0,
                    "failed construction published a borrowed target");
        } else if(stage==1) {
            rejects([&]{Engine e(profile(),backend,factory);},"declaration failure");
            require(backend->target_drains>0 && s->models_destroyed==1,
                    "post-construction failure did not drain the live owner");
        } else {
            {
                Engine e(profile(),backend,factory);
                require(e.submit({1},1).wait().tokens==std::vector<Token>({2}),
                        "borrowed backend consumer failed its request");
            }
            require(backend->target_drains>0 && s->models_destroyed==1 && s->closed==1,
                    "normal cleanup did not drain and release the owner");
        }
        require(!backend->registered && backend->target.expired() && backend->expired_drains==0,
                "construction/final cleanup retained an invalid borrowed target");
        require(observed.usage(Memory::device).live==0 && s->unsafe==0,
                "construction/final cleanup leaked or freed pending storage");
        backend->drain(); // External backend outlives the model and remains safe.
        require(backend->expired_drains==0,"unregistered backend tried to use expired storage");
    }
}
void failed_drain(bool permanent) {
    auto s = std::make_shared<State>();
    auto e = engine(s);
    const auto before = quarantined_engines();
    s->drain_failures = permanent ? -1 : 1;
    rejects([&]{e->submit({1},1).wait();},"backend drain failure");
    require(s->closed == 0 && s->destroyed == 0 && s->pending,"failed drain released live session");
    // The quarantined runtime keeps the request and its charges; only settlement at destruction may release them.
    require(e->usage(Memory::host).live > 0,"quarantined request released its charges");
    rejects([&]{e->submit({1},1);},"unavailable");
    e.reset();
    if(permanent) {
        require(quarantined_engines() == before+1 && s->closed == 0 && s->models_destroyed == 0,"fatal owner was not retained");
    } else {
        require(s->closed == 1 && s->models_destroyed == 1 && !s->pending,"successful cleanup retry failed");
    }
    require(s->unsafe == 0,"failed drain permitted premature storage release");
}
void tracing() {
    auto make=[](const std::shared_ptr<State>& s,Profile p) {
        return std::make_unique<Engine>(p,std::make_shared<FixtureBackend>(s),[s](const Profile&,Services services) {
            return std::make_unique<FixtureModel>(s,services);
        });
    };
    auto disabled=engine(std::make_shared<State>());
    disabled->submit({1},1).wait();
    require(!disabled->trace().enabled && disabled->trace().events.empty(),"disabled tracing recorded work");
    auto p=profile(); p.host_limit=1<<20; p.trace={256,1};
    auto s=std::make_shared<State>(); s->gate=std::make_shared<Gate>();
    auto e=make(s,p);
    require(e->usage(Memory::host).live==256*sizeof(TraceEvent),"trace buffer escaped resource ledger");
    auto h=e->submit({1},2);
    s->gate->observed();
    // No engine/request lock: this snapshot must complete while wait() is blocked.
    auto live=e->trace();
    const bool waiting=std::any_of(live.events.begin(),live.events.end(),[](const auto& event) {
        return event.phase==TracePhase::readiness && event.mark==TraceMark::begin && event.key==RequestKey{1,1,0};
    });
    s->gate->release();
    h.wait();
    require(waiting,"live readiness span unavailable during blocked callback");
    auto finished=e->trace();
    require(finished.slow_steps>0,"blocked step was not classified");
    for(auto phase:{TracePhase::submitted,TracePhase::open,TracePhase::prefill,TracePhase::decode,
                    TracePhase::readiness,TracePhase::commit,TracePhase::publication,TracePhase::drain,
                    TracePhase::retirement,TracePhase::completion})
        require(std::any_of(finished.events.begin(),finished.events.end(),[&](const auto& x){return x.phase==phase;}),
                "lifecycle phase missing");
    require(finished.events.back().phase==TracePhase::completion,"completion not last");
    auto other=make(std::make_shared<State>(),p);
    const auto isolated=other->trace();
    require(std::none_of(isolated.events.begin(),isolated.events.end(),[](const auto& x){return x.key.request!=0;}),
            "trace state leaked between engines");
    auto failure=std::make_shared<State>(); failure->fault="commit";
    auto broken=make(failure,p);
    rejects([&]{broken->submit({1},1).wait();},"commit failure");
    auto errors=broken->trace();
    require(std::any_of(errors.events.begin(),errors.events.end(),[](const auto& x) {
        return x.phase==TracePhase::commit && x.mark==TraceMark::failed;
    }),"throwing commit not recorded");
    require(std::none_of(errors.events.begin(),errors.events.end(),[](const auto& x) {
        return x.phase==TracePhase::publication;
    }),"failed commit recorded publication");
    require(errors.events.back().finish==Finish::failed,"failure completion reason missing");
    p.trace.capacity=3;
    auto small=make(std::make_shared<State>(),p);
    small->submit({1},2).wait();
    const auto wrapped=small->trace();
    require(wrapped.events.size()==3 && wrapped.overwritten>0,"trace ring did not bound retention");
    require(wrapped.events[0].sequence<wrapped.events[1].sequence && wrapped.events[1].sequence<wrapped.events[2].sequence,
            "trace ring order corrupted on wrap");
    auto timed=std::make_shared<State>();
    timed->timings={{DeviceStage::draft,0,1500},{DeviceStage::verify,3,2500}};
    auto untraced=engine(timed);
    untraced->submit({1},2).wait();
    require(timed->timing_calls==0,"device timings requested without tracing");
    p.trace.capacity=256;
    auto traced=make(timed,p);
    traced->submit({1},2).wait();
    const auto devices=traced->trace();
    std::size_t device_events=0;
    for(const auto& x:devices.events) if(x.phase==TracePhase::device) {
        ++device_events;
        require(x.mark==TraceMark::point && x.key==RequestKey{1,1,0},"device timing identity");
        require((x.stage==DeviceStage::draft && x.index==0 && x.duration_ns==1500) ||
                (x.stage==DeviceStage::verify && x.index==3 && x.duration_ns==2500),"device timing payload");
        require(std::any_of(devices.events.begin(),devices.events.end(),[&](const auto& y) {
            return y.phase==TracePhase::step && y.mark==TraceMark::begin && y.operation==x.operation && y.key==x.key;
        }),"device timing outside its step operation");
    }
    require(timed->timing_calls==2 && device_events==4 && !devices.dropped,"device timings not recorded per step");
    auto throwing=std::make_shared<State>(); throwing->fault="timings";
    auto tolerant=make(throwing,p);
    const auto kept=tolerant->submit({1},2).wait();
    const auto lost=tolerant->trace();
    require(kept.finish==Finish::length && kept.tokens.size()==2 && throwing->timing_calls==2,"throwing device timings failed the request");
    require(lost.dropped==2 && std::none_of(lost.events.begin(),lost.events.end(),[](const auto& x){return x.phase==TracePhase::device;}),
            "throwing device timings not counted as dropped");
    p.trace.capacity=65537;
    rejects([&]{make(std::make_shared<State>(),p);},"trace capacity");
    p.trace={100,0}; p.host_limit=1;
    rejects([&]{make(std::make_shared<State>(),p);},"ceiling");
}
// A device round learns its token count only from its readback: advance()
// returns deferred_tokens and Core reads the final span after readiness.
struct DeferredRecord {
    std::atomic<int> waits{0}, reads{0}, commits{0};
    bool read_after_wait = true, invalid = false, missing = false;
};
struct DeferredBackend : Backend { void bind_thread() override {} void drain() override {} };
struct DeferredSession : Session, Readiness {
    std::shared_ptr<DeferredRecord> record;
    std::span<const Token> prompt;
    std::array<Token,3> produced{7,8,9};
    std::uint32_t emitted = 0;
    bool waited = false;
    DeferredSession(std::shared_ptr<DeferredRecord> r, std::span<const Token> p):record(std::move(r)),prompt(p) {}
    void wait() override { waited = true; ++record->waits; }
    Progress advance(WorkBudget budget) override {
        if(!budget.prefill_complete) return {std::uint32_t(prompt.size()),true,{},nullptr};
        waited = false;
        emitted = std::min<std::uint32_t>(budget.output_tokens,2); // known only "after the readback"
        Progress p{0,true,{},this,true};
        if(record->invalid) p.tokens = std::span<const Token>(produced.data(),1);
        return p;
    }
    std::span<const Token> ready_tokens() override {
        if(record->missing) return Session::ready_tokens();
        ++record->reads;
        if(!waited) record->read_after_wait = false;
        return {produced.data(),emitted};
    }
    void commit(std::uint32_t,bool) override { ++record->commits; }
    void close(Finish) noexcept override {}
};
struct DeferredModel : Model {
    std::shared_ptr<DeferredRecord> record;
    explicit DeferredModel(std::shared_ptr<DeferredRecord> r):record(std::move(r)) {}
    ModelInfo info() const override { ModelInfo m; m.implementation="deferred"; m.vocabulary=16; m.context=32; m.prefill_chunk=8; return m; }
    std::unique_ptr<Session> open(std::span<const Token> p) override { return std::make_unique<DeferredSession>(record,p); }
    std::vector<Token> tokenize(std::string_view) override { return {}; }
    std::string detokenize(std::span<const Token>) override { return {}; }
};
void deferred() {
    auto run=[](std::shared_ptr<DeferredRecord> r,std::uint32_t output) {
        Engine e(profile(),std::make_shared<DeferredBackend>(),[r](const Profile&,Services) {return std::make_unique<DeferredModel>(r);});
        return e.submit({1,2},output).wait();
    };
    auto ok=std::make_shared<DeferredRecord>();
    const auto result=run(ok,5);
    require(result.finish==Finish::length && result.tokens==std::vector<Token>({7,8,7,8,7}),"deferred tokens not published");
    require(ok->waits==3 && ok->reads==3,"deferred rounds not read once per readiness");
    require(ok->read_after_wait,"deferred tokens read before readiness");
    auto bad=std::make_shared<DeferredRecord>(); bad->invalid=true;
    rejects([&]{run(bad,4);},"invalid deferred model progress");
    auto missing=std::make_shared<DeferredRecord>(); missing->missing=true;
    rejects([&]{run(missing,4);},"without ready_tokens");
}
// Generation log-probabilities, text and ranked scoring rows from a serial adapter. Ids 60-63 are padding
// (valid_vocabulary 60). Rows are recomputable: each token's own log-probability and its k-th alternative.
float own_logprob(Token t) { return -0.125F-0.001F*float(t); }
TokenLogprob alternative(Token t, std::uint32_t k) { return {Token((t+Token(k))%60),-0.125F*float(k+1)-0.001F*float(t)}; }
// Token t decodes to `repeat` copies of the letter 'a' + t % 26; there is no incremental decoder of its own.
struct Letters final : TextFrontend {
    std::size_t repeat = 1;
    std::vector<Token> tokenize(std::string_view text) const override {
        std::vector<Token> out;
        for(const char c:text) out.push_back(Token(c-'a'));
        return out;
    }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string out;
        for(const auto t:tokens) out.append(repeat,char('a'+t%26));
        return out;
    }
};
struct LogprobRecord {
    std::string fault;                       // breaks the emitting step numbered fault_step
    std::uint32_t fault_step = 1, per_step = 1;
    std::vector<Token> outputs{10,11,12,13};
    bool deferred = false;                   // rounds learn their token count after readiness
    std::shared_ptr<const TextFrontend> frontend;
    std::function<void(ModelInfo&)> declare;
    std::optional<std::uint32_t> seen_logprobs;
    ScoreQueries seen_queries;
    std::vector<std::size_t> open_live, commit_live, score_live; // host usage seen by the adapter
    std::vector<std::uint32_t> accepted;
    std::vector<bool> terminal;
    std::function<void(std::uint32_t)> before_step; // runs before each emitting step, with its number
    std::optional<Finish> closed;
};
struct LogprobSession final : Session, Readiness {
    std::shared_ptr<LogprobRecord> record;
    Resources& resources;
    std::span<const Token> prompt;
    std::optional<std::uint32_t> width;
    std::uint32_t frontier = 0, step = 0, ready = 0;
    std::size_t next = 0;
    std::vector<Token> tokens;
    std::vector<float> chosen;
    std::vector<TokenLogprob> top;
    LogprobSession(std::shared_ptr<LogprobRecord> r, Resources& res, std::span<const Token> p, std::optional<std::uint32_t> n)
        :record(std::move(r)),resources(res),prompt(p),width(n) {}
    void wait() override {}
    Progress advance(WorkBudget budget) override {
        Progress p{0,true,{}};
        if(!budget.prefill_complete) {
            p.prompt_tokens = std::min(budget.prefill_tokens,std::uint32_t(prompt.size())-frontier);
            frontier += p.prompt_tokens;
            p.prefill_complete = frontier == prompt.size();
            if(!p.prefill_complete || record->deferred) return p; // deferred rounds start after prefill
        }
        if(record->before_step) record->before_step(step);
        const auto count = std::min(record->per_step,budget.output_tokens);
        // A deferred round supplies rows for its whole grant and learns its count after readiness.
        const auto rows = record->deferred ? budget.output_tokens : count;
        tokens.clear(); chosen.clear(); top.clear();
        for(std::uint32_t i = 0; i < rows; ++i) {
            const auto t = record->outputs[(next+i)%record->outputs.size()];
            tokens.push_back(t);
            if(!width) continue;
            chosen.push_back(own_logprob(t));
            for(std::uint32_t k = 0; k < *width; ++k) top.push_back(alternative(t,k));
        }
        next += count;
        if(step++ == record->fault_step) corrupt(count);
        p.logprobs = chosen; p.top_logprobs = top;
        if(record->deferred) { ready = count; p.readiness = this; p.deferred_tokens = true; }
        else p.tokens = std::span<const Token>(tokens).first(count);
        return p;
    }
    void corrupt(std::uint32_t count) {
        const auto& f = record->fault;
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        if(f == "token_padding") tokens[0] = 61;
        else if(f == "lp_nan") chosen[0] = nan;
        else if(f == "lp_inf") chosen[0] = inf;
        else if(f == "lp_positive") chosen[0] = 0.01F;
        else if(f == "lp_tolerance") chosen[0] = 0.0005F;
        else if(f == "lp_zero") { chosen[0] = -inf; top[*width-1].logprob = -inf; }
        else if(f == "lp_count") chosen.resize(count-1);
        else if(f == "lp_long") chosen.push_back(-1.0F);
        else if(f == "lp_top_count") top.resize(count**width-1);
        else if(f == "lp_unsorted") std::swap(top[0].logprob,top[1].logprob);
        else if(f == "lp_duplicate") top[1] = top[0];
        else if(f == "lp_padding") top[0].token = 61;
        else if(f == "lp_top_nan") top[1].logprob = nan;
    }
    std::span<const Token> ready_tokens() override { return std::span<const Token>(tokens).first(ready); }
    void commit(std::uint32_t accepted, bool terminal) override {
        record->accepted.push_back(accepted); record->terminal.push_back(terminal);
        record->commit_live.push_back(resources.usage(Memory::host).live);
        // A physical commit may invalidate the borrowed rows.
        std::fill(tokens.begin(),tokens.end(),63); std::fill(chosen.begin(),chosen.end(),1.0F);
        std::fill(top.begin(),top.end(),TokenLogprob{63,1.0F});
    }
    void close(Finish finish) noexcept override { record->closed = finish; }
};
struct LogprobModel final : Model {
    std::shared_ptr<LogprobRecord> record;
    Resources& resources;
    LogprobModel(std::shared_ptr<LogprobRecord> r, Services services):record(std::move(r)),resources(services.resources) {}
    ModelInfo info() const override {
        ModelInfo m{"logprob.fixture",64,32,8,{5},true,true};
        m.query_scoring = true; m.valid_vocabulary = 60; m.token_logprobs = true; m.max_top_logprobs = 3;
        m.score_paths = {"prefill","decode"};
        if(record->declare) record->declare(m);
        return m;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        return std::make_unique<LogprobSession>(record,resources,prompt,std::nullopt);
    }
    std::unique_ptr<Session> open_request(std::span<const Token> prompt, const RequestOptions& options) override {
        record->seen_logprobs = options.logprobs;
        record->open_live.push_back(resources.usage(Memory::host).live);
        return std::make_unique<LogprobSession>(record,resources,prompt,options.logprobs);
    }
    std::shared_ptr<const TextFrontend> text_frontend() const override { return record->frontend; }
    std::vector<float> score_request(std::span<const Token> prompt, std::uint32_t first, const RequestOptions&) override {
        std::vector<float> out;
        for(std::size_t t = first; t < prompt.size(); ++t) out.push_back(own_logprob(prompt[t]));
        return out;
    }
    QueryScores score_queries(std::span<const Token> prompt, std::uint32_t first, const ScoreQueries& queries,
                              const RequestOptions&) override {
        record->seen_queries = queries;
        record->score_live.push_back(resources.usage(Memory::host).live);
        QueryScores out;
        for(std::size_t t = first; t < prompt.size(); ++t) {
            out.target.push_back(own_logprob(prompt[t]));
            for(std::uint32_t k = 0; k < queries.per_target; ++k) out.queries.push_back(-1.0F-0.125F*float(k));
            out.argmax.push_back(prompt[t]);
            for(std::uint32_t k = 0; k < queries.top_k; ++k) out.top.push_back(alternative(prompt[t],k));
        }
        const auto& f = record->fault;
        if(f == "top_count") out.top.pop_back();
        else if(f == "top_unsorted") std::swap(out.top[0].logprob,out.top[1].logprob);
        else if(f == "top_duplicate") out.top[1] = out.top[0];
        else if(f == "top_padding") out.top[0].token = 61;
        else if(f == "top_nan") out.top[0].logprob = std::numeric_limits<float>::quiet_NaN();
        else if(f == "target_positive") out.target[0] = 0.01F;
        else if(f == "query_positive") out.queries[0] = 0.01F;
        else if(f == "target_tolerance") out.target[0] = 0.0005F;
        else if(f == "argmax_padding") out.argmax[0] = 61;
        return out;
    }
};
std::unique_ptr<Engine> logprob_engine(const std::shared_ptr<LogprobRecord>& r, std::size_t host = std::size_t(1) << 16) {
    return std::make_unique<Engine>(Profile{32,8,host,host},std::make_shared<DeferredBackend>(),[r](const Profile&,Services services) {
        return std::make_unique<LogprobModel>(r,services);
    });
}
void check_rows(const Result& r, std::uint32_t width) {
    require(r.logprobs.size() == r.tokens.size() && r.top_logprobs.size() == r.tokens.size()*width,"logprob row counts");
    for(std::size_t i = 0; i < r.tokens.size(); ++i) {
        require(r.logprobs[i] == own_logprob(r.tokens[i]),"chosen log-probability");
        for(std::uint32_t k = 0; k < width; ++k) {
            const auto expected = alternative(r.tokens[i],k), actual = r.top_logprobs[i*width+k];
            require(actual.token == expected.token && actual.logprob == expected.logprob,"ranked alternative");
        }
    }
}
void declarations() {
    const std::vector<std::pair<std::function<void(ModelInfo&)>,const char*>> invalid{
        {[](ModelInfo& m){m.valid_vocabulary = 65;},"valid_vocabulary exceeds vocabulary"},
        {[](ModelInfo& m){m.max_top_logprobs = 61;},"max_top_logprobs exceeds"},
        {[](ModelInfo& m){m.valid_vocabulary = 0; m.vocabulary = 128; m.max_top_logprobs = top_logprobs_limit+1;},"top_logprobs_limit"},
        {[](ModelInfo& m){m.generation = false;},"token_logprobs requires generation"},
        {[](ModelInfo& m){m.token_logprobs = false; m.query_scoring = false;},"max_top_logprobs requires"},
        {[](ModelInfo& m){m.score_paths = {"decode","prefill","decode"};},"score paths"},
        {[](ModelInfo& m){m.score_paths = {""};},"score paths"}};
    for(const auto& [edit,message]:invalid) {
        auto r = std::make_shared<LogprobRecord>(); r->declare = edit;
        rejects([&]{logprob_engine(r);},message);
    }
    // Zero valid_vocabulary means the whole vocabulary; zero alternatives still allow own log-probabilities.
    auto whole = std::make_shared<LogprobRecord>();
    whole->declare = [](ModelInfo& m){m.valid_vocabulary = 0; m.max_top_logprobs = 64;};
    require(logprob_engine(whole)->info().max_top_logprobs == 64,"whole-vocabulary declaration rejected");
    auto own = std::make_shared<LogprobRecord>();
    own->declare = [](ModelInfo& m){m.max_top_logprobs = 0;};
    auto e = logprob_engine(own);
    RequestOptions o; o.logprobs = 0;
    const auto result = e->submit({1,2},2,o).wait();
    require(result.logprobs.size() == 2 && result.top_logprobs.empty(),"own log-probabilities without alternatives");
    o.logprobs = 1;
    rejects([&]{e->submit({1,2},2,o);},"logprobs exceeds max_top_logprobs");
    auto scoring_only = std::make_shared<LogprobRecord>();
    scoring_only->declare = [](ModelInfo& m){m.token_logprobs = false; m.score_paths.clear();};
    auto s = logprob_engine(scoring_only);
    o.logprobs = 0;
    rejects([&]{s->submit({1,2},2,o);},"does not support generation logprobs");
    require(s->submit_score({1,2,3},1,ScoreQueries{{},0,3}).wait().top_logprobs.size() == 6,"query-only alternatives");
    rejects([&]{s->submit_score({1,2,3},1,ScoreQueries{{},0,0,"decode"});},"unknown score path");
}
void logprobs() {
    auto r = std::make_shared<LogprobRecord>();
    auto e = logprob_engine(r);
    const auto base = e->usage(Memory::host).live;
    RequestOptions o; o.logprobs = 3;
    auto result = e->submit({1,2},4,o).wait();
    require(result.tokens == std::vector<Token>({10,11,12,13}) && result.finish == Finish::length,"logprob generation output");
    require(r->seen_logprobs == 3u,"logprobs not forwarded to open_request");
    check_rows(result,3);
    // Prompt, tokens, then one log-probability and n alternatives per output token.
    require(r->open_live.back() == base+2*sizeof(Token)+4*sizeof(Token)+4*sizeof(float)+4*3*sizeof(TokenLogprob),
            "logprob storage accounting");
    require(settled(*e,base),"logprob storage not released after consume");
    // Rows follow a stop token (kept) and the output budget in multi-token steps.
    r->per_step = 3; r->outputs = {10,5,12};
    result = e->submit({1,2},6,o).wait();
    require(result.tokens == std::vector<Token>({10,5}) && result.finish == Finish::stop && r->accepted.back() == 2,"stop token trim");
    check_rows(result,3);
    r->outputs = {10,11,12};
    result = e->submit({1,2},5,o).wait();
    require(result.tokens == std::vector<Token>({10,11,12,10,11}) && r->accepted.back() == 2 && r->terminal.back(),"budget rows");
    check_rows(result,3);
    // Zero alternatives; unrequested rows are never published.
    o.logprobs = 0;
    result = e->submit({1,2},2,o).wait();
    require(result.logprobs.size() == 2 && result.top_logprobs.empty(),"zero alternatives");
    result = e->submit({1,2},2).wait();
    require(result.logprobs.empty() && result.top_logprobs.empty() && !r->seen_logprobs,"unrequested rows published");
    // Rounding slightly above zero and zero probabilities (negative infinity) are valid values.
    r->per_step = 1; r->outputs = {10,11,12,13}; o.logprobs = 3;
    r->fault = "lp_tolerance";
    result = e->submit({1,2},3,o).wait();
    require(result.logprobs[1] == 0.0005F,"rounding above zero rejected");
    r->fault = "lp_zero";
    result = e->submit({1,2},3,o).wait();
    require(std::isinf(result.logprobs[1]) && result.logprobs[1] < 0 && std::isinf(result.top_logprobs[5].logprob),
            "zero probability rejected");
    r->fault.clear();
    // Stop strings trim the rows with the tokens: the completing token is kept, later ones are not.
    auto t = std::make_shared<LogprobRecord>(); t->frontend = std::make_shared<Letters>(); t->per_step = 4;
    auto te = logprob_engine(t);
    RequestOptions s; s.logprobs = 2; s.stop_strings = {"lm"};
    result = te->submit({1,2},8,s).wait();
    require(result.tokens == std::vector<Token>({10,11,12}) && result.finish == Finish::stop && t->accepted.back() == 3 &&
            t->terminal.back(),"stop string trim");
    require(result.text == "k" && result.text_ends == std::vector<std::uint32_t>({1,1,1}),"stop string text");
    check_rows(result,2);
}
void logprob_rows() {
    const std::vector<std::pair<std::string,const char*>> faults{
        {"token_padding","generated token outside the valid vocabulary"},{"lp_nan","nan, infinite or positive"},{"lp_inf","nan, infinite or positive"},
        {"lp_positive","nan, infinite or positive"},{"lp_top_nan","nan, infinite or positive"},{"lp_count","count mismatch"},
        {"lp_long","count mismatch"},{"lp_top_count","count mismatch"},{"lp_unsorted","not ranked"},
        {"lp_duplicate","not distinct"},{"lp_padding","valid vocabulary"}};
    for(const auto& [fault,message]:faults) for(const bool deferred:{false,true}) {
        if(deferred && fault == "lp_long") continue; // deferred rows may cover more than the final count
        auto r = std::make_shared<LogprobRecord>(); r->fault = fault; r->deferred = deferred; r->per_step = 2;
        auto e = logprob_engine(r);
        RequestOptions o; o.logprobs = 3;
        std::vector<Token> delivered;
        rejects([&]{e->submit({1,2},6,o).wait([&](Token token){delivered.push_back(token);});},message);
        // The first step was published; nothing of the faulty one was (deferred rounds commit their prefill first).
        require(delivered == std::vector<Token>({10,11}) && r->accepted.size() == (deferred ? 2u : 1u),"faulty step published or committed");
        r->fault.clear();
        const auto ok = e->submit({1,2},2,o).wait();
        require(ok.tokens.size() == 2 && ok.logprobs.size() == 2,"engine did not recover from a faulty row");
        require(settled(*e,0),"faulty request retained storage");
    }
}
void deferred_logprobs() {
    auto r = std::make_shared<LogprobRecord>(); r->deferred = true; r->per_step = 2;
    auto e = logprob_engine(r);
    RequestOptions o; o.logprobs = 3;
    const auto result = e->submit({1,2},5,o).wait();
    require(result.tokens == std::vector<Token>({10,11,12,13,10}) && r->accepted == std::vector<std::uint32_t>({0,2,2,1}),
            "deferred rounds");
    check_rows(result,3);
}
void wait_output() {
    struct Seen { std::size_t index; Token token; std::optional<float> logprob; std::vector<TokenLogprob> top; std::string text; };
    auto r = std::make_shared<LogprobRecord>(); r->frontend = std::make_shared<Letters>(); r->per_step = 2;
    auto e = logprob_engine(r);
    RequestOptions o; o.logprobs = 2; o.text = true;
    std::vector<Seen> seen;
    auto result = e->submit({1,2},4,o).wait_output([&](const OutputEvent& event) {
        seen.push_back({event.index,event.token,event.logprob,{event.top_logprobs.begin(),event.top_logprobs.end()},std::string(event.text)});
    });
    require(seen.size() == 4 && result.text == "klmn" && result.text_ends == std::vector<std::uint32_t>({1,2,3,4}),"output events");
    for(std::size_t i = 0; i < seen.size(); ++i) {
        require(seen[i].index == i && seen[i].token == result.tokens[i] && seen[i].logprob == result.logprobs[i],"event token row");
        require(seen[i].top.size() == 2 && seen[i].top[1].token == result.top_logprobs[2*i+1].token &&
                seen[i].top[1].logprob == result.top_logprobs[2*i+1].logprob,"event alternatives");
        require(seen[i].text == std::string(1,char('k'+i)),"event text");
    }
    // Unrequested fields stay empty; a null sink behaves like wait().
    int events = 0;
    e->submit({1,2},2).wait_output([&](const OutputEvent& event) {
        require(!event.logprob && event.top_logprobs.empty() && event.text.empty(),"unrequested output fields");
        ++events;
    });
    require(events == 2 && e->submit({1,2},3,o).wait_output({}).text == "klm","null output sink");
    // A throwing sink cancels the request and consumes the handle. Steps after the first wait for the sink to
    // throw and then pause, so the worker observes the cancellation long before the 20-token budget ends.
    auto failed = std::make_shared<std::atomic<bool>>(false);
    r->before_step = [failed](std::uint32_t step) {
        if(!step) return;
        for(int i = 0; i < 5000 && !*failed; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };
    r->accepted.clear(); r->closed.reset();
    auto handle = e->submit({1,2},20,o);
    std::size_t delivered = 0;
    rejects([&]{handle.wait_output([&](const OutputEvent&){++delivered; *failed = true; throw std::runtime_error("sink failure");});},
            "sink failure");
    rejects([&]{handle.wait_output({});},"consumed");
    rejects([&]{handle.wait();},"consumed");
    std::uint32_t committed = 0;
    for(const auto count:r->accepted) committed += count;
    require(delivered == 1 && r->closed == Finish::cancelled && committed < 20,"throwing output sink did not cancel the request");
    require(settled(*e,0),"cancelled output request retained storage");
    r->before_step = nullptr;
    require(e->submit({1,2},2,o).wait().text == "kl" && r->closed == Finish::length,"engine unusable after a sink failure");
}
void text_storage() {
    // Released text is charged as it grows: doubling chunks of at least 256 bytes.
    auto r = std::make_shared<LogprobRecord>();
    auto letters = std::make_shared<Letters>(); letters->repeat = 100;
    r->frontend = letters;
    auto e = logprob_engine(r);
    RequestOptions o; o.text = true;
    const auto result = e->submit({1,2},8,o).wait();
    require(result.text.size() == 800 && result.text_ends.back() == 800,"long text");
    const std::size_t request = 2*sizeof(Token)+8*sizeof(Token)+8*sizeof(std::uint32_t)+6+256;
    require(r->open_live.back() == request,"text request accounting");
    const std::vector<std::size_t> charged{256,256,512,512,512,1024,1024,1024};
    for(std::size_t i = 0; i < charged.size(); ++i) require(r->commit_live[i] == request+charged[i],"text growth charges");
    require(settled(*e,0),"text charges not released after consume");
    // Near the ceiling growth falls back to the exact need, then fails only this request.
    auto tight = std::make_shared<LogprobRecord>(); tight->frontend = letters;
    auto te = logprob_engine(tight,request+600);
    std::vector<Token> delivered;
    rejects([&]{te->submit({1,2},8,o).wait([&](Token t){delivered.push_back(t);});},"ceiling");
    require(delivered.size() == 6 && tight->commit_live[5] == request+600,"exact text growth near the ceiling");
    require(settled(*te,0) && te->submit({1,2},2,o).wait().text.size() == 200,"text ceiling failure poisoned the engine");
}
void logprob_scoring() {
    auto r = std::make_shared<LogprobRecord>();
    auto e = logprob_engine(r);
    // Targets alone: per_target zero with no ids.
    auto scored = e->submit_score({1,2,3,4},1,ScoreQueries{}).wait();
    require(scored.scores == std::vector<float>({own_logprob(2),own_logprob(3),own_logprob(4)}) && scored.query_scores.empty() &&
            scored.argmax == std::vector<Token>({2,3,4}) && scored.top_logprobs.empty(),"targets-only scoring");
    // Ranked alternatives per target on a named path; accounting covers ids, path and every output row.
    ScoreQueries q; q.ids = {7,8,9,10,11,12}; q.per_target = 2; q.top_k = 3; q.path = "decode";
    const auto charged = 4*sizeof(Token)+3*sizeof(float)+q.ids.capacity()*sizeof(Token)+6*sizeof(float)+3*sizeof(Token)+
                         3*3*sizeof(TokenLogprob)+q.path.capacity();
    scored = e->submit_score({1,2,3,4},1,std::move(q)).wait();
    require(r->seen_queries.path == "decode" && r->seen_queries.top_k == 3 && r->seen_queries.per_target == 2,"query forwarding");
    require(r->score_live.back() == charged,"query scoring accounting");
    require(scored.top_logprobs.size() == 9 && scored.query_scores.size() == 6,"ranked alternative rows");
    for(std::size_t t = 0; t < 3; ++t) for(std::uint32_t k = 0; k < 3; ++k) {
        const auto expected = alternative(Token(t+2),k), actual = scored.top_logprobs[t*3+k];
        require(actual.token == expected.token && actual.logprob == expected.logprob,"scored alternative");
    }
    require(settled(*e,0),"scoring storage not released after consume");
    // Submission checks: shape, top_k, path and the valid vocabulary.
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{5},0});},"per_target");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{},0,4});},"top_k exceeds max_top_logprobs");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{},0,0,"beam"});},"unknown score path");
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{{60,1},1});},"query id outside the valid vocabulary");
    // A padding target has no log-probability: submission rejects it before reserving anything or calling the
    // model. Tokens before the first target may be padding.
    const auto calls = r->score_live.size();
    for(const Token target:{60,63})
        rejects([&]{e->submit_score({1,2,target},1,ScoreQueries{});},"scoring target outside the valid vocabulary");
    rejects([&]{e->submit_score({1,62,3},1,ScoreQueries{{5,6},1,2});},"scoring target outside the valid vocabulary");
    require(r->score_live.size() == calls && e->usage(Memory::host).live == 0,"padding target reached the model or kept storage");
    require(e->submit_score({62,2,59},1,ScoreQueries{}).wait().scores == std::vector<float>({own_logprob(2),own_logprob(59)}),
            "padding context before the first target");
    RequestOptions generation; generation.logprobs = 1;
    rejects([&]{e->submit_score({1,2,3},1,ScoreQueries{},generation);},"generation options");
    auto tight = logprob_engine(std::make_shared<LogprobRecord>(),charged-1);
    rejects([&]{tight->submit_score({1,2,3,4},1,ScoreQueries{{7,8,9,10,11,12},2,3,"decode"});},"ceiling");
    require(tight->usage(Memory::host).live == 0,"rejected scoring retained storage");
    // Output checks; each failure fails only that request.
    const std::vector<std::pair<const char*,const char*>> faults{
        {"top_count","count"},{"top_unsorted","not ranked"},{"top_duplicate","not distinct"},{"top_padding","valid vocabulary"},
        {"top_nan","nan"},{"target_positive","above zero"},{"query_positive","above zero"},{"argmax_padding","argmax token outside the valid vocabulary"}};
    for(const auto& [fault,message]:faults) {
        r->fault = fault;
        rejects([&]{e->submit_score({1,2,3,4},1,ScoreQueries{{7,8,9},1,3}).wait();},message);
    }
    r->fault = "target_tolerance";
    require(e->submit_score({1,2,3},1,ScoreQueries{}).wait().scores[0] == 0.0005F,"rounding above zero rejected");
    r->fault.clear();
    require(e->submit_score({1,2,3},1,ScoreQueries{{},0,2}).wait().top_logprobs.size() == 4 && settled(*e,0),
            "scoring unusable after output failures");
}
}
// A request context whose destructor can be slowed. `destroyed` is set when the destructor finishes, so it shows the
// request's options were destroyed, not merely unreferenced. A slow destructor keeps a release by the worker, after
// the waiter returned, observable: the request's storage charge is released only after its options.
struct Probe final : RequestContext {
    std::shared_ptr<std::atomic<bool>> destroyed;
    std::size_t bytes;
    std::chrono::microseconds slow;
    Probe(std::shared_ptr<std::atomic<bool>> d, std::size_t b, std::chrono::microseconds s):destroyed(std::move(d)),bytes(b),slow(s) {}
    ~Probe() override {
        const auto until = std::chrono::steady_clock::now()+slow;
        while(std::chrono::steady_clock::now() < until) {}
        *destroyed = true;
    }
    std::size_t host_bytes() const noexcept override { return bytes; }
};
// Many short requests on every serial path, each checked right after its wait returns or its handle is destroyed:
// generation with log-probabilities, text and deferred rounds, output events, stop strings, plain and query scoring,
// cancellation, abandoned handles, a throwing sink and a failed step.
void release_race() {
    auto r = std::make_shared<LogprobRecord>(); r->frontend = std::make_shared<Letters>(); r->per_step = 2;
    r->declare = [](ModelInfo& m){ m.request_context = true; };
    auto e = logprob_engine(r);
    for(int i = 0; i < 2000; ++i) {
        auto destroyed = std::make_shared<std::atomic<bool>>(false);
        RequestOptions o; o.context = std::make_shared<Probe>(destroyed,512,std::chrono::microseconds(i%3 ? 0 : 100));
        r->deferred = i%2;
        switch(i%8) {
        case 0: o.logprobs = 2; o.text = true; require(e->submit({1,2},4,std::move(o)).wait().tokens.size() == 4,"race generation"); break;
        case 1: {
            o.logprobs = 1; o.stop_strings = {"m"};
            std::size_t events = 0;
            const auto result = e->submit({1,2},6,std::move(o)).wait_output([&](const OutputEvent&){ ++events; });
            require(result.finish == Finish::stop && events == result.tokens.size(),"race output events");
            break;
        }
        case 2: require(e->submit_score({1,2,3},1,std::move(o)).wait().scores.size() == 2,"race scoring"); break;
        case 3: require(e->submit_score({1,2,3,4},1,ScoreQueries{{7,8,9},1,2,"decode"},std::move(o)).wait().top_logprobs.size() == 6,
                        "race query scoring"); break;
        case 4: { auto h = e->submit({1,2},8,std::move(o)); h.cancel(); h.wait(); break; }
        case 5: { auto h = e->submit({1,2},8,std::move(o)); } break; // destroyed without wait
        case 6: rejects([&]{e->submit({1,2},6,std::move(o)).wait([](Token){ throw std::runtime_error("sink failure"); });},"sink failure"); break;
        default:
            r->fault = "lp_nan"; o.logprobs = 1;
            rejects([&]{e->submit({1,2},6,std::move(o)).wait();},"nan");
            r->fault.clear();
        }
        require(*destroyed,"request context outlived its settled handle");
        require(settled(*e,0),"request charges outlived its settled handle");
    }
}
// A sink that leaves wait() or wait_output() by a foreign exception (another language runtime's) or by forced
// unwinding (pthread_exit or thread cancellation, as CPython uses to end daemon threads at shutdown) cancels and settles
// its request like any throwing sink, and the unwinding continues unchanged: the foreign exception reaches the caller,
// the sink is not called again, and forced unwinding ends the waiting thread instead of aborting the process.
void sink_unwind() {
#ifdef CRIT_TEST_SINK_UNWIND
    auto r = std::make_shared<LogprobRecord>(); r->frontend = std::make_shared<Letters>();
    r->declare = [](ModelInfo& m){ m.request_context = true; };
    auto e = logprob_engine(r);
    _Unwind_Exception foreign_exception{};
    std::memcpy(&foreign_exception.exception_class,"CRITTEST",sizeof foreign_exception.exception_class);
    foreign_exception.exception_cleanup = [](_Unwind_Reason_Code,_Unwind_Exception*){};
    for(int i = 0; i < 4; ++i) {
        const bool foreign = i < 2, output = i%2;
        auto destroyed = std::make_shared<std::atomic<bool>>(false);
        RequestOptions o; o.context = std::make_shared<Probe>(destroyed,512,std::chrono::microseconds(0)); o.logprobs = 1; o.text = true;
        auto handle = e->submit({1,2},8,std::move(o));
        int calls = 0;
        bool returned = false, reached = false, empty = false;
        const auto leave = [&] {
            ++calls;
            if(!foreign) pthread_exit(nullptr);
            static_cast<void>(_Unwind_RaiseException(&foreign_exception));
            throw std::logic_error("foreign exception was not raised");
        };
        std::thread waiter([&] {
            try {
                if(output) handle.wait_output([&](const OutputEvent&){ leave(); });
                else handle.wait([&](Token){ leave(); });
                returned = true;
            } catch(...) {
                reached = true; empty = !std::current_exception();
                if(!foreign) throw; // Forced unwinding ends the thread.
            }
        });
        waiter.join();
        require(reached && !returned && empty,foreign ? "foreign sink exception did not reach the waiter" : "forced unwinding did not reach the waiter");
        require(calls == 1,"sink called again after it unwound");
        rejects([&]{handle.wait();},"consumed");
        require(*destroyed && settled(*e,0),"unwound sink left its request charged");
    }
    require(e->submit({1,2},3).wait().tokens.size() == 3 && settled(*e,0),"engine unusable after sink unwinding");
#else
    std::cout << "sink_unwind: forced unwinding and foreign exceptions are exercised only with glibc on x86-64 or AArch64\n";
#endif
}
int main(int argc,char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string,std::function<void()>>> tests{
            {"resources",resources},{"generation",ordinary},{"readiness",[]{asynchronous(false,false);}},
            {"cancel",[]{asynchronous(true,false);}},{"retained_owner",[]{asynchronous(false,true);}},
            {"retained_cancel",[]{asynchronous(true,true);}},{"errors",errors},{"score_ownership",score_ownership},{"abandoned",abandoned},
            {"startup",startup},{"borrowed_startup",borrowed_startup},{"cancel_wait",cancel_while_waiting},
            {"drain_retry",[]{failed_drain(false);}},{"drain_fatal",[]{failed_drain(true);}},{"tracing",tracing},{"deferred",deferred},{"timing",timing},{"reuse",reuse},{"query_scoring",query_scoring},
            {"declarations",declarations},{"logprobs",logprobs},{"logprob_rows",logprob_rows},{"deferred_logprobs",deferred_logprobs},
            {"wait_output",wait_output},{"text_storage",text_storage},{"logprob_scoring",logprob_scoring},
            {"release_race",release_race},{"sink_unwind",sink_unwind}};
        bool found = false;
        for(const auto& [name,run]:tests) if(test == "all" || test == name) {
            run(); found = true;
            std::cout << "crit core host contract passed: " << name << '\n';
        }
        require(found,"unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
