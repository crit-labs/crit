// Synthetic per-session sampling: validates Core forwarding, not model quality.
#include <crit/core.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
using namespace crit::core;
namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
template<class F> void rejects(F action,std::string_view message) {
    try {action();} catch(const std::exception& e) {
        require(std::string_view(e.what()).find(message)!=std::string_view::npos,"unexpected rejection"); return;
    }
    throw std::runtime_error("expected rejection missing");
}
struct Context final:RequestContext {
    Token offset;
    std::size_t bytes;
    Context(Token value,std::size_t extent):offset(value),bytes(extent) {}
    std::size_t host_bytes() const noexcept override {return bytes;}
};
struct State {
    std::atomic<bool> admit{true}, fail_drain{false}; std::atomic<int> opens{0}, closes{0};
    std::optional<RequestOptions> seen; // options a recording adapter received, read after the request completes
};
struct Host final:Backend {
    std::shared_ptr<State> state;
    explicit Host(std::shared_ptr<State> s):state(std::move(s)) {}
    void bind_thread() override {}
    void drain() override {if(state->fail_drain) throw std::runtime_error("fixture drain failure");}
};
struct Turn {
    const RequestOptions& options;
    std::shared_ptr<State> state;
    std::uint32_t prompt_size, frontier=0, generated=0;
    std::vector<Token> pending;
    std::vector<float> chosen;
    std::vector<TokenLogprob> top;
    std::mt19937_64 rng;
    Turn(const RequestOptions& o,std::shared_ptr<State> s,std::size_t prompt)
        :options(o),state(std::move(s)),prompt_size(static_cast<std::uint32_t>(prompt)),rng(o.sampling.seed.value_or(0)) {++state->opens;}
    Progress advance(WorkBudget b) {
        frontier+=b.prefill_tokens;
        pending.clear();
        if(frontier==prompt_size) for(std::uint32_t i=0;i<std::min(4u,b.output_tokens);++i) {
            const auto context=std::dynamic_pointer_cast<const Context>(options.context);
            pending.push_back((options.sampling.seed ? 100+Token(rng()%100) : 2+Token(generated+i))+
                              (context ? context->offset : 0));
        }
        // With logprobs: each token's own log-probability and n ranked alternatives.
        chosen.clear(); top.clear();
        if(options.logprobs) for(const auto token:pending) {
            chosen.push_back(-1.0F);
            for(std::uint32_t k=0;k<*options.logprobs;++k) top.push_back({Token((token+Token(k))%250),-1.0F-float(k)});
        }
        Progress progress{b.prefill_tokens,frontier==prompt_size,pending,nullptr};
        progress.logprobs=chosen; progress.top_logprobs=top;
        return progress;
    }
    void commit(std::uint32_t count) {generated+=count; pending.clear();}
};
struct SerialTurn final:Session {
    Turn turn;
    SerialTurn(const RequestOptions& o,std::shared_ptr<State> s,std::size_t prompt):turn(o,std::move(s),prompt) {}
    Progress advance(WorkBudget b) override {return turn.advance(b);}
    void commit(std::uint32_t n,bool) override {turn.commit(n);}
    void close(Finish) noexcept override {++turn.state->closes;}
};
ModelInfo declaration(bool options=true) {
    auto info=ModelInfo{"options.fixture",256,32,4,{3},true,true,4};
    if(options) {info.sampling={true,true,true,true}; info.request_context=true;}
    return info;
}
struct Legacy:Model {
    std::shared_ptr<State> state;
    const RequestOptions defaults;
    explicit Legacy(std::shared_ptr<State> s):state(std::move(s)) {}
    ModelInfo info() const override {return declaration(false);}
    std::unique_ptr<Session> open(std::span<const Token> p) override {return std::make_unique<SerialTurn>(defaults,state,p.size());}
    std::vector<Token> tokenize(std::string_view) override {return {1};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
};
struct Serial final:Legacy {
    using Legacy::Legacy;
    ModelInfo info() const override {return declaration();}
    std::unique_ptr<Session> open_request(std::span<const Token> p,const RequestOptions& o) override {
        if(o.sampling.temperature) require(*o.sampling.temperature==0.5 && o.sampling.top_p==0.9 && o.sampling.top_k==12u,
                                          "sampling values changed before open");
        return std::make_unique<SerialTurn>(o,state,p.size());
    }
    std::vector<float> score_request(std::span<const Token> p,std::uint32_t first,const RequestOptions& o) override {
        auto c=std::dynamic_pointer_cast<const Context>(o.context);
        return std::vector<float>(p.size()-first,c ? -static_cast<float>(c->offset) : -1.0f);
    }
};
struct ConcurrentTurn final:ConcurrentSession {
    Turn turn;
    explicit ConcurrentTurn(const RequestView& v,std::shared_ptr<State> s):turn(*v.options,std::move(s),v.prompt.size()) {}
    void close(Finish) noexcept override {++turn.state->closes;}
};
struct Settled final:Retirement {bool ready() override {return true;}};
struct Concurrent:ConcurrentModel {
    std::shared_ptr<State> state;
    explicit Concurrent(std::shared_ptr<State> s):state(std::move(s)) {}
    ModelInfo info() const override {return declaration();}
    Admission admission(const RequestView& v) override {
        require(v.options!=nullptr,"Core omitted request options");
        return state->admit ? Admission::ready : Admission::deferred;
    }
    std::unique_ptr<ConcurrentSession> open(const RequestView& v) override {return std::make_unique<ConcurrentTurn>(v,state);}
    RowProgress prefill(const WorkItem& w) override {return {w.key,static_cast<ConcurrentTurn&>(*w.session).turn.advance(w.budget)};}
    std::vector<RowProgress> decode(std::span<const WorkItem> rows) override {
        std::vector<RowProgress> result;
        for(const auto& row:rows) result.push_back(prefill(row));
        return result;
    }
    void commit(std::span<const CommitDecision> rows) override {
        for(const auto& row:rows) static_cast<ConcurrentTurn&>(*row.session).turn.commit(row.accepted);
    }
    std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession&,Finish) override {return std::make_unique<Settled>();}
    void quiesce() override {}
    std::vector<Token> tokenize(std::string_view) override {return {1};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
};
template<class Adapter> std::unique_ptr<Engine> engine(std::shared_ptr<State> s, bool concurrent=false,std::size_t host=4096) {
    return std::make_unique<Engine>(Profile{32,2,host,4096,concurrent ? 3u : 1u,concurrent ? 3u : 0u},std::make_shared<Host>(s),
        [s](const Profile&,Services){return std::make_unique<Adapter>(s);});
}
void stopping() {
    for(bool concurrent:{false,true}) {
        auto s=std::make_shared<State>();
        auto e=concurrent ? engine<Concurrent>(s,true) : engine<Legacy>(s);
        require(e->submit({1},4).wait().tokens==std::vector<Token>({2,3}),"default stops changed");
        RequestOptions o; o.stop_tokens=std::vector<Token>{4};
        auto stopped=e->submit({1},4,o).wait();
        require(stopped.tokens==std::vector<Token>({2,3,4}) && stopped.finish==Finish::stop,"request stop override failed");
        o.stop_tokens=std::vector<Token>{};
        auto unlimited=e->submit({1},4,o).wait();
        require(unlimited.tokens==std::vector<Token>({2,3,4,5}) && unlimited.finish==Finish::length,"empty override did not disable stops");
    }
}
void sampling_and_ownership() {
    auto serial=engine<Serial>(std::make_shared<State>());
    auto s=std::make_shared<State>(); s->admit=false;
    auto batch=engine<Concurrent>(s,true);
    auto context=std::make_shared<Context>(1,20);
    std::weak_ptr<const RequestContext> lifetime=context;
    RequestOptions a; a.sampling={0.5,0.9,12,7}; a.context=context; a.stop_tokens=std::vector<Token>{};
    auto expected=serial->submit({1},8,a).wait().tokens;
    RequestOptions b=a; b.sampling.seed=29;
    auto other=serial->submit({1},8,b).wait().tokens;
    require(expected!=other,"different seeds did not affect fixture output");
    auto first=batch->submit({1},8,a), second=batch->submit({1},8,b);
    // Caller edits cannot alter admitted option values; context ownership survives.
    a.sampling.seed=99; b.sampling.seed=99; a.context.reset(); b.context.reset(); context.reset();
    require(!lifetime.expired(),"queued request lost context");
    require(batch->usage(Memory::host).live==2*(4+32+32+20),"request context escaped accounting");
    s->admit=true;
    require(first.wait().tokens==expected && second.wait().tokens==other,"batch membership changed private RNG streams");
    // Returned waits have already released each request's options and charges.
    require(lifetime.expired() && batch->usage(Memory::host).live==0,"settled handles retained context or charges");
    auto replay=a; replay.sampling.seed=7; replay.context=std::make_shared<Context>(1,20);
    require(batch->submit({1},8,replay).wait().tokens==expected,"lane reuse inherited RNG state");
    RequestOptions score; score.context=std::make_shared<Context>(2,20);
    require(serial->submit_score({1,2,3},1,score).wait().scores==std::vector<float>({-2,-2}),"score context was not forwarded");
}
void validation() {
    auto s=std::make_shared<State>(); auto e=engine<Serial>(s);
    for(auto value:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        RequestOptions o; o.sampling.temperature=value;
        rejects([&]{e->submit({1},1,o);},"temperature");
    }
    for(auto value:{-1.0,0.0,1.1,std::numeric_limits<double>::quiet_NaN()}) {
        RequestOptions o; o.sampling.top_p=value;
        rejects([&]{e->submit({1},1,o);},"top_p");
    }
    RequestOptions o; o.sampling.top_k=257;
    rejects([&]{e->submit({1},1,o);},"top_k");
    o={}; o.stop_tokens=std::vector<Token>{256};
    rejects([&]{e->submit({1},1,o);},"vocabulary");
    o={}; o.sampling.seed=0;
    rejects([&]{e->submit_score({1,2},1,o);},"generation options");
    auto legacy=engine<Legacy>(s);
    rejects([&]{legacy->submit({1},1,o);},"sampling option");
    o={}; o.context=std::make_shared<Context>(0,5);
    rejects([&]{legacy->submit({1},1,o);},"request context");
    o.context=std::make_shared<Context>(0,std::numeric_limits<std::size_t>::max());
    rejects([&]{e->submit({1},1,o);},"overflow");
    o.context=std::make_shared<Context>(0,4096);
    rejects([&]{e->submit({1},1,o);},"ceiling");
    require(s->opens==0 && e->usage(Memory::host).live==0,"invalid options opened a session or retained memory");
    require(e->submit({1},1).wait().tokens==std::vector<Token>{2},"validation poisoned later admission");
}
void missing_hook_and_failed_drain() {
    struct Missing final:Legacy {
        using Legacy::Legacy;
        ModelInfo info() const override {return declaration();}
    };
    auto state=std::make_shared<State>();
    auto missing=engine<Missing>(state);
    RequestOptions seeded; seeded.sampling.seed=0;
    rejects([&]{missing->submit({1},1,seeded).wait();},"implement open_request");
    require(state->opens==0,"legacy adapter silently ignored sampling");
    auto e=engine<Serial>(state);
    RequestOptions o; o.context=std::make_shared<Context>(0,20);
    std::weak_ptr<const RequestContext> lifetime=o.context;
    state->fail_drain=true;
    rejects([&]{e->submit({1},1,std::move(o)).wait();},"drain failure");
    require(!lifetime.expired() && state->closes==0,"failed drain released borrowed options");
    const auto before=quarantined_engines();
    e.reset();
    require(quarantined_engines()==before+1 && !lifetime.expired(),"quarantine lost request options");
    state->fail_drain=false; // Allow the unrelated missing-hook fixture to retire.
}
// Every adapter option: full sampling support, generation logprobs, a text frontend and padding ids 2040-2047.
struct Letters final:TextFrontend {
    std::vector<Token> tokenize(std::string_view text) const override {
        std::vector<Token> result;
        for(const char c:text) result.push_back(Token(c-'a'));
        return result;
    }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string result;
        for(const auto token:tokens) result.push_back(char('a'+token%26));
        return result;
    }
};
ModelInfo full_declaration() {
    auto info=declaration();
    info.vocabulary=2048; info.valid_vocabulary=2040;
    info.sampling={true,true,true,true,true,true,true,true};
    info.token_logprobs=true; info.max_top_logprobs=4;
    return info;
}
// Hook selects an adapter that implements open_request (recording options and host usage) or keeps the default.
template<bool Hook> struct Full final:Model {
    std::shared_ptr<State> state;
    Resources& resources;
    const RequestOptions defaults;
    std::shared_ptr<std::size_t> observed;
    Full(std::shared_ptr<State> s,Resources& r,std::shared_ptr<std::size_t> o):state(std::move(s)),resources(r),observed(std::move(o)) {}
    ModelInfo info() const override {return full_declaration();}
    std::unique_ptr<Session> open(std::span<const Token> p) override {return std::make_unique<SerialTurn>(defaults,state,p.size());}
    std::unique_ptr<Session> open_request(std::span<const Token> p,const RequestOptions& o) override {
        if constexpr(!Hook) return Model::open_request(p,o);
        else {
            state->seen=o; *observed=resources.usage(Memory::host).live;
            return std::make_unique<SerialTurn>(o,state,p.size());
        }
    }
    std::shared_ptr<const TextFrontend> text_frontend() const override {return std::make_shared<Letters>();}
    std::vector<Token> tokenize(std::string_view) override {return {1};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
};
template<bool Hook> std::unique_ptr<Engine> full_engine(std::shared_ptr<State> s,std::shared_ptr<std::size_t> observed=std::make_shared<std::size_t>()) {
    return std::make_unique<Engine>(Profile{32,2,1u<<16,4096},std::make_shared<Host>(s),[s,observed](const Profile&,Services services) {
        return std::make_unique<Full<Hook>>(s,services.resources,observed);
    });
}
struct RecordingConcurrent final:Concurrent {
    using Concurrent::Concurrent;
    ModelInfo info() const override {return full_declaration();}
    std::unique_ptr<ConcurrentSession> open(const RequestView& v) override {state->seen=*v.options; return Concurrent::open(v);}
    std::shared_ptr<const TextFrontend> text_frontend() const override {return std::make_shared<Letters>();}
};
using Edit=std::function<void(RequestOptions&)>;
const float infinity=std::numeric_limits<float>::infinity();
// One edit per adapter option, each valid on a fully capable model.
std::vector<std::pair<const char*,Edit>> adapter_edits() {
    return {{"min_p",[](RequestOptions& o){o.sampling.min_p=0.5;}},
            {"repetition_penalty",[](RequestOptions& o){o.sampling.repetition_penalty=1.1;}},
            {"frequency_penalty",[](RequestOptions& o){o.sampling.frequency_penalty=0.5;}},
            {"presence_penalty",[](RequestOptions& o){o.sampling.presence_penalty=-0.5;}},
            {"logit_bias",[](RequestOptions& o){o.sampling.logit_bias={{7,-infinity}};}},
            {"min_tokens",[](RequestOptions& o){o.sampling.min_tokens=1;}},
            {"logprobs",[](RequestOptions& o){o.logprobs=0;}},
            {"stop_strings",[](RequestOptions& o){o.stop_strings={"q"};}},
            {"text",[](RequestOptions& o){o.text=true;}}};
}
// A fully capable model but for the sampling flags it declares.
struct Lacking final:Model {
    std::shared_ptr<State> state;
    SamplingSupport support;
    const RequestOptions defaults;
    Lacking(std::shared_ptr<State> s,SamplingSupport flags):state(std::move(s)),support(flags) {}
    ModelInfo info() const override {auto info=full_declaration(); info.sampling=support; return info;}
    std::unique_ptr<Session> open(std::span<const Token> p) override {return std::make_unique<SerialTurn>(defaults,state,p.size());}
    std::unique_ptr<Session> open_request(std::span<const Token> p,const RequestOptions& o) override {
        return std::make_unique<SerialTurn>(o,state,p.size());
    }
};
std::unique_ptr<Engine> lacking_engine(std::shared_ptr<State> s,SamplingSupport flags) {
    return std::make_unique<Engine>(Profile{32,2,4096,4096},std::make_shared<Host>(s),[s,flags](const Profile&,Services) {
        return std::make_unique<Lacking>(s,flags);
    });
}
// Every sampling field with the support flag that guards it and a valid value.
std::vector<std::tuple<const char*,bool SamplingSupport::*,Edit>> sampling_flags() {
    return {{"temperature",&SamplingSupport::temperature,[](RequestOptions& o){o.sampling.temperature=0.5;}},
            {"top_p",&SamplingSupport::top_p,[](RequestOptions& o){o.sampling.top_p=0.9;}},
            {"top_k",&SamplingSupport::top_k,[](RequestOptions& o){o.sampling.top_k=12u;}},
            {"seed",&SamplingSupport::seed,[](RequestOptions& o){o.sampling.seed=3u;}},
            {"min_p",&SamplingSupport::min_p,[](RequestOptions& o){o.sampling.min_p=0.5;}},
            {"repetition_penalty",&SamplingSupport::penalties,[](RequestOptions& o){o.sampling.repetition_penalty=1.1;}},
            {"frequency_penalty",&SamplingSupport::penalties,[](RequestOptions& o){o.sampling.frequency_penalty=0.5;}},
            {"presence_penalty",&SamplingSupport::penalties,[](RequestOptions& o){o.sampling.presence_penalty=-0.5;}},
            {"logit_bias",&SamplingSupport::logit_bias,[](RequestOptions& o){o.sampling.logit_bias={{7,-infinity}};}},
            {"min_tokens",&SamplingSupport::min_tokens,[](RequestOptions& o){o.sampling.min_tokens=1u;}}};
}
void adapter_option_validation() {
    auto s=std::make_shared<State>(); auto e=full_engine<true>(s);
    const auto nan=std::numeric_limits<double>::quiet_NaN(), inf=std::numeric_limits<double>::infinity();
    const auto check=[&](const Edit& edit,std::string_view message) {
        RequestOptions o; edit(o);
        rejects([&]{e->submit({1},2,o);},message);
    };
    for(auto v:{-0.1,1.1,nan,inf}) check([v](RequestOptions& o){o.sampling.min_p=v;},"min_p");
    for(auto v:{0.0,-1.0,inf,nan}) check([v](RequestOptions& o){o.sampling.repetition_penalty=v;},"repetition_penalty");
    for(auto v:{-2.5,2.5,inf,nan}) {
        check([v](RequestOptions& o){o.sampling.frequency_penalty=v;},"frequency_penalty");
        check([v](RequestOptions& o){o.sampling.presence_penalty=v;},"presence_penalty");
    }
    check([](RequestOptions& o){for(Token t=0;t<1025;++t) o.sampling.logit_bias.push_back({t,1.0F});},"1024 entries");
    for(Token t:{-1,2040,2047,2048}) check([t](RequestOptions& o){o.sampling.logit_bias={{0,1.0F},{t,1.0F}};},"valid vocabulary");
    for(float b:{std::numeric_limits<float>::quiet_NaN(),infinity})
        check([b](RequestOptions& o){o.sampling.logit_bias={{1,b}};},"finite or negative infinity");
    check([](RequestOptions& o){o.sampling.logit_bias={{4,1.0F},{9,-1.0F},{4,2.0F}};},"unique");
    check([](RequestOptions& o){o.logprobs=5;},"max_top_logprobs");
    check([](RequestOptions& o){o.stop_strings=std::vector<std::string>(17,"x");},"16 entries");
    check([](RequestOptions& o){o.stop_strings={"x",""};},"nonempty");
    check([](RequestOptions& o){o.stop_strings={std::string(257,'x')};},"256 bytes");
    // A stop that is not well-formed UTF-8 could match inside a character of the output.
    for(const char* stop:{"\x97","\xE6\x97","\xC0\x80","\xED\xA0\x80","\xFF"})
        check([stop](RequestOptions& o){o.stop_strings={"ok",stop};},"stop_strings entry is not valid utf-8");
    for(const auto& [name,edit]:adapter_edits()) {
        RequestOptions o; edit(o);
        rejects([&]{e->submit_score({1,2},1,o);},"generation options");
    }
    require(s->opens==0 && e->usage(Memory::host).live==0,"invalid adapter options opened a session or retained memory");
    // Missing capabilities: each sampling flag, generation logprobs and the text frontend.
    auto serial=engine<Serial>(s);
    for(const auto& [name,edit]:adapter_edits()) {
        RequestOptions o; edit(o);
        const std::string_view option=name;
        const std::string message=option=="logprobs" ? "does not support generation logprobs" :
                                  option=="stop_strings" ? "stop strings require an independent text frontend" :
                                  option=="text" ? "text output requires an independent text frontend" :
                                  std::string("does not support a requested sampling option: ")+name;
        rejects([&]{serial->submit({1},2,o);},message);
    }
    require(s->opens==0 && serial->usage(Memory::host).live==0,"unsupported options opened a session or retained memory");
    // Each support flag guards exactly its own fields: with every other flag declared, the rejection names the field.
    for(const auto& [field,flag,edit]:sampling_flags()) {
        SamplingSupport lacking{true,true,true,true,true,true,true,true};
        lacking.*flag=false;
        auto partial=lacking_engine(s,lacking);
        RequestOptions o; edit(o);
        rejects([&]{partial->submit({1},2,o);},std::string("does not support a requested sampling option: ")+field);
        require(s->opens==0 && partial->usage(Memory::host).live==0,"unsupported flag opened a session or retained memory");
        for(const auto& [other,other_flag,other_edit]:sampling_flags()) if(other_flag!=flag) {
            RequestOptions accepted; other_edit(accepted);
            require(partial->submit({1},2,accepted).wait().tokens.size()==2,"a declared sampling option was rejected");
        }
        s->opens=0;
    }
    // Boundary values are valid.
    RequestOptions edge;
    edge.sampling.min_p=0.0; edge.sampling.repetition_penalty=1e-9; edge.sampling.frequency_penalty=-2.0; edge.sampling.presence_penalty=2.0;
    for(Token t=0;t<1024;++t) edge.sampling.logit_bias.push_back({t,t%2 ? -infinity : 0.5F});
    edge.stop_strings=std::vector<std::string>(16,std::string(256,'z')); edge.logprobs=4; edge.sampling.min_tokens=0;
    require(e->submit({1},2,edge).wait().tokens==std::vector<Token>({2,3}),"boundary adapter options rejected");
}
void adapter_option_forwarding() {
    auto s=std::make_shared<State>(); auto observed=std::make_shared<std::size_t>();
    auto e=full_engine<true>(s,observed);
    RequestOptions o;
    o.sampling.min_p=0.25; o.sampling.repetition_penalty=1.5; o.sampling.frequency_penalty=-1.0; o.sampling.presence_penalty=0.75;
    o.sampling.logit_bias.reserve(8); o.sampling.logit_bias={{0,1.5F},{2039,-infinity},{7,-0.5F}};
    o.sampling.min_tokens=2; o.logprobs=3; o.stop_strings={"zz","yyy"};
    const auto sent=o;
    // Retained option storage (logit_bias and stop string capacities), then the logprob and text outputs.
    const std::size_t expected=1*sizeof(Token)+4*sizeof(Token)+o.sampling.logit_bias.capacity()*sizeof(LogitBias)+
                               o.stop_strings.capacity()*sizeof(std::string)+o.stop_strings[0].capacity()+o.stop_strings[1].capacity()+
                               4*sizeof(float)+4*3*sizeof(TokenLogprob)+4*sizeof(std::uint32_t)+(2+3)*2+3*6+256;
    auto moved=std::move(o);
    const auto result=e->submit({1},4,std::move(moved)).wait();
    require(*observed==expected,"adapter option storage accounting");
    require(result.tokens==std::vector<Token>({2,3}) && result.finish==Finish::stop,"forwarded options changed output");
    require(result.text=="c" && result.logprobs==std::vector<float>({-1.0F,-1.0F}) && result.top_logprobs.size()==6,"text/logprob outputs");
    const auto& seen=*s->seen;
    require(seen.sampling.min_p==sent.sampling.min_p && seen.sampling.repetition_penalty==sent.sampling.repetition_penalty &&
            seen.sampling.frequency_penalty==sent.sampling.frequency_penalty && seen.sampling.presence_penalty==sent.sampling.presence_penalty &&
            seen.sampling.min_tokens==sent.sampling.min_tokens && seen.logprobs==sent.logprobs && seen.stop_strings==sent.stop_strings &&
            !seen.text,"serial adapter options changed before open_request");
    require(seen.sampling.logit_bias.size()==3 && seen.sampling.logit_bias[1].token==2039 && seen.sampling.logit_bias[1].bias==-infinity,
            "logit bias changed before open_request");
    // Concurrent admission and open see the same values.
    auto c=std::make_shared<State>();
    auto batch=std::make_unique<Engine>(Profile{32,2,1u<<16,4096,3,3},std::make_shared<Host>(c),
        [c](const Profile&,Services){return std::make_unique<RecordingConcurrent>(c);});
    auto text=sent; text.text=true;
    const auto concurrent=batch->submit({1},4,text).wait();
    require(concurrent.tokens==result.tokens && concurrent.text=="c" && std::equal(concurrent.top_logprobs.begin(),concurrent.top_logprobs.end(),
            result.top_logprobs.begin(),result.top_logprobs.end(),[](TokenLogprob a,TokenLogprob b){return a.token==b.token && a.logprob==b.logprob;}),
            "concurrent adapter outputs");
    require(c->seen && c->seen->stop_strings==sent.stop_strings && c->seen->logprobs==3u && c->seen->text &&
            c->seen->sampling.logit_bias.size()==3 && c->seen->sampling.min_p==0.25,"concurrent adapter options changed before open");
}
// A concurrent query-scoring adapter that, like score_logit_rows, cannot score a padding target (ids 2040-2047).
struct QueryConcurrent final:Concurrent {
    using Concurrent::Concurrent;
    ModelInfo info() const override {auto info=full_declaration(); info.query_scoring=true; return info;}
    QueryScores score_queries(std::span<const Token> p,std::uint32_t first,const ScoreQueries&,const RequestOptions&) override {
        for(std::size_t t=first;t<p.size();++t)
            if(p[t]>=2040) throw std::invalid_argument("target token outside the valid vocabulary");
        QueryScores out;
        out.target.assign(p.size()-first,-0.5F);
        return out;
    }
};
// Submission rejects a padding scoring target, so it never reaches the hook, whose failure would poison the engine.
void concurrent_padding_target() {
    auto s=std::make_shared<State>();
    auto e=engine<QueryConcurrent>(s,true);
    for(const Token target:{2040,2047})
        rejects([&]{e->submit_score({1,2,target},1,ScoreQueries{});},"scoring target outside the valid vocabulary");
    require(e->usage(Memory::host).live==0,"rejected padding target retained memory");
    require(e->submit_score({2045,2,2039},1,ScoreQueries{}).wait().scores==std::vector<float>({-0.5F,-0.5F}),
            "query scoring after a padding target");
    require(e->submit({1},2).wait().tokens==std::vector<Token>({2,3}),"generation after a padding target");
}
void legacy_open_refusal() {
    auto s=std::make_shared<State>(); auto missing=full_engine<false>(s);
    for(const auto& [name,edit]:adapter_edits()) {
        RequestOptions o; edit(o);
        rejects([&]{missing->submit({1},2,o).wait();},"implement open_request");
    }
    require(s->opens==0,"legacy adapter silently ignored adapter options");
    // Stop tokens are Core's own option; legacy open still serves them.
    RequestOptions stops; stops.stop_tokens=std::vector<Token>{};
    require(missing->submit({1},2,stops).wait().tokens==std::vector<Token>({2,3}) && s->opens==1,"legacy stop tokens refused");
}
}
int main() {
    try {
        stopping(); sampling_and_ownership(); validation(); missing_hook_and_failed_drain();
        adapter_option_validation(); adapter_option_forwarding(); legacy_open_refusal(); concurrent_padding_target();
        std::cout<<"Core request option contracts passed\n";
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
