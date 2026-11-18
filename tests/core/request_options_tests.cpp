// Synthetic per-session sampling: validates Core forwarding, not model quality.
#include <crit/core.h>
#include <crit/json.h>
#include <crit/step_engine.h>
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
    std::atomic<bool> admit{true}; std::atomic<int> opens{0};
    std::optional<RequestOptions> seen; // options a recording model received, read after the request completes
    std::size_t observed=0;             // host usage a recording model saw at open
};
struct Turn final:Session {
    const RequestOptions& options;
    std::shared_ptr<State> state;
    std::uint32_t prompt_size, frontier=0, generated=0;
    std::vector<Token> pending;
    std::vector<float> chosen;
    std::vector<TokenLogprob> top;
    std::mt19937_64 rng;
    Turn(const RequestView& v,std::shared_ptr<State> s)
        :options(*v.options),state(std::move(s)),prompt_size(static_cast<std::uint32_t>(v.prompt.size())),
         rng(v.options->sampling.seed.value_or(0)) {++state->opens;}
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
        Progress progress{b.prefill_tokens,frontier==prompt_size,pending};
        progress.logprobs=chosen; progress.top_logprobs=top;
        return progress;
    }
    void commit(std::uint32_t count) {generated+=count; pending.clear();}
};
// Sampling temperature/top_p/top_k/seed and a request context, or (plain) none of them.
ModelInfo declaration(bool options=true) {
    auto info=ModelInfo{"options.fixture",256,32,4,{3},true,true,4};
    if(options) {info.sampling={true,true,true,true}; info.request_context=true;}
    return info;
}
// Every option: full sampling support, generation logprobs and padding ids 2040-2047.
ModelInfo full_declaration() {
    auto info=declaration();
    info.vocabulary=2048; info.valid_vocabulary=2040;
    info.sampling={true,true,true,true,true,true,true,true};
    info.token_logprobs=true; info.max_top_logprobs=4;
    return info;
}
struct Fixture:Model {
    std::shared_ptr<State> state;
    Resources& resources;
    ModelInfo declared;
    Fixture(std::shared_ptr<State> s,Resources& r,ModelInfo info):state(std::move(s)),resources(r),declared(std::move(info)) {}
    ModelInfo info() const override {return declared;}
    Admission admission(const RequestView& v) override {
        require(v.options!=nullptr,"Core omitted request options");
        return state->admit ? Admission::ready : Admission::deferred;
    }
    std::unique_ptr<Session> open(const RequestView& v) override {
        state->seen=*v.options; state->seen->context.reset(); state->observed=resources.usage(Memory::host).live;
        return std::make_unique<Turn>(v,state);
    }
    RowProgress prefill(const WorkItem& w) override {return {w.key,static_cast<Turn&>(*w.session).advance(w.budget)};}
    std::vector<RowProgress> decode(std::span<const WorkItem> rows) override {
        std::vector<RowProgress> result;
        for(const auto& row:rows) result.push_back(prefill(row));
        return result;
    }
    void commit(std::span<const CommitDecision> rows) override {
        for(const auto& row:rows) static_cast<Turn&>(*row.session).commit(row.accepted);
    }
    void retire(RequestKey,Session&,Finish) override {}
    std::vector<float> score_request(std::span<const Token> p,std::uint32_t first,const RequestOptions& o) override {
        auto c=std::dynamic_pointer_cast<const Context>(o.context);
        return std::vector<float>(p.size()-first,c ? -static_cast<float>(c->offset) : -1.0f);
    }
    // Cannot score a padding target (ids 2040-2047).
    QueryScores score_queries(std::span<const Token> p,std::uint32_t first,const ScoreQueries&,const RequestOptions&) override {
        for(std::size_t t=first;t<p.size();++t)
            if(p[t]>=2040) throw std::invalid_argument("target token outside the valid vocabulary");
        QueryScores out;
        out.target.assign(p.size()-first,-0.5F);
        return out;
    }
};
std::unique_ptr<Engine> engine(std::shared_ptr<State> s,ModelInfo info=declaration(),std::size_t host=4096) {
    return std::make_unique<Engine>(Profile{32,2,host,4096,3u,3u},
        [s,info](const Profile&,Resources& r){return std::make_unique<Fixture>(s,r,info);});
}
void stopping() {
    auto s=std::make_shared<State>();
    auto e=engine(s,declaration(false));
    require(e->submit({1},4).wait().tokens==std::vector<Token>({2,3}),"default stops changed");
    RequestOptions o; o.stop_tokens=std::vector<Token>{4};
    auto stopped=e->submit({1},4,o).wait();
    require(stopped.tokens==std::vector<Token>({2,3,4}) && stopped.finish==Finish::stop,"request stop override failed");
    o.stop_tokens=std::vector<Token>{};
    auto unlimited=e->submit({1},4,o).wait();
    require(unlimited.tokens==std::vector<Token>({2,3,4,5}) && unlimited.finish==Finish::length,"empty override did not disable stops");
}
void sampling_and_ownership() {
    auto alone=engine(std::make_shared<State>());
    auto s=std::make_shared<State>(); s->admit=false;
    auto batch=engine(s);
    auto context=std::make_shared<Context>(1,20);
    std::weak_ptr<const RequestContext> lifetime=context;
    RequestOptions a; a.sampling={0.5,0.9,12,7}; a.context=context; a.stop_tokens=std::vector<Token>{};
    auto expected=alone->submit({1},8,a).wait().tokens;
    RequestOptions b=a; b.sampling.seed=29;
    auto other=alone->submit({1},8,b).wait().tokens;
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
    require(alone->submit_score({1,2,3},1,score).wait().scores==std::vector<float>({-2,-2}),"score context was not forwarded");
}
void validation() {
    auto s=std::make_shared<State>(); auto e=engine(s);
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
    auto plain=engine(s,declaration(false));
    rejects([&]{plain->submit({1},1,o);},"sampling option");
    o={}; o.context=std::make_shared<Context>(0,5);
    rejects([&]{plain->submit({1},1,o);},"request context");
    o.context=std::make_shared<Context>(0,std::numeric_limits<std::size_t>::max());
    rejects([&]{e->submit({1},1,o);},"overflow");
    o.context=std::make_shared<Context>(0,4096);
    rejects([&]{e->submit({1},1,o);},"ceiling");
    require(s->opens==0 && e->usage(Memory::host).live==0,"invalid options opened a session or retained memory");
    require(e->submit({1},1).wait().tokens==std::vector<Token>{2},"validation poisoned later admission");
}
using Edit=std::function<void(RequestOptions&)>;
const float infinity=std::numeric_limits<float>::infinity();
// One edit per option beyond the basic four, each valid on a fully capable model.
std::vector<std::pair<const char*,Edit>> further_edits() {
    return {{"min_p",[](RequestOptions& o){o.sampling.min_p=0.5;}},
            {"repetition_penalty",[](RequestOptions& o){o.sampling.repetition_penalty=1.1;}},
            {"frequency_penalty",[](RequestOptions& o){o.sampling.frequency_penalty=0.5;}},
            {"presence_penalty",[](RequestOptions& o){o.sampling.presence_penalty=-0.5;}},
            {"logit_bias",[](RequestOptions& o){o.sampling.logit_bias={{7,-infinity}};}},
            {"min_tokens",[](RequestOptions& o){o.sampling.min_tokens=1;}},
            {"logprobs",[](RequestOptions& o){o.logprobs=0;}}};
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
void option_validation() {
    auto s=std::make_shared<State>(); auto e=engine(s,full_declaration(),1u<<16);
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
    for(const auto& [name,edit]:further_edits()) {
        RequestOptions o; edit(o);
        rejects([&]{e->submit_score({1,2},1,o);},"generation options");
    }
    require(s->opens==0 && e->usage(Memory::host).live==0,"invalid options opened a session or retained memory");
    // Missing capabilities: each further sampling flag and generation logprobs.
    auto basic=engine(s);
    for(const auto& [name,edit]:further_edits()) {
        RequestOptions o; edit(o);
        const std::string message=std::string_view(name)=="logprobs" ? "does not support generation logprobs" :
                                  std::string("does not support a requested sampling option: ")+name;
        rejects([&]{basic->submit({1},2,o);},message);
    }
    require(s->opens==0 && basic->usage(Memory::host).live==0,"unsupported options opened a session or retained memory");
    // Each support flag guards exactly its own fields: with every other flag declared, the rejection names the field.
    for(const auto& [field,flag,edit]:sampling_flags()) {
        auto info=full_declaration();
        info.sampling.*flag=false;
        auto partial=engine(s,info,1u<<16);
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
    edge.logprobs=4; edge.sampling.min_tokens=0;
    require(e->submit({1},2,edge).wait().tokens==std::vector<Token>({2,3}),"boundary options rejected");
}
void option_forwarding() {
    auto s=std::make_shared<State>();
    auto e=engine(s,full_declaration(),1u<<16);
    RequestOptions o;
    o.sampling.min_p=0.25; o.sampling.repetition_penalty=1.5; o.sampling.frequency_penalty=-1.0; o.sampling.presence_penalty=0.75;
    o.sampling.logit_bias.reserve(8); o.sampling.logit_bias={{0,1.5F},{2039,-infinity},{7,-0.5F}};
    o.sampling.min_tokens=2; o.logprobs=3;
    const auto sent=o;
    // The prompt, the result and its staging row, retained option storage (logit_bias capacity), then the logprob outputs.
    const std::size_t expected=1*sizeof(Token)+4*sizeof(Token)+4*sizeof(Token)+o.sampling.logit_bias.capacity()*sizeof(LogitBias)+
                               4*sizeof(float)+4*3*sizeof(TokenLogprob);
    auto moved=std::move(o);
    const auto result=e->submit({1},4,std::move(moved)).wait();
    require(s->observed==expected,"option storage accounting");
    require(result.tokens==std::vector<Token>({2,3}) && result.finish==Finish::stop,"forwarded options changed output");
    require(result.logprobs==std::vector<float>({-1.0F,-1.0F}) && result.top_logprobs.size()==6,"logprob outputs");
    const auto& seen=*s->seen;
    require(seen.sampling.min_p==sent.sampling.min_p && seen.sampling.repetition_penalty==sent.sampling.repetition_penalty &&
            seen.sampling.frequency_penalty==sent.sampling.frequency_penalty && seen.sampling.presence_penalty==sent.sampling.presence_penalty &&
            seen.sampling.min_tokens==sent.sampling.min_tokens && seen.logprobs==sent.logprobs,"options changed before open");
    require(seen.sampling.logit_bias.size()==3 && seen.sampling.logit_bias[1].token==2039 && seen.sampling.logit_bias[1].bias==-infinity,
            "logit bias changed before open");
}
// Submission rejects a padding scoring target, so it never reaches the model, whose failure would poison the engine.
void padding_target() {
    auto s=std::make_shared<State>();
    auto info=full_declaration(); info.query_scoring=true;
    auto e=engine(s,info);
    for(const Token target:{2040,2047})
        rejects([&]{e->submit_score({1,2,target},1,ScoreQueries{});},"scoring target outside the valid vocabulary");
    require(e->usage(Memory::host).live==0,"rejected padding target retained memory");
    require(e->submit_score({2045,2,2039},1,ScoreQueries{}).wait().scores==std::vector<float>({-0.5F,-0.5F}),
            "query scoring after a padding target");
    require(e->submit({1},2).wait().tokens==std::vector<Token>({2,3}),"generation after a padding target");
}
// A step engine hands a request's options to its callbacks as JSON a Python step model reads with json.loads, a
// banned token's -inf bias included (as Python spells it, -Infinity).
std::string step_options;
int step_open(void*,const StepRequest* request,std::uint64_t* session) {step_options=request->options_json; *session=1; return 0;}
int step_rows(void*,const StepRow* rows,std::uint32_t count,StepProgress* progress,std::int32_t* tokens,float*,std::int32_t*,float*) {
    for(std::uint32_t i=0;i<count;++i) {
        progress[i]={};
        progress[i].prompt_tokens=rows[i].prefill_complete ? 0 : rows[i].prefill_tokens;
        progress[i].prefill_complete=1; progress[i].tokens=1; tokens[i]=2;
    }
    return 0;
}
void step_engine_option_forwarding() {
    StepCallbacks callbacks; callbacks.open=step_open; callbacks.step=step_rows;
    auto native=step_engine(R"({"engine":{"vocabulary":8,"context":64,"sampling":{"temperature":true,"logit_bias":true}}})",callbacks);
    RequestOptions o; o.sampling.temperature=0.5;
    o.sampling.logit_bias={{3,-std::numeric_limits<float>::infinity()},{4,1.5F}};
    require(native.engine->submit({1,2},1,o).wait().tokens==std::vector<Token>({2}),"step engine output");
    const auto document=json::Document::parse(step_options,"step options",{.allow_nonfinite=true});
    const auto* bias=document.root().find("logit_bias");
    require(bias && bias->find("3") && bias->find("3")->real()==-std::numeric_limits<double>::infinity() && bias->find("4")->real()==1.5 &&
            document.root().find("temperature")->real()==0.5,"step engine options as JSON with a banned token");
}
}
int main() {
    try {
        stopping(); sampling_and_ownership(); validation(); option_validation(); option_forwarding(); padding_target();
        step_engine_option_forwarding();
        std::cout<<"Core request option contracts passed\n";
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
