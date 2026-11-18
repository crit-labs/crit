#include "crit/core.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

using namespace crit::core;
namespace {
void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
enum class Path { separate, mixed };
struct Step {
    std::uint32_t prompt;
    bool complete;
    std::vector<Token> tokens = {};
    std::uint32_t reused = 0;
    std::optional<std::uint32_t> work = {};
};
struct Record {
    std::vector<Step> steps;
    std::size_t advances=0, commits=0;
    bool closed=false;
    Finish finish=Finish::failed;
};
// A scripted model reports progress through separate prefill/decode callbacks or mixed groups; commit invalidates
// the borrowed tokens.
struct Driver {
    std::shared_ptr<Record> record;
    std::vector<Token> tokens;
    explicit Driver(std::shared_ptr<Record> r):record(std::move(r)) {}
    Progress advance(WorkBudget budget) {
        require(record->advances==record->commits,"advanced past an uncommitted step");
        const auto& step=record->steps.at(record->advances++);
        tokens=step.tokens;
        Progress p{step.prompt,step.complete,tokens,step.reused};
        if(budget.token_work) {
            // Most cases isolate the common progress rules. Explicit work overrides
            // also exercise the mixed path's separate accounting and error precedence.
            p.token_work=step.work.value_or(budget.prefill_complete ? 1U :
                step.prompt-std::min(step.prompt,step.reused));
        }
        return p;
    }
    void commit(std::uint32_t accepted) {
        require(accepted<=tokens.size(),"commit exceeded returned tokens");
        ++record->commits;
        std::fill(tokens.begin(),tokens.end(),-1);
    }
};
ModelInfo info(Path path) {
    ModelInfo m{"progress.contract.fixture",16,32,2,{}};
    m.valid_vocabulary=12; m.max_concurrency=1;
    m.mixed_batches=path==Path::mixed; m.decode_token_work=4;
    return m;
}
struct Lane final:Session {
    Driver driver;
    explicit Lane(std::shared_ptr<Record> r):driver(std::move(r)) {}
};
struct BatchedModel final:Model {
    std::shared_ptr<Record> record;
    Path path;
    Lane* lane=nullptr;
    BatchedModel(std::shared_ptr<Record> r,Path p):record(std::move(r)),path(p) {}
    ModelInfo info() const override { return ::info(path); }
    std::unique_ptr<Session> open(const RequestView&) override {
        auto session=std::make_unique<Lane>(record); lane=session.get(); return session;
    }
    RowProgress prefill(const WorkItem& item) override { return {item.key,lane->driver.advance(item.budget)}; }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        require(items.size()==1,"unexpected fixture group size");
        return {prefill(items.front())};
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override { return decode(items); }
    void commit(std::span<const CommitDecision> group) override {
        require(group.size()==1,"unexpected fixture commit size");
        lane->driver.commit(group.front().accepted);
    }
    void retire(RequestKey,Session&,Finish finish) override {
        require(!record->closed,"retired twice");
        record->closed=true; record->finish=finish;
    }
};
enum class Error { none, progress, work, vocabulary };
void check(Path path,const char* name,std::vector<Step> steps,Error error=Error::none) {
    try {
        auto r=std::make_shared<Record>(); r->steps=std::move(steps);
        Profile p{32,2,1<<20,1<<20};
        if(path==Path::mixed) p.max_batch_tokens=4;
        Engine e(p,[r,path](const Profile&,Resources&)->std::unique_ptr<Model> { return std::make_unique<BatchedModel>(r,path); });
        const auto live=e.usage(Memory::host).live;
        std::vector<Token> published;
        std::string diagnostic;
        Result result;
        try { result=e.submit({1,2,3,4,5},3).wait([&](Token t){published.push_back(t);}); }
        catch(const std::runtime_error& failure) { diagnostic=failure.what(); }
        catch(const std::invalid_argument& failure) { diagnostic=failure.what(); }
        const auto expected=error==Error::none ? std::string{} :
            error==Error::work ? "mixed token work exceeds grant or omits computation" :
            error==Error::vocabulary ? "generated token outside the valid vocabulary" :
            "invalid or stalled concurrent progress";
        require(diagnostic==expected,("unexpected diagnostic: "+diagnostic).c_str());
        const auto commits=r->steps.size()-(error==Error::none ? 0 : 1);
        require(r->commits==commits && r->advances==r->steps.size(),"invalid progress reached commit");
        std::vector<Token> accepted;
        std::uint32_t reused=0;
        for(std::size_t i=0;i<r->steps.size();++i) {
            const auto& step=r->steps[i];
            if(i<commits) {
                accepted.insert(accepted.end(),step.tokens.begin(),step.tokens.end());
                reused+=step.reused;
            }
        }
        require(published==accepted,"invalid progress published tokens or borrowed output was lost");
        require(r->closed==(error==Error::none) && (!r->closed || r->finish==Finish::length),"request not retired with expected finish");
        require(e.usage(Memory::host).live==live,"request charges not released");
        if(error==Error::none) {
            require(result.tokens==accepted && result.finish==Finish::length,"successful output changed");
            require(result.timing.prefill_tokens==5 && result.timing.reused_prompt_tokens==reused,"prompt accounting changed");
        }
    } catch(const std::exception& failure) { throw std::runtime_error(std::string(name)+": "+failure.what()); }
}
void cases(Path path) {
    const Step reuse{5,true,{},5}, output{0,true,{7,8,9}};
    check(path,"chunked prefill",{{2,false},{2,false},{1,true},output});
    check(path,"full prefix reuse",{reuse,output});
    check(path,"incremental prefix reuse",{{2,false,{},2},{2,false,{},2},{1,true},output});
    check(path,"reuse plus computation",{{4,false,{},2},{1,true},output});
    check(path,"output with final prefill",{{2,false},{2,false},{1,true,{7,8,9}}});
    check(path,"stalled prefill",{{0,false}},Error::progress);
    check(path,"stalled decode",{reuse,{0,true}},Error::progress);
    check(path,"prefill over grant",{{3,false}},path==Path::mixed ? Error::work : Error::progress);
    check(path,"reuse exceeds consumed",{{1,false,{},2}},path==Path::mixed ? Error::work : Error::progress);
    check(path,"reuse after computation",{{1,false},{1,false,{},1}},Error::progress);
    check(path,"reuse beyond prompt",{{6,true,{},6}},Error::progress);
    const auto maximum=std::numeric_limits<std::uint32_t>::max();
    check(path,"overflowing prompt frontier",{{1,false,{},1},{maximum,true,{},maximum}},Error::progress);
    check(path,"early prefill completion",{{1,true}},Error::progress);
    check(path,"missing prefill completion",{{5,false,{},5}},Error::progress);
    check(path,"prefill completion reversed",{reuse,{0,false,{7}}},Error::progress);
    check(path,"prompt work after prefill",{reuse,{1,true,{7},1}},Error::progress);
    check(path,"output before prefill",{{1,false,{7}}},Error::progress);
    check(path,"output over grant",{reuse,{0,true,{7,8,9,10}}},Error::progress);
    check(path,"remaining output grant",{reuse,{0,true,{7}},{0,true,{8,9,10}}},Error::progress);
    check(path,"padding token",{reuse,{0,true,{12}}},Error::vocabulary);
    if(path==Path::mixed) {
        check(path,"mixed over grant",{{1,false,{},0,5}},Error::work);
        check(path,"mixed omitted computation",{{1,false,{},0,0}},Error::work);
        check(path,"mixed omitted decode",{reuse,{0,true,{7},0,0}},Error::work);
    }
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"expected progress path");
        const std::string name=argv[1];
        if(name=="progress_separate") cases(Path::separate);
        else if(name=="progress_mixed") cases(Path::mixed);
        else throw std::runtime_error("unknown progress path");
        std::cout<<name<<" passed\n";
    } catch(const std::exception& failure) { std::cerr<<failure.what()<<'\n'; return 1; }
}
