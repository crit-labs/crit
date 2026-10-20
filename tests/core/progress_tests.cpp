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
enum class Path { serial, concurrent, mixed, async };
enum class Delivery { direct, fenced, deferred, missing_fence, eager_deferred };
struct Step {
    std::uint32_t prompt;
    bool complete;
    std::vector<Token> tokens = {};
    std::uint32_t reused = 0;
    Delivery delivery = Delivery::direct;
    std::optional<std::uint32_t> work = {};
};
struct Record {
    std::vector<Step> steps;
    std::size_t advances=0, commits=0, waits=0, reads=0, batches=0;
    bool settled=true, closed=false;
    Finish finish=Finish::failed;
};
// The same scripted model reports progress through each public execution interface.
// Readiness fills borrowed token storage; commit invalidates it.
struct Driver : Readiness {
    std::shared_ptr<Record> record;
    std::vector<Token> tokens;
    bool waited=false;
    explicit Driver(std::shared_ptr<Record> r):record(std::move(r)) {}
    Progress advance(WorkBudget budget) {
        require(record->advances==record->commits,"advanced past an uncommitted step");
        const auto& step=record->steps.at(record->advances++);
        record->settled=false; waited=false;
        tokens=step.tokens;
        Progress p{step.prompt,step.complete,{},nullptr,false,step.reused};
        if(budget.token_work) {
            // Most cases isolate the common progress rules. Explicit work overrides
            // also exercise the mixed path's separate accounting and error precedence.
            p.token_work=step.work.value_or(budget.prefill_complete ? 1U :
                step.prompt-std::min(step.prompt,step.reused));
        }
        const auto delivery=step.delivery;
        p.deferred_tokens=delivery==Delivery::deferred || delivery==Delivery::missing_fence ||
            delivery==Delivery::eager_deferred;
        if(!p.deferred_tokens || delivery==Delivery::eager_deferred) p.tokens=tokens;
        if(delivery!=Delivery::direct && delivery!=Delivery::missing_fence) {
            p.readiness=this;
            std::fill(tokens.begin(),tokens.end(),-1);
        }
        return p;
    }
    void wait() override {
        require(!waited,"readiness waited twice");
        waited=true; ++record->waits;
        const auto& source=record->steps.at(record->advances-1).tokens;
        std::copy(source.begin(),source.end(),tokens.begin());
    }
    std::span<const Token> ready_tokens() {
        require(waited,"deferred tokens read before readiness");
        ++record->reads;
        return tokens;
    }
    void commit(std::uint32_t accepted) {
        require(accepted<=tokens.size(),"commit exceeded returned tokens");
        ++record->commits;
        std::fill(tokens.begin(),tokens.end(),-1);
    }
    void close(Finish finish) {
        require(record->settled && !record->closed,"closed before settlement or twice");
        record->closed=true; record->finish=finish;
    }
};
struct Host final:Backend {
    std::shared_ptr<Record> record;
    explicit Host(std::shared_ptr<Record> r):record(std::move(r)) {}
    void bind_thread() override {}
    void drain() override { record->settled=true; }
};
ModelInfo info(Path path) {
    ModelInfo m{"progress.contract.fixture",16,32,2,{}};
    m.valid_vocabulary=12; m.max_concurrency=1;
    m.mixed_batches=path==Path::mixed || path==Path::async;
    m.async_batches=path==Path::async; m.decode_token_work=4;
    return m;
}
struct SerialSession final:Session {
    Driver driver;
    explicit SerialSession(std::shared_ptr<Record> r):driver(std::move(r)) {}
    Progress advance(WorkBudget budget) override { return driver.advance(budget); }
    std::span<const Token> ready_tokens() override { return driver.ready_tokens(); }
    void commit(std::uint32_t accepted,bool) override { driver.commit(accepted); }
    void close(Finish finish) noexcept override { driver.close(finish); }
};
struct SerialModel final:Model {
    std::shared_ptr<Record> record;
    explicit SerialModel(std::shared_ptr<Record> r):record(std::move(r)) {}
    ModelInfo info() const override { return ::info(Path::serial); }
    std::unique_ptr<Session> open(std::span<const Token>) override { return std::make_unique<SerialSession>(record); }
};
struct Lane final:ConcurrentSession {
    Driver driver;
    explicit Lane(std::shared_ptr<Record> r):driver(std::move(r)) {}
    void close(Finish finish) noexcept override { driver.close(finish); }
};
struct Settled final:Retirement {
    std::shared_ptr<Record> record;
    explicit Settled(std::shared_ptr<Record> r):record(std::move(r)) {}
    bool ready() override { record->settled=true; return true; }
};
struct BatchedModel final:ConcurrentModel {
    std::shared_ptr<Record> record;
    Path path;
    Lane* lane=nullptr;
    BatchedModel(std::shared_ptr<Record> r,Path p):record(std::move(r)),path(p) {}
    ModelInfo info() const override { return ::info(path); }
    std::unique_ptr<ConcurrentSession> open(const RequestView&) override {
        auto session=std::make_unique<Lane>(record); lane=session.get(); return session;
    }
    RowProgress prefill(const WorkItem& item) override { return {item.key,lane->driver.advance(item.budget)}; }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        require(items.size()==1,"unexpected fixture group size");
        return {prefill(items.front())};
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override { return decode(items); }
    std::span<const Token> ready_tokens(const RequestKey&) override { return lane->driver.ready_tokens(); }
    void commit(std::span<const CommitDecision> group) override {
        require(group.size()==1,"unexpected fixture commit size");
        lane->driver.commit(group.front().accepted);
    }
    std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession&,Finish) override {
        return std::make_unique<Settled>(record);
    }
    void quiesce() override {}
    std::unique_ptr<BatchExecution> prepare_batch(std::span<const WorkItem>) override;
};
struct Batch final:BatchExecution {
    BatchedModel& model;
    std::vector<WorkItem> items;
    std::vector<RowProgress> rows;
    bool submitted=false;
    Batch(BatchedModel& m,std::span<const WorkItem> work):model(m),items(work.begin(),work.end()) {}
    void submit(Notify notify) override {
        rows=model.execute(items); submitted=true; ++model.record->batches; notify();
    }
    bool ready() override { return submitted; }
    std::vector<RowProgress> progress() override { return rows; }
    std::span<const Token> ready_tokens(const RequestKey& key) override { return model.ready_tokens(key); }
    void commit(std::span<const CommitDecision> group) override { model.commit(group); }
    void quiesce() override {}
};
std::unique_ptr<BatchExecution> BatchedModel::prepare_batch(std::span<const WorkItem> items) {
    return std::make_unique<Batch>(*this,items);
}
enum class Error { none, progress, deferred, work, vocabulary };
void check(Path path,const char* name,std::vector<Step> steps,Error error=Error::none) {
    try {
        auto r=std::make_shared<Record>(); r->steps=std::move(steps);
        Profile p{32,2,1<<20,1<<20};
        if(path==Path::mixed || path==Path::async) p.max_batch_tokens=4;
        if(path==Path::async) p.max_inflight_batches=1;
        Engine e(p,std::make_shared<Host>(r),[r,path](const Profile&,Services)->std::unique_ptr<Model> {
            if(path==Path::serial) return std::make_unique<SerialModel>(r);
            return std::make_unique<BatchedModel>(r,path);
        });
        const auto live=e.usage(Memory::host).live;
        std::vector<Token> published;
        std::string diagnostic;
        Result result;
        try { result=e.submit({1,2,3,4,5},3).wait([&](Token t){published.push_back(t);}); }
        catch(const std::runtime_error& failure) { diagnostic=failure.what(); }
        catch(const std::invalid_argument& failure) { diagnostic=failure.what(); }
        const auto suffix=path==Path::serial ? "model progress" : "concurrent progress";
        const auto expected=error==Error::none ? std::string{} :
            error==Error::deferred ? "invalid deferred "+std::string(suffix) :
            error==Error::work ? "mixed token work exceeds grant or omits computation" :
            error==Error::vocabulary ? "generated token outside the valid vocabulary" :
            "invalid or stalled "+std::string(suffix);
        require(diagnostic==expected,("unexpected diagnostic: "+diagnostic).c_str());
        const auto commits=r->steps.size()-(error==Error::none ? 0 : 1);
        require(r->commits==commits && r->advances==r->steps.size(),"invalid progress reached commit");
        std::vector<Token> accepted;
        std::size_t waits=0,reads=0;
        std::uint32_t reused=0;
        for(std::size_t i=0;i<r->steps.size();++i) {
            const auto& step=r->steps[i];
            if(i<commits) {
                accepted.insert(accepted.end(),step.tokens.begin(),step.tokens.end());
                reused+=step.reused;
            }
            const bool shape_rejected=i==commits && error==Error::deferred;
            if(step.delivery!=Delivery::direct && step.delivery!=Delivery::missing_fence &&
               !(shape_rejected && path==Path::serial)) ++waits;
            if(step.delivery==Delivery::deferred && !shape_rejected) ++reads;
        }
        require(published==accepted,"invalid progress published tokens or borrowed output was lost");
        require(r->waits==waits && r->reads==reads,"readiness/materialization order changed");
        require(r->closed && r->finish==(error==Error::none ? Finish::length : Finish::failed),"request not closed with expected finish");
        require(e.usage(Memory::host).live==live,"request charges not released");
        require(path!=Path::async || r->batches==r->advances,"async path not exercised");
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
    check(path,"fenced output",{reuse,{0,true,{7,8,9},0,Delivery::fenced}});
    check(path,"deferred output",{reuse,{0,true,{7,8,9},0,Delivery::deferred}});
    check(path,"stalled prefill",{{0,false}},Error::progress);
    check(path,"stalled decode",{reuse,{0,true}},Error::progress);
    check(path,"prefill over grant",{{3,false}},path==Path::mixed || path==Path::async ? Error::work : Error::progress);
    check(path,"reuse exceeds consumed",{{1,false,{},2}},path==Path::mixed || path==Path::async ? Error::work : Error::progress);
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
    check(path,"deferred without readiness",{reuse,{0,true,{7},0,Delivery::missing_fence}},Error::deferred);
    check(path,"deferred with eager tokens",{reuse,{0,true,{7},0,Delivery::eager_deferred}},Error::deferred);
    check(path,"deferred before prefill",{{1,false,{7},0,Delivery::deferred}},Error::deferred);
    check(path,"empty deferred result",{reuse,{0,true,{},0,Delivery::deferred}},Error::progress);
    check(path,"oversized deferred result",{reuse,{0,true,{7,8,9,10},0,Delivery::deferred}},Error::progress);
    check(path,"padding token",{reuse,{0,true,{12}}},Error::vocabulary);
    check(path,"invalid deferred token",{reuse,{0,true,{-1},0,Delivery::deferred}},Error::vocabulary);
    if(path==Path::mixed || path==Path::async) {
        check(path,"mixed over grant",{{1,false,{},0,Delivery::direct,5}},Error::work);
        check(path,"mixed omitted computation",{{1,false,{},0,Delivery::direct,0}},Error::work);
        check(path,"mixed omitted decode",{reuse,{0,true,{7},0,Delivery::direct,0}},Error::work);
    }
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"expected progress path");
        const std::string name=argv[1];
        if(name=="progress_serial") cases(Path::serial);
        else if(name=="progress_concurrent") cases(Path::concurrent);
        else if(name=="progress_mixed") cases(Path::mixed);
        else if(name=="progress_async") cases(Path::async);
        else throw std::runtime_error("unknown progress path");
        std::cout<<name<<" passed\n";
    } catch(const std::exception& failure) { std::cerr<<failure.what()<<'\n'; return 1; }
}
