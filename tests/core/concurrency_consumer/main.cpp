// Independent installed-SDK consumer: no private core header or donor controller.
#include <crit/core.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace crit::core;
namespace {
struct Host final:Backend { void bind_thread() override {} void drain() override {} };
struct Context final:ConcurrentSession {
    RequestView request;
    Reservation state;
    std::uint32_t frontier=0, pending_frontier=0, outputs=0;
    std::vector<Token> pending;
    Context(RequestView v,Resources& r):request(v),state(r.reserve(Memory::device,8)) {}
    void close(Finish) noexcept override {}
};
struct Settled final:Retirement { bool ready() override {return true;} };
struct Example final:ConcurrentModel {
    Resources& resources;
    Reservation shared_weights;
    Example(Resources& r):resources(r),shared_weights(r.reserve(Memory::device,64)) {}
    ModelInfo info() const override {
        ModelInfo result{"independent.installed.concurrent.example",256,32,4,{},true,false,3};
        result.mixed_batches=true;
        return result;
    }
    std::unique_ptr<ConcurrentSession> open(const RequestView& v) override {return std::make_unique<Context>(v,resources);}
    RowProgress prefill(const WorkItem& item) override {
        auto& c=static_cast<Context&>(*item.session);
        c.pending_frontier=c.frontier+item.budget.prefill_tokens;
        c.pending.clear();
        return {item.key,{item.budget.prefill_tokens,c.pending_frontier==c.request.prompt.size(),{},nullptr}};
    }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        for(const auto& item:items) {
            auto& c=static_cast<Context&>(*item.session);
            c.pending.assign(1,c.request.prompt.front()+Token(c.outputs)+1);
            rows.push_back({item.key,{0,true,c.pending,nullptr}});
        }
        return rows;
    }
    void commit(std::span<const CommitDecision> decisions) override {
        for(const auto& decision:decisions) {
            auto& c=static_cast<Context&>(*decision.session);
            c.frontier=c.pending_frontier; c.outputs+=decision.accepted; c.pending.clear();
        }
    }
    std::vector<RowProgress> execute(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        for(const auto& item:items) {
            auto row=item.budget.prefill_complete ? decode(std::span<const WorkItem>(&item,1)).front() : prefill(item);
            row.progress.token_work=item.budget.prefill_complete ? 1 : row.progress.prompt_tokens;
            rows.push_back(row);
        }
        return rows;
    }
    std::unique_ptr<Retirement> retire(RequestKey,ConcurrentSession&,Finish) override {return std::make_unique<Settled>();}
    void quiesce() override {}
    std::vector<Token> tokenize(std::string_view) override {return {10};}
    std::string detokenize(std::span<const Token>) override {return "example";}
};
}
int main() {
    try {
      for(auto batch_tokens:{0U,3U}) {
        Profile profile{32,2,65536,4096,3,0,3000,3000};
        profile.trace={256,1};
        profile.max_batch_tokens=batch_tokens;
        if(batch_tokens) profile.batch_policy=BatchPolicy::decode_first;
        profile.admission_lookahead=2;
        Engine engine(profile,std::make_shared<Host>(),
            [](const Profile&,Services s){return std::make_unique<Example>(s.resources);});
        auto a=engine.submit({10,11,12},3),b=engine.submit({20},2),c=engine.submit({30,31},4);
        if(a.wait().tokens!=std::vector<Token>({11,12,13}) || b.wait().tokens!=std::vector<Token>({21,22}) ||
           c.wait().tokens!=std::vector<Token>({31,32,33,34}) || engine.usage(Memory::device).live!=64 ||
           engine.usage(Memory::host).live!=profile.trace.capacity*sizeof(TraceEvent))
            throw std::runtime_error("installed SDK request isolation/ownership failed");
        const auto trace=engine.trace();
        if(!trace.enabled || std::count_if(trace.events.begin(),trace.events.end(),[](const auto& event) {
            return event.phase==TracePhase::completion && event.finish==Finish::length;
        })!=3) throw std::runtime_error("installed SDK trace lost completed requests");
      }
        std::cout<<"crit 0.2.0 installed SDK concurrent consumer passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
