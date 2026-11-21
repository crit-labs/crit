// Independent installed-SDK consumer: no private core header or donor controller.
#include <crit/core.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace crit::core;
namespace {
struct Context final:Session {
    RequestView request;
    Reservation state;
    std::uint32_t frontier=0, pending_frontier=0, outputs=0;
    std::vector<Token> pending;
    Context(RequestView v,Resources& r):request(v),state(r.reserve(Memory::device,8)) {}
};
struct Example final:Model {
    Resources& resources;
    Reservation shared_weights;
    Example(Resources& r):resources(r),shared_weights(r.reserve(Memory::device,64)) {}
    ModelInfo info() const override {
        ModelInfo result{"independent.installed.concurrent.example",256,32,4,{},true,false,3};
        result.mixed_batches=true;
        return result;
    }
    std::unique_ptr<Session> open(const RequestView& v) override {return std::make_unique<Context>(v,resources);}
    RowProgress prefill(const WorkItem& item) override {
        auto& c=static_cast<Context&>(*item.session);
        c.pending_frontier=c.frontier+item.budget.prefill_tokens;
        c.pending.clear();
        return {item.key,{item.budget.prefill_tokens,c.pending_frontier==c.request.prompt.size(),{}}};
    }
    std::vector<RowProgress> decode(std::span<const WorkItem> items) override {
        std::vector<RowProgress> rows;
        for(const auto& item:items) {
            auto& c=static_cast<Context&>(*item.session);
            c.pending.assign(1,c.request.prompt.front()+Token(c.outputs)+1);
            rows.push_back({item.key,{0,true,c.pending}});
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
    void retire(RequestKey,Session&,Finish) override {}
};
}
int main() {
    try {
      for(auto batch_tokens:{0U,3U}) {
        Profile profile{32,2,65536,4096,3,0,3000};
        profile.max_batch_tokens=batch_tokens;
        if(batch_tokens) profile.batch_policy=BatchPolicy::decode_first;
        Engine engine(profile,[](const Profile&,Resources& r){return std::make_unique<Example>(r);});
        auto a=engine.submit({10,11,12},3),b=engine.submit({20},2),c=engine.submit({30,31},4);
        if(a.wait().tokens!=std::vector<Token>({11,12,13}) || b.wait().tokens!=std::vector<Token>({21,22}) ||
           c.wait().tokens!=std::vector<Token>({31,32,33,34}) || engine.usage(Memory::device).live!=64 ||
           engine.usage(Memory::host).live!=0)
            throw std::runtime_error("installed SDK request isolation/ownership failed");
      }
        std::cout<<"crit 0.22.0 installed SDK concurrent consumer passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
