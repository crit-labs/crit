// Independently compiled extension fixture, using only the installed public SDK.
#include <crit/cuda.h>
#include <crit/cuda_graph.h>
#include <crit/cuda_memory.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace crit::core;
struct TinySession : Session {
    CudaBackend& backend;
    CudaBuffer device,host;
    CudaEvent event;
    std::uint32_t remaining;
    TinySession(CudaBackend& b,std::uint32_t count)
        :backend(b),device(b.allocate(Memory::device,sizeof(Token))),host(b.allocate(Memory::host,sizeof(Token))),event(b.event()),remaining(count) {}
    Progress advance(WorkBudget budget) override {
        const auto consumed=std::min(remaining,budget.prefill_tokens);
        remaining-=consumed;
        if(remaining) return {consumed,false,{},nullptr};
        backend.fill_async(device,0);
        backend.copy_async(host,0,device,0,sizeof(Token),Queue::compute);
        event.record(backend.stream());
        return {consumed,true,{static_cast<const Token*>(host.data()),1},&event};
    }
    void commit(std::uint32_t accepted,bool terminal) override {
        if((accepted==0 && !terminal) || (accepted==1 && terminal)) return;
        throw std::runtime_error("consumer commit mismatch");
    }
    void close(Finish) noexcept override {}
};
struct TinyModel : Model {
    CudaBackend& backend;
    explicit TinyModel(Services s):backend(dynamic_cast<CudaBackend&>(s.backend)) {}
    ModelInfo info() const override {return {"independent installed-SDK consumer",8,32,2,{},true,false};}
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {return std::make_unique<TinySession>(backend,static_cast<std::uint32_t>(prompt.size()));}
    std::vector<Token> tokenize(std::string_view) override {return {1};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
};
int main() {
    try {
        auto backend=std::make_shared<CudaBackend>();
        Profile profile{32,2,65536,4096};
        profile.trace={256,1};
        Engine engine(profile,backend,[](const Profile&,Services s){return std::make_unique<TinyModel>(s);});
        const auto result=engine.submit({1,2,3},1).wait();
        if(result.tokens!=std::vector<Token>{0} || result.finish!=Finish::length || engine.usage(Memory::device).live)
            throw std::runtime_error("installed consumer result/cleanup mismatch");
        const auto trace=engine.trace();
        if(!trace.enabled || std::none_of(trace.events.begin(),trace.events.end(),[&](const auto& event) {
            return event.phase==TracePhase::completion && event.key.request==result.request_id && event.finish==Finish::length;
        })) throw std::runtime_error("installed SDK tracing unavailable");
        if(quarantined_staged_graphs() || quarantined_cuda_mappings()) throw std::runtime_error("installed graph harness retained an owner");
        std::cout<<"Installed SDK consumer passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
