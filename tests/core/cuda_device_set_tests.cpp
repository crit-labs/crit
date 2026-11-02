// CudaDeviceSet: members with their own streams and ledgers, peer access, copies between members, cross-member
// ordering, the barrier and per-member category timers. Every case runs a set that lists the machine's first device
// twice (two members on one device, as a one-GPU machine allows) and, where the machine has two devices, a set of
// devices 0 and 1. Host callbacks hold streams for known durations, so the fixture needs no kernels.
#include "crit/cuda_devices.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace crit::core;
namespace {
void check(cudaError_t s) {if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorName(s));}
void require(bool c,const char* message) {if(!c) throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message) {
    try {f();} catch(const std::exception& e) {
        require(std::string(e.what()).find(message) != std::string::npos,"unexpected diagnostic"); return;
    }
    throw std::runtime_error("expected rejection");
}
struct Members {
    std::vector<Resources> resources;
    CudaDeviceSet set;
    explicit Members(std::vector<int> devices):set(devices) {
        // One ledger per member: a copied Resources shares its ledger.
        for(std::size_t i=0;i<set.size();++i) resources.emplace_back(1<<22,1<<22);
        for(std::size_t i=0;i<set.size();++i) set.member(i).initialize(resources[i]);
    }
};
std::vector<std::vector<int>> sets() {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    require(count > 0,"no CUDA device");
    std::vector<std::vector<int>> out{{0,0}};
    if(count > 1) out.push_back({0,1});
    return out;
}
// A stream callback that spins for a while, then marks a flag: ordering shows in what a later callback sees.
struct Slow {int milliseconds; std::atomic<int>* flag; int value;};
void CUDART_CB slow(void* pointer) {
    auto* step = static_cast<Slow*>(pointer);
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(step->milliseconds);
    while(std::chrono::steady_clock::now() < until) std::this_thread::yield();
    step->flag->store(step->value);
}
struct Seen {std::atomic<int>* flag; int seen = -1;};
void CUDART_CB see(void* pointer) {auto* look = static_cast<Seen*>(pointer); look->seen = look->flag->load();}

void members() {
    for(auto devices:sets()) {
        Members m(devices);
        require(m.set.size() == 2,"two members");
        require(m.set.device(0) == devices[0] && m.set.device(1) == devices[1],"member devices");
        require(m.set.peer(0,0) && m.set.peer(1,1),"a member reads its own memory");
        if(devices[0] == devices[1]) require(m.set.peer(0,1) && m.set.peer(1,0),"one device is its own peer");
        require(m.set.member(0).stream() != m.set.member(1).stream(),"members have their own streams");
        auto a = m.set.member(0).allocate(Memory::device,4096);
        require(m.resources[0].usage(Memory::device).live == 4096 && m.resources[1].usage(Memory::device).live == 0,
                "each member charges its own ledger");
        require(m.set.member(0).owns(a) && !m.set.member(1).owns(a),"ownership");
        auto timer = m.set.category_timer(1,4);
        timer.start(m.set.member(1).stream());
        timer.begin(0); timer.end();
        require(timer.finish(1).complete,"a member's category timer");
        rejects([&]{m.set.member(2);},"no such device set member");
        rejects([&]{(void)m.set.peer(0,3);},"no such device set member");
    }
    rejects([]{CudaDeviceSet set(std::vector<int>{});},"at least one device");
    rejects([]{CudaDeviceSet set(std::vector<int>{0,4096});},"invalid CUDA device");
}

void copies() {
    for(auto devices:sets()) {
        Members m(devices);
        auto src = m.set.member(0).allocate(Memory::device,1<<20);
        auto dst = m.set.member(1).allocate(Memory::device,1<<20);
        std::vector<std::byte> in(1<<20), out(1<<20);
        for(std::size_t i=0;i<in.size();++i) in[i] = std::byte(i*131 % 251);
        src.upload(in);
        m.set.member(1).fill_async(dst,0,Queue::transfer);
        m.set.copy_async(1,dst,4096,0,src,0,(1<<20)-8192);
        m.set.member(1).drain();
        dst.download(out);
        require(std::memcmp(out.data()+4096,in.data(),(1<<20)-8192) == 0,"copied bytes");
        require(out[0] == std::byte{0} && out[(1<<20)-1] == std::byte{0},"bytes outside the range stay");
        m.set.copy_async(1,dst,0,0,src,0,0);   // zero bytes
        rejects([&]{m.set.copy_async(0,dst,0,1,src,0,16);},"belong to its member");
        rejects([&]{m.set.copy_async(1,dst,(1<<20)-8,0,src,0,16);},"range exceeded");
        auto host = m.set.member(1).allocate(Memory::host,64);
        rejects([&]{m.set.copy_async(1,host,0,0,src,0,16);},"device buffers");
        // Releasing the source right after the copy waits for it: the copy finishes before its memory is freed.
        auto later = m.set.member(0).allocate(Memory::device,1<<20);
        later.upload(in);
        m.set.copy_async(1,dst,0,0,later,0,1<<20);
        later = {};
        m.set.member(1).drain();
        dst.download(out);
        require(std::memcmp(out.data(),in.data(),1<<20) == 0,"a copy outlives its source's release");
    }
}

void ordering() {
    for(auto devices:sets()) {
        Members m(devices);
        std::atomic<int> flag{0};
        Slow step{60,&flag,1};
        Seen look{&flag};
        check(cudaSetDevice(m.set.device(0)));
        check(cudaLaunchHostFunc(m.set.member(0).stream(),slow,&step));
        m.set.order(1,Queue::compute,0);
        check(cudaSetDevice(m.set.device(1)));
        check(cudaLaunchHostFunc(m.set.member(1).stream(),see,&look));
        m.set.member(1).drain();
        require(look.seen == 1,"the waiting member ran after the source's work");
        m.set.member(0).drain();
        m.set.order(0,Queue::compute,0,Queue::compute);   // a stream after itself: nothing to do
    }
}

void barrier() {
    for(auto devices:sets()) {
        Members m(devices);
        std::atomic<int> first{0}, second{0};
        Slow a{50,&first,1}, b{80,&second,1};
        Seen after_first{&first}, after_second{&second};
        check(cudaSetDevice(m.set.device(0)));
        check(cudaLaunchHostFunc(m.set.member(0).stream(),slow,&a));
        check(cudaSetDevice(m.set.device(1)));
        check(cudaLaunchHostFunc(m.set.member(1).stream(),slow,&b));
        m.set.barrier();
        check(cudaSetDevice(m.set.device(0)));
        check(cudaLaunchHostFunc(m.set.member(0).stream(),see,&after_second));
        check(cudaSetDevice(m.set.device(1)));
        check(cudaLaunchHostFunc(m.set.member(1).stream(),see,&after_first));
        m.set.member(0).drain();
        m.set.member(1).drain();
        require(after_second.seen == 1 && after_first.seen == 1,"each member waited for the other's work");
    }
}

// Every call leaves the calling thread's current device as it found it, so an engine that bound a member before
// launching into its streams keeps that binding across Core's calls.
int current() {int device = -1; check(cudaGetDevice(&device)); return device;}
void binding() {
    for(auto devices:sets()) {
        for(const int mine:{devices.back(),devices.front()}) {
            check(cudaSetDevice(mine));
            Members m(devices);
            require(current() == mine,"construction keeps the current device");
            auto src = m.set.member(0).allocate(Memory::device,4096);
            auto dst = m.set.member(1).allocate(Memory::device,4096);
            std::vector<std::byte> bytes(4096,std::byte{7});
            src.upload(bytes);
            m.set.copy_async(1,dst,0,0,src,0,4096);
            m.set.order(0,Queue::compute,1);
            m.set.barrier();
            auto event = m.set.member(1).event();
            event.record(m.set.member(1).stream());
            event.wait();
            m.set.member(0).drain();
            m.set.member(1).drain();
            dst.download(bytes);
            require(bytes[4095] == std::byte{7} && current() == mine,"calls on the set and its members keep the current device");
        }
        require(current() == devices.front(),"destruction keeps the current device");
    }
}

void rejects_case() {
    Members m({0,0});
    rejects([&]{m.set.order(0,static_cast<Queue>(7),1);},"invalid CUDA queue");
    auto foreign_resources = Resources{1<<20,1<<20};
    CudaBackend foreign;
    foreign.initialize(foreign_resources);
    auto stranger = foreign.allocate(Memory::device,64);
    auto mine = m.set.member(1).allocate(Memory::device,64);
    rejects([&]{m.set.copy_async(1,mine,0,0,stranger,0,16);},"belong to its member");
}
}

int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("one fixture name required");
        const std::pair<const char*,void(*)()> groups[]{
            {"members",members},{"copies",copies},{"ordering",ordering},{"barrier",barrier},{"binding",binding},
            {"rejects",rejects_case}};
        for(auto [name,test]:groups) if(argv[1]==std::string(name)) {test(); std::cout<<"PASS "<<name<<'\n'; return 0;}
        throw std::invalid_argument("unknown fixture");
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n'; return 1;}
}
