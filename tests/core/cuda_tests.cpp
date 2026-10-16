#include "crit/cuda.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>

using namespace crit::core;
namespace {
#ifdef CRIT_TEST_LINKER_WRAP
std::atomic<int> fail_sync{0}, fail_malloc{0}, fail_pinned{0}, fail_free{0}, fail_create{0};
std::atomic<int> frees{0}, pinned_frees{0}, creates{0}, destroys{0};
bool consume(std::atomic<int>& count) {auto n=count.load(); while(n>0) {if(count.compare_exchange_weak(n,n-1)) return true;} return false;}
#endif
void check(cudaError_t s) {if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorName(s));}
void require(bool c,const char* message) {if(!c) throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message) {
    try {f();} catch(const std::exception& e) {require(std::string(e.what()).find(message) != std::string::npos,"unexpected diagnostic"); return;}
    throw std::runtime_error("expected rejection");
}
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered=false,released=false;
    void wait() {std::unique_lock lock(mutex); entered=true; cv.notify_all(); cv.wait(lock,[&]{return released;});}
    void observed() {std::unique_lock lock(mutex); cv.wait(lock,[&]{return entered;});}
    void release() {std::lock_guard lock(mutex); released=true; cv.notify_all();}
    ~Gate() {release();}
};
void CUDART_CB gated(void* pointer) {static_cast<Gate*>(pointer)->wait();}
struct Ungate {Gate& gate; ~Ungate() {gate.release();}};
struct Fixture {
    Resources resources{1<<20,1<<20};
    std::shared_ptr<CudaBackend> backend=std::make_shared<CudaBackend>();
    Fixture() {backend->initialize(resources);}
};
void allocations() {
    Fixture f;
    int device=0; cudaDeviceProp actual{};
    check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&actual,device));
    require(f.backend->properties().major==actual.major && f.backend->properties().minor==actual.minor,
            "backend device properties differ from CUDA");
    require(f.backend->sync_schedule()==SyncSchedule::spin,"applied sync schedule differs");
    auto device_buffer=f.backend->allocate(Memory::device,4096);
    auto alias=device_buffer;
    auto host=f.backend->allocate(Memory::host,4096);
    require(f.resources.usage(Memory::device).live==4096,"alias charged twice");
    require(f.resources.usage(Memory::host).live==4096,"pinned memory uncharged");
    device_buffer={}; require(f.resources.usage(Memory::device).live==4096,"alias released early");
    rejects([&]{f.backend->allocate(Memory::device,1<<20);},"ceiling");
    rejects([&]{f.backend->allocate(Memory::host,std::numeric_limits<std::size_t>::max());},"ceiling");
    rejects([&]{f.backend->initialize(f.resources);},"already initialized");
    std::array<std::byte,64> in,out;
    in.fill(std::byte{0x29}); out.fill(std::byte{0});
    alias.upload(in,32); alias.download(out,32);
    require(in==out,"pageable transfer mismatch");
    rejects([&]{alias.upload(in,4090);},"range");
    rejects([&]{f.backend->copy_async(alias,0,alias,1,32);},"overlapping");
    Arena arena(alias.data(),alias.size());
    auto first=arena.allocate(256); const auto mark=arena.mark(); arena.allocate(1024); arena.rewind(mark);
    require(arena.allocate(1024).data()==first.data()+256,"arena failed exact reuse");
    require(f.resources.usage(Memory::device).live==4096,"arena grew backing allocation");
    alias={}; host={};
    require(!f.resources.usage(Memory::device).live && !f.resources.usage(Memory::host).live,"allocation leak");
}
void ordering() {
    Fixture f;
    auto src=f.backend->allocate(Memory::host,4096),dev=f.backend->allocate(Memory::device,4096),out=f.backend->allocate(Memory::host,4096);
    std::memset(src.data(),0x5b,src.size()); std::memset(out.data(),0,out.size());
    auto event=f.backend->event();
    rejects([&]{event.ready();},"not been recorded");
    rejects([&]{event.record(nullptr);},"foreign/null");
    Gate gate;
    check(cudaLaunchHostFunc(f.backend->stream(Queue::transfer),gated,&gate));
    f.backend->copy_async(dev,0,src,0,src.size());
    event.record(f.backend->stream(Queue::transfer));
    event.order(f.backend->stream());
    f.backend->copy_async(out,0,dev,0,dev.size(),Queue::compute);
    gate.observed();
    require(!event.ready(),"pending event reported ready");
    gate.release(); event.wait(); f.backend->drain();
    require(event.ready(),"settled event not ready");
    require(std::memcmp(src.data(),out.data(),src.size())==0,"cross-stream dependency lost");
    auto foreign=std::make_shared<CudaBackend>();
    rejects([&]{event.order(foreign->stream());},"foreign/null");
    Resources other(4096,4096); foreign->initialize(other);
    auto foreign_buffer=foreign->allocate(Memory::device,4096);
    rejects([&]{f.backend->copy_async(foreign_buffer,0,src,0,64);},"foreign/empty");
}
void release() {
    Fixture f;
    auto src=f.backend->allocate(Memory::host,4096),dev=f.backend->allocate(Memory::device,4096),out=f.backend->allocate(Memory::host,4096);
    std::memset(src.data(),0x39,src.size());
    Gate gate;
    check(cudaLaunchHostFunc(f.backend->stream(Queue::transfer),gated,&gate));
    f.backend->copy_async(dev,0,src,0,src.size());
    auto event=f.backend->event(); event.record(f.backend->stream(Queue::transfer)); event.order(f.backend->stream());
    f.backend->copy_async(out,0,dev,0,dev.size(),Queue::compute);
    gate.observed();
    std::promise<void> entered;
    auto dropping=std::async(std::launch::async,[owned=std::move(src),&entered]() mutable {entered.set_value(); owned={};});
    Ungate release_on_failure{gate};
    entered.get_future().wait();
    require(f.resources.usage(Memory::host).live==8192,"pending pinned source released early");
    gate.release(); dropping.get();
    require(f.resources.usage(Memory::host).live==4096,"pinned source not released after drain");
    for(std::size_t i=0;i<out.size();++i) require(static_cast<std::byte*>(out.data())[i]==std::byte{0x39},"copy used released storage");
    dev={}; require(f.resources.usage(Memory::device).live==0,"device owner leak");
}
void partial() {
    Fixture f;
    Gate gate;
    std::promise<void> throwing;
    auto failed=std::async(std::launch::async,[&] {
        try {
            auto weights=f.backend->allocate(Memory::device,4096);
            check(cudaLaunchHostFunc(f.backend->stream(),gated,&gate));
            f.backend->fill_async(weights,7);
            throwing.set_value();
            throw std::runtime_error("partial model construction");
        } catch(const std::runtime_error&) {}
    });
    Ungate release_on_failure{gate};
    throwing.get_future().wait(); gate.observed();
    require(f.resources.usage(Memory::device).live==4096,"partial construction freed pending weights");
    gate.release(); failed.get();
    require(f.resources.usage(Memory::device).live==0,"partial construction allocation leaked");
    // Repeat through the actual Engine factory. Initialization unwind must
    // protect local weight owners before the core itself can catch the failure.
    auto backend=std::make_shared<CudaBackend>();
    Resources observed(0,0);
    Gate startup_gate;
    std::promise<void> submitted;
    auto startup=std::async(std::launch::async,[&] {
        rejects([&]{Engine failed_engine({32768,1024,1<<20,1<<20},backend,[&](const Profile&,Services s)->std::unique_ptr<Model> {
            observed=s.resources;
            auto weights=backend->allocate(Memory::device,4096);
            check(cudaLaunchHostFunc(backend->stream(),gated,&startup_gate));
            backend->fill_async(weights,11);
            submitted.set_value();
            throw std::runtime_error("model factory failure");
        });},"model factory failure");
    });
    Ungate release_startup{startup_gate};
    submitted.get_future().wait(); startup_gate.observed();
    require(observed.usage(Memory::device).live==4096,"failed factory freed in-flight weight storage");
    startup_gate.release(); startup.get();
    require(observed.usage(Memory::device).live==0,"failed factory retained drained storage");
}
struct AsyncSession : Session {
    CudaBackend& backend;
    CudaBuffer device,host;
    CudaEvent ready;
    std::uint32_t prompt;
    bool prefilled=false;
    std::atomic<int>& closed;
    Gate* gate;
    AsyncSession(CudaBackend& b,std::uint32_t n,std::atomic<int>& c,Gate* g)
        :backend(b),device(b.allocate(Memory::device,sizeof(Token))),host(b.allocate(Memory::host,sizeof(Token))),ready(b.event()),prompt(n),closed(c),gate(g) {}
    Progress advance(WorkBudget budget) override {
        if(!prefilled) {require(budget.prefill_tokens==prompt,"fixture prompt budget"); prefilled=true; return {prompt,true,{},nullptr};}
        if(gate) check(cudaLaunchHostFunc(backend.stream(),gated,gate));
        backend.fill_async(device,0); // Token 0 is a valid output after asynchronous D2H.
        backend.copy_async(host,0,device,0,sizeof(Token),Queue::compute);
        ready.record(backend.stream());
        return {0,true,{static_cast<Token*>(host.data()),1},&ready};
    }
    void commit(std::uint32_t accepted,bool) override {require(accepted<=1,"fixture accepted count");}
    void close(Finish) noexcept override {++closed;}
};
struct AsyncModel : Model {
    CudaBackend& backend;
    std::atomic<int>& closed;
    Gate* gate;
    AsyncModel(Services s,std::atomic<int>& c,Gate* g):backend(dynamic_cast<CudaBackend&>(s.backend)),closed(c),gate(g) {}
    ModelInfo info() const override {return {"CUDA fixture",16,32768,1024,{},true,false};}
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        return std::make_unique<AsyncSession>(backend,static_cast<std::uint32_t>(prompt.size()),closed,gate);
    }
    std::vector<Token> tokenize(std::string_view) override {return {1};}
    std::string detokenize(std::span<const Token>) override {return "fixture";}
};
void engine() {
    auto backend=std::make_shared<CudaBackend>();
    std::atomic<int> closed{0}; Gate gate;
    Engine engine({32768,1024,1<<20,1<<20},backend,[&](const Profile&,Services s){return std::make_unique<AsyncModel>(s,closed,&gate);});
    auto handle=engine.submit({1,2,3},1);
    Ungate release_on_failure{gate};
    gate.observed(); handle.cancel(); gate.release();
    const auto cancelled=handle.wait();
    require(cancelled.finish==Finish::cancelled && cancelled.tokens.empty() && closed==1,"CUDA cancellation published pending output");
    const auto result=engine.submit({1,2},1).wait();
    require(result.tokens==std::vector<Token>{0} && closed==2,"CUDA runtime readiness result mismatch");
    require(engine.usage(Memory::device).live==0,"session device storage retained after close");
}
#ifdef CRIT_TEST_LINKER_WRAP
void faults() {
    Fixture f;
    fail_malloc=1;
    rejects([&]{f.backend->allocate(Memory::device,4096);},"cudaErrorMemoryAllocation");
    require(f.resources.usage(Memory::device).live==0,"failed malloc retained reservation");
    fail_pinned=1;
    rejects([&]{f.backend->allocate(Memory::host,4096);},"cudaErrorMemoryAllocation");
    require(f.resources.usage(Memory::host).live==0,"failed pinned allocation retained reservation");
    // Inject a second-stream creation failure; the already-created first stream
    // must be destroyed by constructor unwinding, without touching source trees.
    const int before_destroy=destroys,before_create=creates;
    fail_create=2;
    rejects([]{CudaBackend failed;},"cudaErrorMemoryAllocation");
    require(creates==before_create+1 && destroys==before_destroy+1,"partial stream construction leaked");
    auto buffer=f.backend->allocate(Memory::device,4096);
    fail_sync=2; const int before_free=frees;
    buffer={};
    require(quarantined_cuda_allocations()==1 && f.resources.usage(Memory::device).live==4096 && frees==before_free,"failed drain freed unknown storage");
    auto second=f.backend->allocate(Memory::device,64);
    fail_free=1; second={};
    require(quarantined_cuda_allocations()==2 && f.resources.usage(Memory::device).live==4160,"failed free lost reservation");
    // Failed synchronous upload retains its owned staging until proven complete.
    auto third=f.backend->allocate(Memory::device,64);
    std::array<std::byte,64> source{};
    fail_sync=3;
    rejects([&]{third.upload(source);},"cudaErrorUnknown");
    require(quarantined_cuda_allocations()==3 && f.resources.usage(Memory::host).live==64,"failed upload lost staging owner");
    f.backend->drain(); // Injected statuses did not corrupt the actual CUDA context.
}
#endif
void timer() {
    Fixture f;
    auto buffer=f.backend->allocate(Memory::device,4096);
    auto timer=f.backend->timer();
    rejects([&]{timer.record_stop();},"not been recorded");
    timer.start(); f.backend->fill_async(buffer,5);
    const auto elapsed=timer.stop_ms();
    require(elapsed>=0 && elapsed<10000,"invalid CUDA timer result");
    timer.start(); rejects([&]{timer.elapsed_ms();},"not been recorded");
    timer.stop_ms();
}
void readback() {
    Fixture f;
    auto dev=f.backend->allocate(Memory::device,4096),host=f.backend->allocate(Memory::host,4096),other=f.backend->allocate(Memory::device,64);
    f.backend->fill_async(dev,0x11); f.backend->drain();
    std::memset(host.data(),0,host.size());
    auto done=f.backend->event();
    Gate gate;
    check(cudaLaunchHostFunc(f.backend->stream(),gated,&gate));
    f.backend->fill_async(dev,0x42);
    f.backend->readback_async(host,0,dev,0,dev.size(),done);
    Ungate release_on_failure{gate};
    gate.observed();
    require(!done.ready(),"readback completed before its producing compute work");
    gate.release(); done.wait();
    for(std::size_t i=0;i<host.size();++i) require(static_cast<std::byte*>(host.data())[i]==std::byte{0x42},"readback not ordered after compute work");
    f.backend->readback_async(host,0,dev,0,0,done); done.wait();
    rejects([&]{f.backend->readback_async(other,0,dev,0,64,done);},"pinned host destination");
    rejects([&]{f.backend->readback_async(host,0,host,64,64,done);},"device source");
    rejects([&]{f.backend->readback_async(host,4090,dev,0,64,done);},"range");
    CudaEvent empty;
    rejects([&]{f.backend->readback_async(host,0,dev,0,64,empty);},"foreign/empty CUDA event");
    auto foreign=std::make_shared<CudaBackend>();
    auto foreign_event=foreign->event();
    rejects([&]{f.backend->readback_async(host,0,dev,0,64,foreign_event);},"foreign/empty CUDA event");
}
void stages() {
    Fixture f;
    auto buffer=f.backend->allocate(Memory::device,1<<16);
    auto timer=f.backend->stage_timer(2);
    require(timer.capacity()==2 && timer.collect().empty(),"fresh stage timer state");
    rejects([&]{timer.end();},"not open");
    Gate gate;
    timer.begin(DeviceStage::draft,0);
    check(cudaLaunchHostFunc(f.backend->stream(),gated,&gate));
    f.backend->fill_async(buffer,1);
    timer.end();
    {
        Ungate release_on_failure{gate};
        gate.observed();
        rejects([&]{timer.collect();},"not ready");
    }
    timer.begin(DeviceStage::verify,7); f.backend->fill_async(buffer,2); timer.end();
    rejects([&]{timer.begin(DeviceStage::accept);},"capacity");
    f.backend->drain();
    const auto timings=timer.collect();
    require(timings.size()==2 && timings[0].stage==DeviceStage::draft && timings[1].stage==DeviceStage::verify &&
            timings[1].index==7 && timings[0].duration_ns<10000000000ull,"stage timings mismatch");
    timer.reset();
    require(timer.collect().empty(),"reset retained stage timings");
    cudaGraph_t graph=nullptr;
    check(cudaStreamBeginCapture(f.backend->stream(),cudaStreamCaptureModeRelaxed));
    bool rejected=false;
    try {timer.begin(DeviceStage::forward);} catch(const std::logic_error& e) {rejected=std::string(e.what()).find("capture")!=std::string::npos;}
    check(cudaStreamEndCapture(f.backend->stream(),&graph));
    if(graph) check(cudaGraphDestroy(graph));
    require(rejected,"stage timer recorded inside stream capture");
    timer.begin(DeviceStage::head); timer.end(); f.backend->drain();
    require(timer.collect().size()==1,"stage timer unusable after capture rejection");
}
void retained() {
    Resources resources(4096,4096);
    CudaBuffer buffer;
    {
        auto backend=std::make_unique<CudaBackend>(); backend->initialize(resources);
        buffer=backend->allocate(Memory::device,4096); backend->fill_async(buffer,9);
    }
    std::array<std::byte,64> out{}; buffer.download(out);
    for(auto byte:out) require(byte==std::byte{9},"buffer did not retain CUDA backend");
    buffer={}; require(resources.usage(Memory::device).live==0,"retained backend storage leak");
#ifdef CRIT_TEST_LINKER_WRAP
    auto backend=std::make_unique<CudaBackend>();
    fail_sync=2; backend.reset();
    require(quarantined_cuda_backends()==1,"failed backend destruction lost stream owner");
#endif
}
}
// Linker fault injection exercises core error branches while successful calls
// still execute the real CUDA runtime. It never injects a failing GPU kernel.
#ifdef CRIT_TEST_LINKER_WRAP
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t s) {return consume(fail_sync) ? cudaErrorUnknown : __real_cudaStreamSynchronize(s);}
extern "C" cudaError_t __real_cudaMalloc(void**,std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** p,std::size_t n) {return consume(fail_malloc) ? cudaErrorMemoryAllocation : __real_cudaMalloc(p,n);}
extern "C" cudaError_t __real_cudaMallocHost(void**,std::size_t);
extern "C" cudaError_t __wrap_cudaMallocHost(void** p,std::size_t n) {return consume(fail_pinned) ? cudaErrorMemoryAllocation : __real_cudaMallocHost(p,n);}
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaFree(void* p) {if(consume(fail_free)) return cudaErrorUnknown; ++frees; return __real_cudaFree(p);}
extern "C" cudaError_t __real_cudaFreeHost(void*);
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) {++pinned_frees; return __real_cudaFreeHost(p);}
extern "C" cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s,unsigned f) {
    if(fail_create>0 && --fail_create==0) return cudaErrorMemoryAllocation;
    const auto status=__real_cudaStreamCreateWithFlags(s,f); if(status==cudaSuccess) ++creates; return status;
}
extern "C" cudaError_t __real_cudaStreamDestroy(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t s) {++destroys; return __real_cudaStreamDestroy(s);}
#endif
void spare() {
    Fixture f;  // a 1 MiB device budget, far below any device's free memory
    const auto all=f.backend->spare_device_bytes();
    require(all==(1u<<20),"spare memory is bounded by the engine's device budget");
    auto first=f.backend->allocate_spare(512u<<10,0);
    require(first && first->memory()==Memory::device && first->size()==(512u<<10),"spare allocation");
    require(first->device_address()==first->data(),"device memory is addressed directly");
    require(f.backend->spare_device_bytes()==all-(512u<<10),"spare memory shrinks by the allocation");
    require(f.backend->spare_device_bytes(256u<<10)==(256u<<10),"the margin stays free");
    require(!f.backend->allocate_spare(1u<<20,0),"no allocation beyond the budget");
    require(!f.backend->allocate_spare(256u<<10,512u<<10),"no allocation into the margin");
    auto host=f.backend->allocate(Memory::host,64);
    require(host.device_address()!=nullptr,"pinned host memory has a device address");
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("one fixture name required");
        const std::pair<const char*,void(*)()> groups[]{
            {"allocations",allocations},{"ordering",ordering},{"release",release},{"partial",partial},
            {"engine",engine},{"timer",timer},{"readback",readback},{"stages",stages},{"retained",retained},{"spare",spare},
#ifdef CRIT_TEST_LINKER_WRAP
            {"faults",faults},
#endif
        };
        for(auto [name,test]:groups) if(argv[1]==std::string(name)) {test(); std::cout<<"PASS "<<name<<'\n'; return 0;}
        throw std::invalid_argument("unknown fixture");
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n'; return 1;}
}
