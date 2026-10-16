// Core CUDA graph capture harness: captured cuBLAS calls, eager bridges, padded staging, cross-stream input
// order, failure poisoning and complete-owner retention. Requires a CUDA device.
#include <crit/cuda_graph.h>
#include <cublas_v2.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

using namespace crit::core;
namespace {
#ifdef CRIT_TEST_LINKER_WRAP
std::atomic<bool> fail_launch{false}, fail_drain{false};
std::atomic<int> launches{0};
#endif
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
void check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
void blas(cublasStatus_t status) { if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure"); }
template<class F> void rejects(F action,const char* message) {
    try { action(); } catch(const std::exception& e) {
        require(std::string(e.what()).find(message)!=std::string::npos,"unexpected rejection"); return;
    }
    throw std::runtime_error("expected rejection missing");
}
struct Fixture {
    Resources resources{1<<20,1<<20};
    std::shared_ptr<CudaBackend> backend=std::make_shared<CudaBackend>();
    Fixture() {backend->initialize(resources);}
};
struct BlasOwner {
    cublasHandle_t handle=nullptr;
    explicit BlasOwner(cudaStream_t stream) {blas(cublasCreate(&handle)); blas(cublasSetStream(handle,stream));}
    ~BlasOwner() {if(handle) cublasDestroy(handle);}
};
GraphStage copy_stage(std::shared_ptr<BlasOwner> library) {
    return [library](cudaStream_t,const StagedBuffers& buffers) {
        const int count=int(buffers.input_layout.rows*buffers.input_layout.columns);
        blas(cublasScopy(library->handle,count,static_cast<float*>(buffers.input.data()),1,
                        static_cast<float*>(buffers.output.data()),1));
    };
}
template<std::size_t N> void upload(CudaBuffer& buffer,const std::array<float,N>& data) {
    buffer.upload(std::as_bytes(std::span(data)));
}
template<std::size_t N> auto download(CudaBuffer& buffer) {
    std::array<float,N> data{};
    buffer.download(std::as_writable_bytes(std::span(data)));
    return data;
}
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered=false,released=false;
    void hold() {std::unique_lock lock(mutex); entered=true; cv.notify_all(); cv.wait_for(lock,std::chrono::seconds(10),[&]{return released;});}
    void observed() {std::unique_lock lock(mutex); require(cv.wait_for(lock,std::chrono::seconds(5),[&]{return entered;}),"CUDA gate did not enter");}
    void release() {std::lock_guard lock(mutex); released=true; cv.notify_all();}
};
void CUDART_CB gated(void* value) {static_cast<Gate*>(value)->hold();}
struct Release {Gate& gate; cudaStream_t stream; ~Release() {gate.release(); cudaStreamSynchronize(stream);}};

void arithmetic() {
    Fixture f;
    auto library=std::make_shared<BlasOwner>(f.backend->stream());
    MatrixLayout layout{2,3,4,MatrixElement::f32};
    auto in=f.backend->allocate(Memory::device,32),out=f.backend->allocate(Memory::device,32);
    auto all_rows=f.backend->allocate(Memory::device,4*3*4);
    GraphStage scale=[library,all_rows](cudaStream_t stream,const StagedBuffers& b) {
        const float three=3;
        blas(cublasSscal(library->handle,12,&three,static_cast<float*>(b.output.data()),1));
        check(cudaMemcpyAsync(all_rows.data(),b.output.data(),48,cudaMemcpyDeviceToDevice,stream));
    };
    GraphStage bridge=[library](cudaStream_t,const StagedBuffers& b) {
        const float one=1;
        blas(cublasSaxpy(library->handle,12,&one,static_cast<float*>(b.input.data()),1,
                        static_cast<float*>(b.output.data()),1));
    };
    CudaStagedGraph graph(f.backend,layout,layout,4,{copy_stage(library),scale},{bridge});
#ifdef CRIT_TEST_LINKER_WRAP
    const auto capture_launches=launches.load();
#endif
    auto changed=layout; changed.stride=3;
    require(!graph.compatible(changed,layout),"changed stride accepted");
    rejects([&]{graph.replay({in,changed},{out,layout});},"layout differs");
#ifdef CRIT_TEST_LINKER_WRAP
    require(launches==capture_launches,"unsupported layout enqueued work");
#endif
    upload(in,std::array<float,8>{1,2,3,999,4,5,6,999});
    upload(out,std::array<float,8>{0,0,0,-9,0,0,0,-9});
    graph.replay({in,layout},{out,layout}); graph.wait();
    require(download<8>(out)==std::array<float,8>{6,12,18,-9,24,30,36,-9},"captured library/bridge result wrong");
    require(download<12>(all_rows)==std::array<float,12>{6,12,18,24,30,36,0,0,0,0,0,0},"padded rows not initialized");
    upload(in,std::array<float,8>{2,3,4,888,5,6,7,888});
    graph.replay({in,layout},{out,layout}); graph.wait();
    require(download<8>(out)==std::array<float,8>{12,18,24,-9,30,36,42,-9},"second replay used stale inputs");
    graph.close(); require(!graph.compatible(layout,layout),"closed graph usable");
}
void ordering() {
    Fixture f;
    auto library=std::make_shared<BlasOwner>(f.backend->stream());
    MatrixLayout layout{1,4,4,MatrixElement::f32};
    auto in=f.backend->allocate(Memory::device,16),out=f.backend->allocate(Memory::device,16);
    auto source=f.backend->allocate(Memory::host,16);
    const std::array<float,4> values{2,4,6,8}; std::memcpy(source.data(),values.data(),16);
    CudaStagedGraph graph(f.backend,layout,layout,1,{copy_stage(library)});
    Gate gate; Release release{gate,f.backend->stream(Queue::transfer)};
    check(cudaLaunchHostFunc(f.backend->stream(Queue::transfer),gated,&gate)); gate.observed();
    f.backend->copy_async(in,0,source,0,16);
    auto event=std::make_shared<CudaEvent>(f.backend->event()); event->record(f.backend->stream(Queue::transfer));
    const std::array dependencies{event};
    graph.replay({in,layout},{out,layout},dependencies);
    require(!graph.ready(),"graph ignored pending input dependency");
    rejects([&]{graph.replay({in,layout},{out,layout});},"in-flight");
    in={}; // Replay retains the allocation through the pending copy and capture.
    gate.release(); graph.wait();
    require(download<4>(out)==values,"cross-stream dependency did not order input");
    graph.close();
    require(f.resources.usage(Memory::device).live==16,"replay/capture owners leaked after completion");
}
void failures() {
    Fixture f;
    auto library=std::make_shared<BlasOwner>(f.backend->stream());
    MatrixLayout layout{1,4,4,MatrixElement::f32};
    rejects([&]{CudaStagedGraph bad(f.backend,layout,layout,1,{[](auto,const auto&){throw std::runtime_error("capture failed");}});},
            "capture failed");
    require(f.resources.usage(Memory::device).live==0,"failed capture leaked settled owners");
    auto in=f.backend->allocate(Memory::device,16),out=f.backend->allocate(Memory::device,16);
    upload(in,std::array<float,4>{1,2,3,4}); upload(out,std::array<float,4>{-7,-7,-7,-7});
    CudaStagedGraph graph(f.backend,layout,layout,1,{copy_stage(library),copy_stage(library)},
        {[](auto stream,const auto& b){check(cudaMemsetAsync(b.output.data(),0,b.output.size(),stream));
                                       throw std::runtime_error("bridge failed after enqueue");}});
    rejects([&]{graph.replay({in,layout},{out,layout});},"bridge failed");
    require(graph.compatible(layout,layout),"runtime failure disguised as unsupported layout");
    rejects([&]{graph.replay({in,layout},{out,layout});},"failed");
    rejects([&]{graph.ready();},"failed");
    graph.close();
    require(download<4>(out)==std::array<float,4>{-7,-7,-7,-7},"failed phase published output");
    require(f.resources.usage(Memory::device).live==32,"failed bridge leaked after successful drain");
}
void widths() {
    Fixture first, second;
    const MatrixLayout narrow{1,4,4,MatrixElement::f32}, wide{8,4,4,MatrixElement::f32};
    auto small_in=first.backend->allocate(Memory::device,16),small_out=first.backend->allocate(Memory::device,16);
    auto large_in=second.backend->allocate(Memory::device,128),large_out=second.backend->allocate(Memory::device,128);
    auto small_library=std::make_shared<BlasOwner>(first.backend->stream());
    auto large_library=std::make_shared<BlasOwner>(second.backend->stream());
    CudaStagedGraph small(first.backend,narrow,narrow,1,{copy_stage(small_library)});
    upload(small_in,std::array<float,4>{1,2,3,4});
    small.replay({small_in,narrow},{small_out,narrow}); small.wait();
    require(download<4>(small_out)==std::array<float,4>{1,2,3,4},"initial small graph failed");
    CudaStagedGraph large(second.backend,wide,wide,8,{copy_stage(large_library)});
    std::array<float,32> values{};
    for(std::size_t i=0;i<values.size();++i) values[i]=static_cast<float>(i+1);
    upload(large_in,values); large.replay({large_in,wide},{large_out,wide}); large.wait();
    require(download<32>(large_out)==values,"large graph failed");
    for(unsigned round=0;round<3;++round) {
        const std::array<float,4> tiny{float(round+4),float(round+3),float(round+2),float(round+1)};
        for(std::size_t i=0;i<values.size();++i) values[i]=static_cast<float>((31-i)+round*32);
        upload(small_in,tiny); upload(large_in,values);
        small.replay({small_in,narrow},{small_out,narrow});
        large.replay({large_in,wide},{large_out,wide});
        small.wait(); large.wait();
        require(download<4>(small_out)==tiny,"larger capture corrupted an existing small graph");
        require(download<32>(large_out)==values,"independent wide graph used stale rows");
    }
    small.close(); large.close();
    require(first.resources.usage(Memory::device).live==32 && second.resources.usage(Memory::device).live==256,
            "independent graph slots retained staging after close");
}
#ifdef CRIT_TEST_LINKER_WRAP
void quarantine() {
    Fixture f;
    auto library=std::make_shared<BlasOwner>(f.backend->stream());
    const auto before=quarantined_staged_graphs();
    MatrixLayout layout{1,4,4,MatrixElement::f32};
    {
        auto in=f.backend->allocate(Memory::device,16),out=f.backend->allocate(Memory::device,16);
        upload(in,std::array<float,4>{1,2,3,4});
        CudaStagedGraph graph(f.backend,layout,layout,1,{copy_stage(library)});
        fail_launch=true;
        rejects([&]{graph.replay({in,layout},{out,layout});},"launch graph phase");
        require(graph.compatible(layout,layout),"ambiguous launch disguised as unsupported layout");
        in={}; out={}; library.reset();
        require(f.resources.usage(Memory::device).live==64,"ambiguous launch released external owners");
        fail_drain=true;
    }
    fail_drain=false;
    require(quarantined_staged_graphs()==before+1 && f.resources.usage(Memory::device).live==64,
            "failed drain did not retain complete capture and replay owners");
}
#endif
}
#ifdef CRIT_TEST_LINKER_WRAP
extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t,cudaStream_t);
extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t graph,cudaStream_t stream) {
    ++launches;
    const auto status=__real_cudaGraphLaunch(graph,stream);
    return fail_launch.exchange(false) && status==cudaSuccess ? cudaErrorInvalidValue : status;
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    return fail_drain ? cudaErrorUnknown : __real_cudaStreamSynchronize(stream);
}
#endif
int main(int argc,char** argv) {
    try {
        const std::string test=argc>1 ? argv[1] : "arithmetic";
        if(test=="arithmetic") arithmetic();
        else if(test=="ordering") ordering();
        else if(test=="failures") failures();
        else if(test=="widths") widths();
#ifdef CRIT_TEST_LINKER_WRAP
        else if(test=="quarantine") quarantine();
#endif
        else throw std::runtime_error("unknown test");
        std::cout<<"crit staged graph passed: "<<test<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
