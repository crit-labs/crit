// Standalone borrowed-view POC. No model, Core allocator or tensor framework.
#include "crit/rmsnorm.h"
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace crit::rmsnorm;
namespace {
void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
// Settle work before releasing either the stream or its borrowed allocations,
// including exception unwinding. If settlement fails, retain the allocation.
struct Stream {
    cudaStream_t value{};
    Stream() {check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream() {if(cudaStreamSynchronize(value)==cudaSuccess) cudaStreamDestroy(value);}
};
struct Buffer {
    void* value{};
    cudaStream_t stream;
    Buffer(std::size_t bytes,cudaStream_t s):stream(s) {check(cudaMalloc(&value,bytes));}
    ~Buffer() {if(cudaStreamSynchronize(stream)==cudaSuccess) cudaFree(value);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
std::uint16_t encode(float value) {
    auto bits=std::bit_cast<std::uint32_t>(value);
    bits+=0x7fffU+((bits>>16)&1U);
    return static_cast<std::uint16_t>(bits>>16);
}
float decode(std::uint16_t value) {return std::bit_cast<float>(std::uint32_t(value)<<16);}
template<class F> void rejects(F call) {
    try {call();} catch(const std::invalid_argument&) {return;}
    throw std::runtime_error("invalid metadata was admitted");
}
void run(std::int64_t width,std::int64_t rows,Mode mode,int alias) {
    const auto count=static_cast<std::size_t>(width*rows), bytes=count*2;
    std::vector<std::uint16_t> x(count),g(count),weight(static_cast<std::size_t>(width)),out(count);
    for(std::size_t i=0;i<count;++i) {
        x[i]=encode(static_cast<float>(int(i%37)-18)/8);
        g[i]=encode(static_cast<float>(int(i%31)-15)/2);
    }
    for(std::size_t i=0;i<weight.size();++i) weight[i]=encode(static_cast<float>(int(i%19)-9)/8);
    Stream stream;
    Buffer dx(bytes,stream.value),dg(bytes,stream.value),dw(weight.size()*2,stream.value),dy(bytes,stream.value);
    check(cudaMemcpy(dx.value,x.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dg.value,g.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.value,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
    ReadView input{dx.value,width,rows,2*width},gain{dw.value,width,1,2*width},gate{dg.value,width,rows,2*width};
    WriteView output{alias==1 ? dx.value : alias==2 ? dg.value : dy.value,width,rows,2*width};
    const ReadView* gated=mode==Mode::gated ? &gate : nullptr;
    Parameters parameters{mode,1e-5F};
    auto invalid=input; invalid.width=0;
    rejects([&]{validate(invalid,gain,gated,output,parameters,{});});
    rejects([&]{validate(input,gain,gated,output,{mode,0},{});});
    auto partial=output; partial.data=bytes>2 ? static_cast<char*>(dx.value)+2 : dw.value;
    rejects([&]{validate(input,gain,gated,partial,parameters,{});});
    rejects([&]{validate(input,gain,gated,output,parameters,{dy.value,1});});
    const auto dispatch=enqueue(input,gain,gated,output,parameters,{stream.value});
    if(dispatch.kernel_launches!=1) throw std::runtime_error("unexpected dispatch");
    check(cudaStreamSynchronize(stream.value));
    check(cudaMemcpy(out.data(),output.data,bytes,cudaMemcpyDeviceToHost));
    for(std::int64_t row=0;row<rows;++row) {
        const auto offset=static_cast<std::size_t>(row*width);
        double squares=0;
        for(std::int64_t j=0;j<width;++j) {const double v=decode(x[offset+j]); squares+=v*v;}
        const double inverse=1/std::sqrt(squares/static_cast<double>(width)+parameters.epsilon);
        for(std::int64_t j=0;j<width;++j) {
            const auto i=offset+static_cast<std::size_t>(j);
            double v=decode(x[i])*inverse*(decode(weight[j])+(mode==Mode::unit_offset ? 1 : 0));
            if(gated) {const double a=decode(g[i]); v*=a/(1+std::exp(-a));}
            const double expected=decode(encode(static_cast<float>(v))), actual=decode(out[i]);
            // Allow one BF16 rounding step around a boundary: the device uses
            // FP32 tree reduction and rsqrt, while this oracle sums in double.
            if(!std::isfinite(actual) || std::abs(actual-expected)>0.008*std::abs(expected)+0.0001)
                throw std::runtime_error("BF16 output differs from independent formula");
        }
    }
}
}
int main() {
    try {
        unsigned cases=0;
        for(const auto width:{1,3,255,256,257,8192}) for(const auto rows:{1,3})
            for(const auto mode:{Mode::plain,Mode::unit_offset,Mode::gated})
                for(int alias=0;alias<(mode==Mode::gated ? 3 : 2);++alias) {run(width,rows,mode,alias); ++cases;}
        for(const auto mode:{Mode::plain,Mode::unit_offset,Mode::gated})
            for(int alias=0;alias<(mode==Mode::gated ? 3 : 2);++alias) {run(1,max_rows,mode,alias); ++cases;}
        std::cout<<"PASS "<<cases<<" RMSNorm mode/shape/alias cases and metadata rejections\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
