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
void run(std::int64_t width,std::int64_t rows,Mode mode,int alias,Rounding rounding) {
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
    Parameters parameters{mode,1e-5F,rounding};
    auto invalid=input; invalid.width=0;
    rejects([&]{validate(invalid,gain,gated,output,parameters,{});});
    invalid.width=max_width+1;
    rejects([&]{validate(invalid,gain,gated,output,parameters,{});});
    invalid=input; invalid.rows=max_rows+1;
    rejects([&]{validate(invalid,gain,gated,output,parameters,{});});
    rejects([&]{validate(input,gain,gated,output,{mode,1e-5F,static_cast<Rounding>(2)},{});});
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
            const double scale=decode(weight[j])+(mode==Mode::unit_offset ? 1 : 0);
            double v=decode(x[i])*inverse;
            // normalize_first rounds at the same points as the kernel: x * r, then the gain product.
            if(rounding==Rounding::normalize_first) v=decode(encode(static_cast<float>(decode(encode(static_cast<float>(v)))*scale)));
            else v*=scale;
            if(gated) {const double a=decode(g[i]); v*=a/(1+std::exp(-a));}
            const double expected=decode(encode(static_cast<float>(v))), actual=decode(out[i]);
            // Allow one BF16 rounding step around a boundary: the device uses
            // FP32 tree reduction and rsqrt, while this oracle sums in double.
            if(!std::isfinite(actual) || std::abs(actual-expected)>0.008*std::abs(expected)+0.0001)
                throw std::runtime_error("BF16 output differs from independent formula");
        }
    }
}
// Exact check of the normalize_first rounding order. A unit gain in single mode
// gives the device's own BF16 x * r; normalize_first must then equal that value
// times the gain rounded once more to BF16, bit for bit (one step with a gate,
// whose host and device exp may differ).
void exact_normalize_first(std::int64_t width,std::int64_t rows,Mode mode) {
    const auto count=static_cast<std::size_t>(width*rows), bytes=count*2;
    std::vector<std::uint16_t> x(count),g(count),ones(static_cast<std::size_t>(width),encode(1.0F)),
        weight(static_cast<std::size_t>(width)),normalized(count),single(count),out(count);
    std::uint32_t state=12345;
    const auto next=[&]{state=state*1664525U+1013904223U; return static_cast<float>(state>>8)/16777216.0F;};
    for(std::size_t i=0;i<count;++i) {x[i]=encode((next()-0.5F)*6); g[i]=encode((next()-0.5F)*8);}
    for(auto& w:weight) w=encode(0.7F+next()*0.6F);
    Stream stream;
    Buffer dx(bytes,stream.value),dg(bytes,stream.value),dw(weight.size()*2,stream.value),
        d1(weight.size()*2,stream.value),dy(bytes,stream.value);
    check(cudaMemcpy(dx.value,x.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dg.value,g.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw.value,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(d1.value,ones.data(),ones.size()*2,cudaMemcpyHostToDevice));
    ReadView input{dx.value,width,rows,2*width},gain{dw.value,width,1,2*width},unit{d1.value,width,1,2*width},
        gate{dg.value,width,rows,2*width};
    WriteView output{dy.value,width,rows,2*width};
    const ReadView* gated=mode==Mode::gated ? &gate : nullptr;
    const auto fetch=[&](std::vector<std::uint16_t>& host) {
        check(cudaStreamSynchronize(stream.value));
        check(cudaMemcpy(host.data(),dy.value,bytes,cudaMemcpyDeviceToHost));
    };
    enqueue(input,unit,nullptr,output,{Mode::plain,1e-6F},{stream.value}); fetch(normalized);
    enqueue(input,gain,gated,output,{mode,1e-6F},{stream.value}); fetch(single);
    enqueue(input,gain,gated,output,{mode,1e-6F,Rounding::normalize_first},{stream.value}); fetch(out);
    std::size_t differ=0;
    for(std::size_t i=0;i<count;++i) {
        const float scale=decode(weight[i%static_cast<std::size_t>(width)])+(mode==Mode::unit_offset ? 1.0F : 0.0F);
        auto expected=encode(decode(normalized[i])*scale);
        if(gated) {const float a=decode(g[i]); expected=encode(decode(expected)*(a/(1+std::exp(-a))));}
        const int step=std::abs(int(std::int16_t(out[i]))-int(std::int16_t(expected)));
        if(step>(gated ? 1 : 0)) throw std::runtime_error("normalize_first differs from its exact rounding order");
        // Each extra rounding moves the result by at most one BF16 step.
        if(std::abs(int(std::int16_t(out[i]))-int(std::int16_t(single[i])))>(gated ? 2 : 1))
            throw std::runtime_error("rounding orders differ by more than their extra roundings");
        differ+=out[i]!=single[i];
    }
    if(width>=256 && differ==0) throw std::runtime_error("normalize_first did not change any rounding");
}
}
int main() {
    try {
        unsigned cases=0;
        for(const auto rounding:{Rounding::single,Rounding::normalize_first}) {
            for(const auto width:{1,3,255,256,257,8192,12288,16384}) for(const auto rows:{1,3})
                for(const auto mode:{Mode::plain,Mode::unit_offset,Mode::gated})
                    for(int alias=0;alias<(mode==Mode::gated ? 3 : 2);++alias) {run(width,rows,mode,alias,rounding); ++cases;}
            // Past the former 49152-row cap; each row is one grid-x block.
            for(const auto mode:{Mode::plain,Mode::unit_offset,Mode::gated})
                for(int alias=0;alias<(mode==Mode::gated ? 3 : 2);++alias) {run(1,65537,mode,alias,rounding); ++cases;}
        }
        for(const auto width:{4096,12288,16384})
            for(const auto mode:{Mode::plain,Mode::unit_offset,Mode::gated}) {exact_normalize_first(width,64,mode); ++cases;}
        std::cout<<"PASS "<<cases<<" RMSNorm mode/rounding/shape/alias cases and metadata rejections\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
