#include "fp8_block.h"
#include <array>
#include <iostream>
#include <random>
#include <string>
using namespace crit;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"Invalid quantization input was accepted");}
void close(double a,double b,double rel=4e-6,double abs=2e-5){require(std::abs(a-b)<=abs+rel*std::abs(b),"Quantized arithmetic differs from independent decoded oracle");}
// The matched FP8 order's host twin (fp8_block.h): E8M0 scales, the declared activation quantizer, the FP64 bound.
void fp8_order(){
 namespace fb=crit::fp8_block;
 require(fb::e8m0(127)==1.f && fb::e8m0(130)==8.f && fb::e8m0(0)==std::ldexp(1.f,-127),"E8M0 scales");
 rejects([]{(void)fb::e8m0(255);});
 std::array<float,32> v{};for(std::size_t k=0;k<32;++k)v[k]=static_cast<float>(static_cast<int>(k)-16)/5.f;
 std::array<std::uint8_t,32> codes{};
 const float mx=fb::quantize_group(v.data(),32,fb::ScaleKind::mx,codes.data());
 require(mx==std::ldexp(1.f,-7),"MXFP8 shared exponent 2^(floor(log2 amax) - 8)");
 for(std::size_t k=0;k<32;++k){require(codes[k]==fp8::encode(v[k]/mx),"MXFP8 codes");close(fp8::decode(codes[k])*mx,v[k],1./16,1e-6);}
 std::array<std::uint8_t,32> amax_codes{};
 const float amax=fb::quantize_group(v.data(),32,fb::ScaleKind::amax,amax_codes.data());
 require(amax==3.2f/448.f,"amax / 448 scale");
 for(std::size_t k=0;k<32;++k)require(amax_codes[k]==fp8::encode(v[k]/amax),"amax codes");
 std::array<float,32> zero{};std::array<std::uint8_t,32> zc{};zc.fill(7);
 require(fb::quantize_group(zero.data(),32,fb::ScaleKind::mx,zc.data())==1.f,"all-zero MX group scale 1");
 for(auto c:zc)require(c==0,"all-zero group codes");
 require(fb::quantize_group(zero.data(),32,fb::ScaleKind::amax,zc.data())==0.f,"all-zero amax group scale 0");
 const float ws[1]={2.f},as[1]={mx};double magnitude=0,expected=0;
 for(std::size_t k=0;k<32;++k){const double p=static_cast<double>(fp8::decode(codes[k]))*fp8::decode(codes[k])*2.*mx;expected+=p;}
 require(fb::project_fp64(codes.data(),ws,32,codes.data(),as,32,32,&magnitude)==expected && magnitude==expected,"FP64 projection");
 std::cout<<"FP8 order host twin: E8M0, both quantizers and the FP64 bound passed\n";
}
int main(int argc,char** argv){try{const std::string mode=argc>1?argv[1]:"all";if(mode=="fp8_block" || mode=="all")fp8_order();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
