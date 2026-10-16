#pragma once
// This is Inference's symmetric signed codec, not GGML Q4_0/Q8_0.
// 32 values per group; FP32 scales; adjacent signed nibbles (even = low).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#if defined(CRIT_GROUPED_INT_AVX2)
#include <immintrin.h>
#endif

namespace crit::quant {
inline std::size_t extent(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max()/b) throw std::length_error("Quantized extent overflow");
  return a*b;
}
inline int even(double value) {
  const double low=std::floor(value), fraction=value-low;
  return static_cast<int>(low+(fraction>.5 || (fraction==.5 && std::fmod(low,2.)!=0.)));
}
inline float scale32(std::span<const float> values, int limit) {
  float maximum=0;
  for(float x:values) {
    if(!std::isfinite(x)) throw std::invalid_argument("Finite quantization input required");
    maximum=std::max(maximum,std::abs(x));
  }
  const float scale=maximum==0?1.f:maximum/static_cast<float>(limit);
  if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Positive represented FP32 scale required");
  return scale;
}
class Rows {
  std::span<const std::uint8_t> codes_;
  std::span<const float> scales_;
  std::size_t rows_=0,width_=0;
  int bits_=0;
  Rows(std::span<const std::uint8_t> c,std::span<const float> s,std::size_t n,std::size_t k,int b,bool)
      :codes_(c),scales_(s),rows_(n),width_(k),bits_(b) {}
public:
  Rows()=default;
  // Borrowed immutable bytes: retain the owner and do not mutate after validation.
  Rows(std::span<const std::uint8_t> codes,std::span<const float> scales,std::size_t n,std::size_t k,int bits)
      :codes_(codes),scales_(scales),rows_(n),width_(k),bits_(bits) {
    if(!n || !k || k%32 || (bits!=4 && bits!=8)) throw std::invalid_argument("Complete positive G32 rows required");
    const auto count=extent(n,k);
    if(codes.size()!=count/static_cast<std::size_t>(8/bits) || scales.size()!=count/32)
      throw std::invalid_argument("Quantized payload extent differs");
    for(float s:scales) if(!std::isfinite(s) || s<=0) throw std::invalid_argument("Invalid quantized scale");
  }
  std::size_t rows()const{return rows_;}
  std::size_t width()const{return width_;}
  int bits()const{return bits_;}
  auto codes()const{return codes_;}
  auto scales()const{return scales_;}
  int code(std::size_t i)const {
    if(i>=rows_*width_)throw std::out_of_range("Code index outside quantized tensor");
    int q=bits_==8?codes_[i]:(codes_[i/2]>>(4*(i%2)))&15;
    return q>=(1<<(bits_-1))?q-(1<<bits_):q;
  }
  Rows subrows(std::size_t first,std::size_t count)const {
    if(!count || first>rows_ || count>rows_-first) throw std::out_of_range("Quantized row slice outside tensor");
    const auto per=width_/static_cast<std::size_t>(8/bits_);
    return Rows(codes_.subspan(first*per,count*per),scales_.subspan(first*width_/32,count*width_/32),count,width_,bits_,true);
  }
  void row_range(std::size_t i,std::size_t first,std::span<float> output)const {
    if(i>=rows_ || first>width_ || output.size()>width_-first) throw std::out_of_range("Quantized row output extent");
#if defined(CRIT_GROUPED_INT_AVX2)
    if(bits_==8 && first%32==0 && output.size()%32==0) {
      for(std::size_t j=0;j<output.size();j+=32) {
        const auto at=i*width_+first+j;const auto scale=_mm256_set1_ps(scales_[at/32]);
        for(std::size_t lane=0;lane<32;lane+=8){
          const auto q=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_.data()+at+lane));
          const auto values=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),scale);
          const auto finite=_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f),values),_mm256_set1_ps(std::numeric_limits<float>::max()),_CMP_LE_OQ);
          if(_mm256_movemask_ps(finite)!=255)throw std::overflow_error("Decoded FP32 row overflow");
          _mm256_storeu_ps(output.data()+j+lane,values);
        }
      }
      return;
    }
#endif
    for(std::size_t j=0;j<output.size();++j){const auto at=i*width_+first+j;output[j]=static_cast<float>(code(at))*scales_[at/32];if(!std::isfinite(output[j]))throw std::overflow_error("Decoded FP32 row overflow");}
  }
  void row(std::size_t i,std::span<float> output)const {if(output.size()!=width_)throw std::out_of_range("Quantized row output extent");row_range(i,0,output);}
};
struct PackedRows {
  std::size_t rows,width; int bits;
  std::vector<std::uint8_t> codes; std::vector<float> scales;
  Rows view()const{return Rows(codes,scales,rows,width,bits);}
};
inline PackedRows quantize(std::span<const float> values,std::size_t n,std::size_t k,int bits=8) {
  if(!n || !k || k%32 || (bits!=4 && bits!=8) || values.size()!=extent(n,k))
    throw std::invalid_argument("Finite complete G32 tensor required");
  const int limit=(1<<(bits-1))-1;
  PackedRows result{n,k,bits,std::vector<std::uint8_t>(values.size()/static_cast<std::size_t>(8/bits)),{}};
  result.scales.reserve(values.size()/32);
  for(std::size_t start=0;start<values.size();start+=32) {
    const float s=scale32(values.subspan(start,32),limit);result.scales.push_back(s);
    for(std::size_t j=0;j<32;++j) {
      const auto q=std::clamp(even(static_cast<double>(values[start+j])/s),-limit,limit);
      const auto code=static_cast<std::uint8_t>(q&((1<<bits)-1));
      if(bits==8) result.codes[start+j]=code;
      else result.codes[(start+j)/2]|=static_cast<std::uint8_t>(code<<(4*((start+j)%2)));
    }
  }
  return result;
}
#if defined(CRIT_GROUPED_INT_AVX2)
inline __m256i group_codes(const Rows& w,std::size_t at) {
  if(w.bits()==8) return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.codes().data()+at));
  const auto bytes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.codes().data()+at/2));
  const auto mask=_mm_set1_epi8(15), sign=_mm_set1_epi8(8);
  auto low=_mm_and_si128(bytes,mask),high=_mm_and_si128(_mm_srli_epi16(bytes,4),mask);
  low=_mm_sub_epi8(_mm_xor_si128(low,sign),sign);high=_mm_sub_epi8(_mm_xor_si128(high,sign),sign);
  return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(low,high)),_mm_unpackhi_epi8(low,high),1);
}
inline float sum8(__m256 v) {
  auto s=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
  s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);return _mm_cvtss_f32(s);
}
inline float dot32(const Rows& w,std::size_t at,const float* x) {
  const auto q=group_codes(w,at);
  auto low=_mm256_castsi256_si128(q),high=_mm256_extracti128_si256(q,1);
  auto result=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)),_mm256_loadu_ps(x));
  result=_mm256_add_ps(result,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(low,8))),_mm256_loadu_ps(x+8)));
  result=_mm256_add_ps(result,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)),_mm256_loadu_ps(x+16)));
  result=_mm256_add_ps(result,_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(high,8))),_mm256_loadu_ps(x+24)));
  return sum8(result);
}
inline std::int32_t dot8(const Rows& w,std::size_t at,const std::int8_t* x) {
  const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
  const auto weights=group_codes(w,at);
  // Widen before multiply: externally supplied -128/-8 codes remain exact.
  auto totals=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)),_mm256_cvtepi8_epi16(_mm256_castsi256_si128(weights)));
  totals=_mm256_add_epi32(totals,_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a,1)),_mm256_cvtepi8_epi16(_mm256_extracti128_si256(weights,1))));
  auto s=_mm_add_epi32(_mm256_castsi256_si128(totals),_mm256_extracti128_si256(totals,1));
  s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);return _mm_cvtsi128_si32(s);
}
#endif
// Token-major FP32 inputs/outputs; optional dynamically quantized G32 int8 input.
// SIMD builds require AVX2 on the executing CPU. CPU dispatch belongs to the engine.
inline std::vector<float> project(const Rows& w,std::span<const float> x,std::size_t tokens,bool activation8=false,bool simd=false) {
  if(!w.rows() || !tokens || x.size()!=extent(tokens,w.width())) throw std::invalid_argument("Projection input extent differs");
  for(float a:x) if(!std::isfinite(a)) throw std::invalid_argument("Finite activation required");
#if !defined(CRIT_GROUPED_INT_AVX2)
  if(simd) throw std::invalid_argument("AVX2 implementation was not compiled");
#endif
  std::vector<std::int8_t> aq;std::vector<float> as;
  if(activation8) {
    aq.resize(x.size());as.reserve(x.size()/32);
    for(std::size_t at=0;at<x.size();at+=32) {
      const auto scale=scale32(x.subspan(at,32),127);as.push_back(scale);
      for(std::size_t j=0;j<32;++j) aq[at+j]=static_cast<std::int8_t>(std::clamp(even(static_cast<double>(x[at+j])/scale),-127,127));
    }
  }
  std::vector<float> out(extent(tokens,w.rows()));
  for(std::size_t t=0;t<tokens;++t) for(std::size_t r=0;r<w.rows();++r) {
    double total=0;
    for(std::size_t g=0;g<w.width();g+=32) {
      const auto at=r*w.width()+g, act=t*w.width()+g;double dot=0;
#if defined(CRIT_GROUPED_INT_AVX2)
      if(simd) dot=activation8?static_cast<double>(dot8(w,at,aq.data()+act)):static_cast<double>(dot32(w,at,x.data()+act));
      else
#endif
      if(activation8) {std::int32_t sum=0;for(std::size_t j=0;j<32;++j)sum+=w.code(at+j)*static_cast<int>(aq[act+j]);dot=sum;}
      else {for(std::size_t j=0;j<32;++j)dot+=static_cast<double>(w.code(at+j))*x[act+j];}
      total+=dot*static_cast<double>(w.scales()[at/32])*(activation8?static_cast<double>(as[act/32]):1.);
    }
    if(!std::isfinite(total) || std::abs(total)>std::numeric_limits<float>::max()) throw std::overflow_error("Quantized projection overflow");
    out[t*w.rows()+r]=static_cast<float>(total);
  }
  return out;
}
} // namespace crit::quant
