#include "grouped_int.h"
#include "encoded_kv.h"
#include <crit/state.h>
#include "batch_attention.h"
#include <array>
#include "attention.h"
#include <iostream>
#include <random>
#include <string>
using namespace crit;
#if defined(CRIT_GROUPED_INT_AVX2)
constexpr bool fast=true;
#else
constexpr bool fast=false;
#endif
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"Invalid quantization input was accepted");}
void close(double a,double b,double rel=4e-6,double abs=2e-5){require(std::abs(a-b)<=abs+rel*std::abs(b),"Quantized arithmetic differs from independent decoded oracle");}
void arithmetic(){
 std::mt19937 rng(401);std::size_t cases=0;
 for(int bits:{4,8})for(std::size_t n:{1U,7U,33U})for(std::size_t k:{32U,96U,512U})for(std::size_t t:{1U,3U,8U}){
  std::vector<float> w(n*k),x(t*k);
  for(auto& f:w)f=static_cast<float>(static_cast<int>(rng()%10001)-5000)/1137;
  for(auto& f:x)f=static_cast<float>(static_cast<int>(rng()%10001)-5000)/4211;
  auto packed=quant::quantize(w,n,k,bits);auto v=packed.view();
  for(bool a8:{false,true}){
   const auto y=quant::project(v,x,t,a8,fast);std::vector<float> row(k);
   auto activ=quant::quantize(x,t,k,8);
   for(std::size_t token=0;token<t;++token)for(std::size_t out=0;out<n;++out){v.row(out,row);double expected=0;
    for(std::size_t c=0;c<k;++c){double a=x[token*k+c];if(a8)a=static_cast<double>(static_cast<std::int8_t>(activ.codes[token*k+c]))*activ.scales[(token*k+c)/32];expected+=static_cast<double>(row[c])*a;}
    close(y[token*n+out],expected,1e-5,5e-5);
   }
   ++cases;
  }
 }
 for(int bits:{4,8}){
  std::vector<std::uint8_t> bytes(bits==4?16:32,bits==4?0x88:0x80);std::vector<float> scales{.25f};quant::Rows w(bytes,scales,1,32,bits);
  std::vector<float> x(32,-127);for(bool a8:{false,true})close(quant::project(w,x,1,a8,fast)[0],32.*(bits==4?-8:-128)*.25*-127);
 }
 std::cout<<cases<<" native/scalar independently decoded projection comparisons passed\n";
}
void failures(){
 std::vector<float> z(32,0);auto packed=quant::quantize(z,1,32,4);for(auto x:packed.codes)require(x==0,"Zero group packing");require(packed.scales[0]==1,"Zero scale convention");
 require(quant::even(.5)==0 && quant::even(1.5)==2 && quant::even(-1.5)==-2 && quant::even(-2.5)==-2,"Ties-to-even rounding");
 rejects([&]{quant::quantize(z,1,31,4);});rejects([&]{quant::quantize(z,1,32,3);});
 z[0]=std::numeric_limits<float>::infinity();rejects([&]{quant::quantize(z,1,32,8);});
 z[0]=std::numeric_limits<float>::denorm_min();rejects([&]{quant::quantize(z,1,32,8);});
 packed.scales[0]=0;rejects([&]{packed.view();});packed.scales[0]=1;
 rejects([&]{packed.view().subrows(1,1);});rejects([&]{quant::project(packed.view(),std::span(z).first(31),1);});
 rejects([]{quant::Rows{}.code(0);});
}
void kv(){
 using namespace techniques;
 // Core's StateCache keeps sequences, COW, rollback, rewind and retirement; EncodedPages the G32 rows.
 core::Resources resources(std::size_t{1}<<24,0);
 const auto config=[](std::uint32_t pages,std::uint32_t tokens){core::StateCacheConfig c;c.pages=pages;c.page_tokens=4;c.sequences=4;c.max_tokens=tokens;c.max_round_rows=tokens;return c;};
 const std::array<core::Token,1> token{0};
 EncodedPages pages(3,4,32,32);require(pages.reserved_bytes()==3*4*(64+8),"Compressed physical arena accounting");
 rejects([]{EncodedPages bad(1,4,31,33);});
 std::vector<float> k(32),v(32);for(std::size_t i=0;i<32;++i){k[i]=static_cast<float>(i)/31;v[i]=-static_cast<float>(i)/15;}
 {core::StateCache cache(config(3,12),pages,resources);
  const auto a=cache.open();auto round=pages.stage(cache,a,k,v);cache.commit(a,round.id,token);
  auto held=pages.view(cache,a);auto initial=held.block(0);std::vector<float> old(initial.values.begin(),initial.values.end());
  const auto branch=cache.fork(a);round=pages.stage(cache,branch,k,v);cache.commit(branch,round.id,token);require(cache.stats().live_pages==2,"Shared tail must copy on write");
  require(held.rows()==1 && held.block(0).values[32]==old[32],"Committed read view changed");
  round=pages.stage(cache,branch,k,v);cache.discard(branch,round.id);require(cache.committed(branch)==2,"Rollback retained rows");cache.rewind(branch,1);require(cache.committed(branch)==1,"Rewind failed");
  std::vector<float> hk(16),hv(16);held.gather_head(0,0,1,1,16,16,hk,hv);
  for(std::size_t i=0;i<16;++i){close(hk[i],old[16+i],0,1e-7);close(hv[i],old[48+i],0,1e-7);}
  rejects([&]{held.gather_head(0,0,1,2,32,32,k,v);});
  std::vector<std::size_t> positions{0};BasicAttentionPlan<EncodedPagesView> plan(held,{1,1,32,32,1/std::sqrt(32.)},positions,1,4,8192);
  require(plan.workspace_bytes()>=4*64*4,"Encoded page decode scratch must be charged");auto answer=plan.run({k,1,1,32,32,32});for(std::size_t i=0;i<32;++i)close(answer.values[i],old[32+i],0,1e-6);
  const auto before=cache.stats().live_pages;k[0]=std::numeric_limits<float>::infinity();rejects([&]{pages.stage(cache,branch,k,v);});
  require(cache.stats().live_pages==before && cache.committed(branch)==1 && cache.staged(branch)==0,"Invalid append mutated KV");k[0]=0;
  cache.close(a);cache.close(branch);cache.retire(cache.epoch());
  require(cache.stats().live_pages==0 && cache.stats().free_pages==3,"Closed encoded sequences failed to retire pages");
 }
 {EncodedPages one(1,4,32,32);core::StateCache cache(config(1,4),one,resources);const auto a=cache.open();auto round=one.stage(cache,a,k,v);cache.commit(a,round.id,token);
  const auto child=cache.fork(a);rejects([&]{one.stage(cache,child,k,v);});
  require(cache.committed(child)==1 && cache.staged(child)==0 && cache.stats().live_pages==1,"Capacity failure mutated encoded KV");}
 {EncodedPages many(8,4,64,64);core::StateCache cache(config(8,16),many,resources);const auto sequence=cache.open();std::vector<float> keys(10*64),values(keys.size());
  for(std::size_t i=0;i<keys.size();++i){keys[i]=static_cast<float>(static_cast<int>(i%97)-48)/97;values[i]=static_cast<float>(static_cast<int>(i%41)-20)/41;}
  const auto round=many.stage(cache,sequence,keys,values);cache.commit(sequence,round.id,std::vector<core::Token>(10,0));auto image=many.view(cache,sequence);std::vector<float> dense;
  for(std::size_t page=0;page<image.pages();++page){const auto b=image.block(page);dense.insert(dense.end(),b.values.begin(),b.values.end());}
  std::vector<float> query(keys.end()-64,keys.end());std::vector<std::size_t> positions{9};BasicAttentionPlan<EncodedPagesView> plan(image,{2,2,32,32,1/std::sqrt(32.)},positions,1,3,16384);auto result=plan.run({query,1,2,32,64,32});
  const std::array<AttentionBatchSpec<EncodedPagesView>,2> specs{{{{1,2},image,{2,2,32,32,1/std::sqrt(32.)},positions},{{1,1},image,{2,2,32,32,1/std::sqrt(32.)},positions}}};
  BasicAttentionBatch<EncodedPagesView> batch(specs,1,3,{2,2,32768,32768,32768});
  const std::array<AttentionBatchInput,2> inputs{{{{1,2},{query,1,2,32,64,32}},{{1,1},{query,1,2,32,64,32}}}};
  const auto batched=batch.run(inputs);
  require(batched[0].key.generation==1 && batched[1].key.generation==2,"Encoded batch generation identity");
  for(const auto& row:batched)require(row.attention.values==result.values,"Encoded batch differs from decoded-KV-checked plan");
  for(std::size_t head=0;head<2;++head){std::vector<double> scores(10);double maximum=-std::numeric_limits<double>::infinity();
   for(std::size_t t=0;t<10;++t){double dot=0;for(std::size_t d=0;d<32;++d)dot+=static_cast<double>(query[head*32+d])*dense[t*128+head*32+d];scores[t]=dot/std::sqrt(32.);maximum=std::max(maximum,scores[t]);}
   double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
   for(std::size_t d=0;d<32;++d){double expected=0;for(std::size_t t=0;t<10;++t)expected+=scores[t]/denominator*dense[t*128+64+head*32+d];close(result.values[head*32+d],expected,1e-5,1e-6);}
  }
 }
 std::cout<<"Encoded pages under StateCache: COW, rollback, bounded attention, invalid input and retirement passed\n";
}
int main(int argc,char** argv){try{const std::string mode=argc>1?argv[1]:"all";if(mode=="arithmetic" || mode=="all")arithmetic();if(mode=="failures" || mode=="all")failures();if(mode=="kv" || mode=="all")kv();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
