#include "latency_chunks.h"
#include <limits>
#include <random>
#include <iostream>
using namespace crit::techniques;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try {
  ChunkLatency dense{10,1,1,2};
  ChunkWindow w{0,100,100,100,1,16,0,100};
  auto short_context=choose_prefill_chunk(w,dense); w.history=100;
  require(choose_prefill_chunk(w,dense).tokens<short_context.tokens,"History did not reduce chunk");
  auto linear=ChunkLatency{10,3,0,0};
  auto long_linear=choose_prefill_chunk(w,linear);w.history=0;
  require(choose_prefill_chunk(w,linear).tokens==long_linear.tokens,"Linear model gained false context cost");
  require(choose_prefill_chunk(w,{}).tokens==16,"Missing estimate fallback");
  dense.call_ns=std::numeric_limits<double>::quiet_NaN();
  require(choose_prefill_chunk(w,dense).fallback,"Invalid calibration did not fall back");
  auto minimum=choose_prefill_chunk({100,20,1,20,8,8,0,1},linear);
  require(minimum.tokens==1 && !minimum.target_met,"Tight budget stalled or hid target miss");
  std::mt19937 rng(782);
  const auto next=[&]{return static_cast<std::uint32_t>(rng());};
  for(int trial=0;trial<6000;++trial) {
    ChunkLatency cost{double(next()%20),double(next()%8),double(next()%4),double(next()%3)};
    ChunkWindow query{next()%100,1+next()%100,1+next()%100,1+next()%100,1+next()%8,16,next()%9,double(1+next()%10000)};
    auto actual=choose_prefill_chunk(query,cost);
    const auto cap=std::min({query.remaining,query.token_budget,query.workspace_tokens});
    const auto quantum=std::min(query.alignment,cap);
    auto expected=quantum;
    for(std::uint32_t n=quantum;n<=cap;n+=quantum)
      if(cost.estimate(query.history,n,query.decode_work)<=query.target_ns)expected=n;
    if(query.remaining<=cap && cost.estimate(query.history,query.remaining,query.decode_work)<=query.target_ns)expected=query.remaining;
    require(actual.tokens==expected,"Binary policy disagrees with exhaustive feasible search");
  }
  std::cout<<"Latency chunk oracle passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
