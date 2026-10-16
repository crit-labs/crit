#include "attention.h"
#include "encoded_kv.h"
#include <crit/state.h>
#include <array>
#include <iostream>

int main() {
  try {
    namespace cc = crit::core;
    using namespace crit::techniques;
    // G32 INT8 rows in EncodedPages; Core's StateCache keeps the bookkeeping.
    cc::Resources resources(std::size_t{1} << 20, 0);
    EncodedPages pages(4,2,64,64);
    cc::StateCacheConfig config;
    config.pages=4;config.page_tokens=2;config.sequences=2;config.max_tokens=4;config.max_round_rows=3;
    cc::StateCache cache(config,pages,resources);
    const auto state=cache.open();
    std::vector<float> keys(3*64),values(3*64),query(64,.125f);
    for(std::size_t i=0;i<keys.size();++i) {keys[i]=static_cast<float>(i%7)/8;values[i]=static_cast<float>(i%11)/16;}
    {
      const auto round=pages.stage(cache,state,keys,values);
      BasicAttentionPlan<EncodedPagesView> plan(pages.view(cache,state,true),{2,2,32,32,.125},std::array<std::size_t,1>{2},1,2,16384);
      const auto result=plan.run({query,1,2,32,64,32});
      cache.commit(state,round.id,std::array<cc::Token,3>{1,2,3});
      std::cout<<"Packed KV payload="<<pages.reserved_bytes()<<", attention scratch="<<plan.workspace_bytes()
        <<", first="<<result.values.front()<<'\n';
    }
    const auto branch=cache.fork(state);cache.rewind(branch,2);
    const auto round=pages.stage(cache,branch,std::span(keys).last(64),std::span(values).last(64));cache.discard(branch,round.id);
    if(cache.committed(state)!=3 || cache.committed(branch)!=2) throw std::runtime_error("Branch changed committed state");
    // A plan reads its pages until cache.retire() covers the epoch that released them.
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
