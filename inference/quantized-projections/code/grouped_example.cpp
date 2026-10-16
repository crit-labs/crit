#include "grouped_int.h"
#include <iostream>

int main() {
  try {
    std::vector<float> weights(3*64),activations(2*64);
    for(std::size_t i=0;i<weights.size();++i) weights[i]=static_cast<float>(static_cast<int>(i%19)-9)/16;
    for(std::size_t i=0;i<activations.size();++i) activations[i]=static_cast<float>(static_cast<int>(i%13)-6)/8;
    for(int bits:{4,8}) {
      const auto packed=crit::quant::quantize(weights,3,64,bits);
      const auto result=crit::quant::project(packed.view(),activations,2,true);
      std::cout<<"W"<<bits<<"/A8 payload="<<packed.codes.size()+packed.scales.size()*sizeof(float)
        <<" bytes, output rows="<<result.size()<<", first="<<result.front()<<'\n';
    }
    // Keep packed bytes/scales alive and immutable while any borrowed Rows is used.
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
