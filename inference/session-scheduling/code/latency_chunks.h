#pragma once
// Independently implemented shape-cost policy. No device or model dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace crit::techniques {
struct ChunkLatency {
  double call_ns=0, token_ns=0, pair_ns=0, decode_token_ns=0;
  bool valid() const {
    return std::isfinite(call_ns) && call_ns>=0 && std::isfinite(token_ns) && token_ns>=0 &&
        std::isfinite(pair_ns) && pair_ns>=0 && std::isfinite(decode_token_ns) && decode_token_ns>=0;
  }
  double estimate(std::uint32_t history, std::uint32_t tokens, std::uint32_t decode_work=0) const {
    const double n=tokens;
    return call_ns + n*token_ns + (double(history)*n+n*(n+1)/2)*pair_ns + double(decode_work)*decode_token_ns;
  }
};
struct ChunkWindow {
  std::uint32_t history, remaining, token_budget, workspace_tokens;
  std::uint32_t alignment=1, fallback_tokens=1, decode_work=0;
  double target_ns=0;
};
struct ChunkChoice {
  std::uint32_t tokens=0;
  std::optional<double> estimated_ns;
  bool target_met=false, fallback=false;
};
inline ChunkChoice choose_prefill_chunk(const ChunkWindow& w, std::optional<ChunkLatency> cost) {
  if(!w.alignment || !w.fallback_tokens || !std::isfinite(w.target_ns) || w.target_ns<=0)
    throw std::invalid_argument("Positive chunk alignment, fallback and latency target required");
  const auto cap=std::min({w.remaining,w.token_budget,w.workspace_tokens});
  if(!cap)return {};
  const auto aligned=[&](std::uint32_t n) { return n<w.alignment ? n : n/w.alignment*w.alignment; };
  const auto fallback=[&] { return ChunkChoice{aligned(std::min(cap,w.fallback_tokens)),{},false,true}; };
  if(!cost || !cost->valid())return fallback();
  const auto estimate=[&](std::uint32_t n){return cost->estimate(w.history,n,w.decode_work);};
  if(!std::isfinite(estimate(cap)))return fallback();
  // Nonnegative coefficients make this monotonic. Retain the call cost even
  // when history is nonzero: each chunk still performs an execution call.
  const auto quantum=std::min(cap,w.alignment);
  std::uint32_t lo=1,hi=cap/quantum;
  while(lo<hi) {
    const auto middle=lo+(hi-lo+1)/2;
    if(estimate(middle*quantum)<=w.target_ns)lo=middle; else hi=middle-1;
  }
  auto count=lo*quantum;
  // The last prompt tail can be unaligned; it must still fit every bound.
  if(w.remaining<=cap && estimate(w.remaining)<=w.target_ns)count=w.remaining;
  const auto prediction=estimate(count);
  return {count,prediction,prediction<=w.target_ns,false};
}
} // namespace crit::techniques
