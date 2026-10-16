#pragma once
// Whole-round cost check from Core DeviceTiming values. Host-only C++20.
#include <crit/core.h>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace crit::techniques::device_round {
struct RoundBudgetReport {
    double step_ns = 0, round_ns = 0;
    double tokens_per_round = 0;
    double ratio = 0;   // round_ns / step_ns
    double speedup = 0; // tokens_per_round / ratio, versus ordinary decoding
    double threshold = 0;
    bool pass = false;  // ratio <= threshold
};

// A round of K drafts plus K+1 verify columns should cost little more than one
// ordinary decode step. A failing ratio means round overhead (host syncs,
// serial launches, extra readbacks, uncaptured work): fix it before tuning kernels.
class RoundBudget {
public:
    static constexpr double default_threshold = 1.3;
    explicit RoundBudget(double threshold = default_threshold) : threshold_(threshold) {
        if(!std::isfinite(threshold) || threshold <= 0) throw std::invalid_argument("Positive finite round/step threshold required");
    }
    static std::uint64_t total_ns(std::span<const core::DeviceTiming> timings) {
        std::uint64_t total = 0;
        for(const auto& timing : timings) {
            if(timing.duration_ns > UINT64_MAX - total) throw std::overflow_error("Device timing overflow");
            total += timing.duration_ns;
        }
        return total;
    }
    // One ordinary (non-speculative) decode step: all its device stages.
    void observe_step(std::span<const core::DeviceTiming> timings) { observe_step_ns(total_ns(timings)); }
    void observe_step_ns(std::uint64_t ns) {
        if(!ns) throw std::invalid_argument("Ordinary step timing must be positive");
        step_ns_ += double(ns); ++steps_;
    }
    // One whole speculative round and the tokens it published.
    void observe_round(std::span<const core::DeviceTiming> timings, std::uint32_t emitted) {
        observe_round_ns(total_ns(timings), emitted);
    }
    void observe_round_ns(std::uint64_t ns, std::uint32_t emitted) {
        if(!ns) throw std::invalid_argument("Round timing must be positive");
        round_ns_ += double(ns); tokens_ += emitted; ++rounds_;
    }
    RoundBudgetReport report() const {
        if(!steps_ || !rounds_) throw std::logic_error("Observe at least one ordinary step and one round");
        RoundBudgetReport r;
        r.step_ns = step_ns_ / double(steps_);
        r.round_ns = round_ns_ / double(rounds_);
        r.tokens_per_round = double(tokens_) / double(rounds_);
        r.ratio = r.round_ns / r.step_ns;
        r.speedup = r.tokens_per_round / r.ratio;
        r.threshold = threshold_;
        r.pass = r.ratio <= threshold_;
        return r;
    }
private:
    double threshold_, step_ns_ = 0, round_ns_ = 0;
    std::uint64_t steps_ = 0, rounds_ = 0, tokens_ = 0;
};
}
