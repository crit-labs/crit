// A tiny hand-authored next-token model using Core and native Inference arithmetic.
#include <crit/core.h>
#include "grouped_int.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace crit::core;
constexpr std::size_t vocabulary = 4, width = 32;
constexpr std::size_t weight_bytes = vocabulary * width / 2 + vocabulary * sizeof(float);

crit::quant::PackedRows make_weights() {
    std::array<float, vocabulary * width> values{};
    // A -> B -> C -> D -> A. G32 signed int4 encodes these weights exactly.
    for(std::size_t token = 0; token < vocabulary; ++token)
        values[((token + 1) % vocabulary) * width + token] = 7.f;
    return crit::quant::quantize(values, vocabulary, width, 4);
}

struct HostBackend final : Backend {
    void bind_thread() override {}
    void drain() override {} // Every projection completes before advance returns.
};

class ProjectionSession final : public Session {
    std::span<const Token> prompt_;
    const crit::quant::PackedRows& weights_;
    Reservation state_charge_;
    std::size_t frontier_ = 0, pending_frontier_ = 0;
    Token state_ = 0, pending_state_ = 0, pending_token_ = 0;
    bool decoding_ = false;
public:
    ProjectionSession(std::span<const Token> prompt, const crit::quant::PackedRows& weights,
                      Reservation charge)
        : prompt_(prompt), weights_(weights), state_charge_(std::move(charge)) {}

    Progress advance(WorkBudget budget) override {
        pending_frontier_ = frontier_;
        pending_state_ = state_;
        decoding_ = budget.prefill_complete;
        if(!decoding_) {
            const auto count = std::min<std::size_t>(budget.prefill_tokens, prompt_.size() - frontier_);
            for(std::size_t i = 0; i < count; ++i) pending_state_ = prompt_[frontier_ + i];
            pending_frontier_ += count;
            return {static_cast<std::uint32_t>(count), pending_frontier_ == prompt_.size(), {}, nullptr};
        }
        std::array<float, width> activations{};
        activations[static_cast<std::size_t>(state_)] = 1.f;
        const auto logits = crit::quant::project(weights_.view(), activations, 1);
        pending_token_ = static_cast<Token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        pending_state_ = pending_token_;
        // Core borrows this token until commit; the owning session stays alive.
        return {0, true, std::span<const Token>(&pending_token_, 1), nullptr};
    }

    void commit(std::uint32_t accepted, bool) override {
        if(accepted > (decoding_ ? 1u : 0u)) throw std::logic_error("Invalid accepted prefix");
        frontier_ = pending_frontier_;
        if(!decoding_ || accepted) state_ = pending_state_;
    }
    void close(Finish) noexcept override {
        prompt_ = {};
        state_charge_ = {};
    }
};

class ProjectionModel final : public Model {
    Resources& resources_;
    Reservation weight_charge_;
    const crit::quant::PackedRows weights_;
public:
    explicit ProjectionModel(Resources& resources)
        : resources_(resources), weight_charge_(resources.reserve(Memory::host, weight_bytes)),
          weights_(make_weights()) {}
    ModelInfo info() const override {
        ModelInfo declaration;
        declaration.implementation = "core-inference-projection-example";
        declaration.vocabulary = static_cast<std::uint32_t>(vocabulary);
        declaration.context = 32;
        declaration.prefill_chunk = 2;
        return declaration;
    }
    std::unique_ptr<Session> open(std::span<const Token> prompt) override {
        if(prompt.empty()) throw std::invalid_argument("A nonempty prompt is required");
        // Reserve the session and projection-result payload before allocating it.
        auto charge = resources_.reserve(Memory::host, sizeof(ProjectionSession) + vocabulary * sizeof(float));
        return std::make_unique<ProjectionSession>(prompt, weights_, std::move(charge));
    }
    std::vector<Token> tokenize(std::string_view text) override {
        std::vector<Token> tokens;
        for(char c : text) {
            if(c < 'A' || c > 'D') throw std::invalid_argument("The toy vocabulary is A, B, C, D");
            tokens.push_back(c - 'A');
        }
        return tokens;
    }
    std::string detokenize(std::span<const Token> tokens) override {
        std::string text;
        for(Token token : tokens) {
            if(token < 0 || token >= static_cast<Token>(vocabulary)) throw std::invalid_argument("Invalid toy token");
            text.push_back(static_cast<char>('A' + token));
        }
        return text;
    }
};
}

int main() {
    try {
        crit::core::Profile profile;
        profile.context = 32;
        profile.prefill_chunk = 2;
        profile.host_limit = 4096;
        profile.device_limit = 1; // Core requires positive limits; this engine uses no device memory.
        crit::core::Engine engine(profile, std::make_shared<HostBackend>(),
            [](const crit::core::Profile&, crit::core::Services services) {
                return std::make_unique<ProjectionModel>(services.resources);
            });
        // Multiple prefill chunks, then six decode/commit/publication rounds.
        const auto first = engine.submit(engine.tokenize("ABCDAB"), 6).wait();
        if(first.finish != crit::core::Finish::length || engine.detokenize(first.tokens) != "CDABCD")
            throw std::runtime_error("Chunked Core + Inference generation failed");
        // A new request has private state even though it reuses the model weights.
        const auto second = engine.submit(engine.tokenize("D"), 3).wait();
        if(second.finish != crit::core::Finish::length || engine.detokenize(second.tokens) != "ABC")
            throw std::runtime_error("Consecutive request isolation failed");
        std::cout << "ABCDAB -> " << engine.detokenize(first.tokens)
                  << "\nD -> " << engine.detokenize(second.tokens) << '\n';
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
