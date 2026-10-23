// Throughput against concurrency for the batched synthetic engine: for each concurrency c, c requests with distinct
// prompts are submitted at once and the completion tokens are divided by the wall time from submission to the last
// completion (Eval's concurrent rate: completion tokens over the wall interval). Prints one JSON object per level.
//
//   crit_batched_engine_throughput [hidden layers heads kv_heads head_dim ffn vocab prompt output levels...]
#include "engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace be = crit::batched_engine;
namespace cc = crit::core;

int main(int argc, char** argv) {
    try {
        const auto arg = [&](int i, int fallback) { return argc > i ? std::atoi(argv[i]) : fallback; };
        be::Config config;
        config.shape = {arg(7, 32000), arg(1, 2048), arg(2, 8), arg(3, 16), arg(4, 4), arg(5, 128), arg(6, 5632), 10000.f, 5};
        const int prompt = arg(8, 128), output = arg(9, 128);
        std::vector<int> levels;
        for(int i = 10; i < argc; ++i) levels.push_back(std::atoi(argv[i]));
        if(levels.empty()) levels = {1, 2, 4, 8, 16, 32, 64};
        int top = 1;
        for(const int c : levels) top = std::max(top, c);
        config.context = static_cast<std::uint32_t>(prompt + output);
        config.max_concurrency = static_cast<std::uint32_t>(top);
        config.max_batch_tokens = static_cast<std::uint32_t>(std::max(top, 512));
        config.prefill_chunk = 512;
        config.prefix_cache = false;
        cudaDeviceProp prop{};
        crit::paged_attention::check(cudaGetDeviceProperties(&prop, 0), "device");
        auto stats = std::make_shared<be::Stats>();
        auto engine = be::make_engine(config, stats);
        const double weight_gb = static_cast<double>(config.shape.elements()) * 2 / 1e9;
        std::printf("{\"device\": \"%s\", \"weights_gb\": %.3f, \"prompt\": %d, \"output\": %d}\n", prop.name, weight_gb, prompt, output);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> token(0, config.shape.vocab - 1);
        const auto make = [&] {
            std::vector<cc::Token> p(static_cast<std::size_t>(prompt));
            for(auto& t : p) t = token(rng);
            return p;
        };
        engine->submit(make(), 8).wait();  // warm up
        for(const int c : levels) {
            std::vector<std::vector<cc::Token>> prompts;
            for(int i = 0; i < c; ++i) prompts.push_back(make());
            const auto start = std::chrono::steady_clock::now();
            std::vector<cc::Handle> handles;
            for(auto& p : prompts) handles.push_back(engine->submit(p, static_cast<std::uint32_t>(output)));
            std::size_t tokens = 0;
            double rate = 0;  // per-request decode rate after the first token, averaged over the requests
            for(auto& h : handles) {
                const auto r = h.wait();
                tokens += r.tokens.size();
                if(r.timing.decode_ns) rate += static_cast<double>(r.tokens.size() - 1) / (static_cast<double>(r.timing.decode_ns) / 1e9) / c;
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("{\"concurrency\": %d, \"completion_tokens\": %zu, \"seconds\": %.3f, \"tokens_per_s\": %.1f, \"per_request_decode_tokens_per_s\": %.1f}\n",
                        c, tokens, seconds, static_cast<double>(tokens) / seconds, rate);
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
