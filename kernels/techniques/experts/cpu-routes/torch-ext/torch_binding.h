#pragma once

#include <vector>

#include <torch/torch.h>

// A routed layer's CPU routes over a slot of one of the host's handoffs (routes_cuda/cpu_routes.cu). A slot is named by
// its device addresses (ready, done, status, payload: crit_plugins.host's Handoff.words(slot)); hidden, capacity
// (routes) and token_capacity size its payload (layout).
std::vector<std::int64_t> layout(std::int64_t hidden, std::int64_t routes, std::int64_t tokens);
void post(std::int64_t ready, std::int64_t done, std::int64_t status, std::int64_t payload, std::int64_t hidden,
          std::int64_t capacity, std::int64_t token_capacity, const torch::Tensor& ids, const torch::Tensor& slots,
          const torch::Tensor& x, std::int64_t key_base, const torch::Tensor& cpu_routes,
          const torch::Tensor& cpu_count, const torch::Tensor& counters);
void scatter(std::int64_t ready, std::int64_t done, std::int64_t status_word, std::int64_t payload,
             std::int64_t hidden, std::int64_t capacity, std::int64_t token_capacity, const torch::Tensor& y,
             const torch::Tensor& cpu_routes, const torch::Tensor& cpu_count, const torch::Tensor& status,
             std::int64_t routes);
