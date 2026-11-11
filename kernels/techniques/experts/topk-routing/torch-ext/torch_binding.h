#pragma once

#include <optional>

#include <torch/torch.h>

// logits [tokens][experts] FP32 (rows of unit stride), bias [experts] FP32 or none -> ids [tokens][top] int64 and
// weights [tokens][top] FP32, by the rule (softmax or sigmoid, scored before or after selection, renormalization with
// epsilon and floor, scaling).
void route(const torch::Tensor& logits, int64_t top, const std::optional<torch::Tensor>& bias, bool softmax,
           bool before, bool renormalize, double epsilon, double floor, double scaling, torch::Tensor& ids,
           torch::Tensor& weights);
