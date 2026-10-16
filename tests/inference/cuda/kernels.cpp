#include "fused_moe.h"
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
cudaError_t unfused_activation(const float*, const float*, std::uint16_t*, std::size_t, cudaStream_t);

void check(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(double got, double expected, double relative = 3e-5, double absolute = 3e-5) {
  if (!std::isfinite(got) || std::abs(got - expected) > absolute + relative * std::abs(expected))
    throw std::runtime_error("Numeric mismatch: " + std::to_string(got) + " versus " + std::to_string(expected));
}
template<class F> void rejects(F function) {
  try { function(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("Invalid GPU geometry was admitted");
}
template<class T> struct Device {
  T* pointer = nullptr;
  std::size_t count;
  explicit Device(std::size_t size) : count(size) { check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T))); }
  explicit Device(const std::vector<T>& source) : Device(source.size()) { upload(source); }
  ~Device() { cudaFree(pointer); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  std::span<T> span() { return {pointer, count}; }
  std::span<const T> read() const { return {pointer, count}; }
  void upload(std::span<const T> source) {
    require(source.size() == count, "Upload extent differs");
    check(cudaMemcpy(pointer, source.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> output(count);
    check(cudaMemcpy(output.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost)); return output;
  }
};
struct Weights {
  crit::fp8::RowWeights host;
  Device<std::uint8_t> codes;
  Device<std::uint16_t> scales;
  explicit Weights(crit::fp8::RowWeights input) : host(std::move(input)), codes(host.codes), scales(host.scales) { host.validate(); }
  crit::fp8::DeviceRows view() const { return {codes.read(), scales.read(), host.rows, host.width}; }
};
crit::fp8::RowWeights weights(std::size_t rows, std::size_t width, unsigned seed) {
  std::mt19937 rng(seed);
  crit::fp8::RowWeights result{rows, width, std::vector<std::uint8_t>(rows * width), std::vector<std::uint16_t>(rows)};
  for (auto& code : result.codes) { const auto magnitude = rng() % 127; code = static_cast<std::uint8_t>(magnitude | ((rng() & 1) << 7)); }
  for (std::size_t i = 0; i < rows; ++i) result.scales[i] = crit::fp8::bf16_word(static_cast<float>(i % 5 + 1) / 1024);
  return result;
}
std::vector<std::uint16_t> inputs(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<std::uint16_t> result(count);
  for (auto& value : result) value = crit::fp8::bf16_word(static_cast<float>(static_cast<int>(rng() % 65) - 32) / 32);
  return result;
}
void projections(cudaStream_t stream) {
  std::size_t cases = 0;
  for (std::size_t tokens = 1; tokens <= 8; ++tokens)
    for (std::size_t n : {1U, 7U, 8U, 9U, 17U}) for (std::size_t k : {1U, 3U, 31U, 127U, 128U, 129U, 256U, 511U}) {
      Weights w(weights(n, k, static_cast<unsigned>(100 + k + tokens)));
      const auto x = inputs(tokens * k, static_cast<unsigned>(k));
      Device<std::uint16_t> dx(x); Device<float> dy(tokens * n);
      crit::fp8::project_cuda(w.view(), dx.read(), tokens, dy.span(), stream);
      check(cudaStreamSynchronize(stream));
      const auto output = dy.download(); const auto expected = crit::fp8::project(w.host, x, tokens);
      for (std::size_t i = 0; i < output.size(); ++i) near(output[i], expected[i]);
      ++cases;
    }
  Weights w(weights(4, 128, 13)); Device<std::uint16_t> dx(inputs(128, 3)); Device<float> dy(4);
  rejects([&] { crit::fp8::project_cuda(w.view(), dx.read(), 0, dy.span(), stream); });
  rejects([&] { crit::fp8::project_cuda(w.view(), dx.read(), 1, {reinterpret_cast<float*>(dx.pointer), 4}, stream); });
  std::cout << cases << " FP8 projection cases passed\n";
}
void fusion(cudaStream_t stream) {
  std::size_t cases = 0;
  for (std::size_t tokens = 1; tokens <= 8; ++tokens) for (std::size_t k : {3U, 128U, 257U}) {
    constexpr std::size_t experts = 19, middle = 17, outputs = 13, top = 5;
    Weights router(weights(experts, k, 81)), gate(weights(middle, k, 31)), up(weights(middle, k, 111));
    const auto x = inputs(tokens * k, 13), down = inputs(outputs * middle, 43);
    Device<std::uint16_t> dx(x), da(tokens * middle), dd(down);
    Device<float> ds(tokens * experts), dw(tokens * top), dout(tokens * outputs), separate_gate(tokens * middle), separate_up(tokens * middle);
    Device<std::uint16_t> separate_activation(tokens * middle);
    Device<std::int32_t> di(tokens * top);
    crit::moe::entry_cuda(router.view(), gate.view(), up.view(), dx.read(), tokens, ds.span(), da.span(), stream);
    crit::moe::route_down_cuda(ds.read(), experts, top, da.read(), middle, dd.read(), outputs, tokens, di.span(), dw.span(), dout.span(), stream);
    crit::fp8::project_cuda(gate.view(), dx.read(), tokens, separate_gate.span(), stream);
    crit::fp8::project_cuda(up.view(), dx.read(), tokens, separate_up.span(), stream);
    check(unfused_activation(separate_gate.pointer, separate_up.pointer, separate_activation.pointer, tokens * middle, stream));
    check(cudaStreamSynchronize(stream));
    const auto scores = ds.download(), route_weights = dw.download(), shared = dout.download();
    const auto activation = da.download(); const auto ids = di.download();
    require(activation == separate_activation.download(), "Fusion changed the unfused BF16 activation words");
    const auto rs = crit::fp8::project(router.host, x, tokens), gs = crit::fp8::project(gate.host, x, tokens), us = crit::fp8::project(up.host, x, tokens);
    for (std::size_t i = 0; i < scores.size(); ++i) near(scores[i], rs[i]);
    for (std::size_t i = 0; i < activation.size(); ++i) {
      const double silu = gs[i] >= 0 ? gs[i] / (1 + std::exp(-gs[i])) : gs[i] * std::exp(gs[i]) / (1 + std::exp(gs[i]));
      const auto expected = crit::fp8::bf16(crit::fp8::bf16_word(static_cast<float>(silu * us[i])));
      near(crit::fp8::bf16(activation[i]), expected, .009, 2e-5);
    }
    for (std::size_t t = 0; t < tokens; ++t) {
      std::vector<std::int32_t> order(experts); std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return scores[t * experts + a] > scores[t * experts + b] || (scores[t * experts + a] == scores[t * experts + b] && a < b);
      });
      double sum = 0;
      for (std::size_t r = 0; r < top; ++r) sum += std::exp(static_cast<double>(scores[t * experts + order[r]]) - scores[t * experts + order[0]]);
      for (std::size_t r = 0; r < top; ++r) {
        require(ids[t * top + r] == order[r], "Route expert identity differs");
        near(route_weights[t * top + r], std::exp(static_cast<double>(scores[t * experts + order[r]]) - scores[t * experts + order[0]]) / sum);
      }
      for (std::size_t row = 0; row < outputs; ++row) {
        double expected = 0;
        for (std::size_t i = 0; i < middle; ++i) expected += static_cast<double>(crit::fp8::bf16(activation[t * middle + i])) * crit::fp8::bf16(down[row * middle + i]);
        near(shared[t * outputs + row], expected);
      }
    }
    rejects([&] { crit::moe::route_down_cuda(ds.read(), experts, experts + 1, da.read(), middle, dd.read(), outputs, tokens, di.span(), dw.span(), dout.span(), stream); });
    rejects([&] { crit::moe::entry_cuda(router.view(), gate.view(), up.view(), dx.read(), tokens, ds.span(), {reinterpret_cast<std::uint16_t*>(ds.pointer), tokens * middle}, stream); });
    ++cases;
  }
  // Equal logits must produce distinct original IDs ordered by ID, for each row.
  Device<float> scores(std::vector<float>(2 * 33, 0.f)), route_weights(2 * 32), shared(2);
  Device<std::int32_t> ids(2 * 32);
  Device<std::uint16_t> a(inputs(2, 5)), down(inputs(1, 6));
  crit::moe::route_down_cuda(scores.read(), 33, 32, a.read(), 1, down.read(), 1, 2, ids.span(), route_weights.span(), shared.span(), stream);
  check(cudaStreamSynchronize(stream));
  const auto selected = ids.download(); const auto probabilities = route_weights.download();
  for (std::size_t i = 0; i < selected.size(); ++i) { require(selected[i] == static_cast<int>(i % 32), "Top-k tie/uniqueness differs"); near(probabilities[i], 1.0 / 32); }
  std::cout << cases << " MoE fusion cases and deterministic tie checks passed\n";
}
void graph(cudaStream_t stream) {
  Weights w(weights(9, 129, 14));
  auto x = inputs(2 * 129, 5); Device<std::uint16_t> dx(x); Device<float> dy(18);
  cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  crit::fp8::project_cuda(w.view(), dx.read(), 2, dy.span(), stream);
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    x = inputs(x.size(), 42 + iteration); dx.upload(x);
    check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
    const auto actual = dy.download(); const auto expected = crit::fp8::project(w.host, x, 2);
    for (std::size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i]);
  }
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  std::cout << "Repeated graph replay with changed activations passed\n";
}
void width_invariance(cudaStream_t stream) {
  constexpr std::size_t k = 129, experts = 19, middle = 17, outputs = 13, top = 5;
  Weights router(weights(experts, k, 81)), gate(weights(middle, k, 31)), up(weights(middle, k, 111));
  const auto rows = inputs(8 * k, 91), down = inputs(outputs * middle, 43);
  Device<std::uint16_t> dd(down);
  struct Result {
    std::vector<float> scores, route_weights, shared;
    std::vector<std::uint16_t> activation;
    std::vector<std::int32_t> ids;
  };
  const auto run = [&](const std::vector<std::uint16_t>& x, std::size_t tokens) {
    Device<std::uint16_t> dx(x), activation(tokens * middle);
    Device<float> scores(tokens * experts), rw(tokens * top), shared(tokens * outputs);
    Device<std::int32_t> ids(tokens * top);
    crit::moe::entry_cuda(router.view(), gate.view(), up.view(), dx.read(), tokens, scores.span(), activation.span(), stream);
    crit::moe::route_down_cuda(scores.read(), experts, top, activation.read(), middle, dd.read(), outputs,
                             tokens, ids.span(), rw.span(), shared.span(), stream);
    check(cudaStreamSynchronize(stream));
    return Result{scores.download(), rw.download(), shared.download(), activation.download(), ids.download()};
  };
  std::vector<Result> baseline;
  for (std::size_t row = 0; row < 8; ++row)
    baseline.push_back(run({rows.begin() + static_cast<std::ptrdiff_t>(row * k),
                            rows.begin() + static_cast<std::ptrdiff_t>((row + 1) * k)}, 1));
  for (std::size_t width : {1U, 2U, 4U, 8U}) for (std::size_t rotation : {0U, 3U, 7U}) {
    std::vector<std::uint16_t> x;
    for (std::size_t row = 0; row < width; ++row) {
      const auto source = (rotation + 7 * row) % 8;
      x.insert(x.end(), rows.begin() + static_cast<std::ptrdiff_t>(source * k),
               rows.begin() + static_cast<std::ptrdiff_t>((source + 1) * k));
    }
    const auto actual = run(x, width);
    for (std::size_t row = 0; row < width; ++row) {
      const auto& expected = baseline[(rotation + 7 * row) % 8];
      for (std::size_t j = 0; j < experts; ++j) near(actual.scores[row * experts + j], expected.scores[j]);
      for (std::size_t j = 0; j < middle; ++j)
        near(crit::fp8::bf16(actual.activation[row * middle + j]), crit::fp8::bf16(expected.activation[j]), .009, 2e-5);
      for (std::size_t j = 0; j < top; ++j) {
        require(actual.ids[row * top + j] == expected.ids[j], "Batch width or row position changed expert IDs");
        near(actual.route_weights[row * top + j], expected.route_weights[j]);
      }
      for (std::size_t j = 0; j < outputs; ++j) near(actual.shared[row * outputs + j], expected.shared[j]);
    }
  }
  std::cout << "MoE row permutation and widths 1/2/4/8 invariance passed\n";
}
void multiple_graphs(cudaStream_t first_stream) {
  Weights w(weights(17, 257, 71));
  struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
  } second;
  struct Slot {
    std::size_t width;
    Device<std::uint16_t> x;
    Device<float> y;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaStream_t stream;
    Slot(std::size_t tokens, Weights& weights, cudaStream_t queue)
      : width(tokens), x(tokens * weights.host.width), y(tokens * weights.host.rows), stream(queue) {
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      crit::fp8::project_cuda(weights.view(), x.read(), width, y.span(), stream);
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, 0));
    }
    ~Slot() {
      cudaStreamSynchronize(stream);
      cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
    }
    void launch(const std::vector<std::uint16_t>& values) {
      x.upload(values); check(cudaGraphLaunch(executable, stream));
    }
    void verify(Weights& weights, const std::vector<std::uint16_t>& values) {
      check(cudaStreamSynchronize(stream));
      const auto actual = y.download();
      const auto expected = crit::fp8::project(weights.host, values, width);
      for (std::size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i]);
    }
  };
  Slot small(1, w, first_stream);
  auto a = inputs(257, 1); small.launch(a); small.verify(w, a);
  Slot large(8, w, second.value);
  auto b = inputs(8 * 257, 2); large.launch(b); large.verify(w, b);
  a = inputs(257, 3); small.launch(a); small.verify(w, a); // Capture large must not invalidate small.
  for (unsigned round = 0; round < 3; ++round) {
    a = inputs(257, 10 + round); b = inputs(8 * 257, 20 + round);
    small.launch(a); large.launch(b);
    small.verify(w, a); large.verify(w, b); // Independent graph buffers, overlapping streams.
  }
  std::cout << "Small/large/small captures and independent concurrent graph slots passed\n";
}
int main() {
  try {
    cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device, 0));
    require(device.major >= 8, "SM80+ GPU required");
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    projections(stream); fusion(stream); graph(stream); width_invariance(stream); multiple_graphs(stream);
    check(cudaStreamDestroy(stream));
    std::cout << "GPU: " << device.name << "; all operator checks passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
