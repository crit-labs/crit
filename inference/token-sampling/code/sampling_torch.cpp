// token-sampling's sampling law (sampling.cu) for torch tensors: the launches device_rounds.py makes inside a captured
// round (enqueue_verify, enqueue_sample, enqueue_logprobs) on torch's current stream, and the Request rows it stages.
// Host code only; built at load with sampling.cu by device_rounds.native() through torch's extension builder. An empty
// tensor stands for an absent buffer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "sampling.h"

namespace ts = crit::techniques::token_sampling;

namespace {

void check(cudaError_t error, const char* what) {
    TORCH_CHECK(error == cudaSuccess, "token-sampling: ", what, " failed: ", cudaGetErrorString(error));
}

template<typename T> T* ptr(const torch::Tensor& t) {
    return t.defined() && t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}

void cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(!t.defined() || t.numel() == 0 || (t.is_cuda() && t.is_contiguous()), "token-sampling: ", name,
                " is a contiguous CUDA tensor");
}

void logit_rows(const torch::Tensor& logits, int64_t rows) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2 && logits.stride(1) == 1,
                "token-sampling: logits are FP32 CUDA rows with unit stride");
    TORCH_CHECK(logits.size(0) == rows, "token-sampling: ", rows, " logits rows, not ", logits.size(0));
}

// One Request row, as Core's resolve_sampling would hold a request of the sampler slot's rules: no bias, penalties
// or suppression (repetition 1). position is the draw position of column 0; a round rewrites it on the device.
torch::Tensor pack_request(double temperature, double top_p, double min_p, int64_t top_k, uint64_t seed, int64_t stream,
                           int64_t position) {
    ts::Request q;
    q.temperature = temperature;
    q.top_p = top_p;
    q.min_p = min_p;
    q.top_k = std::int32_t(top_k);
    q.repetition = 1.;
    q.stream = std::uint32_t(stream);
    q.seed = seed;
    q.position = std::uint64_t(position);
    auto out = torch::zeros({std::int64_t(sizeof(ts::Request))}, torch::kUInt8);
    std::memcpy(out.data_ptr(), &q, sizeof q);
    return out;
}

py::dict layout() {
    py::dict d;
    d["request"] = sizeof(ts::Request);
    d["position"] = offsetof(ts::Request, position);
    d["sparse"] = sizeof(ts::SparseLaw);
    d["sparse_ids"] = offsetof(ts::SparseLaw, ids);
    d["sparse_probabilities"] = offsetof(ts::SparseLaw, probabilities);
    d["max_top_k"] = ts::kMaxTopK;
    return d;
}

int64_t scratch_bytes(int64_t vocab, int64_t rows) {
    return int64_t(ts::scratch_floats(std::uint32_t(vocab), std::uint32_t(rows)) * sizeof(float));
}

int64_t logprob_scratch_bytes(int64_t vocab, int64_t rows, int64_t n) {
    return int64_t(ts::logprob_scratch_bytes(int(vocab), int(rows), int(n)));
}

ts::Rows rows_of(const torch::Tensor& logits, int64_t vocab, const torch::Tensor& requests, int64_t count, int64_t columns,
                 int64_t position_base, const torch::Tensor& scratch) {
    logit_rows(logits, count * columns);
    cuda_tensor(requests, "requests");
    cuda_tensor(scratch, "scratch");
    TORCH_CHECK(requests.numel() >= count * std::int64_t(sizeof(ts::Request)), "token-sampling: too few staged requests");
    TORCH_CHECK(scratch.numel() * scratch.element_size() >= scratch_bytes(vocab, count * columns),
                "token-sampling: the scratch is smaller than scratch_bytes(vocab, rows)");
    ts::Rows rows;
    rows.logits = logits.data_ptr<float>();
    rows.stride = std::uint32_t(logits.stride(0));
    rows.vocab = std::uint32_t(vocab);
    rows.requests = std::uint32_t(count);
    rows.columns = std::uint32_t(columns);
    rows.request = ptr<const ts::Request>(requests);
    rows.position_base = std::uint32_t(position_base);
    rows.scratch = ptr<float>(scratch);
    return rows;
}

ts::Logprobs logprobs_of(int64_t n, const torch::Tensor& chosen, const torch::Tensor& top_ids, const torch::Tensor& top_values) {
    cuda_tensor(chosen, "chosen log-probabilities");
    cuda_tensor(top_ids, "top ids");
    cuda_tensor(top_values, "top log-probabilities");
    ts::Logprobs lp;
    lp.n = std::uint32_t(n);
    lp.chosen = ptr<float>(chosen);
    lp.top_ids = ptr<std::int32_t>(top_ids);
    lp.top_logprobs = ptr<float>(top_values);
    return lp;
}

// predictions [count][columns] int32 of a verify round: drafts [count][columns - 1] int32; sparse laws (SparseLaw
// rows, [count][columns - 1]) or dense laws ([count][columns - 1][vocab] FP32) or neither (greedy drafts).
void verify(const torch::Tensor& logits, int64_t vocab, const torch::Tensor& requests, int64_t count, int64_t columns,
            const torch::Tensor& drafts, const torch::Tensor& sparse, const torch::Tensor& dense,
            const torch::Tensor& predictions, const torch::Tensor& scratch, int64_t lp_n, const torch::Tensor& chosen,
            const torch::Tensor& top_ids, const torch::Tensor& top_values) {
    const c10::cuda::CUDAGuard guard(logits.device());
    cuda_tensor(drafts, "drafts");
    cuda_tensor(sparse, "sparse laws");
    cuda_tensor(dense, "dense laws");
    cuda_tensor(predictions, "predictions");
    TORCH_CHECK(predictions.numel() >= count * columns, "token-sampling: predictions hold every column");
    TORCH_CHECK(columns == 1 || drafts.numel() >= count * (columns - 1), "token-sampling: drafts hold every draft");
    ts::Verify v;
    v.target = rows_of(logits, vocab, requests, count, columns, 0, scratch);
    v.drafts = ptr<const std::int32_t>(drafts);
    v.draft_laws = ptr<const ts::SparseLaw>(sparse);
    v.draft_probabilities = ptr<const float>(dense);
    v.predictions = ptr<std::int32_t>(predictions);
    v.logprobs = logprobs_of(lp_n, chosen, top_ids, top_values);
    check(ts::enqueue_verify(v, at::cuda::getCurrentCUDAStream()), "enqueue_verify");
}

// One draw per row (count requests x columns) with purpose `purpose` at position + position_base + column; out int32;
// sparse (SparseLaw rows) and dense ([rows][vocab] FP32) laws optional. With tree ([count][columns] int64 root-to-node
// path masks) and tree_tokens ([count][columns] int32 node tokens), the columns are a token tree's nodes: each draws at
// position + position_base + its depth and stages its path's tokens (Rows::tree).
void sample(const torch::Tensor& logits, int64_t vocab, const torch::Tensor& requests, int64_t count, int64_t columns,
            int64_t position_base, int64_t purpose, const torch::Tensor& out, const torch::Tensor& sparse,
            const torch::Tensor& dense, const torch::Tensor& scratch, int64_t lp_n, const torch::Tensor& chosen,
            const torch::Tensor& top_ids, const torch::Tensor& top_values, const torch::Tensor& tree,
            const torch::Tensor& tree_tokens) {
    const c10::cuda::CUDAGuard guard(logits.device());
    cuda_tensor(out, "tokens");
    cuda_tensor(sparse, "sparse laws");
    cuda_tensor(dense, "dense laws");
    cuda_tensor(tree, "tree masks");
    cuda_tensor(tree_tokens, "tree tokens");
    TORCH_CHECK(out.numel() >= count * columns, "token-sampling: one token per row");
    TORCH_CHECK(!ptr<void>(tree) || (tree.numel() >= count * columns && tree_tokens.numel() >= count * columns),
                "token-sampling: one path mask and token per tree node");
    ts::Rows rows = rows_of(logits, vocab, requests, count, columns, position_base, scratch);
    rows.tree = ptr<const std::uint64_t>(tree);
    rows.tree_tokens = ptr<const std::int32_t>(tree_tokens);
    check(ts::enqueue_sample(rows, static_cast<crit::core::DrawPurpose>(std::uint32_t(purpose)), ptr<std::int32_t>(out),
                             ptr<float>(dense), ptr<ts::SparseLaw>(sparse), logprobs_of(lp_n, chosen, top_ids, top_values),
                             at::cuda::getCurrentCUDAStream()),
          "enqueue_sample");
}

// Raw rows' log-probabilities: chosen [rows] int32 tokens -> chosen_values [rows], and the n best alternatives. given:
// the rows are log-probabilities already (enqueue_ranked_logprobs: their own values, ranked).
void logprobs(const torch::Tensor& logits, int64_t vocab, const torch::Tensor& chosen, int64_t n,
              const torch::Tensor& chosen_values, const torch::Tensor& top_ids, const torch::Tensor& top_values,
              const torch::Tensor& scratch, bool given) {
    const c10::cuda::CUDAGuard guard(logits.device());
    logit_rows(logits, logits.size(0));
    cuda_tensor(chosen, "chosen tokens");
    cuda_tensor(scratch, "scratch");
    const ts::Logprobs lp = logprobs_of(n, chosen_values, top_ids, top_values);
    const auto enqueue = given ? ts::enqueue_ranked_logprobs : ts::enqueue_logprobs;
    check(enqueue(logits.data_ptr<float>(), int(logits.stride(0)), int(logits.size(0)), int(vocab),
                  ptr<const std::int32_t>(chosen), int(n), lp.chosen, lp.top_ids, lp.top_logprobs, ptr<void>(scratch),
                  at::cuda::getCurrentCUDAStream()),
          given ? "enqueue_ranked_logprobs" : "enqueue_logprobs");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_request", &pack_request);
    m.def("layout", &layout);
    m.def("scratch_bytes", &scratch_bytes);
    m.def("logprob_scratch_bytes", &logprob_scratch_bytes);
    m.def("verify", &verify);
    m.def("sample", &sample);
    m.def("logprobs", &logprobs);
}
