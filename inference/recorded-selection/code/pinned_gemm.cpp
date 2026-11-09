// One recorded cuBLASLt algorithm per GEMM shape and format, run for every row count (pinned_gemm.py builds and
// drives it).
//
// Layout: a [M, K] row-major (the prompt's rows), b [N, K] row-major (a linear layer's weight), out [M, N] row-major
// = alpha (scale_a a) (scale_b b)^T with FP32 accumulation (CUBLAS_COMPUTE_32F, no TF32). A format names a's and b's
// element types (fp16, bf16, fp32, e4m3, e5m2, e2m1; e2m1 packed two per byte, the first element in the low
// nibble), the output's (bf16, fp16, fp32) and the scale mode: none; tensor, one FP32 element per operand (absent
// means 1); row, FP32 vectors of a's M rows and b's N rows (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F); vec32, one
// UE8M0 scale per 32 elements along K (VEC32_UE8M0); vec16, one UE4M3 scale per 16 (VEC16_UE4M3). Block scales are
// in cuBLASLt's blocked layout (128-row by 4-column tiles, rows padded to 128 and columns to 4; pinned_gemm.py lays
// them out), and a vec16 plan reads alpha, the per-tensor FP32 global scale, and beta from device memory. cuBLASLt
// is column-major: b is its A (column-major [K, N], op T), a its B (column-major [K, M], op N), and out its
// column-major [N, M] D. The fixed dimensions N and K are the shape; M, the row count, is free.
//
// plan() takes the index-th split-K-free algorithm (split-K count 1, no reduction scheme) of the heuristic list that
// cublasLtMatmulAlgoGetHeuristic returns at a representative row count, the recorded choice. run() multiplies any
// row count with that algorithm, so no row's sum is split differently at another count and a row's bits do not
// depend on how many rows the call has. cublasLtMatmulAlgoCheck runs once per row count (cached); a count the
// algorithm cannot run, or one needing more workspace than the plan has, is an error. A plan is used by one host
// thread at a time.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define CHECK_LT(x)                                                                        \
  do {                                                                                     \
    cublasStatus_t s_ = (x);                                                               \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x " failed with cuBLAS status ", (int)s_);   \
  } while (0)

namespace {

constexpr int kHeuristics = 64;

cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) CHECK_LT(cublasLtCreate(&h));
  return h;
}

cudaDataType_t element_type(const std::string& name) {
  if (name == "fp16") return CUDA_R_16F;
  if (name == "bf16") return CUDA_R_16BF;
  if (name == "fp32") return CUDA_R_32F;
  if (name == "e4m3") return CUDA_R_8F_E4M3;
  if (name == "e5m2") return CUDA_R_8F_E5M2;
  if (name == "e2m1") return CUDA_R_4F_E2M1;
  TORCH_CHECK(false, "pinned GEMM element types are fp16, bf16, fp32, e4m3, e5m2 or e2m1, not ", name);
}

int element_bits(const std::string& name) {
  return name == "fp32" ? 32 : name == "fp16" || name == "bf16" ? 16 : name == "e2m1" ? 4 : 8;
}

enum class Scale { none, tensor, row, vec32, vec16 };

Scale scale_mode(const std::string& name) {
  if (name == "none") return Scale::none;
  if (name == "tensor") return Scale::tensor;
  if (name == "row") return Scale::row;
  if (name == "vec32") return Scale::vec32;
  if (name == "vec16") return Scale::vec16;
  TORCH_CHECK(false, "pinned GEMM scale modes are none, tensor, row, vec32 or vec16, not ", name);
}

struct Format {
  std::string a, b, out;
  Scale scale = Scale::none;
};

Format format(const std::vector<std::string>& f) {
  TORCH_CHECK(f.size() == 4, "a format is [a type, b type, output type, scale mode]");
  Format r{f[0], f[1], f[2], scale_mode(f[3])};
  element_type(r.a);
  element_type(r.b);
  TORCH_CHECK(r.out == "bf16" || r.out == "fp16" || r.out == "fp32", "pinned GEMM outputs are bf16, fp16 or fp32");
  return r;
}

struct Rows {
  cublasLtMatrixLayout_t a = nullptr, out = nullptr;
  bool checked = false;
};

struct Plan {
  int64_t n = 0, k = 0;
  Format fmt;
  size_t workspace = 0;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatmulAlgo_t algo{};
  std::map<int64_t, Rows> rows;
};

std::vector<std::unique_ptr<Plan>> plans;

Rows& rows(Plan& p, int64_t m) {
  auto it = p.rows.find(m);
  if (it != p.rows.end()) return it->second;
  Rows r;
  CHECK_LT(cublasLtMatrixLayoutCreate(&r.a, element_type(p.fmt.a), p.k, m, p.k));
  CHECK_LT(cublasLtMatrixLayoutCreate(&r.out, element_type(p.fmt.out), p.n, m, p.n));
  return p.rows.emplace(m, r).first->second;
}

std::unique_ptr<Plan> describe_shape(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t workspace) {
  TORCH_CHECK(n > 0 && k > 0 && workspace >= 0, "positive N and K and a non-negative workspace required");
  auto p = std::make_unique<Plan>();
  p->n = n; p->k = k; p->fmt = format(fmt); p->workspace = (size_t)workspace;
  CHECK_LT(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  int32_t mode = -1;
  switch (p->fmt.scale) {
    case Scale::none: case Scale::tensor: break;
    case Scale::row: mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F; break;
    case Scale::vec32: mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0; break;
    case Scale::vec16: mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3; break;
  }
  if (mode >= 0) {
    CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    // The heuristic offers vector- and block-scaled algorithms only once scale pointers are set; run() sets the
    // real ones before every multiply.
    static const float* placeholder = reinterpret_cast<const float*>(256);
    CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder,
                                            sizeof(placeholder)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder,
                                            sizeof(placeholder)));
  }
  if (p->fmt.scale == Scale::vec16) {
    cublasLtPointerMode_t device = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_LT(cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &device, sizeof(device)));
  }
  CHECK_LT(cublasLtMatrixLayoutCreate(&p->b, element_type(p->fmt.b), k, n, k));
  return p;
}

template <typename T>
T config(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attribute, T fallback) {
  T value = fallback;
  size_t got = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, attribute, &value, sizeof(value), &got);
  return value;
}

// The split-K-free algorithms of the heuristic list at `rep_rows` rows, in the list's order.
std::vector<cublasLtMatmulAlgo_t> split_free(Plan& p, int64_t rep_rows) {
  TORCH_CHECK(rep_rows > 0, "the representative row count must be positive");
  Rows& rep = rows(p, rep_rows);
  cublasLtMatmulPreference_t pref;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = p.workspace;
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                sizeof(reduction)));
  cublasLtMatmulHeuristicResult_t found[kHeuristics];
  int count = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(handle(), p.desc, p.b, rep.a, rep.out, rep.out, pref,
                                                         kHeuristics, found, &count);
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<cublasLtMatmulAlgo_t> out;
  if (status != CUBLAS_STATUS_SUCCESS) return out;
  for (int i = 0; i < count; ++i) {
    if (found[i].state != CUBLAS_STATUS_SUCCESS) continue;
    int32_t split = config<int32_t>(found[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
    uint32_t scheme = config<uint32_t>(found[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                       CUBLASLT_REDUCTION_SCHEME_NONE);
    if (split <= 1 && scheme == CUBLASLT_REDUCTION_SCHEME_NONE) out.push_back(found[i].algo);
  }
  return out;
}

// The bytes of `rows` rows of block scales in the blocked layout: rows padded to 128, K / block columns to 4.
int64_t block_scale_bytes(int64_t rows, int64_t k, int64_t block) {
  return (rows + 127) / 128 * 128 * ((k / block + 3) / 4 * 4);
}

const void* scale_pointer(const c10::optional<at::Tensor>& s, const Plan& p, int64_t rows, const at::Tensor& a,
                          const char* name) {
  if (!s || !s->defined()) {
    TORCH_CHECK(p.fmt.scale == Scale::none || p.fmt.scale == Scale::tensor, "the format needs ", name);
    return nullptr;
  }
  TORCH_CHECK(p.fmt.scale != Scale::none, "the format takes no scales");
  TORCH_CHECK(s->device() == a.device() && s->is_contiguous(), name, " must be contiguous on the operands' device");
  if (p.fmt.scale == Scale::tensor) {
    TORCH_CHECK(s->scalar_type() == at::kFloat && s->numel() == 1, "a per-tensor ", name, " is one FP32 element");
  } else if (p.fmt.scale == Scale::row) {
    TORCH_CHECK(s->scalar_type() == at::kFloat && s->numel() == rows, "a per-row ", name,
                " has one FP32 element per row (", rows, ")");
  } else {
    int64_t bytes = block_scale_bytes(rows, p.k, p.fmt.scale == Scale::vec32 ? 32 : 16);
    TORCH_CHECK(s->element_size() == 1 && s->numel() == bytes, "a block-scaled ", name, " has ", bytes,
                " one-byte scales in the blocked layout");
  }
  return s->data_ptr();
}

at::ScalarType scalar_type(const std::string& type) {
  if (type == "fp16") return at::kHalf;
  if (type == "bf16") return at::kBFloat16;
  if (type == "fp32") return at::kFloat;
  if (type == "e4m3") return at::kFloat8_e4m3fn;
  return at::kFloat8_e5m2;
}

void check_operand(const at::Tensor& t, const std::string& type, int64_t rows, int64_t k, const char* name) {
  int64_t bits = element_bits(type);
  bool typed = type == "e2m1" ? t.element_size() == 1 : t.scalar_type() == scalar_type(type);
  TORCH_CHECK(typed, name, " must hold ", type, " elements", type == "e2m1" ? " packed two per byte" : "");
  TORCH_CHECK(t.dim() == 2 && t.is_contiguous() && t.size(0) == rows &&
              t.size(1) * (int64_t)t.element_size() * 8 == k * bits,
              name, " must be a contiguous row-major [", rows, ", ", k, "] of ", type);
}

}  // namespace

// How many split-K-free algorithms the heuristic lists for the shape and format at rep_rows rows.
int64_t candidates(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows, int64_t workspace) {
  auto p = describe_shape(fmt, n, k, workspace);
  return (int64_t)split_free(*p, rep_rows).size();
}

// A plan running the index-th of those algorithms; an index beyond the list is an error.
int64_t plan(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows, int64_t workspace,
             int64_t index) {
  auto p = describe_shape(fmt, n, k, workspace);
  auto found = split_free(*p, rep_rows);
  TORCH_CHECK(index >= 0 && index < (int64_t)found.size(), "no split-K-free cuBLASLt algorithm ", index, " for ", n,
              "x", k, " at ", rep_rows, " rows (", found.size(), " listed)");
  p->algo = found[index];
  plans.push_back(std::move(p));
  return (int64_t)plans.size() - 1;
}

// The plan's algorithm configuration: algorithm, tile, stages, split-K count, reduction scheme, swizzle, custom
// option. A recorded table stores it, so a replayed index that names another algorithm is caught.
std::vector<int64_t> describe(int64_t id) {
  const cublasLtMatmulAlgo_t& a = plans.at(id)->algo;
  return {config<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID, 0), config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0),
          config<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0)};
}

void run(int64_t id, const at::Tensor& a, const at::Tensor& b, at::Tensor& out,
         const c10::optional<at::Tensor>& scale_a, const c10::optional<at::Tensor>& scale_b,
         const c10::optional<at::Tensor>& alpha, const c10::optional<at::Tensor>& beta, at::Tensor& workspace) {
  Plan& p = *plans.at(id);
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() && out.device() == a.device(), "operands on one CUDA device");
  int64_t m = a.dim() == 2 ? a.size(0) : -1;
  check_operand(a, p.fmt.a, m, p.k, "a");
  check_operand(b, p.fmt.b, p.n, p.k, "b");
  check_operand(out, p.fmt.out, m, p.n, "out");
  TORCH_CHECK(workspace.is_cuda() && (size_t)workspace.nbytes() >= p.workspace, "workspace too small");
  const void* sa = scale_pointer(scale_a, p, m, a, "scale_a");
  const void* sb = scale_pointer(scale_b, p, p.n, a, "scale_b");
  bool device_alpha = p.fmt.scale == Scale::vec16;
  for (auto* s : {&alpha, &beta}) {
    TORCH_CHECK(!device_alpha || (*s && (*s)->defined() && (*s)->scalar_type() == at::kFloat &&
                                  (*s)->numel() == 1 && (*s)->device() == a.device()),
                "a vec16 plan reads alpha and beta as one FP32 element each on the operands' device");
  }
  if (m == 0) return;
  Rows& r = rows(p, m);
  if (!r.checked) {
    cublasLtMatmulHeuristicResult_t result;
    cublasStatus_t status = cublasLtMatmulAlgoCheck(handle(), p.desc, p.b, r.a, r.out, r.out, &p.algo, &result);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "the recorded cuBLASLt algorithm cannot run ", m, " rows of ", p.n,
                "x", p.k, " (cuBLAS status ", (int)status, ")");
    TORCH_CHECK(result.workspaceSize <= p.workspace, "the recorded cuBLASLt algorithm needs ", result.workspaceSize,
                " workspace bytes at ", m, " rows, more than the plan's ", p.workspace);
    r.checked = true;
  }
  // b is cuBLASLt's A and a its B, so b's scale is the A scale.
  if (p.fmt.scale != Scale::none) {
    CHECK_LT(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa)));
  }
  float one = 1.0f, zero = 0.0f;
  const void* pa = device_alpha ? alpha->data_ptr() : (const void*)&one;
  const void* pb = device_alpha ? beta->data_ptr() : (const void*)&zero;
  CHECK_LT(cublasLtMatmul(handle(), p.desc, pa, b.data_ptr(), p.b, a.data_ptr(), r.a, pb, out.data_ptr(), r.out,
                          out.data_ptr(), r.out, &p.algo, workspace.data_ptr(), p.workspace,
                          at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("candidates", &candidates);
  m.def("plan", &plan);
  m.def("describe", &describe);
  m.def("run", &run);
  m.def("version", []() { return (int64_t)cublasLtGetVersion(); });
}
