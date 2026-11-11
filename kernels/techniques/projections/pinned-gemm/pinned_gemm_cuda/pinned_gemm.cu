// One cuBLASLt algorithm per GEMM shape and format, run for every row count. Adapted from crit's recorded-selection
// technique (pinned GEMMs, Apache-2.0), as general code.
//
// cuBLAS's heuristics pick a GEMM's algorithm by its row count, and algorithms split a row's sum differently, so a
// prompt row computes other bits when the prompt is longer (or padded to a captured bucket). plan() takes the
// index-th split-K-free algorithm (split-K count 1, no reduction scheme) of the list cublasLtMatmulAlgoGetHeuristic
// returns at a representative row count; run() multiplies any row count with it, so no row's sum is split differently
// at another count. cublasLtMatmulAlgoCheck runs once per row count (cached); a count the algorithm cannot run, or
// one needing more workspace than the plan has, is an error. A plan is used by one host thread at a time.
//
// Layout: a [M, K] row major (the rows), b [N, K] row major (a linear layer's weight), out [M, N] row major
// = alpha (scale_a a) (scale_b b)^T with FP32 accumulation (CUBLAS_COMPUTE_32F, no TF32). cuBLASLt is column major:
// b is its A (column-major [K, N], op T), a its B (column-major [K, M], op N), and out its column-major [N, M] D. The
// fixed dimensions N and K are the shape; M is free.
//
// cuBLASLt itself is the library torch loaded (opened by its soname), so the plans run the same kernels as torch's
// own GEMMs and nothing links against a second copy.
#include "pinned_gemm.h"
#include "crit_kcore/formats.cuh"   // the blocked block-scale layout's size (kernels/core)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct Lt {
  decltype(&cublasLtCreate) create = nullptr;
  decltype(&cublasLtGetVersion) get_version = nullptr;
  decltype(&cublasLtMatmulDescCreate) desc_create = nullptr;
  decltype(&cublasLtMatmulDescSetAttribute) desc_set = nullptr;
  decltype(&cublasLtMatrixLayoutCreate) layout_create = nullptr;
  decltype(&cublasLtMatmulPreferenceCreate) pref_create = nullptr;
  decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set = nullptr;
  decltype(&cublasLtMatmulPreferenceDestroy) pref_destroy = nullptr;
  decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic = nullptr;
  decltype(&cublasLtMatmulAlgoConfigGetAttribute) config_get = nullptr;
  decltype(&cublasLtMatmulAlgoCheck) check = nullptr;
  decltype(&cublasLtMatmul) matmul = nullptr;
};

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

const Lt& lt() {
  static const Lt functions = [] {
    std::string major = std::to_string(CUBLAS_VER_MAJOR);
#if defined(_WIN32)
    std::string name = "cublasLt64_" + major + ".dll";
    void* lib = GetModuleHandleA(name.c_str());
    if (!lib) lib = LoadLibraryA(name.c_str());
    TORCH_CHECK(lib, "pinned-gemm: cannot open ", name);
#else
    std::string name = "libcublasLt.so." + major;
    void* lib = dlopen(name.c_str(), RTLD_NOW | RTLD_NOLOAD);   // the copy torch loaded
    if (!lib) lib = dlopen(name.c_str(), RTLD_NOW);
    TORCH_CHECK(lib, "pinned-gemm: cannot open ", name, ": ", dlerror());
#endif
    Lt f;
#define PINNED_LOAD(field, sym)                                                \
  f.field = reinterpret_cast<decltype(&sym)>(symbol(lib, #sym));               \
  TORCH_CHECK(f.field, "pinned-gemm: ", name, " has no ", #sym);
    PINNED_LOAD(create, cublasLtCreate)
    PINNED_LOAD(get_version, cublasLtGetVersion)
    PINNED_LOAD(desc_create, cublasLtMatmulDescCreate)
    PINNED_LOAD(desc_set, cublasLtMatmulDescSetAttribute)
    PINNED_LOAD(layout_create, cublasLtMatrixLayoutCreate)
    PINNED_LOAD(pref_create, cublasLtMatmulPreferenceCreate)
    PINNED_LOAD(pref_set, cublasLtMatmulPreferenceSetAttribute)
    PINNED_LOAD(pref_destroy, cublasLtMatmulPreferenceDestroy)
    PINNED_LOAD(heuristic, cublasLtMatmulAlgoGetHeuristic)
    PINNED_LOAD(config_get, cublasLtMatmulAlgoConfigGetAttribute)
    PINNED_LOAD(check, cublasLtMatmulAlgoCheck)
    PINNED_LOAD(matmul, cublasLtMatmul)
#undef PINNED_LOAD
    return f;
  }();
  return functions;
}

#define CHECK_LT(x)                                                                        \
  do {                                                                                     \
    cublasStatus_t s_ = (x);                                                               \
    TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x " failed with cuBLAS status ", (int)s_);   \
  } while (0)

constexpr int kHeuristics = 64;

cublasLtHandle_t handle() {
  static cublasLtHandle_t h = [] {
    cublasLtHandle_t made = nullptr;
    CHECK_LT(lt().create(&made));
    return made;
  }();
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
  std::map<std::pair<int64_t, int64_t>, Rows> rows;   // by row count and the output's leading dimension
};

std::mutex plans_lock;
std::vector<std::unique_ptr<Plan>> plans;

Plan& plan_at(int64_t id) {
  std::lock_guard<std::mutex> guard(plans_lock);
  TORCH_CHECK(id >= 0 && id < (int64_t)plans.size(), "no pinned GEMM plan ", id);
  return *plans[id];
}

Rows& rows(Plan& p, int64_t m, int64_t ld) {
  auto it = p.rows.find({m, ld});
  if (it != p.rows.end()) return it->second;
  Rows r;
  CHECK_LT(lt().layout_create(&r.a, element_type(p.fmt.a), p.k, m, p.k));
  CHECK_LT(lt().layout_create(&r.out, element_type(p.fmt.out), p.n, m, ld));
  return p.rows.emplace(std::make_pair(m, ld), r).first->second;
}

std::unique_ptr<Plan> describe_shape(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t workspace) {
  TORCH_CHECK(n > 0 && k > 0 && workspace >= 0, "positive N and K and a non-negative workspace required");
  auto p = std::make_unique<Plan>();
  p->n = n;
  p->k = k;
  p->fmt = format(fmt);
  p->workspace = (size_t)workspace;
  CHECK_LT(lt().desc_create(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  int32_t mode = -1;
  switch (p->fmt.scale) {
    case Scale::none:
    case Scale::tensor:
      break;
    case Scale::row:
      mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
      break;
    case Scale::vec32:
      mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
      break;
    case Scale::vec16:
      mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
      break;
  }
  if (mode >= 0) {
    CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    // The heuristic offers vector- and block-scaled algorithms only once scale pointers are set; run() sets the
    // real ones before every multiply.
    static const float* placeholder = reinterpret_cast<const float*>(256);
    CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)));
    CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)));
  }
  if (p->fmt.scale == Scale::vec16) {
    cublasLtPointerMode_t device = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_LT(lt().desc_set(p->desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &device, sizeof(device)));
  }
  CHECK_LT(lt().layout_create(&p->b, element_type(p->fmt.b), k, n, k));
  return p;
}

template <typename T>
T config(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attribute, T fallback) {
  T value = fallback;
  size_t got = 0;
  lt().config_get(&algo, attribute, &value, sizeof(value), &got);
  return value;
}

// The split-K-free algorithms of the heuristic list at `rep_rows` rows, in the list's order.
std::vector<cublasLtMatmulAlgo_t> split_free(Plan& p, int64_t rep_rows) {
  TORCH_CHECK(rep_rows > 0, "the representative row count must be positive");
  Rows& rep = rows(p, rep_rows, p.n);
  cublasLtMatmulPreference_t pref;
  CHECK_LT(lt().pref_create(&pref));
  uint64_t ws = p.workspace;
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
  CHECK_LT(lt().pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  CHECK_LT(lt().pref_set(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)));
  cublasLtMatmulHeuristicResult_t found[kHeuristics];
  int count = 0;
  cublasStatus_t status =
      lt().heuristic(handle(), p.desc, p.b, rep.a, rep.out, rep.out, pref, kHeuristics, found, &count);
  lt().pref_destroy(pref);
  std::vector<cublasLtMatmulAlgo_t> out;
  if (status != CUBLAS_STATUS_SUCCESS) return out;
  for (int i = 0; i < count; ++i) {
    if (found[i].state != CUBLAS_STATUS_SUCCESS) continue;
    int32_t split = config<int32_t>(found[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
    uint32_t scheme =
        config<uint32_t>(found[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_REDUCTION_SCHEME_NONE);
    if (split <= 1 && scheme == CUBLASLT_REDUCTION_SCHEME_NONE) out.push_back(found[i].algo);
  }
  return out;
}

const void* scale_pointer(const std::optional<torch::Tensor>& s, const Plan& p, int64_t rows, const torch::Tensor& a,
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
    int64_t bytes = crit::kcore::blocked_scale_bytes(rows, p.k, p.fmt.scale == Scale::vec32 ? 32 : 16);
    TORCH_CHECK(s->element_size() == 1 && s->numel() >= bytes, "a block-scaled ", name, " has at least ", bytes,
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

void check_operand(const torch::Tensor& t, const std::string& type, int64_t rows, int64_t k, const char* name) {
  int64_t bits = element_bits(type);
  bool typed = type == "e2m1" ? t.element_size() == 1 : t.scalar_type() == scalar_type(type);
  TORCH_CHECK(typed, name, " must hold ", type, " elements", type == "e2m1" ? " packed two per byte" : "");
  TORCH_CHECK(t.dim() == 2 && t.is_contiguous() && t.size(0) == rows &&
                  t.size(1) * (int64_t)t.element_size() * 8 == k * bits,
              name, " must be a contiguous row-major [", rows, ", ", k, "] of ", type);
}

}  // namespace

int64_t pinned_candidates(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows,
                          int64_t workspace) {
  auto p = describe_shape(fmt, n, k, workspace);
  return (int64_t)split_free(*p, rep_rows).size();
}

int64_t pinned_plan(const std::vector<std::string>& fmt, int64_t n, int64_t k, int64_t rep_rows, int64_t workspace,
                    int64_t index) {
  auto p = describe_shape(fmt, n, k, workspace);
  auto found = split_free(*p, rep_rows);
  TORCH_CHECK(index >= 0 && index < (int64_t)found.size(), "no split-K-free cuBLASLt algorithm ", index, " for ", n,
              "x", k, " at ", rep_rows, " rows (", found.size(), " listed)");
  p->algo = found[index];
  std::lock_guard<std::mutex> guard(plans_lock);
  plans.push_back(std::move(p));
  return (int64_t)plans.size() - 1;
}

std::vector<int64_t> pinned_describe(int64_t id) {
  const cublasLtMatmulAlgo_t& a = plan_at(id).algo;
  return {config<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0),
          config<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0),
          config<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0),
          config<uint16_t>(a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 0),
          config<uint16_t>(a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 0)};
}

void pinned_run(int64_t id, const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& out,
                const std::optional<torch::Tensor>& scale_a, const std::optional<torch::Tensor>& scale_b,
                const std::optional<torch::Tensor>& alpha, const std::optional<torch::Tensor>& beta,
                torch::Tensor& workspace) {
  Plan& p = plan_at(id);
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() && out.device() == a.device(), "operands on one CUDA device");
  int64_t m = a.dim() == 2 ? a.size(0) : -1;
  check_operand(a, p.fmt.a, m, p.k, "a");
  check_operand(b, p.fmt.b, p.n, p.k, "b");
  // out may be a block of columns of a wider row-major matrix (its rows ld elements apart).
  TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == p.n && out.stride(1) == 1 &&
                  (out.stride(0) >= p.n || m == 1) && out.scalar_type() == scalar_type(p.fmt.out),
              "out must be [", m, ", ", p.n, "] of ", p.fmt.out, " with contiguous rows");
  int64_t ld = m == 1 ? p.n : out.stride(0);
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
  c10::cuda::CUDAGuard device_guard(a.device());
  Rows& r = rows(p, m, ld);
  if (!r.checked) {
    cublasLtMatmulHeuristicResult_t result;
    cublasStatus_t status = lt().check(handle(), p.desc, p.b, r.a, r.out, r.out, &p.algo, &result);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "the pinned cuBLASLt algorithm cannot run ", m, " rows of ", p.n,
                "x", p.k, " (cuBLAS status ", (int)status, ")");
    TORCH_CHECK(result.workspaceSize <= p.workspace, "the pinned cuBLASLt algorithm needs ", result.workspaceSize,
                " workspace bytes at ", m, " rows, more than the plan's ", p.workspace);
    r.checked = true;
  }
  // b is cuBLASLt's A and a its B, so b's scale is the A scale.
  if (p.fmt.scale != Scale::none) {
    CHECK_LT(lt().desc_set(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sb, sizeof(sb)));
    CHECK_LT(lt().desc_set(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sa, sizeof(sa)));
  }
  float one = 1.0f, zero = 0.0f;
  const void* pa = device_alpha ? alpha->data_ptr() : (const void*)&one;
  const void* pb = device_alpha ? beta->data_ptr() : (const void*)&zero;
  CHECK_LT(lt().matmul(handle(), p.desc, pa, b.data_ptr(), p.b, a.data_ptr(), r.a, pb, out.data_ptr(), r.out,
                       out.data_ptr(), r.out, &p.algo, workspace.data_ptr(), p.workspace,
                       at::cuda::getCurrentCUDAStream()));
}

int64_t pinned_version() { return (int64_t)lt().get_version(); }
