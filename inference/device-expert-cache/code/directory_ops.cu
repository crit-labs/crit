// device-expert-cache's directory on the device for its device form (expert_cache.py, device_directory.py): the plan
// of one routed layer and the bundle copies it admits, built at load through torch's extension builder
// (native.directory_ops); device_directory.py's torch ops are the fallback where it does not build, and
// device_cache.py's LayerDirectory is the rule both follow bit for bit. Mechanism, not arithmetic: no expert math.
//
// plan: one block of 256 threads per routed layer over a step's routes (ids [tokens * top], -1 not routed, duplicates
// allowed). The distinct experts in the order they are first routed (atomicMin of route indices, then a block scan);
// hits pinned with the plan's stamp and referenced; misses decided in that order by thread 0 (quota, quota by miss
// count, second touch; a miss not admitted goes to the CPU tier, -2); admitted misses take clock victims by warp 0, a
// 32-slot window per step with a ballot, several victims per window, the same victims, reference bits and hand as
// LayerDirectory's one-slot-at-a-time scan; a search scans at most 2 * slots slots. A prompt plan (ring given) reads hits
// in place without touching the clock and gives the misses ring slots in route order. Per-row counters: each row's
// distinct routed experts and those resident before the plan. Capture-safe: device counts only, no allocation.
//
// fill: the plan's copies (key -> slot), bundle words loaded from the store's device address (registered host memory
// read over the link) and stored into the slots, by a fixed grid that exits once the copies are done. fill_program:
// the same from a store in another layout (expert_store.csf_layout: CSF scales), rebuilt into the slot layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int32_t kEmpty = -1, kReserved = -2, kUnavailable = -3;
constexpr int32_t kNotRouted = -1, kCpu = -2;

struct Directory {
  int32_t* key_slot;
  int32_t* slot_key;
  uint8_t* slot_ref;
  int32_t* slot_pin;
  int32_t* state;               // [0] hand, [1] stamp (wraps; compared for equality only), [2] failure bits
  long long* counters;          // reads, hits, fills, cpu, prompt reads, prompt hits, prompt copies
  uint8_t* seen;                // second touch, or null
  long long* route_counts;      // or null
  const int32_t* quota;         // by miss count, or null
  int quota_len, admit_quota, slots, experts;
};

// Exclusive prefix of `flag` over the block, and the block's total.
__device__ __forceinline__ int block_scan(bool flag, int* total, int* s_warp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned mask = __ballot_sync(0xffffffffu, flag);
  const int prefix = __popc(mask & ((1u << lane) - 1u));
  if (lane == 0) s_warp[warp] = __popc(mask);
  __syncthreads();
  int offset = 0, sum = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const int c = s_warp[w];
    offset += w < warp ? c : 0;
    sum += c;
  }
  __syncthreads();
  *total = sum;
  return offset + prefix;
}

template <typename Index>
__global__ void __launch_bounds__(kThreads)
plan_kernel(Directory d, int layer, const Index* ids, int n, int top, const int64_t* rows, long long* row_counters,
            const int32_t* ring, int ring_count, int32_t* out, int32_t* jobs, int32_t* job_count) {
  extern __shared__ int smem[];
  const int E = d.experts;
  int* first = smem;                                   // [E] first route of each expert (n: not routed); then admits
  int* list = first + E;                               // [E] distinct experts in route order
  uint8_t* hit = reinterpret_cast<uint8_t*>(list + E); // [E] resident before the plan
  __shared__ int s_warp[kWarps];
  __shared__ int s_hits, s_stamp, s_bad, s_admitted, s_cpu;
  const int tid = threadIdx.x;
  const bool prompt = ring != nullptr;
  if (tid == 0) {
    s_hits = 0;
    s_bad = 0;
    s_stamp = prompt ? 0 : ++d.state[1];
  }
  for (int e = tid; e < E; e += kThreads) {
    first[e] = n;
    hit[e] = 0;
    out[e] = kNotRouted;
  }
  __syncthreads();
  for (int i = tid; i < n; i += kThreads) {
    const long long v = static_cast<long long>(ids[i]);
    if (v >= 0 && v < E) atomicMin(&first[v], i);
    else if (v != -1) s_bad = 1;
  }
  __syncthreads();
  int count = 0;
  for (int base = 0; base < n; base += kThreads) {
    const int i = base + tid;
    int e = -1;
    if (i < n) {
      const long long v = static_cast<long long>(ids[i]);
      if (v >= 0 && v < E && first[v] == i) e = static_cast<int>(v);
    }
    int total;
    const int at = block_scan(e >= 0, &total, s_warp);
    if (e >= 0) list[count + at] = e;
    count += total;
  }
  const int D = count;
  __syncthreads();
  for (int k = tid; k < D; k += kThreads) {
    const int e = list[k], key = layer * E + e;
    const int s = d.key_slot[key];
    if (!prompt && d.route_counts) d.route_counts[key] += 1;
    if (s >= 0) {
      hit[e] = 1;
      out[e] = s;
      atomicAdd(&s_hits, 1);
      if (!prompt) {
        d.slot_ref[s] = 1;
        d.slot_pin[s] = s_stamp;
      }
    }
  }
  __syncthreads();

  if (prompt) {
    int m = 0;
    for (int base = 0; base < D; base += kThreads) {
      const int k = base + tid;
      const bool miss = k < D && !hit[list[k]];
      int total;
      const int at = block_scan(miss, &total, s_warp);
      if (miss) {
        const int r = m + at, e = list[k];
        if (r < ring_count) {
          out[e] = ring[r];
          jobs[2 * r] = layer * E + e;
          jobs[2 * r + 1] = ring[r];
        } else {
          out[e] = kCpu;
        }
      }
      m += total;
    }
    if (tid == 0) {
      const int copied = m < ring_count ? m : ring_count;
      *job_count = copied;
      const int fail = (s_bad ? 1 : 0) | (m > ring_count ? 2 : 0);
      if (fail) d.state[2] |= fail;
      d.counters[4] += D;
      d.counters[5] += s_hits;
      d.counters[6] += copied;
    }
    return;
  }

  if (rows != nullptr && row_counters != nullptr) {
    for (int i = tid; i < n; i += kThreads) {
      const long long v = static_cast<long long>(ids[i]);
      if (v < 0 || v >= E) continue;
      const long long r = rows[i / top];
      bool lead = true;
      for (int j = i - 1; j >= 0 && rows[j / top] == r; --j)
        if (static_cast<long long>(ids[j]) == v) {
          lead = false;
          break;
        }
      if (lead) {
        atomicAdd(reinterpret_cast<unsigned long long*>(row_counters + 2 * r), 1ull);
        if (hit[v]) atomicAdd(reinterpret_cast<unsigned long long*>(row_counters + 2 * r + 1), 1ull);
      }
    }
  }

  int* admit = first;   // the misses admitted, in route order (first[] is done)
  if (tid == 0) {
    const int misses = D - s_hits;
    const int quota = d.quota_len ? d.quota[misses < d.quota_len ? misses : d.quota_len - 1] : d.admit_quota;
    int a = 0, cpu = 0;
    for (int k = 0; k < D; ++k) {
      const int e = list[k];
      if (hit[e]) continue;
      const int key = layer * E + e;
      bool ok = quota < 0 || a < quota;
      if (d.seen && !d.seen[key]) {
        d.seen[key] = 1;
        ok = false;
      }
      if (ok) admit[a++] = e;
      else {
        out[e] = kCpu;
        ++cpu;
      }
    }
    s_admitted = a;
    s_cpu = cpu;
  }
  __syncthreads();
  if (tid >= 32) return;
  const int lane = tid, A = s_admitted, S = d.slots, stamp = s_stamp;
  const int W = S < 32 ? S : 32;
  int hand = d.state[0], got = 0, since = 0;
  while (got < A && since < 2 * S) {
    const int width = W < 2 * S - since ? W : 2 * S - since;
    const bool active = lane < width;
    const int c = (hand + lane) % S;
    bool blocked = true, referenced = false;
    if (active) {
      const int k = d.slot_key[c];
      blocked = d.slot_pin[c] == stamp || k == kReserved || k == kUnavailable;
      referenced = d.slot_ref[c] != 0;
    }
    const bool free = active && !blocked && !referenced;
    const bool second = active && !blocked && referenced;
    const unsigned fm = __ballot_sync(0xffffffffu, free);
    const int rank = __popc(fm & ((1u << lane) - 1u));
    const int need = A - got, found = __popc(fm), take = found < need ? found : need;
    int limit, advance;
    if (take == need) {   // the last victim: lanes after it are not scanned
      const int p = __ffs(__ballot_sync(0xffffffffu, free && rank == take - 1)) - 1;
      limit = p;
      advance = p + 1;
      since = 0;
    } else if (take > 0) {
      const int last = 31 - __clz(fm);
      limit = advance = width;
      since = width - last - 1;
    } else {
      limit = advance = width;
      since += width;
    }
    if (second && lane < limit) d.slot_ref[c] = 0;
    if (free && rank < take) {
      const int e = admit[got + rank], key = layer * E + e;
      const int old = d.slot_key[c];
      if (old >= 0) d.key_slot[old] = kEmpty;
      d.slot_key[c] = key;
      d.key_slot[key] = c;
      d.slot_ref[c] = 1;
      d.slot_pin[c] = stamp;
      out[e] = c;
      jobs[2 * (got + rank)] = key;
      jobs[2 * (got + rank) + 1] = c;
    }
    got += take;
    hand = (hand + advance) % S;
    __syncwarp();
  }
  for (int r = got + lane; r < A; r += 32) out[admit[r]] = kCpu;
  if (lane == 0) {
    d.state[0] = hand;
    *job_count = got;
    const int fail = (s_bad ? 1 : 0) | (got < A ? 2 : 0);
    if (fail) d.state[2] |= fail;
    d.counters[0] += D;
    d.counters[1] += s_hits;
    d.counters[2] += got;
    d.counters[3] += s_cpu + (A - got);
  }
}

template <typename Vec>
__global__ void __launch_bounds__(kThreads)
fill_kernel(const Vec* __restrict__ src, Vec* __restrict__ dst, const int32_t* jobs, const int32_t* job_count,
            long long words) {
  const int count = *job_count;
  const long long stride = static_cast<long long>(gridDim.x) * kThreads;
  const long long start = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
  for (int j = 0; j < count; ++j) {
    const Vec* s = src + static_cast<long long>(jobs[2 * j]) * words;
    Vec* t = dst + static_cast<long long>(jobs[2 * j + 1]) * words;
    long long i = start;
    for (; i + 3 * stride < words; i += 4 * stride) {
      const Vec a = s[i], b = s[i + stride], c = s[i + 2 * stride], e = s[i + 3 * stride];
      t[i] = a;
      t[i + stride] = b;
      t[i + 2 * stride] = c;
      t[i + 3 * stride] = e;
    }
    for (; i < words; i += stride) t[i] = s[i];
  }
}

// A fill by a program (expert_store.csf_layout): per copy, each op of the store bundle rebuilt into the slot layout:
// COPY copies a component (16-byte words where both ends allow, bytes otherwise), CSF decodes a compressed scale
// matrix one slab of 16 rows per block at a time, staged in shared memory (scale = base + 4-bit code). The exception
// words follow in program_exceptions_kernel, launched after this one.
template <int Threads>
__global__ void __launch_bounds__(Threads)
program_kernel(const uint8_t* src, uint8_t* dst, const int32_t* jobs, const int32_t* job_count, const long long* program,
               int ops, long long src_bytes, long long dst_bytes) {
  extern __shared__ uint4 slab4[];
  const uint8_t* slab = reinterpret_cast<const uint8_t*>(slab4);
  const int count = *job_count;
  const long long stride = static_cast<long long>(gridDim.x) * Threads;
  const long long start = static_cast<long long>(blockIdx.x) * Threads + threadIdx.x;
  for (int j = 0; j < count; ++j) {
    const uint8_t* s = src + static_cast<long long>(jobs[2 * j]) * src_bytes;
    uint8_t* t = dst + static_cast<long long>(jobs[2 * j + 1]) * dst_bytes;
    for (int o = 0; o < ops; ++o) {
      const long long* p = program + 8 * o;
      if (p[0] == 0) {
        const long long n = p[3];
        const long long words = ((p[1] | p[2]) & 15) == 0 ? n / 16 : 0;
        const uint4* a = reinterpret_cast<const uint4*>(s + p[1]);
        uint4* b = reinterpret_cast<uint4*>(t + p[2]);
        for (long long i = start; i < words; i += stride) b[i] = a[i];
        for (long long i = words * 16 + start; i < n; i += stride) t[p[2] + i] = s[p[1] + i];
        continue;
      }
      const int groups = static_cast<int>(p[4]), width = 16 * (1 + groups / 2), slabs = static_cast<int>(p[3] / 16);
      for (int sl = blockIdx.x; sl < slabs; sl += gridDim.x) {
        const uint4* in = reinterpret_cast<const uint4*>(s + p[1] + static_cast<long long>(sl) * width);
        for (int i = threadIdx.x; i < width / 16; i += Threads) slab4[i] = in[i];
        __syncthreads();
        uint8_t* out = t + p[2] + static_cast<long long>(sl) * 16 * groups;
        for (int i = threadIdx.x; i < 16 * groups; i += Threads) {
          const int r = i / groups, c = i % groups;
          const int code = (slab[16 + r * (groups / 2) + c / 2] >> (4 * (c & 1))) & 15;
          out[i] = static_cast<uint8_t>((slab[r] + code) & 255);
        }
        __syncthreads();
      }
    }
  }
}

template <int Threads>
__global__ void __launch_bounds__(Threads)
program_exceptions_kernel(const uint8_t* src, uint8_t* dst, const int32_t* jobs, const int32_t* job_count,
                          const long long* program, int ops, long long src_bytes, long long dst_bytes) {
  const int count = *job_count;
  const long long stride = static_cast<long long>(gridDim.x) * Threads;
  const long long start = static_cast<long long>(blockIdx.x) * Threads + threadIdx.x;
  for (int j = 0; j < count; ++j) {
    const uint8_t* s = src + static_cast<long long>(jobs[2 * j]) * src_bytes;
    uint8_t* t = dst + static_cast<long long>(jobs[2 * j + 1]) * dst_bytes;
    for (int o = 0; o < ops; ++o) {
      const long long* p = program + 8 * o;
      if (p[0] != 1) continue;
      const int32_t* words = reinterpret_cast<const int32_t*>(s + p[5]);
      const long long size = p[3] * p[4];
      for (long long i = start; i < p[6]; i += stride) {
        const uint32_t w = static_cast<uint32_t>(words[i]);
        if (w == 0xffffffffu) continue;
        const uint32_t pos = w & 0xffffffu;
        if (pos < size) t[p[2] + pos] = static_cast<uint8_t>(w >> 24);
      }
    }
  }
}

template <typename T>
T* optional(const torch::Tensor& t) {
  return t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}

void check(const torch::Tensor& t, torch::ScalarType type, const char* name) {
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
  TORCH_CHECK(t.numel() == 0 || (t.is_cuda() && t.is_contiguous()), name, " is a contiguous CUDA tensor");
}

}  // namespace

void plan(torch::Tensor key_slot, torch::Tensor slot_key, torch::Tensor slot_ref, torch::Tensor slot_pin,
          torch::Tensor state, torch::Tensor counters, torch::Tensor seen, torch::Tensor route_counts,
          torch::Tensor quota, int64_t admit_quota, int64_t layer, int64_t experts, torch::Tensor ids, int64_t top,
          torch::Tensor rows, torch::Tensor row_counters, torch::Tensor ring, torch::Tensor out, torch::Tensor jobs,
          torch::Tensor job_count) {
  check(key_slot, torch::kInt32, "key_slot");
  check(slot_key, torch::kInt32, "slot_key");
  check(slot_ref, torch::kUInt8, "slot_ref");
  check(slot_pin, torch::kInt32, "slot_pin");
  check(state, torch::kInt32, "state");
  check(counters, torch::kInt64, "counters");
  check(seen, torch::kUInt8, "seen");
  check(route_counts, torch::kInt64, "route_counts");
  check(quota, torch::kInt32, "quota");
  check(rows, torch::kInt64, "rows");
  check(row_counters, torch::kInt64, "row_counters");
  check(ring, torch::kInt32, "ring");
  check(out, torch::kInt32, "out");
  check(jobs, torch::kInt32, "jobs");
  check(job_count, torch::kInt32, "job_count");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && (ids.scalar_type() == torch::kInt64 ||
              ids.scalar_type() == torch::kInt32), "ids are contiguous int64 or int32 CUDA routes");
  TORCH_CHECK(out.numel() == experts && jobs.numel() >= 2 * experts && top >= 1, "a table and copies per expert");
  const c10::cuda::CUDAGuard guard(ids.device());
  Directory d{key_slot.data_ptr<int32_t>(), slot_key.data_ptr<int32_t>(), slot_ref.data_ptr<uint8_t>(),
              slot_pin.data_ptr<int32_t>(), state.data_ptr<int32_t>(),
              reinterpret_cast<long long*>(counters.data_ptr<int64_t>()), optional<uint8_t>(seen),
              optional<long long>(route_counts), optional<int32_t>(quota), static_cast<int>(quota.numel()),
              static_cast<int>(admit_quota), static_cast<int>(slot_key.numel() - 1), static_cast<int>(experts)};
  const int n = static_cast<int>(ids.numel());
  const size_t shared = static_cast<size_t>(experts) * 9;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* row_ids = optional<const int64_t>(rows);
  auto* row_out = optional<long long>(row_counters);
  const auto* ring_ids = optional<const int32_t>(ring);
  const int ring_count = static_cast<int>(ring.numel());
  if (ids.scalar_type() == torch::kInt64) {
    auto kernel = plan_kernel<int64_t>;
    if (shared > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    kernel<<<1, kThreads, shared, stream>>>(d, int(layer), ids.data_ptr<int64_t>(), n, int(top), row_ids, row_out,
                                            ring_ids, ring_count, out.data_ptr<int32_t>(), jobs.data_ptr<int32_t>(),
                                            job_count.data_ptr<int32_t>());
  } else {
    auto kernel = plan_kernel<int32_t>;
    if (shared > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    kernel<<<1, kThreads, shared, stream>>>(d, int(layer), ids.data_ptr<int32_t>(), n, int(top), row_ids, row_out,
                                            ring_ids, ring_count, out.data_ptr<int32_t>(), jobs.data_ptr<int32_t>(),
                                            job_count.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill(torch::Tensor source, torch::Tensor store, torch::Tensor jobs, torch::Tensor job_count, int64_t blocks,
          int64_t vector) {
  TORCH_CHECK(source.is_cuda() && store.is_cuda() && source.scalar_type() == torch::kUInt8 &&
              store.scalar_type() == torch::kUInt8 && source.is_contiguous() && store.is_contiguous(),
              "source and store are contiguous uint8 CUDA tensors");
  TORCH_CHECK(store.dim() == 2, "the store is [slots][bundle bytes]");
  const long long bundle = store.size(1);
  TORCH_CHECK(bundle % 16 == 0, "bundles are a multiple of 16 bytes");
  TORCH_CHECK(vector == 4 || vector == 8 || vector == 16, "vector is 4, 8 or 16 bytes");
  check(jobs, torch::kInt32, "jobs");
  check(job_count, torch::kInt32, "job_count");
  const c10::cuda::CUDAGuard guard(store.device());
  if (blocks < 1) {
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, store.device().index()));
    blocks = sms / 2 > 0 ? sms / 2 : 1;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = static_cast<int>(blocks);
  const int32_t* j = jobs.data_ptr<int32_t>();
  const int32_t* c = job_count.data_ptr<int32_t>();
  if (vector == 16)
    fill_kernel<uint4><<<grid, kThreads, 0, stream>>>(reinterpret_cast<const uint4*>(source.data_ptr()),
                                                      reinterpret_cast<uint4*>(store.data_ptr()), j, c, bundle / 16);
  else if (vector == 8)
    fill_kernel<uint2><<<grid, kThreads, 0, stream>>>(reinterpret_cast<const uint2*>(source.data_ptr()),
                                                      reinterpret_cast<uint2*>(store.data_ptr()), j, c, bundle / 8);
  else
    fill_kernel<uint32_t><<<grid, kThreads, 0, stream>>>(reinterpret_cast<const uint32_t*>(source.data_ptr()),
                                                         reinterpret_cast<uint32_t*>(store.data_ptr()), j, c,
                                                         bundle / 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_program(torch::Tensor source, torch::Tensor store, torch::Tensor jobs, torch::Tensor job_count,
                  torch::Tensor program, int64_t shared, int64_t blocks) {
  TORCH_CHECK(source.is_cuda() && store.is_cuda() && source.scalar_type() == torch::kUInt8 &&
              store.scalar_type() == torch::kUInt8 && source.is_contiguous() && store.is_contiguous(),
              "source and store are contiguous uint8 CUDA tensors");
  TORCH_CHECK(source.dim() == 2 && store.dim() == 2, "source [keys][store bytes] and store [slots][bundle bytes]");
  TORCH_CHECK(program.is_cuda() && program.scalar_type() == torch::kInt64 && program.is_contiguous() &&
              program.dim() == 2 && program.size(1) == 8, "a program of [ops][8] int64 on the device");
  TORCH_CHECK(source.size(1) % 16 == 0 && shared % 16 == 0 && shared >= 16, "16-byte bundles and slab staging");
  check(jobs, torch::kInt32, "jobs");
  check(job_count, torch::kInt32, "job_count");
  const c10::cuda::CUDAGuard guard(store.device());
  if (blocks < 1) {
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, store.device().index()));
    blocks = sms / 2 > 0 ? sms / 2 : 1;
  }
  if (shared > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(program_kernel<kThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(shared)));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int grid = static_cast<int>(blocks), ops = static_cast<int>(program.size(0));
  program_kernel<kThreads><<<grid, kThreads, shared, stream>>>(source.data_ptr<uint8_t>(), store.data_ptr<uint8_t>(),
                                                     jobs.data_ptr<int32_t>(), job_count.data_ptr<int32_t>(),
                                                     reinterpret_cast<const long long*>(program.data_ptr<int64_t>()),
                                                     ops, source.size(1), store.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  program_exceptions_kernel<kThreads><<<grid, kThreads, 0, stream>>>(
      source.data_ptr<uint8_t>(), store.data_ptr<uint8_t>(), jobs.data_ptr<int32_t>(), job_count.data_ptr<int32_t>(),
      reinterpret_cast<const long long*>(program.data_ptr<int64_t>()), ops, source.size(1), store.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &plan, "plan one routed layer (or a prompt chunk's layer into ring slots) on the device directory");
  m.def("fill", &fill, "copy a plan's bundles from the device-readable store into the slots");
  m.def("fill_program", &fill_program, "rebuild a plan's bundles from a store layout (CSF scales) into the slots");
}
