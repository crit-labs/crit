// A C interface over the host rows (host_int.h, host_nvfp4.h) for bindings that load the library at run time (ctypes):
// plain pointers and sizes, every function returning 0 on success or 1 with the error text in `error`. Rows of one call
// are a range [row0, row1): a caller may split a matrix's rows over threads, any split giving the same bits. Compile
// without FP contraction (-ffp-contract=off): the combine's FMAs are written as std::fma.
#include "host_int.h"
#include "host_nvfp4.h"
#include "nvfp4.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace hi = crit::techniques::host_int;
namespace hn = crit::techniques::host_nvfp4;

namespace {

const float* e4m3_table() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> t{};
        for(int b = 0; b < 256; ++b) t[std::size_t(b)] = crit::nvfp4::e4m3(std::uint8_t(b));
        return t;
    }();
    return table.data();
}

int fail(const char* what, char* error, std::size_t bytes) {
    if(error && bytes) {
        const std::size_t n = std::min(std::strlen(what), bytes - 1);
        std::memcpy(error, what, n);
        error[n] = 0;
    }
    return 1;
}

template <class Fn> int guarded(char* error, std::size_t bytes, Fn&& fn) {
    try {
        fn();
        return 0;
    } catch(const std::exception& e) {
        return fail(e.what(), error, bytes);
    } catch(...) {
        return fail("host rows: an unknown exception", error, bytes);
    }
}

}  // namespace

extern "C" {

// The integer path and the FP32 rows' path this CPU runs ("avx512", "avx2", "scalar" / "portable").
const char* crit_hx_int_isa() { return hn::isa_name(hi::best_int_isa()); }
const char* crit_hx_isa() { return hn::isa_name(hn::best_isa()); }

// Integer block rows [row0, row1) of a [rows][cols] matrix: codes [rows][cols / 2] (E2M1 x 2 or INT4 nibbles), E4M3
// block scales [rows][cols / 16], post [rows] (each row times factor * post[r], rounded once); values [tokens][cols]
// int8 with act_scales [tokens][cols / block] -> out [tokens][ldo] at columns row0..row1. isa: -1 the fastest.
int crit_hx_int_rows(const std::uint8_t* codes, const std::uint8_t* scales, const float* post, float factor,
                     std::size_t rows, std::size_t cols, const std::int8_t* values, const float* act_scales, int tokens,
                     int block, int weights, std::size_t row0, std::size_t row1, int isa, float* out, std::size_t ldo,
                     char* error, std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        hi::IntMatrix m;
        m.codes = codes;
        m.scales = scales;
        m.rows = rows;
        m.cols = cols;
        m.weights = weights ? hi::Weights::int4 : hi::Weights::e2m1x2;
        hn::ScaleDecoder decoder;
        decoder.byte_table = e4m3_table();
        const hi::Isa path = isa < 0 ? hi::best_int_isa() : hi::Isa(isa);
        const std::size_t lds = cols / std::size_t(block);
        for(int t0 = 0; t0 < tokens; t0 += hi::kMaxTokens) {
            const int nt = tokens - t0 < hi::kMaxTokens ? tokens - t0 : hi::kMaxTokens;
            const hi::IntActs acts{values + std::size_t(t0) * cols, cols, act_scales + std::size_t(t0) * lds, lds, block};
            float* o = out + std::size_t(t0) * ldo;
            hi::dot_rows_int(path, {}, m, decoder, acts, nt, 0xffu, row0, row1, post ? 1.0f : factor, o, ldo);
            if(!post) continue;
            for(std::size_t r = row0; r < row1; ++r) {
                const float p = factor * post[r];
                for(int t = 0; t < nt; ++t) o[std::size_t(t) * ldo + r] = o[std::size_t(t) * ldo + r] * p;
            }
        }
    });
}

// NVFP4 rows [row0, row1) in the declared FP32 order (scaled_elements, 16 lanes, halving) against FP32 x
// [tokens][cols], times post[r] (rounded once): W4A16 rows with the device's FP32 bits.
int crit_hx_nvfp4_rows(const std::uint8_t* codes, const std::uint8_t* scales, const float* post, std::size_t rows,
                       std::size_t cols, const float* x, int tokens, std::size_t row0, std::size_t row1, int isa,
                       float* out, std::size_t ldo, char* error, std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        const hn::MatrixView m{codes, scales, rows, cols};
        const hn::ScaleDecoder decoder = hn::e4m3_scales(m);
        const hn::Isa path = isa < 0 ? hn::best_isa() : hn::Isa(isa);
        for(int t0 = 0; t0 < tokens; t0 += hn::kMaxTokens) {
            const int nt = tokens - t0 < hn::kMaxTokens ? tokens - t0 : hn::kMaxTokens;
            float* o = out + std::size_t(t0) * ldo;
            hn::dot_rows(path, {}, m, decoder, x + std::size_t(t0) * cols, cols, nt, (1u << nt) - 1u, row0, row1, 1.0f, o,
                         ldo);
            if(!post) continue;
            for(std::size_t r = row0; r < row1; ++r)
                for(int t = 0; t < nt; ++t) o[std::size_t(t) * ldo + r] = o[std::size_t(t) * ldo + r] * post[r];
        }
    });
}

// The declared quantizers of `tokens` rows of n values (x [tokens][n] FP32 -> values [tokens][n] int8, scales
// [tokens][n / block]).
int crit_hx_quantize_e2m1x2(const float* x, int tokens, std::size_t n, float global, std::int8_t* values, float* scales,
                            char* error, std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        for(int t = 0; t < tokens; ++t)
            hi::quantize_e2m1x2(x + std::size_t(t) * n, n, global, values + std::size_t(t) * n, scales + std::size_t(t) * (n / 16));
    });
}
int crit_hx_quantize_int8(const float* x, int tokens, std::size_t n, int block, std::int8_t* values, float* scales,
                          char* error, std::size_t error_bytes) {
    return guarded(error, error_bytes, [&] {
        for(int t = 0; t < tokens; ++t)
            hi::quantize_int8(x + std::size_t(t) * n, n, block, values + std::size_t(t) * n,
                              scales + std::size_t(t) * (n / std::size_t(block)));
    });
}

// out[r][k] = swiglu_declared(gate_up[r][k], gate_up[r][inter + k]).
void crit_hx_swiglu(const float* gate_up, std::size_t rows, std::size_t inter, float* out) {
    for(std::size_t r = 0; r < rows; ++r)
        for(std::size_t k = 0; k < inter; ++k)
            out[r * inter + k] = hi::swiglu_declared(gate_up[r * 2 * inter + k], gate_up[r * 2 * inter + inter + k]);
}
float crit_hx_exp(float x) { return hi::exp_declared(x); }

// The routed sum in route order: out[t][h] = acc, acc = fma(weights[t][j], y[t * top + j][h], acc) for j ascending
// over ids[t][j] >= 0 (ids int64 when wide, else int32), acc from 0.
void crit_hx_combine(const float* y, const void* ids, int wide, const float* weights, std::size_t tokens, std::size_t top,
                     std::size_t hidden, float* out) {
    for(std::size_t t = 0; t < tokens; ++t) {
        float* row = out + t * hidden;
        for(std::size_t h = 0; h < hidden; ++h) row[h] = 0.0f;
        for(std::size_t j = 0; j < top; ++j) {
            const std::int64_t id = wide ? static_cast<const std::int64_t*>(ids)[t * top + j]
                                         : static_cast<const std::int32_t*>(ids)[t * top + j];
            if(id < 0) continue;
            const float w = weights[t * top + j];
            const float* yr = y + (t * top + j) * hidden;
            for(std::size_t h = 0; h < hidden; ++h) row[h] = std::fma(w, yr[h], row[h]);
        }
    }
}

}  // extern "C"
