// Rotated KV rows on CUDA: rotation and inverse against an FP64 Walsh-Hadamard,
// quantized codes and scales, and lower attention error than unrotated
// quantization for rows with outlier channels.
#include "rotated_kv.h"
#include <crit/cuda.h>
#include <cmath>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crit::core;
namespace rk = crit::techniques::rotated_kv;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
struct Fixture {
    Resources resources{1 << 26, 1 << 28};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
    template<class T> CudaBuffer put(const std::vector<T>& v) {
        auto b = backend->allocate(Memory::device, std::max<std::size_t>(v.size(), 1) * sizeof(T));
        if(!v.empty()) b.upload(std::as_bytes(std::span(v)), 0, Queue::compute);
        return b;
    }
    CudaBuffer bytes(std::size_t n) { return backend->allocate(Memory::device, std::max<std::size_t>(n, 1)); }
    template<class T> std::vector<T> get(const CudaBuffer& b, std::size_t n) {
        std::vector<T> out(n);
        b.download(std::as_writable_bytes(std::span(out)), 0, Queue::compute);
        return out;
    }
};
std::vector<double> hadamard(std::vector<double> y) {
    const auto n = y.size();
    for(std::size_t h = 1; h < n; h *= 2)
        for(std::size_t i = 0; i < n; i += 2 * h)
            for(std::size_t j = i; j < i + h; ++j) { const double a = y[j], b = y[j + h]; y[j] = a + b; y[j + h] = a - b; }
    for(auto& v : y) v /= std::sqrt(double(n));
    return y;
}
// Attention over stored (dequantized) keys/values for one query, FP64.
std::vector<double> attend(const std::vector<double>& q, const std::vector<std::vector<double>>& k, const std::vector<std::vector<double>>& v) {
    std::vector<double> s(k.size());
    double top = -INFINITY;
    for(std::size_t i = 0; i < k.size(); ++i) {
        double d = 0;
        for(std::size_t j = 0; j < q.size(); ++j) d += q[j] * k[i][j];
        s[i] = d / std::sqrt(double(q.size()));
        top = std::max(top, s[i]);
    }
    double total = 0;
    for(auto& x : s) total += x = std::exp(x - top);
    std::vector<double> out(v[0].size(), 0.0);
    for(std::size_t i = 0; i < v.size(); ++i)
        for(std::size_t j = 0; j < out.size(); ++j) out[j] += s[i] / total * v[i][j];
    return out;
}
std::vector<double> plain_quantized(const std::vector<double>& x, int qmax) {
    std::vector<double> out(x.size());
    for(std::size_t g = 0; g < x.size(); g += rk::group) {
        double amax = 0;
        for(std::size_t i = g; i < g + rk::group; ++i) amax = std::max(amax, std::fabs(x[i]));
        const double scale = amax > 0 ? amax / qmax : 1.0;
        for(std::size_t i = g; i < g + rk::group; ++i) out[i] = std::nearbyint(x[i] / scale) * scale;
    }
    return out;
}
}

int main() {
    try {
        Fixture f;
        constexpr std::uint32_t rows = 64, dim = 128;
        std::mt19937 random(3);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> signs(dim), x(std::size_t(rows) * dim);
        for(auto& s : signs) s = random() & 1 ? 1.0f : -1.0f;
        for(std::uint32_t r = 0; r < rows; ++r)
            for(std::uint32_t j = 0; j < dim; ++j) x[r * dim + j] = normal(random) * (j % 41 == 7 ? 40.0f : 1.0f);  // outlier channels
        auto signs_d = f.put(signs), x_d = f.put(x), y_d = f.put(std::vector<float>(x.size())), back_d = f.put(std::vector<float>(x.size()));
        auto codes_d = f.bytes(x.size()), scales_d = f.put(std::vector<float>(rows * dim / rk::group)), deq_d = f.put(std::vector<float>(x.size()));
        const rk::Rows p{rows, dim, static_cast<const float*>(signs_d.data())};
        const auto s = f.backend->stream();
        auto* xp = static_cast<float*>(x_d.data());
        require(rk::enqueue_rotate(p, xp, dim, static_cast<float*>(y_d.data()), dim, s) == cudaSuccess, "rotate");
        require(rk::enqueue_unrotate(p, static_cast<float*>(y_d.data()), dim, static_cast<float*>(back_d.data()), dim, s) == cudaSuccess, "unrotate");
        require(rk::enqueue_rotate_quantize(p, xp, dim, 8, static_cast<std::int8_t*>(codes_d.data()), static_cast<float*>(scales_d.data()), s) == cudaSuccess, "quantize");
        require(rk::enqueue_dequantize(p, static_cast<const std::int8_t*>(codes_d.data()), static_cast<const float*>(scales_d.data()),
                                       static_cast<float*>(deq_d.data()), dim, s) == cudaSuccess, "dequantize");
        const auto y = f.get<float>(y_d, x.size()), back = f.get<float>(back_d, x.size()), deq = f.get<float>(deq_d, x.size());
        const auto scales = f.get<float>(scales_d, rows * dim / rk::group);
        std::vector<std::vector<double>> keys(rows), plain(rows), rotated(rows), stored(rows);
        for(std::uint32_t r = 0; r < rows; ++r) {
            std::vector<double> row(dim), signed_row(dim);
            for(std::uint32_t j = 0; j < dim; ++j) { row[j] = x[r * dim + j]; signed_row[j] = row[j] * signs[j]; }
            const auto ref = hadamard(signed_row);
            for(std::uint32_t j = 0; j < dim; ++j) {
                require(std::fabs(y[r * dim + j] - ref[j]) < 1e-4 * (1 + std::fabs(ref[j])), "rotation differs from FP64");
                require(std::fabs(back[r * dim + j] - row[j]) < 1e-4 * (1 + std::fabs(row[j])), "inverse rotation");
                const double step = scales[r * (dim / rk::group) + j / rk::group];
                require(std::fabs(deq[r * dim + j] - ref[j]) <= 0.5 * step + 1e-4 * (1 + std::fabs(ref[j])), "quantization error above half a step");
            }
            keys[r] = row;
            plain[r] = plain_quantized(row, 127);
            rotated[r] = ref;
            stored[r].assign(deq.begin() + std::ptrdiff_t(r * dim), deq.begin() + std::ptrdiff_t((r + 1) * dim));
        }
        // Attention error: exact vs unrotated int8 vs rotated int8 (scores in the rotated basis, output rotated back).
        double err_plain = 0, err_rot = 0;
        for(std::uint32_t qi = 0; qi < 8; ++qi) {
            std::vector<double> q(dim), signed_q(dim);
            for(std::uint32_t j = 0; j < dim; ++j) { q[j] = 0.1 * normal(random); signed_q[j] = q[j] * signs[j]; }
            const auto exact = attend(q, keys, keys);
            const auto a = attend(q, plain, plain);
            const auto rq = hadamard(signed_q);
            auto b = hadamard(attend(rq, stored, stored));
            for(std::uint32_t j = 0; j < dim; ++j) {
                b[j] *= signs[j];
                err_plain += (a[j] - exact[j]) * (a[j] - exact[j]);
                err_rot += (b[j] - exact[j]) * (b[j] - exact[j]);
            }
        }
        require(err_rot < err_plain, "rotation did not reduce attention error with outlier channels");
        // In-place rotation and rejected launches.
        require(rk::enqueue_rotate(p, xp, dim, xp, dim, s) == cudaSuccess, "in-place rotate");
        const auto inplace = f.get<float>(x_d, x.size());
        for(std::size_t i = 0; i < x.size(); ++i) require(inplace[i] == y[i], "in-place rotation differs");
        rk::Rows bad = p;
        bad.dim = 96;
        require(rk::enqueue_rotate(bad, xp, dim, xp, dim, s) == cudaErrorInvalidValue, "non-power-of-two dim accepted");
        require(rk::enqueue_rotate_quantize(p, xp, dim, 6, static_cast<std::int8_t*>(codes_d.data()), static_cast<float*>(scales_d.data()), s) == cudaErrorInvalidValue, "bits 6 accepted");
        require(rk::enqueue_rotate(p, xp, dim - 1, xp, dim, s) == cudaErrorInvalidValue, "short stride accepted");
        std::cout << "rotated kv ok (attention squared error " << err_rot << " rotated vs " << err_plain << " unrotated)\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
