#include "openblas.h"
#include <iostream>
#include <iomanip>
#include <vector>

namespace {
void check(const crit_kernels::OpenBlas& blas, std::size_t m, std::size_t n, std::size_t k,
           std::size_t padding, bool ta, bool tb, float alpha, float beta) {
    const auto ar = ta ? k : m, ac = ta ? m : k, br = tb ? n : k, bc = tb ? k : n;
    const auto lda = ac + padding, ldb = bc + padding, ldc = n + padding;
    constexpr float sentinel = 99999.f;
    std::vector<float> a(ar * lda, sentinel), b(br * ldb, sentinel), c(m * ldc + 8, sentinel);
    for (std::size_t i = 0; i < ar; ++i)
        for (std::size_t j = 0; j < ac; ++j) a[i * lda + j] = static_cast<float>(static_cast<int>((i * 11 + j * 3) % 29) - 14) / 17.f;
    for (std::size_t i = 0; i < br; ++i)
        for (std::size_t j = 0; j < bc; ++j) b[i * ldb + j] = static_cast<float>(static_cast<int>((i * 5 + j * 7) % 31) - 15) / 13.f;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) c[4 + i * ldc + j] = beta == 0 ?
            std::numeric_limits<float>::quiet_NaN() : static_cast<float>((i + j) % 5) / 7.f;
    const auto original_a = a, original_b = b, original_c = c;
    blas.gemm(ta, tb, alpha, {a, ar, ac, lda}, {b, br, bc, ldb}, beta,
              {std::span<float>(c).subspan(4, m * ldc), m, n, ldc});
    if (a != original_a || b != original_b) throw std::runtime_error("Input mutation");
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double expected = beta == 0 ? 0 : beta * static_cast<double>(original_c[4 + i * ldc + j]);
            double magnitude = std::abs(expected);
            for (std::size_t p = 0; p < k; ++p) {
                const double term = alpha * static_cast<double>(a[ta ? p * lda + i : i * lda + p]) * b[tb ? j * ldb + p : p * ldb + j];
                expected += term; magnitude += std::abs(term);
            }
            if (!std::isfinite(c[4 + i * ldc + j]) || std::abs(c[4 + i * ldc + j] - expected) > 3e-6 + 2e-5 * magnitude)
                throw std::runtime_error("SGEMM reference mismatch");
        }
        for (std::size_t j = n; j < ldc; ++j)
            if (c[4 + i * ldc + j] != sentinel) throw std::runtime_error("Output padding overwritten");
    }
    for (std::size_t i = 0; i < 4; ++i)
        if (c[i] != sentinel || c[c.size() - 1 - i] != sentinel) throw std::runtime_error("Output guard overwritten");
}

void quoted(const std::string& value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32 || c >= 127) std::cout << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else std::cout << c;
    }
    std::cout << '"';
}
template<class Character> std::string ascii(const Character* value) {
    std::string result;
    for (; *value; ++value) {
        if (static_cast<std::uint32_t>(*value) > 127 || result.size() >= 128)
            throw std::invalid_argument("ABI and symbol arguments must be bounded ASCII");
        result += static_cast<char>(*value);
    }
    return result;
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 6) throw std::invalid_argument("Usage: openblas_check ABSOLUTE_LIBRARY 32|64 SGEMM_SYMBOL CONFIG_SYMBOL THREADS_SYMBOL");
        const std::string bits = ascii(argv[2]);
        if (bits != "32" && bits != "64") throw std::invalid_argument("Choose integer bits 32 or 64");
        crit_kernels::OpenBlas blas(std::filesystem::path(argv[1]), bits == "64" ? 64 : 32,
                                  ascii(argv[3]), ascii(argv[4]), ascii(argv[5]));
        int cases = 0;
        for (std::size_t m : {1, 7, 31}) for (std::size_t n : {1, 11, 33}) for (std::size_t k : {1, 16, 37})
            for (std::size_t padding : {0, 3}) for (bool ta : {false, true}) for (bool tb : {false, true})
                for (const auto& scalars : {std::pair{1.25f, -.5f}, std::pair{1.f, 0.f}, std::pair{0.f, 1.f}}) {
                    check(blas, m, n, k, padding, ta, tb, scalars.first, scalars.second); ++cases;
                }
        std::cout << "{\"schema_version\":1,\"record_kind\":\"openblas_check\",\"cases\":" << cases
                  << ",\"integer_bits\":" << bits << ",\"configuration\":";
        quoted(blas.configuration());
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
