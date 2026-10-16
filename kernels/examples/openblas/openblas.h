#pragma once
// Native CBLAS ABI facts: OpenMathLib/OpenBLAS v0.3.30 cblas.h and common.h.
// This is an independently written, synchronous integration wrapper, not OpenBLAS source.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace crit_kernels {
class Library {
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
public:
    explicit Library(const std::filesystem::path& path) {
        if (!path.is_absolute() || !std::filesystem::is_regular_file(path))
            throw std::invalid_argument("Library requires an absolute regular-file path");
#ifdef _WIN32
        handle_ = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) throw std::runtime_error("Could not load the exact library and its runtime dependencies");
    }
    ~Library() {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    template<class Function> Function symbol(const std::string& name) const {
#ifdef _WIN32
        const auto address = GetProcAddress(handle_, name.c_str());
#else
        const auto address = dlsym(handle_, name.c_str());
#endif
        if (!address) throw std::runtime_error("Missing symbol: " + name);
        Function result;
        static_assert(sizeof result == sizeof address);
        std::memcpy(&result, &address, sizeof result);
        return result;
    }
};

struct Matrix {
    std::span<const float> storage;
    std::size_t rows, columns, stride;
};
struct Output {
    std::span<float> storage;
    std::size_t rows, columns, stride;
};

class OpenBlas {
    template<class Integer> using Gemm = void (*)(int, int, int, Integer, Integer, Integer,
        float, const float*, Integer, const float*, Integer, float, float*, Integer);
    Library library_; // Outlives every resolved function pointer and synchronous call.
    Gemm<std::int32_t> gemm32_ = nullptr;
    Gemm<std::int64_t> gemm64_ = nullptr;
    std::size_t maximum_;
    std::string configuration_;

    template<class View> void validate(const View& matrix) const {
        if (!matrix.rows || !matrix.columns || matrix.stride < matrix.columns ||
            matrix.rows > maximum_ || matrix.columns > maximum_ || matrix.stride > maximum_)
            throw std::invalid_argument("Matrix dimensions/stride exceed the declared BLAS ABI");
        const auto limit = std::numeric_limits<std::size_t>::max();
        if (matrix.rows - 1 > (limit - matrix.columns) / matrix.stride)
            throw std::invalid_argument("Matrix extent overflow");
        const auto required = (matrix.rows - 1) * matrix.stride + matrix.columns;
        if (!matrix.storage.data() || required > matrix.storage.size() || matrix.storage.size() > limit / sizeof(float))
            throw std::invalid_argument("Matrix storage does not cover its extent");
    }
    template<class A, class B> static bool overlap(const A& a, const B& b) {
        const auto start_a = reinterpret_cast<std::uintptr_t>(a.data());
        const auto start_b = reinterpret_cast<std::uintptr_t>(b.data());
        const auto bytes_a = a.size() * sizeof(float), bytes_b = b.size() * sizeof(float);
        const auto limit = std::numeric_limits<std::uintptr_t>::max();
        if (start_a > limit - bytes_a || start_b > limit - bytes_b)
            throw std::invalid_argument("Matrix address extent overflow");
        return start_a < start_b + bytes_b && start_b < start_a + bytes_a;
    }
    template<class Integer> static void call(Gemm<Integer> fn, bool transpose_a, bool transpose_b,
        float alpha, const Matrix& a, const Matrix& b, float beta, const Output& c, std::size_t k) {
        // CBLAS enum values are 32-bit; only dimensions/leading dimensions vary.
        fn(101, transpose_a ? 112 : 111, transpose_b ? 112 : 111,
           static_cast<Integer>(c.rows), static_cast<Integer>(c.columns), static_cast<Integer>(k),
           alpha, a.storage.data(), static_cast<Integer>(a.stride), b.storage.data(),
           static_cast<Integer>(b.stride), beta, c.storage.data(), static_cast<Integer>(c.stride));
    }
public:
    OpenBlas(const std::filesystem::path& path, int integer_bits, const std::string& gemm_symbol,
             const std::string& config_symbol, const std::string& threads_symbol, int threads = 1)
        : library_(path), maximum_(integer_bits == 32 ? std::numeric_limits<std::int32_t>::max() :
                                  static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        static_assert(sizeof(int) == 4 && sizeof(std::size_t) == 8, "This example requires a 64-bit host with 32-bit C int");
        if ((integer_bits != 32 && integer_bits != 64) || threads < 1)
            throw std::invalid_argument("Choose BLAS integer_bits 32 or 64 and a positive thread count");
        const char* config = library_.symbol<const char*(*)()>(config_symbol)();
        if (!config) throw std::runtime_error("OpenBLAS returned no configuration");
        for (std::size_t i = 0; i < 4096 && config[i]; ++i) configuration_ += config[i];
        if (configuration_.empty() || configuration_.size() == 4096 || configuration_.rfind("OpenBLAS ", 0) != 0)
            throw std::runtime_error("Unrecognized OpenBLAS configuration");
        std::istringstream tokens(configuration_);
        std::string token;
        bool use64 = false;
        while (tokens >> token) use64 = use64 || token == "USE64BITINT";
        if (use64 != (integer_bits == 64))
            throw std::invalid_argument("BLAS integer ABI mismatch; refusing matrix calls");
        if (integer_bits == 64) gemm64_ = library_.symbol<Gemm<std::int64_t>>(gemm_symbol);
        else gemm32_ = library_.symbol<Gemm<std::int32_t>>(gemm_symbol);
        library_.symbol<void(*)(int)>(threads_symbol)(threads);
    }
    const std::string& configuration() const noexcept { return configuration_; }
    void gemm(bool transpose_a, bool transpose_b, float alpha, Matrix a, Matrix b, float beta, Output c) const {
        validate(a); validate(b); validate(c);
        if (!std::isfinite(alpha) || !std::isfinite(beta))
            throw std::invalid_argument("This wrapper requires finite alpha and beta");
        const auto m = transpose_a ? a.columns : a.rows;
        const auto k = transpose_a ? a.rows : a.columns;
        const auto kb = transpose_b ? b.columns : b.rows;
        const auto n = transpose_b ? b.rows : b.columns;
        if (m != c.rows || n != c.columns || k != kb)
            throw std::invalid_argument("GEMM shape mismatch");
        if (overlap(a.storage, c.storage) || overlap(b.storage, c.storage))
            throw std::invalid_argument("GEMM output must not overlap its inputs");
        if (gemm64_) call(gemm64_, transpose_a, transpose_b, alpha, a, b, beta, c, k);
        else call(gemm32_, transpose_a, transpose_b, alpha, a, b, beta, c, k);
    }
};
} // namespace crit_kernels
