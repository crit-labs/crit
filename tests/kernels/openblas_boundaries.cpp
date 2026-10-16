#include "openblas.h"
#include <iostream>
#include <vector>

template<class Action> void rejects(Action action, const std::string& expected) {
    try { action(); }
    catch (const std::exception& error) {
        if (std::string(error.what()).find(expected) != std::string::npos) return;
        throw;
    }
    throw std::runtime_error("Expected rejection: " + expected);
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Need both fixture library paths");
        using crit_kernels::OpenBlas;
        rejects([&] { OpenBlas b(argv[1], 64, "fixture_sgemm", "fixture_config", "fixture_threads"); }, "ABI mismatch");
        rejects([&] { OpenBlas b(argv[2], 32, "fixture_sgemm", "fixture_config", "fixture_threads"); }, "ABI mismatch");
        rejects([&] { OpenBlas b(argv[1], 32, "missing", "fixture_config", "fixture_threads"); }, "Missing symbol");
        rejects([&] { OpenBlas b("relative.dll", 32, "x", "y", "z"); }, "absolute regular-file");
        OpenBlas blas(argv[1], 32, "fixture_sgemm", "fixture_config", "fixture_threads");
        std::vector<float> a(6, 1), b(6, 2), c(4, 0);
        const crit_kernels::Matrix av{a, 2, 3, 3}, bv{b, 3, 2, 2};
        const crit_kernels::Output cv{c, 2, 2, 2};
        rejects([&] { blas.gemm(false, false, 1, {a, 2, 3, 2}, bv, 0, cv); }, "dimensions/stride");
        rejects([&] { blas.gemm(false, false, 1, {std::span<const float>(a).first(5), 2, 3, 3}, bv, 0, cv); }, "storage");
        rejects([&] { blas.gemm(true, false, 1, av, bv, 0, cv); }, "shape mismatch");
        rejects([&] { blas.gemm(false, false, 1, av, bv, 0, {a, 2, 2, 2}); }, "overlap");
        rejects([&] { blas.gemm(false, false, std::numeric_limits<float>::infinity(), av, bv, 0, cv); }, "finite");
        rejects([&] { blas.gemm(false, false, 1, {a, 2, 3, std::size_t{1} << 32}, bv, 0, cv); }, "BLAS ABI");
        OpenBlas wide(argv[2], 64, "fixture_sgemm", "fixture_config", "fixture_threads");
        rejects([&] { wide.gemm(false, false, 1, {a, std::size_t{1} << 62, 3, 16}, bv, 0, cv); }, "extent overflow");
        blas.gemm(false, false, 1, av, bv, 0, cv);
        for (float value : c) if (value != 6) throw std::runtime_error("Valid call failed after rejections");
        std::cout << "11 boundary rejections and valid recovery passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
