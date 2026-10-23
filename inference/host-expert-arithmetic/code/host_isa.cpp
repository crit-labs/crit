// CPU and OS feature detection for the host ISA files; see host_isa.h.
#include "host_isa.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace crit::techniques::host_isa {
namespace {

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid(int leaf, int sub, unsigned r[4]) {
    int v[4];
    __cpuidex(v, leaf, sub);
    for(int i = 0; i < 4; ++i) r[i] = unsigned(v[i]);
}
unsigned long long xcr0() { return _xgetbv(0); }
#define CRIT_HOST_ISA_CPUID 1
#elif defined(__x86_64__) || defined(__i386__)
void cpuid(int leaf, int sub, unsigned r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
unsigned long long xcr0() {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#define CRIT_HOST_ISA_CPUID 1
#endif

struct Features {
    bool avx2 = false, avx512 = false;
    Features() {
#ifdef CRIT_HOST_ISA_CPUID
        unsigned r[4];
        cpuid(0, 0, r);
        const unsigned max_leaf = r[0];
        cpuid(1, 0, r);
        const bool osxsave = r[2] & (1u << 27), avx = r[2] & (1u << 28), fma = r[2] & (1u << 12);
        if(!osxsave || !avx || max_leaf < 7) return;
        const unsigned long long x = xcr0();
        const bool ymm = (x & 0x6) == 0x6, zmm = (x & 0xe6) == 0xe6;  // SSE+AVX state; plus opmask and ZMM
        cpuid(7, 0, r);
        avx2 = ymm && fma && (r[1] & (1u << 5));
        avx512 = zmm && fma && (r[1] & (1u << 16));
#endif
    }
};
[[maybe_unused]] const Features& features() {
    static const Features f;
    return f;
}

}  // namespace

const char* isa_name(Isa isa) noexcept {
    switch(isa) {
        case Isa::avx512: return "avx512";
        case Isa::avx2: return "avx2";
        default: return "scalar";
    }
}

bool supported(Isa isa) noexcept {
    switch(isa) {
#if defined(CRIT_HOST_ISA_FILES)
        case Isa::avx512: return features().avx512;
        case Isa::avx2: return features().avx2;
#endif
        case Isa::scalar: return true;
        default: return false;
    }
}

Isa best_isa() noexcept {
    if(supported(Isa::avx512)) return Isa::avx512;
    if(supported(Isa::avx2)) return Isa::avx2;
    return Isa::scalar;
}

}  // namespace crit::techniques::host_isa
