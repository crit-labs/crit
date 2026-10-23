// Run-time choice among a host path's scalar, AVX2 and AVX-512 files. The ISA files are compiled with their own flags
// and built under one condition for every library that has them (crit_inference_isa_sources in
// inference/CMakeLists.txt, which defines CRIT_HOST_ISA_FILES for those libraries), so supported() answers for all of
// them: compiled into this build and reported by the CPU and the OS (cpuid, xgetbv).
#pragma once

namespace crit::techniques::host_isa {

enum class Isa { scalar, avx2, avx512 };
const char* isa_name(Isa) noexcept;
bool supported(Isa) noexcept;
Isa best_isa() noexcept;

}  // namespace crit::techniques::host_isa
