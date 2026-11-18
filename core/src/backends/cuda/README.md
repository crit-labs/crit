# The CUDA backend

[driver_rounds.cpp](driver_rounds.cpp) is the one source of Core that names a device runtime: the launcher of a
step engine's captured decode rounds (`crit_rounds_cuda_*`, exported by `crit_core_steps`;
[step engines](../../runtime/README.md#step-engines)). It loads the CUDA driver API at run time, so Core builds
without a CUDA toolkit and shares the process's one driver with the framework that captured the graphs.

Everything else Core does on a device it does in Python over torch: a device's memory
(`crit_plugins.memory`), captured rounds and prompt chunks (`crit_plugins.rounds`, `crit_plugins.prompt_graphs`),
the host's handoffs (`crit_plugins.host`) and native code built at load (`crit_plugins.native`); see
[crit plugins](../../../../docs/PLUGINS.md). A second backend adds its launcher beside this one and its device to
those services.

[gpu_runtime.h](../../../include/crit/gpu_runtime.h) is a header for native sources that should build on CUDA and
on ROCm's HIP: with `CRIT_CORE_HIP` defined it maps the CUDA runtime names it lists to their HIP equivalents.
device-expert-cache's kernels include it. Written against ROCm 6.4 and not yet built on an AMD toolchain.
