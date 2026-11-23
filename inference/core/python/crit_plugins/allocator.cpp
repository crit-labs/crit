// Core's device allocator: the segments torch's caching allocator takes for an engine's steps.
//
// crit_plugins.memory runs every step Core serves inside a torch memory pool (torch.cuda.MemPool) whose allocator is
// this library (torch.cuda.memory.CUDAPluggableAllocator over crit_alloc and crit_free). Torch keeps caching blocks
// inside the pool as it does everywhere; only a new segment comes here. The device makes it where it has the bytes
// free (the lesser of the driver's reading and what the card has left, as NVML counts it), and every segment made,
// refused or freed goes into the device's log, which the device's ledger reads at its step frames
// (crit_alloc_events): so every byte a step takes is counted, and none crosses the headroom unseen.
//
// Nothing here calls Python. crit_alloc runs inside torch's allocation, under torch's allocator lock, often on a
// thread that let the GIL go; a Python hook here would wait for the GIL there, while a thread holding the GIL and
// reading torch's allocator (torch.cuda.memory_stats, a telemetry read) waits for that lock: the two never move again.
// The log's own lock is held only to append or read it.
//
// The memory itself is the driver's (cuMemAlloc in the device's primary context, reached through the driver library
// at run time: nothing links against CUDA), and the card's reading NVML's (the handle crit_plugins.memory found).
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#define CRIT_EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define CRIT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = unsigned long long;

constexpr int DEVICES = 64;
constexpr std::uint64_t UNKNOWN = ~std::uint64_t(0);   // a free reading nobody could take

// One entry of a device's log: a segment made (with the free bytes the device had before it), one refused (the device
// did not have the bytes, or the driver did not make them) or one freed.
enum Kind : std::int32_t { MADE = 1, REFUSED = 2, FREED = 3 };

struct Event {
    std::int32_t kind;
    std::int32_t reserved;
    std::uint64_t bytes;
    std::uint64_t free;
};

struct Driver {
    CUresult (*alloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*free)(CUdeviceptr) = nullptr;
    CUresult (*retain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*push)(CUcontext) = nullptr;
    CUresult (*pop)(CUcontext*) = nullptr;
    CUresult (*info)(std::size_t*, std::size_t*) = nullptr;
    bool ready = false;
};

struct NvmlMemoryV2 {   // nvmlMemory_v2_t
    unsigned int version;
    unsigned long long total, reserved, free, used;
};

struct NvmlMemory {     // nvmlMemory_t
    unsigned long long total, free, used;
};

struct Nvml {
    int (*info_v2)(void*, NvmlMemoryV2*) = nullptr;
    int (*info)(void*, NvmlMemory*) = nullptr;
    bool tried = false;
};

std::mutex lock;        // the driver's and NVML's symbols, the contexts and the cards
std::mutex logged;      // the logs
Driver driver;
Nvml nvml;
CUcontext contexts[DEVICES] = {};
void* cards[DEVICES] = {};
std::vector<Event> logs[DEVICES];

void* symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

void* open_library(const char* name) {
#if defined(_WIN32)
    return LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

bool load_driver() {
    if (driver.ready) return true;
#if defined(_WIN32)
    void* library = open_library("nvcuda.dll");
#else
    void* library = open_library("libcuda.so.1");
#endif
    if (!library) return false;
    driver.alloc = reinterpret_cast<CUresult (*)(CUdeviceptr*, std::size_t)>(symbol(library, "cuMemAlloc_v2"));
    driver.free = reinterpret_cast<CUresult (*)(CUdeviceptr)>(symbol(library, "cuMemFree_v2"));
    driver.retain = reinterpret_cast<CUresult (*)(CUcontext*, CUdevice)>(symbol(library, "cuDevicePrimaryCtxRetain"));
    driver.push = reinterpret_cast<CUresult (*)(CUcontext)>(symbol(library, "cuCtxPushCurrent_v2"));
    driver.pop = reinterpret_cast<CUresult (*)(CUcontext*)>(symbol(library, "cuCtxPopCurrent_v2"));
    driver.info = reinterpret_cast<CUresult (*)(std::size_t*, std::size_t*)>(symbol(library, "cuMemGetInfo_v2"));
    driver.ready = driver.alloc && driver.free && driver.retain && driver.push && driver.pop && driver.info;
    return driver.ready;
}

// NVML's memory readings, found once (the library is initialized by whoever found the card's handle).
void load_nvml() {
    if (nvml.tried) return;
    nvml.tried = true;
#if defined(_WIN32)
    void* library = open_library("nvml.dll");
#else
    void* library = open_library("libnvidia-ml.so.1");
#endif
    if (!library) return;
    nvml.info_v2 = reinterpret_cast<int (*)(void*, NvmlMemoryV2*)>(symbol(library, "nvmlDeviceGetMemoryInfo_v2"));
    nvml.info = reinterpret_cast<int (*)(void*, NvmlMemory*)>(symbol(library, "nvmlDeviceGetMemoryInfo"));
}

CUcontext context(int device) {
    if (device < 0 || device >= DEVICES) return nullptr;
    std::lock_guard<std::mutex> held(lock);
    if (!contexts[device]) {
        CUcontext found = nullptr;
        if (driver.retain(&found, device) == 0) contexts[device] = found;
    }
    return contexts[device];
}

// What the card has left (NVML), UNKNOWN where its handle was not given or the library cannot say.
std::uint64_t card_free(int device) {
    void* card = nullptr;
    {
        std::lock_guard<std::mutex> held(lock);
        card = cards[device];
    }
    if (!card) return UNKNOWN;
    if (nvml.info_v2) {
        NvmlMemoryV2 found{};
        found.version = static_cast<unsigned int>(sizeof(NvmlMemoryV2)) | (2u << 24);
        if (nvml.info_v2(card, &found) == 0) return found.free;
    }
    if (nvml.info) {
        NvmlMemory found{};
        if (nvml.info(card, &found) == 0) return found.free;
    }
    return UNKNOWN;
}

void log(int device, Kind kind, std::uint64_t bytes, std::uint64_t free) {
    std::lock_guard<std::mutex> held(logged);
    logs[device].push_back(Event{kind, 0, bytes, free});
}

}  // namespace

// The card's NVML handle for `device` (crit_plugins.memory finds it, the device by its UUID): a segment is then made
// only where the card has the bytes left as well. nullptr: the driver's reading alone.
CRIT_EXPORT void crit_alloc_card(int device, void* handle) {
    if (device < 0 || device >= DEVICES) return;
    std::lock_guard<std::mutex> held(lock);
    load_nvml();
    cards[device] = handle;
}

// Up to `capacity` of `device`'s logged segments, oldest first, into `out`; they leave the log. Returns how many.
CRIT_EXPORT int crit_alloc_events(int device, Event* out, int capacity) {
    if (device < 0 || device >= DEVICES || capacity <= 0) return 0;
    std::lock_guard<std::mutex> held(logged);
    std::vector<Event>& found = logs[device];
    int n = static_cast<int>(found.size() < static_cast<std::size_t>(capacity) ? found.size() : capacity);
    for (int i = 0; i < n; ++i) out[i] = found[i];
    found.erase(found.begin(), found.begin() + n);
    return n;
}

// torch's pluggable allocator: a segment of `size` bytes on `device` (the stream orders nothing here: a segment is
// whole device memory, which torch hands out in blocks on its streams). Made where the device has the bytes free;
// refused (torch raises its out-of-memory error) where it does not.
CRIT_EXPORT void* crit_alloc(std::size_t size, int device, void* /*stream*/) {
    if (device < 0 || device >= DEVICES) return nullptr;
    {
        std::lock_guard<std::mutex> held(lock);
        if (!load_driver()) return nullptr;
    }
    CUcontext ctx = context(device);
    if (!ctx || driver.push(ctx) != 0) return nullptr;
    std::size_t driver_free = 0, total = 0;
    std::uint64_t free = driver.info(&driver_free, &total) == 0 ? driver_free : UNKNOWN;
    std::uint64_t left = card_free(device);
    if (left < free) free = left;
    CUdeviceptr found = 0;
    bool made = (free == UNKNOWN || free >= size) && driver.alloc(&found, size) == 0;
    CUcontext popped = nullptr;
    driver.pop(&popped);
    log(device, made ? MADE : REFUSED, size, free);
    return made ? reinterpret_cast<void*>(found) : nullptr;
}

CRIT_EXPORT void crit_free(void* pointer, std::size_t size, int device, void* /*stream*/) {
    if (!pointer || !driver.ready || device < 0 || device >= DEVICES) return;
    CUcontext ctx = context(device);
    if (ctx && driver.push(ctx) == 0) {
        driver.free(reinterpret_cast<CUdeviceptr>(pointer));
        CUcontext popped = nullptr;
        driver.pop(&popped);
    }
    log(device, FREED, size, UNKNOWN);
}
