// Included by one translation unit per fixture executable. Fail allocations only on the calling
// thread, so a narrow test scope can exercise cleanup without affecting independent workers.
#pragma once
#include <cstdlib>
#include <new>

namespace allocation_failure {
inline thread_local bool enabled = false;
struct Reset {
    bool previous = enabled;
    ~Reset() { enabled = previous; }
};
}
#if defined(_MSC_VER)
#define ALLOCATION_TEST_NOINLINE __declspec(noinline)
#else
#define ALLOCATION_TEST_NOINLINE [[gnu::noinline]]
#endif
ALLOCATION_TEST_NOINLINE void* operator new(std::size_t size) {
    if(allocation_failure::enabled) throw std::bad_alloc();
    if(void* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}
ALLOCATION_TEST_NOINLINE void operator delete(void* memory) noexcept { std::free(memory); }
ALLOCATION_TEST_NOINLINE void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
ALLOCATION_TEST_NOINLINE void* operator new[](std::size_t size) { return ::operator new(size); }
ALLOCATION_TEST_NOINLINE void operator delete[](void* memory) noexcept { ::operator delete(memory); }
ALLOCATION_TEST_NOINLINE void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }
#undef ALLOCATION_TEST_NOINLINE
