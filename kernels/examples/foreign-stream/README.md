# Foreign stream binding

[stream_binding.h](stream_binding.h) is a C++20 scope that binds a foreign launcher's actual
thread-local stream slot. Write an adapter whose `current() noexcept` returns that slot and
whose `select(stream) noexcept` updates it; a framework stream context counts only if it updates
the slot the native launcher reads. Validate device, ABI and stream compatibility first.

```cpp
{
    crit::kernels::ScopedStreamBinding<ForeignAdapter> bound(backend.stream());
    // Invoke the foreign operator, which reads ForeignAdapter's actual slot.
    // Record completion on the same stream after every submitted sibling copy.
}
// The old slot is restored, including exception unwind and nested scopes.
```

The scope is not copyable or movable; destroying it on another thread terminates. It does not
fence, allocate or retain buffers. Submit Core-owned views only on the backend's streams, make a
restore wait for every prior producer, and keep owners until settlement. APIs with fallible
restoration need their own failure protocol.

The [CUDA fixture](../../../tests/core/foreign_stream_tests.cpp) (`CRIT_CORE_CUDA=ON`) checks
ordering, nested and exception restoration with a synthetic launcher; qualify each real
dependency separately.
