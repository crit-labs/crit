// recurrent-state-recovery's device commit (recurrent_cuda.h): the checkpoint select and the deferred count record,
// each from a count in device memory, against the host's choice for every count. The scans that write the checkpoints
// and the stash are crit-labs' recurrent-scans and are tested there.
#include "recurrent_cuda.h"
#include <crit/cuda.h>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace crit::core;
namespace dc = crit::techniques::recurrent::cuda;

namespace {
void check(cudaError_t s, const char* what) { if(s != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s)); }
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
struct Fixture {
    Resources resources{1 << 26, 1 << 28};
    std::shared_ptr<CudaBackend> backend = std::make_shared<CudaBackend>();
    Fixture() { backend->initialize(resources); }
    CudaBuffer put(const void* data, std::size_t bytes) {
        auto b = backend->allocate(Memory::device, std::max<std::size_t>(bytes, 4));
        if(bytes) b.upload(std::span(static_cast<const std::byte*>(data), bytes), 0, Queue::compute);
        return b;
    }
    CudaBuffer put(const std::vector<float>& values) { return put(values.data(), values.size() * sizeof(float)); }
    template <class T> std::vector<T> get(const CudaBuffer& b, std::size_t count) {
        std::vector<T> out(count);
        b.download(std::as_writable_bytes(std::span(out)), 0, Queue::compute);
        return out;
    }
};

// For every count -1 .. columns + 2 the state is the start (count <= 0) or checkpoint count - 1 (clamped to columns),
// bit for bit, over a few geometries.
void select() {
    Fixture f;
    std::mt19937 random(77);
    std::uniform_real_distribution<float> unit(-1, 1);
    std::size_t cases = 0;
    for(std::uint32_t heads : {1u, 3u})
    for(std::uint32_t keys : {1u, 16u, 128u})
    for(std::uint32_t values : {1u, 32u, 128u})
    for(std::uint32_t columns : {1u, 4u}) {
        const dc::DeviceGeometry g{heads, keys, values};
        const auto elements = dc::state_elements(g);
        std::vector<float> start(elements), checkpoints(columns * elements);
        for(auto& x : start) x = unit(random);
        for(auto& x : checkpoints) x = unit(random);
        auto d_start = f.put(start), d_ckpt = f.put(checkpoints), d_state = f.put(start);
        const auto stream = f.backend->stream();
        for(std::int32_t n = -1; n <= std::int32_t(columns) + 2; ++n) {
            auto d_count = f.put(&n, sizeof n);
            check(dc::enqueue_select_accepted(g, static_cast<const float*>(d_start.data()), static_cast<const float*>(d_ckpt.data()),
                                              static_cast<const std::int32_t*>(d_count.data()), columns,
                                              static_cast<float*>(d_state.data()), stream), "select");
            const auto got = f.get<float>(d_state, elements);
            const std::int32_t chosen = n <= 0 ? 0 : std::min(n, std::int32_t(columns));
            const float* want = chosen ? checkpoints.data() + std::size_t(chosen - 1) * elements : start.data();
            require(std::memcmp(got.data(), want, elements * sizeof(float)) == 0, "select chose another state for " + std::to_string(n));
            ++cases;
        }
    }
    std::cout << "select: " << cases << " counts\n";
}

// The single record and the many-count record write the clamped count into `count` ints of every group, `stride`
// apart, and nothing else (a deferred scan's per-block counts).
void record() {
    Fixture f;
    const auto stream = f.backend->stream();
    std::vector<std::int32_t> sentinel(10, -7);
    auto pending = f.put(sentinel.data(), sentinel.size() * sizeof(std::int32_t));
    auto* pending_ptr = static_cast<std::int32_t*>(pending.data());
    for(const auto& [n, want] : {std::pair{4, 3}, std::pair{-2, 0}, std::pair{2, 2}}) {
        pending.upload(std::as_bytes(std::span(sentinel)), 0, Queue::compute);
        auto accepted = f.put(&n, sizeof n);
        const auto* accepted_ptr = static_cast<const std::int32_t*>(accepted.data());
        check(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 3, 2, 5, stream), "record many");
        auto got = f.get<std::int32_t>(pending, 10);
        for(int i = 0; i < 10; ++i)
            require(got[std::size_t(i)] == (i % 5 < 3 ? want : -7), "record many wrote " + std::to_string(got[std::size_t(i)]) + " at " + std::to_string(i));
        pending.upload(std::as_bytes(std::span(sentinel)), 0, Queue::compute);
        check(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, stream), "record");
        got = f.get<std::int32_t>(pending, 10);
        require(got[0] == want, "record wrote " + std::to_string(got[0]));
        for(int i = 1; i < 10; ++i) require(got[std::size_t(i)] == -7, "record wrote past its count");
        require(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 3, 2, 2, stream) == cudaErrorInvalidValue, "overlapping groups recorded");
        require(dc::enqueue_record_accepted(accepted_ptr, 3, pending_ptr, 0, 1, 0, stream) == cudaErrorInvalidValue, "zero counts recorded");
    }
    std::cout << "record: every group's counts and nothing else\n";
}

void rejects() {
    Fixture f;
    std::vector<float> zeros(16);
    auto buffer = f.put(zeros);
    auto* p = static_cast<float*>(buffer.data());
    auto* n = reinterpret_cast<std::int32_t*>(p);
    std::int32_t* none = nullptr;
    const auto stream = f.backend->stream();
    require(dc::enqueue_select_accepted({1, 2, 0}, p, p, n, 1, p, stream) == cudaErrorInvalidValue, "zero values accepted");
    require(dc::enqueue_select_accepted({1, 2, 2000}, p, p, n, 1, p, stream) == cudaErrorInvalidValue, "too many value columns accepted");
    require(dc::enqueue_select_accepted({1, 2, 2}, p, p, none, 1, p, stream) == cudaErrorInvalidValue, "missing count accepted");
    require(dc::enqueue_select_accepted({1, 2, 2}, p, nullptr, n, 1, p, stream) == cudaErrorInvalidValue, "missing checkpoints accepted");
    require(dc::enqueue_record_accepted(nullptr, 1, n, stream) == cudaErrorInvalidValue, "missing accepted count recorded");
    require(dc::enqueue_record_accepted(n, 1, nullptr, stream) == cudaErrorInvalidValue, "missing pending count recorded");
}
}

int main(int argc, char** argv) {
    try {
        if(argc != 2) throw std::invalid_argument("test group required");
        const std::string group = argv[1];
        if(group == "select") select();
        else if(group == "record") record();
        else if(group == "rejects") rejects();
        else throw std::invalid_argument("unknown test group");
        std::cout << "PASS " << group << '\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
