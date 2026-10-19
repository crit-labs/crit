// Host-work mailbox on CUDA: captured graphs replayed with both capturable joins, the eager join, release of a
// stuck window, handler failure answering every later doorbell, a window aborted between post and join, and the
// join probe and cost measurement.
#include "mailbox.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace hm = crit::techniques::host_mailbox;

cudaError_t fixture_produce(const hm::SlotView&, std::uint32_t* counter, bool skip, cudaStream_t);
cudaError_t fixture_consume(const hm::SlotView&, unsigned long long* sums, cudaStream_t);

namespace {
void require(bool c, const std::string& m) {
    if(!c) throw std::runtime_error(m);
}
void ok(cudaError_t e, const char* what) {
    if(e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr std::size_t kSlots = 8, kPayload = 64;

struct Region {
    void* host = nullptr;
    void* device = nullptr;
    std::shared_ptr<void> owner;
    Region() {
        ok(cudaHostAlloc(&host, hm::region_bytes(kSlots, kPayload), cudaHostAllocMapped | cudaHostAllocPortable), "alloc");
        owner = std::shared_ptr<void>(host, [](void* p) { cudaFreeHost(p); });
        ok(cudaHostGetDevicePointer(&device, host, 0), "device pointer");
    }
    hm::MappedRegion view() const { return {host, device, hm::region_bytes(kSlots, kPayload), owner}; }
};

struct DeviceState {
    std::uint32_t* counter = nullptr;
    unsigned long long* sums = nullptr;  // [0] sum of results, [1..5] status counts
    DeviceState() {
        ok(cudaMalloc(&counter, 4), "counter");
        ok(cudaMalloc(&sums, 8 * sizeof(unsigned long long)), "sums");
        reset();
    }
    ~DeviceState() {
        cudaFree(counter);
        cudaFree(sums);
    }
    void reset() {
        ok(cudaMemset(counter, 0, 4), "reset counter");
        ok(cudaMemset(sums, 0, 8 * sizeof(unsigned long long)), "reset sums");
    }
    std::vector<unsigned long long> read() const {
        std::vector<unsigned long long> out(8);
        ok(cudaMemcpy(out.data(), sums, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read sums");
        return out;
    }
};

void triple(std::size_t, std::span<std::byte> payload) {
    std::uint32_t in;
    std::memcpy(&in, payload.data(), 4);
    const std::uint32_t out = in * 3u + 1u;
    std::memcpy(payload.data() + 4, &out, 4);
}

// sum over k = first..last of (3k + 1)
unsigned long long expected(unsigned long long first, unsigned long long last) {
    unsigned long long s = 0;
    for(unsigned long long k = first; k <= last; ++k) s += 3 * k + 1;
    return s;
}

// One window: every slot posts (slot `skip_slot` is answered by the device itself), joins and consumes.
void window(hm::Mailbox& box, DeviceState& d, cudaStream_t s, std::size_t skip_slot = kSlots, bool join = true) {
    for(std::size_t i = 0; i < box.slots(); ++i) {
        ok(fixture_produce(box.slot(i), d.counter, i == skip_slot, s), "produce");
        if(join) {
            ok(box.join(i, s), "join");
            ok(fixture_consume(box.slot(i), d.sums, s), "consume");
        }
    }
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    template <class F> Graph(cudaStream_t s, F record) {
        ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
        record();
        ok(cudaStreamEndCapture(s, &graph), "end capture");
        ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    }
    ~Graph() {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
    void launch(cudaStream_t s) const { ok(cudaGraphLaunch(exec, s), "launch"); }
};

bool finishes(cudaStream_t s, std::chrono::milliseconds within) {
    const auto end = std::chrono::steady_clock::now() + within;
    while(std::chrono::steady_clock::now() < end) {
        const cudaError_t q = cudaStreamQuery(s);
        if(q == cudaSuccess) return true;
        ok(q == cudaErrorNotReady ? cudaSuccess : q, "query");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// Device-reset doorbells survive replay: each of 200 replays serves 7 new jobs and skips one.
void replay(hm::Join join) {
    Region r;
    DeviceState d;
    hm::Mailbox box(r.view(), kSlots, kPayload, join, triple);
    cudaStream_t s;
    ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    {
        Graph g(s, [&] { window(box, d, s, 3); });
        const int replays = 200;
        for(int i = 0; i < replays; ++i) g.launch(s);
        require(finishes(s, std::chrono::seconds(20)), "replays did not finish");
        const auto sums = d.read();
        const unsigned long long jobs = replays * (kSlots - 1);
        require(sums[1 + hm::served] == jobs && sums[1 + hm::skipped] == replays, "status counts");
        require(sums[0] == expected(1, jobs), "every replay saw its own job's answer");
        require(box.served_jobs() == jobs, "served count");
    }
    cudaStreamDestroy(s);
    std::cout << "replay " << hm::join_name(join) << " ok\n";
}

void eager() {
    Region r;
    DeviceState d;
    hm::Mailbox box(r.view(), kSlots, kPayload, hm::Join::host_eager, triple);
    cudaStream_t s;
    ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    for(int i = 0; i < 3; ++i) window(box, d, s, 5);
    ok(cudaStreamSynchronize(s), "sync");
    const auto sums = d.read();
    require(sums[0] == expected(1, 3 * (kSlots - 1)) && sums[1 + hm::skipped] == 3, "eager results");
    ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
    const cudaError_t rejected = box.join(0, s);
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(s, &g);
    if(g) cudaGraphDestroy(g);
    cudaGetLastError();
    require(rejected == cudaErrorStreamCaptureUnsupported, "eager join must reject capture");
    cudaStreamDestroy(s);
    std::cout << "eager ok\n";
}

// A handler that never returns holds the window; release() completes it, rearm() starts the next one.
void release(hm::Join join) {
    Region r;
    DeviceState d;
    std::atomic<bool> hold{true};
    std::atomic<int> calls{0};
    hm::Mailbox box(r.view(), kSlots, kPayload, join, [&](std::size_t slot, std::span<std::byte> p) {
        if(calls.fetch_add(1) == 2)
            while(hold.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        triple(slot, p);
    });
    cudaStream_t s;
    ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    {
        Graph g(s, [&] { window(box, d, s); });
        g.launch(s);
        require(!finishes(s, std::chrono::milliseconds(200)), "a held job must hold the window");
        box.release();
        require(finishes(s, std::chrono::seconds(5)), "release must complete the window");
        require(!box.rearm(std::chrono::milliseconds(50)), "rearm must wait for the running job");
        hold = false;
        require(box.rearm(), "rearm after the job returned");
        d.reset();
        g.launch(s);
        require(finishes(s, std::chrono::seconds(5)), "window after rearm");
        const auto sums = d.read();
        require(sums[0] == expected(1, kSlots) && sums[1 + hm::served] == kSlots, "results after rearm");
    }
    cudaStreamDestroy(s);
    std::cout << "release " << hm::join_name(join) << " ok\n";
}

// The 5th job throws: that window and every later one complete at once with status failed.
void failure(hm::Join join) {
    Region r;
    DeviceState d;
    std::atomic<int> calls{0};
    hm::Mailbox box(r.view(), kSlots, kPayload, join, [&](std::size_t slot, std::span<std::byte> p) {
        if(calls.fetch_add(1) == 4) throw std::runtime_error("injected handler failure");
        triple(slot, p);
    });
    cudaStream_t s;
    ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    {
        Graph g(s, [&] { window(box, d, s); });
        for(int i = 0; i < 10; ++i) g.launch(s);
        require(finishes(s, std::chrono::seconds(5)), "a failed service must not block any window");
        const auto sums = d.read();
        require(box.failed() && box.error() == "injected handler failure", "failure recorded");
        require(calls.load() == 5, "the handler is never called after a failure");
        require(sums[1 + hm::served] == 4 && sums[1 + hm::failed] == 10 * kSlots - 4, "failed status answers");
        require(!box.rearm(), "failure is terminal");
    }
    cudaStreamDestroy(s);
    std::cout << "failure " << hm::join_name(join) << " ok\n";
}

// A window aborted between post and join leaves raised doorbells; the service answers them anyway, and after
// rearm() the next window gets its own answers (posts in reverse order, too).
void aborted(hm::Join join) {
    Region r;
    DeviceState d;
    hm::Mailbox box(r.view(), kSlots, kPayload, join, triple);
    cudaStream_t s;
    ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    window(box, d, s, kSlots, false);  // posts only
    ok(cudaStreamSynchronize(s), "sync");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while(box.served_jobs() < kSlots && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(box.served_jobs() == kSlots, "orphaned doorbells are answered");
    require(box.rearm(), "rearm after an aborted window");
    d.reset();
    {
        Graph g(s, [&] {
            for(std::size_t i = kSlots; i-- > 0;) {
                ok(fixture_produce(box.slot(i), d.counter, false, s), "produce");
                ok(box.join(i, s), "join");
                ok(fixture_consume(box.slot(i), d.sums, s), "consume");
            }
        });
        for(int i = 0; i < 20; ++i) g.launch(s);
        require(finishes(s, std::chrono::seconds(10)), "window after abort");
    }
    const auto sums = d.read();
    require(sums[0] == expected(1, 20 * kSlots), "answers after an aborted window");
    cudaStreamDestroy(s);
    std::cout << "aborted " << hm::join_name(join) << " ok\n";
}

void costs() {
    const hm::JoinCosts c = hm::measure_joins(0);
    std::cout << "probe: mapped=" << c.support.mapped << " stream_wait=" << c.support.stream_wait
              << " captured=" << c.support.stream_wait_captured << " " << c.support.detail << "\n";
    std::cout << "post+join us: stream_wait=" << (c.stream_wait_us ? std::to_string(*c.stream_wait_us) : "n/a")
              << " device_poll=" << (c.device_poll_us ? std::to_string(*c.device_poll_us) : "n/a")
              << " choice=" << hm::join_name(c.choice()) << "\n";
    require(c.support.mapped && c.device_poll_us, "mapped memory and the poll join are required by this fixture");
    require(c.choice() != hm::Join::host_eager, "a capturable join is available");
}
}  // namespace

int main() {
    try {
        const hm::JoinSupport support = hm::probe_joins(0);
        std::vector<hm::Join> joins{hm::Join::device_poll};
        if(support.stream_wait_captured) joins.push_back(hm::Join::stream_wait);
        else std::cout << "stream_wait unsupported here: " << support.detail << "\n";
        for(hm::Join j : joins) {
            replay(j);
            release(j);
            failure(j);
            aborted(j);
        }
        eager();
        costs();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
