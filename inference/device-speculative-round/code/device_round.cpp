#include "device_round.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace crit::techniques::device_round {
namespace {
void check(cudaError_t status, const char* what) {
    if(status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
}

struct DeviceSpeculativeRound::Impl {
    std::shared_ptr<core::CudaBackend> backend;
    RoundConfig config;
    RoundSteps steps;
    cudaStream_t stream;
    core::CudaBuffer control, drafts, predictions, result, stops; // device
    core::CudaBuffer result_host, input_host;                     // pinned
    core::CudaEvent completion, staged;
    core::CudaStageTimer timer;
    cudaGraph_t graphs[2] = {nullptr, nullptr};
    cudaGraphExec_t execs[2] = {nullptr, nullptr};
    std::uint32_t frontier = 0, allowance = 0, emitted = 0;
    std::uint64_t rounds = 0, readbacks = 0;
    bool launched = false, pending = false, read = false, started = false;

    RoundView view() const {
        return {static_cast<std::int32_t*>(control.data()), static_cast<std::int32_t*>(drafts.data()),
                static_cast<std::int32_t*>(predictions.data()), static_cast<const std::int32_t*>(result.data()),
                config.draft_tokens};
    }
    std::size_t result_bytes() const { return result_width(config.draft_tokens) * sizeof(std::int32_t); }
    // Forward: per-round allowance from pinned memory, then engine draft + verify.
    void forward(cudaStream_t s) {
        auto* budget = static_cast<std::int32_t*>(control.data()) + control::output_budget;
        check(cudaMemcpyAsync(budget, input_host.data(), sizeof(std::int32_t), cudaMemcpyHostToDevice, s), "round allowance copy");
        const auto v = view();
        if(config.draft_tokens) steps.draft(s, v);
        steps.verify(s, v);
    }
    // Finish: device acceptance, frontier advance, then engine state selection.
    void finish(cudaStream_t s) {
        const auto v = view();
        // stops holds {count, ids...}; the count is read on the device.
        const auto* stop_block = static_cast<const std::int32_t*>(stops.data());
        AcceptRows rows{v.drafts, v.predictions, stop_block + 1, 0, v.control,
                        static_cast<std::int32_t*>(result.data()), 1, config.draft_tokens, stop_block};
        check(enqueue_greedy_accept(rows, s), "greedy accept");
        steps.commit_state(s, v);
    }
    void capture(int which) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin round capture");
        try {
            which ? finish(stream) : forward(stream);
        } catch(...) {
            cudaGraph_t broken = nullptr;
            cudaStreamEndCapture(stream, &broken);
            if(broken) cudaGraphDestroy(broken);
            throw;
        }
        check(cudaStreamEndCapture(stream, &graphs[which]), "end round capture");
        check(cudaGraphInstantiate(&execs[which], graphs[which], 0), "instantiate round graph");
    }
    void destroy() noexcept {
        for(auto& exec : execs) if(exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        for(auto& graph : graphs) if(graph) { cudaGraphDestroy(graph); graph = nullptr; }
    }
};

DeviceSpeculativeRound::DeviceSpeculativeRound(std::shared_ptr<core::CudaBackend> backend, RoundConfig config, RoundSteps steps)
    : impl_(std::make_unique<Impl>()) {
    if(!backend) throw std::invalid_argument("CUDA backend required");
    if(!config.vocabulary || config.vocabulary > std::uint32_t(INT32_MAX) || !config.context)
        throw std::invalid_argument("Positive signed-32-bit vocabulary and context required");
    if(config.draft_tokens > 1024) throw std::invalid_argument("Draft width above the reference bound of 1024");
    if((config.draft_tokens && !steps.draft) || !steps.verify || !steps.commit_state)
        throw std::invalid_argument("Draft, verify and commit_state device steps required");
    for(auto stop : config.stop_tokens)
        if(stop < 0 || std::uint32_t(stop) >= config.vocabulary) throw std::invalid_argument("Stop token outside vocabulary");
    config.max_stops = std::max<std::uint32_t>(config.max_stops, std::uint32_t(config.stop_tokens.size()));
    auto& p = *impl_;
    p.backend = std::move(backend);
    p.config = std::move(config);
    p.steps = std::move(steps);
    p.stream = p.backend->stream(core::Queue::compute);
    const auto k = p.config.draft_tokens;
    p.control = p.backend->allocate(core::Memory::device, control::fields * sizeof(std::int32_t));
    p.drafts = p.backend->allocate(core::Memory::device, std::max<std::size_t>(k, 1) * sizeof(std::int32_t));
    p.predictions = p.backend->allocate(core::Memory::device, (k + 1) * sizeof(std::int32_t));
    p.result = p.backend->allocate(core::Memory::device, p.result_bytes());
    p.stops = p.backend->allocate(core::Memory::device, (std::size_t(p.config.max_stops) + 1) * sizeof(std::int32_t));
    p.result_host = p.backend->allocate(core::Memory::host, p.result_bytes());
    p.input_host = p.backend->allocate(core::Memory::host, std::max<std::size_t>(control::fields, p.config.max_stops + 1) * sizeof(std::int32_t));
    p.completion = p.backend->event();
    p.staged = p.backend->event();
    p.timer = p.backend->stage_timer(2);
    p.backend->fill_async(p.control, 0);
    p.backend->fill_async(p.drafts, 0);
    p.backend->fill_async(p.predictions, 0);
    p.backend->fill_async(p.result, 0);
    std::vector<std::int32_t> block{std::int32_t(p.config.stop_tokens.size())}; // Loading, not per-round work.
    block.insert(block.end(), p.config.stop_tokens.begin(), p.config.stop_tokens.end());
    p.stops.upload(std::as_bytes(std::span(block)), 0, core::Queue::compute);
}

void DeviceSpeculativeRound::set_stops(std::span<const std::int32_t> stops) {
    auto& p = *impl_;
    if(p.pending) throw std::logic_error("Commit or abandon the pending round before changing stops");
    if(stops.size() > p.config.max_stops) throw std::length_error("Stop list exceeds the configured capacity");
    for(auto stop : stops)
        if(stop < 0 || std::uint32_t(stop) >= p.config.vocabulary) throw std::invalid_argument("Stop token outside vocabulary");
    if(p.launched) p.completion.wait(); // Pinned input may still feed the last Forward copy.
    auto* host = static_cast<std::int32_t*>(p.input_host.data());
    host[0] = std::int32_t(stops.size());
    std::copy(stops.begin(), stops.end(), host + 1);
    check(cudaMemcpyAsync(p.stops.data(), host, (stops.size() + 1) * sizeof(std::int32_t), cudaMemcpyHostToDevice, p.stream), "stop list copy");
    p.staged.record(p.stream);
    p.staged.wait();
}

void DeviceSpeculativeRound::abandon() {
    auto& p = *impl_;
    if(p.launched) p.completion.wait();
    p.pending = false;
    p.read = false;
    p.started = false;
}

DeviceSpeculativeRound::~DeviceSpeculativeRound() {
    if(!impl_) return;
    // Settle captured work before destroying graphs; on failure keep them alive.
    if(const auto status = cudaStreamSynchronize(impl_->stream); status != cudaSuccess) {
        std::cerr << "device round drain failed: " << cudaGetErrorString(status) << '\n';
        return;
    }
    impl_->destroy();
}

void DeviceSpeculativeRound::start(std::uint32_t frontier, std::int32_t last_token) {
    auto& p = *impl_;
    if(p.pending) throw std::logic_error("Commit the pending round before restarting");
    if(frontier >= p.config.context) throw std::length_error("Start frontier exceeds context");
    if(last_token < 0 || std::uint32_t(last_token) >= p.config.vocabulary) throw std::invalid_argument("Start token outside vocabulary");
    if(p.launched) p.completion.wait(); // Pinned input may still feed the last Forward copy.
    auto* host = static_cast<std::int32_t*>(p.input_host.data());
    std::fill(host, host + control::fields, 0);
    host[control::frontier] = std::int32_t(frontier);
    host[control::last_token] = last_token;
    host[control::base_frontier] = std::int32_t(frontier);
    host[control::base_last] = last_token;
    check(cudaMemcpyAsync(p.control.data(), host, control::fields * sizeof(std::int32_t), cudaMemcpyHostToDevice, p.stream), "start control copy");
    p.staged.record(p.stream);
    p.staged.wait();
    p.frontier = frontier;
    p.started = true;
}

void DeviceSpeculativeRound::prepare() {
    auto& p = *impl_;
    if(!p.config.graphs || p.execs[0]) return;
    try { p.capture(0); p.capture(1); }
    catch(...) { p.destroy(); throw; }
}

core::Readiness* DeviceSpeculativeRound::launch(std::uint32_t output_tokens) {
    auto& p = *impl_;
    if(!p.started) throw std::logic_error("Start the device frontier before launching rounds");
    if(p.pending) throw std::logic_error("Commit the pending round first");
    if(!output_tokens) throw std::invalid_argument("A round needs a positive output allowance");
    if(p.frontier + p.config.draft_tokens + 1 > p.config.context) throw std::length_error("Round would exceed context capacity");
    if(p.launched && !p.completion.ready()) throw std::logic_error("Previous round readback not settled");
    prepare();
    p.allowance = std::min(output_tokens, p.config.draft_tokens + 1);
    *static_cast<std::int32_t*>(p.input_host.data()) = std::int32_t(p.allowance);
    p.timer.reset();
    p.timer.begin(core::DeviceStage::forward, 0);
    if(p.execs[0]) check(cudaGraphLaunch(p.execs[0], p.stream), "launch Forward graph");
    else p.forward(p.stream);
    p.timer.end();
    p.timer.begin(core::DeviceStage::accept, 0);
    if(p.execs[1]) check(cudaGraphLaunch(p.execs[1], p.stream), "launch Finish graph");
    else p.finish(p.stream);
    p.timer.end();
    p.backend->readback_async(p.result_host, 0, p.result, 0, p.result_bytes(), p.completion);
    ++p.readbacks; ++p.rounds;
    p.launched = p.pending = true;
    p.read = false;
    return &p.completion;
}

std::span<const std::int32_t> DeviceSpeculativeRound::tokens() {
    auto& p = *impl_;
    if(!p.pending) throw std::logic_error("No pending round");
    if(!p.completion.ready()) throw std::logic_error("Round readback not ready; wait on its readiness first");
    const auto* host = static_cast<const std::int32_t*>(p.result_host.data());
    if(!p.read) {
        const auto n = host[0];
        if(n < 1 || std::uint32_t(n) > p.allowance) throw std::runtime_error("Device round emitted an invalid token count");
        for(std::int32_t i = 0; i < n; ++i)
            if(host[1 + i] < 0 || std::uint32_t(host[1 + i]) >= p.config.vocabulary)
                throw std::runtime_error("Device round emitted a token outside vocabulary");
        p.emitted = std::uint32_t(n);
        p.read = true;
    }
    return {host + 1, p.emitted};
}

void DeviceSpeculativeRound::commit(std::uint32_t accepted) {
    auto& p = *impl_;
    if(!p.pending) throw std::logic_error("No pending round");
    tokens();
    if(accepted > p.emitted) throw std::out_of_range("Commit exceeds the device-chosen tokens");
    p.pending = false; // A failed truncation leaves device state undefined; the caller fails the request.
    if(accepted < p.emitted) {
        check(enqueue_truncate(static_cast<std::int32_t*>(p.control.data()), static_cast<std::int32_t*>(p.result.data()),
                               p.config.draft_tokens, 0, accepted, p.stream), "device truncation");
        p.steps.commit_state(p.stream, p.view());
    }
    p.frontier += accepted;
}

std::span<const core::DeviceTiming> DeviceSpeculativeRound::timings() { return impl_->timer.collect(); }
RoundView DeviceSpeculativeRound::view() const noexcept { return impl_->view(); }
std::uint32_t DeviceSpeculativeRound::frontier() const noexcept { return impl_->frontier; }
std::uint64_t DeviceSpeculativeRound::rounds() const noexcept { return impl_->rounds; }
std::uint64_t DeviceSpeculativeRound::readbacks() const noexcept { return impl_->readbacks; }
bool DeviceSpeculativeRound::captured() const noexcept { return impl_->execs[0] && impl_->execs[1]; }
bool DeviceSpeculativeRound::pending() const noexcept { return impl_->pending; }
}
