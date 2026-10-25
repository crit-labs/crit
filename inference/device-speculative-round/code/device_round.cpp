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
    core::CudaBuffer control, drafts, predictions, result, stops, block; // device
    core::CudaBuffer logprob, top_ids, top_logprobs;                    // device, with RoundConfig::logprobs
    core::CudaBuffer block_host, input_host;                            // pinned
    std::vector<float> chosen;                                          // [K+1] log-probabilities of the tokens
    std::vector<core::TokenLogprob> ranked;                             // [K+1][top_width] alternatives
    core::CudaEvent completion, staged;
    core::CudaStageTimer timer;
    struct Pair { cudaGraph_t graphs[2] = {nullptr, nullptr}; cudaGraphExec_t execs[2] = {nullptr, nullptr}; };
    std::vector<Pair> widths; // Forward/Finish per draft width 0..config.draft_tokens
    std::uint32_t frontier = 0, allowance = 0, emitted = 0, width = 0; // width: K of the pending round
    std::uint32_t matched = 0, top_width = 0;
    std::uint64_t rounds = 0, readbacks = 0;
    bool launched = false, pending = false, read = false, started = false;

    std::uint32_t top() const { return config.logprobs ? config.top_logprobs : 0u; }
    RoundView view(std::uint32_t k) const {
        return {static_cast<std::int32_t*>(control.data()), static_cast<std::int32_t*>(drafts.data()),
                static_cast<std::int32_t*>(predictions.data()), static_cast<const std::int32_t*>(result.data()),
                k, config.draft_tokens, static_cast<float*>(logprob.data()), static_cast<std::int32_t*>(top_ids.data()),
                static_cast<float*>(top_logprobs.data()), top()};
    }
    PackedLayout layout(std::uint32_t k) const { return {k, top(), config.logprobs}; }
    static std::size_t result_bytes(std::uint32_t k) { return result_width(k) * sizeof(std::int32_t); }
    std::size_t block_bytes(std::uint32_t k) const { return layout(k).words() * sizeof(std::int32_t); }
    // Forward: per-round allowance from pinned memory, then engine draft + verify.
    // Zero copy: the allowance is read in place from mapped pinned memory and the block written to it.
    void forward(cudaStream_t s, std::uint32_t k) {
        auto* budget = static_cast<std::int32_t*>(control.data()) + control::output_budget;
        if(config.zero_copy)
            check(enqueue_set_budget(static_cast<std::int32_t*>(control.data()),
                                     static_cast<const std::int32_t*>(input_host.device_address()), 1, s), "round allowance");
        else
            check(cudaMemcpyAsync(budget, input_host.data(), sizeof(std::int32_t), cudaMemcpyHostToDevice, s), "round allowance copy");
        const auto v = view(k);
        if(k) steps.draft(s, v);
        steps.verify(s, v);
    }
    // Finish: device acceptance, frontier advance, then engine state selection.
    void finish(cudaStream_t s, std::uint32_t k) {
        const auto v = view(k);
        // stops holds {count, ids...}; the count is read on the device.
        const auto* stop_block = static_cast<const std::int32_t*>(stops.data());
        AcceptRows rows{v.drafts, v.predictions, stop_block + 1, 0, v.control,
                        static_cast<std::int32_t*>(result.data()), 1, k, stop_block};
        if(steps.accept) steps.accept(s, rows);
        else check(enqueue_greedy_accept(rows, s), "greedy accept");
        auto* packed = static_cast<std::int32_t*>(config.zero_copy ? block_host.device_address() : block.data());
        const PackRows pack{v.control, v.result, packed, v.logprob, v.top_ids, v.top_logprobs, 1, k, v.top};
        check(enqueue_pack(pack, s), "readback block");
        steps.commit_state(s, v);
    }
    void capture(int which, std::uint32_t k) {
        auto& pair = widths[k];
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin round capture");
        try {
            which ? finish(stream, k) : forward(stream, k);
        } catch(...) {
            cudaGraph_t broken = nullptr;
            cudaStreamEndCapture(stream, &broken);
            if(broken) cudaGraphDestroy(broken);
            throw;
        }
        check(cudaStreamEndCapture(stream, &pair.graphs[which]), "end round capture");
        check(cudaGraphInstantiate(&pair.execs[which], pair.graphs[which], 0), "instantiate round graph");
    }
    static void destroy(Pair& pair) noexcept {
        for(auto& exec : pair.execs) if(exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        for(auto& graph : pair.graphs) if(graph) { cudaGraphDestroy(graph); graph = nullptr; }
    }
    void destroy() noexcept { for(auto& pair : widths) destroy(pair); }
};

DeviceSpeculativeRound::DeviceSpeculativeRound(std::shared_ptr<core::CudaBackend> backend, RoundConfig config, RoundSteps steps)
    : impl_(std::make_unique<Impl>()) {
    if(!backend) throw std::invalid_argument("CUDA backend required");
    if(!config.vocabulary || config.vocabulary > std::uint32_t(INT32_MAX) || !config.context)
        throw std::invalid_argument("Positive signed-32-bit vocabulary and context required");
    if(config.draft_tokens > 1024) throw std::invalid_argument("Draft width above the reference bound of 1024");
    if(config.top_logprobs > core::top_logprobs_limit || (config.top_logprobs && !config.logprobs))
        throw std::invalid_argument("top_logprobs needs logprobs and at most core::top_logprobs_limit");
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
    p.widths.resize(std::size_t(k) + 1);
    p.control = p.backend->allocate(core::Memory::device, control::fields * sizeof(std::int32_t));
    p.drafts = p.backend->allocate(core::Memory::device, std::max<std::size_t>(k, 1) * sizeof(std::int32_t));
    p.predictions = p.backend->allocate(core::Memory::device, (k + 1) * sizeof(std::int32_t));
    p.result = p.backend->allocate(core::Memory::device, p.result_bytes(k));
    if(!p.config.zero_copy) p.block = p.backend->allocate(core::Memory::device, p.block_bytes(k));
    if(p.config.logprobs) {
        const auto rows = std::size_t(k) + 1, top = std::max<std::size_t>(p.top(), 1);
        p.logprob = p.backend->allocate(core::Memory::device, rows * sizeof(float));
        p.top_ids = p.backend->allocate(core::Memory::device, rows * top * sizeof(std::int32_t));
        p.top_logprobs = p.backend->allocate(core::Memory::device, rows * top * sizeof(float));
        for(auto* b : {&p.logprob, &p.top_ids, &p.top_logprobs}) p.backend->fill_async(*b, 0);
        p.top_width = p.top();
        p.chosen.assign(rows, 0.0f);
        p.ranked.assign(rows * p.top_width, {});
    }
    p.stops = p.backend->allocate(core::Memory::device, (std::size_t(p.config.max_stops) + 1) * sizeof(std::int32_t));
    p.block_host = p.backend->allocate(core::Memory::host, p.block_bytes(k));
    p.input_host = p.backend->allocate(core::Memory::host, std::max<std::size_t>(control::fields, p.config.max_stops + 1) * sizeof(std::int32_t));
    p.completion = p.backend->event();
    p.staged = p.backend->event();
    p.timer = p.backend->stage_timer(2);
    p.backend->fill_async(p.control, 0);
    p.backend->fill_async(p.drafts, 0);
    p.backend->fill_async(p.predictions, 0);
    p.backend->fill_async(p.result, 0);
    if(!p.config.zero_copy) p.backend->fill_async(p.block, 0);
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

void DeviceSpeculativeRound::set_top_logprobs(std::uint32_t top) {
    auto& p = *impl_;
    if(p.pending) throw std::logic_error("Commit or abandon the pending round before changing the logprob width");
    if(top > p.top()) throw std::length_error("Logprob width exceeds RoundConfig::top_logprobs");
    p.top_width = top;
    p.ranked.assign(p.chosen.size() * top, {});
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

void DeviceSpeculativeRound::seed(std::uint32_t frontier, std::int32_t last_token, std::int32_t previous) {
    auto& p = *impl_;
    if(p.pending) throw std::logic_error("Commit the pending round before restarting");
    if(!frontier || frontier >= p.config.context) throw std::length_error("Seed frontier outside 1..context - 1");
    for(auto token : {last_token, previous})
        if(token < 0 || std::uint32_t(token) >= p.config.vocabulary) throw std::invalid_argument("Seed token outside vocabulary");
    check(enqueue_seed(static_cast<std::int32_t*>(p.control.data()), static_cast<std::int32_t*>(p.result.data()),
                       p.config.draft_tokens, 0, frontier, last_token, previous, p.stream), "seed round state");
    p.staged.record(p.stream);
    p.staged.wait();
    p.frontier = frontier;
    p.started = true;
}

void DeviceSpeculativeRound::prepare() { prepare(impl_->config.draft_tokens); }

void DeviceSpeculativeRound::prepare(std::uint32_t draft_tokens) {
    auto& p = *impl_;
    if(draft_tokens > p.config.draft_tokens) throw std::out_of_range("Draft width above the configured widest width");
    auto& pair = p.widths[draft_tokens];
    if(!p.config.graphs || pair.execs[0]) return;
    try { p.capture(0, draft_tokens); p.capture(1, draft_tokens); }
    catch(...) { Impl::destroy(pair); throw; }
}

core::Readiness* DeviceSpeculativeRound::launch(std::uint32_t output_tokens) {
    return launch(output_tokens, impl_->config.draft_tokens);
}

core::Readiness* DeviceSpeculativeRound::launch(std::uint32_t output_tokens, std::uint32_t draft_tokens) {
    auto& p = *impl_;
    if(!p.started) throw std::logic_error("Start the device frontier before launching rounds");
    if(p.pending) throw std::logic_error("Commit the pending round first");
    if(!output_tokens) throw std::invalid_argument("A round needs a positive output allowance");
    if(draft_tokens > p.config.draft_tokens) throw std::out_of_range("Draft width above the configured widest width");
    if(p.frontier + draft_tokens + 1 > p.config.context) throw std::length_error("Round would exceed context capacity");
    if(p.launched && !p.completion.ready()) throw std::logic_error("Previous round readback not settled");
    prepare(draft_tokens);
    const auto& pair = p.widths[draft_tokens];
    p.width = draft_tokens;
    p.allowance = std::min(output_tokens, draft_tokens + 1);
    *static_cast<std::int32_t*>(p.input_host.data()) = std::int32_t(p.allowance);
    p.timer.reset();
    p.timer.begin(core::DeviceStage::forward, 0);
    if(pair.execs[0]) check(cudaGraphLaunch(pair.execs[0], p.stream), "launch Forward graph");
    else p.forward(p.stream, draft_tokens);
    p.timer.end();
    p.timer.begin(core::DeviceStage::accept, 0);
    if(pair.execs[1]) check(cudaGraphLaunch(pair.execs[1], p.stream), "launch Finish graph");
    else p.finish(p.stream, draft_tokens);
    p.timer.end();
    if(p.config.zero_copy) p.completion.record(p.stream); // Finish wrote the block in place
    else p.backend->readback_async(p.block_host, 0, p.block, 0, p.block_bytes(draft_tokens), p.completion);
    ++p.readbacks; ++p.rounds;
    p.launched = p.pending = true;
    p.read = false;
    return &p.completion;
}

std::span<const std::int32_t> DeviceSpeculativeRound::tokens() {
    auto& p = *impl_;
    if(!p.pending) throw std::logic_error("No pending round");
    if(!p.completion.ready()) throw std::logic_error("Round readback not ready; wait on its readiness first");
    const auto* host = static_cast<const std::int32_t*>(p.block_host.data());
    if(!p.read) {
        const auto layout = p.layout(p.width);
        const auto n = host[0];
        if(n < 1 || std::uint32_t(n) > p.allowance) throw std::runtime_error("Device round emitted an invalid token count");
        for(std::int32_t i = 0; i < n; ++i)
            if(host[1 + i] < 0 || std::uint32_t(host[1 + i]) >= p.config.vocabulary)
                throw std::runtime_error("Device round emitted a token outside vocabulary");
        if(host[layout.frontier()] != std::int32_t(p.frontier) + n) throw std::runtime_error("Device frontier differs from the host mirror");
        if(host[layout.matched()] < 0 || host[layout.matched()] > n) throw std::runtime_error("Device round matched count outside 0..n_emitted");
        for(std::uint32_t c = 0; layout.logprobs && c < std::uint32_t(n); ++c) {
            std::memcpy(&p.chosen[c], host + layout.logprob() + c, sizeof(float));
            for(std::uint32_t i = 0; i < p.top_width; ++i) {
                auto& entry = p.ranked[std::size_t(c) * p.top_width + i];
                entry.token = host[layout.top_ids() + c * layout.top + i];
                std::memcpy(&entry.logprob, host + layout.top_logprobs() + c * layout.top + i, sizeof(float));
            }
        }
        p.emitted = std::uint32_t(n);
        p.matched = std::uint32_t(host[layout.matched()]);
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
                               p.width, 0, accepted, p.stream), "device truncation");
        p.steps.commit_state(p.stream, p.view(p.width));
    }
    p.frontier += accepted;
}

std::span<const core::DeviceTiming> DeviceSpeculativeRound::timings() { return impl_->timer.collect(); }
std::uint32_t DeviceSpeculativeRound::matched() {
    tokens();
    return impl_->matched;
}
std::span<const float> DeviceSpeculativeRound::logprobs() const noexcept { return impl_->chosen; }
std::span<const core::TokenLogprob> DeviceSpeculativeRound::top_logprobs() const noexcept { return impl_->ranked; }
RoundView DeviceSpeculativeRound::view() const noexcept { return impl_->view(impl_->config.draft_tokens); }
std::uint32_t DeviceSpeculativeRound::frontier() const noexcept { return impl_->frontier; }
std::uint64_t DeviceSpeculativeRound::rounds() const noexcept { return impl_->rounds; }
std::uint64_t DeviceSpeculativeRound::readbacks() const noexcept { return impl_->readbacks; }
bool DeviceSpeculativeRound::captured() const noexcept { return captured(impl_->config.draft_tokens); }
bool DeviceSpeculativeRound::captured(std::uint32_t draft_tokens) const noexcept {
    if(draft_tokens > impl_->config.draft_tokens) return false;
    const auto& pair = impl_->widths[draft_tokens];
    return pair.execs[0] && pair.execs[1];
}
bool DeviceSpeculativeRound::pending() const noexcept { return impl_->pending; }
}
