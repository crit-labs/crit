// Independent installed-SDK consumer: includes every header the Core install ships and calls one symbol per
// component, so a header missing from install(FILES ...), an installed header that includes one that is not
// installed, or a strict warning inside an installed header fails this build. CMakeLists.txt checks that the
// includes below name every installed crit/*.h header and that those are exactly the checkout's
// core/include/crit headers (the cuda*.h headers are covered by the CUDA SDK consumer).
#include <crit/constraints.h>
#include <crit/core.h>
#include <crit/device_graph.h>
#include <crit/device_memory.h>
#include <crit/file.h>
#include <crit/host_block.h>
#include <crit/host_memory.h>
#include <crit/json.h>
#include <crit/model_config.h>
#include <crit/native_engine.h>
#include <crit/safetensors.h>
#include <crit/sampling.h>
#include <crit/sha256.h>
#include <crit/state.h>
#include <crit/text.h>
#include <crit/trace.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cc = crit::core;
namespace {
void require(bool ok, const char* what) {
    if(!ok) throw std::runtime_error(what);
}

void write_file(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    require(static_cast<bool>(out), "fixture write failed");
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::random_device entropy;
        const auto base = std::filesystem::temp_directory_path();
        for(int attempt = 0; attempt < 16; ++attempt) {
            auto candidate = base / ("crit-installed-headers-" + std::to_string(entropy()));
            if(std::filesystem::create_directory(candidate)) {
                path_ = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("no fresh temporary directory");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    const std::filesystem::path& path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

// crit/core.h
void check_resources() {
    cc::Resources resources(1024, 0);
    {
        const auto held = resources.reserve(cc::Memory::host, 256);
        require(held.bytes() == 256 && resources.usage(cc::Memory::host).live == 256, "core.h: reservation");
    }
    require(resources.usage(cc::Memory::host).live == 0, "core.h: release at the last owner");
}

// crit/text.h
void check_text() {
    cc::StopMatcher matcher({"</s>"});
    const auto held = matcher.push("hello </");
    require(!held.stopped && held.released == "hello ", "text.h: holdback");
    const auto stopped = matcher.push("s> tail");
    require(stopped.stopped && stopped.released.empty() && stopped.stop_index == 0, "text.h: stop");
}

// crit/constraints.h
void check_constraints() {
    auto vocabulary = std::make_shared<const cc::ByteVocabulary>(std::vector<std::string>{"{", "}", "x", ""}, 3);
    cc::JsonObjectMatcher matcher(vocabulary);
    require(matcher.allowed() == std::vector<cc::Token>{0}, "constraints.h: JSON object start");
    matcher.append(0);
    matcher.append(1);
    const std::vector<float> logits{5.f, 1.f, 9.f, 0.f};
    cc::Sampler sampler(4);
    const auto allowed = matcher.allowed();
    require(sampler.sample(logits, cc::AllowedTokens{allowed}, cc::SamplingParams{}, cc::TokenHistory({})).token == 3,
            "constraints.h: constrained sampling");
}

// crit/sampling.h
void check_sampling() {
    const std::array<float, 3> logits{0.0f, 1.0f, 2.0f};
    std::array<float, 3> logprobs{};
    cc::log_softmax(logits, logprobs);
    double total = 0;
    for(const float logprob : logprobs) total += std::exp(static_cast<double>(logprob));
    require(std::abs(total - 1.0) < 1e-6 && logprobs[2] > logprobs[1], "sampling.h: log_softmax");
    const double draw = cc::uniform(7, 0);
    require(draw >= 0.0 && draw < 1.0, "sampling.h: uniform");
}

// crit/state.h
struct NoCopies final : cc::StateStorage {
    void copy_rows(cc::PageId, cc::PageId, std::uint32_t) override {}
    void copy_slot(std::uint32_t, cc::SlotId, cc::SlotId) override {}
};

void check_state() {
    cc::Resources resources(std::size_t(1) << 20, 0);
    NoCopies storage;
    cc::StateCacheConfig config;
    config.pages = 4;
    config.page_tokens = 2;
    config.max_tokens = 8;
    cc::StateCache cache(config, storage, resources);
    const auto sequence = cache.open();
    const auto round = cache.begin(sequence, 1);
    const std::array<cc::Token, 1> input{5};
    cache.commit(sequence, round.id, input);
    require(cache.committed(sequence) == 1 && cache.block_table(sequence).size() == 1, "state.h: commit");
}

// crit/json.h
void check_json() {
    const auto document = cc::json::Document::parse(R"({"name":"crit","ids":[1,2]})", "inline");
    const auto* ids = document.root().find("ids");
    const auto* name = document.root().find("name");
    require(ids && name && ids->array().size() == 2 && ids->array()[1].u64() == 2 && name->string() == "crit",
            "json.h: parse");
}

// crit/file.h and crit/sha256.h
void check_file(const std::filesystem::path& directory) {
    write_file(directory / "abc.bin", "abc");
    const auto file = cc::ReadOnlyFile::open(directory / "abc.bin");
    std::array<std::byte, 3> bytes{};
    file.read_exact(0, bytes);
    require(file.identity().size == 3 && bytes[0] == std::byte{0x61} && bytes[2] == std::byte{0x63}, "file.h: read");
    const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    require(cc::sha256_hex(bytes) == abc && cc::sha256_file(file) == abc, "sha256.h: digest");
}

// crit/trace.h
void check_trace() {
    crit::trace::Recorder recorder(2);
    { const crit::trace::Scope scope(recorder, crit::trace::Work{"consumer", 1}); }
    const auto snapshot = recorder.snapshot();
    require(snapshot.ranges.size() == 1 && snapshot.ranges[0].name() == "consumer" && snapshot.ranges[0].tokens == 1,
            "trace.h: recorder");
}

// crit/host_memory.h
void check_host_memory() {
    const auto admission = cc::admit_host(cc::HostWorkingSet{}, cc::observe_host_memory());
    require(admission.admitted, "host_memory.h: zero demand is admitted");
}

// crit/host_block.h
void check_host_block() {
    cc::HostBlockPool pool;
    bool reused = true;
    const auto block = pool.acquire("weights", 4096, true, &reused, "checkpoint");
    require(!reused && block->size() == 4096 && pool.cached_bytes() == 4096, "host_block.h: pool allocates and caches");
}

// crit/safetensors.h
void check_safetensors(const std::filesystem::path& directory) {
    const std::string header = R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    std::string bytes(8, '\0');
    const auto length = static_cast<std::uint64_t>(header.size());
    for(std::size_t i = 0; i < 8; ++i) bytes[i] = static_cast<char>((length >> (8 * i)) & 0xffu);
    bytes += header;
    bytes += std::string(8, '\0');
    write_file(directory / "model.safetensors", bytes);
    const auto checkpoint = cc::Checkpoint::open(directory / "model.safetensors");
    const auto& tensor = checkpoint.at("w");
    require(tensor.dtype == cc::DType::f32 && tensor.shape == std::vector<std::uint64_t>{2} && tensor.nbytes() == 8,
            "safetensors.h: open");
}

// crit/device_graph.h
void check_device_graph() {
    const cc::MatrixLayout layout{2, 3, 4, cc::MatrixElement::f32};
    const cc::StagedGraphPlan plan(layout, layout, 4, 1, 0);
    require(plan.input_rows().size() == 2 && plan.input_rows()[1].source == 16 && plan.staged_input().stride == 3,
            "device_graph.h: staged geometry");
}

// crit/device_memory.h
void check_device_memory() {
    const auto owner = std::make_shared<int>(0);
    const cc::MappedSource source(1, {{4096, 65536, 64, owner}, {4160, 65600, 64, owner}});
    const auto resolved = source.resolve(4128, 64);
    require(resolved.device_address == 65568 && resolved.owners.size() == 1, "device_memory.h: mapped source");
}

// crit/native_engine.h
cc::NativeEngine refusing_factory(std::string_view) { throw std::runtime_error("consumer factory refused"); }

void check_native_engine() {
    cc::ReplayContinuation replay;
    replay.tokens = {1, 2, 3};
    require(replay.host_bytes() == 3 * sizeof(cc::Token) && cc::native_api::abi_version == 1, "native_engine.h: types");
    require(cc::native_api::open(&refusing_factory, "{}") == nullptr &&
                std::string_view(cc::native_api::last_error()) == "consumer factory refused",
            "native_engine.h: a failing factory");
}

// crit/model_config.h
void check_model_config(const std::filesystem::path& directory) {
    write_file(directory / "config.json", R"({"model_type":"llama","vocab_size":32,"hidden_size":64,)"
                                          R"("num_attention_heads":4,"num_hidden_layers":2,"eos_token_id":[3,2]})");
    const auto config = cc::load_model_config(directory);
    require(config.vocab_size == 32u && config.effective_head_dim() == 16 && config.layer_kinds().size() == 2,
            "model_config.h: load");
    require(cc::stop_token_ids(config, nullptr) == std::vector<cc::Token>{2, 3}, "model_config.h: stop ids");
}
}

int main() {
    try {
        check_resources();
        check_text();
        check_constraints();
        check_sampling();
        check_state();
        check_json();
        check_host_memory();
        check_host_block();
        check_trace();
        check_device_graph();
        check_device_memory();
        check_native_engine();
        const TemporaryDirectory directory;
        check_file(directory.path());
        check_safetensors(directory.path());
        check_model_config(directory.path());
        std::cout << "every installed header compiled and one call per component passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "installed headers consumer failed: " << error.what() << '\n';
        return 1;
    }
}
