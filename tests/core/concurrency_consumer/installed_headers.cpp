// Independent installed-SDK consumer: includes every header the Core install ships and calls one symbol per
// component, so a header missing from install(FILES ...), an installed header that includes one that is not
// installed, or a strict warning inside an installed header fails this build. CMakeLists.txt checks that the
// includes below name every installed crit/*.h and crit/core/*.h header and that those are exactly the checkout's
// inference/core/include/crit headers (the cuda*.h headers are covered by the CUDA SDK consumer).
#include <crit/constraints.h>
#include <crit/core.h>
#include <crit/core/scheduling.h>
#include <crit/core/engine.h>
#include <crit/core/model.h>
#include <crit/core/profile.h>
#include <crit/core/request.h>
#include <crit/core/resources.h>
#include <crit/file.h>
#include <crit/json.h>
#include <crit/native_engine.h>
#include <crit/sampling.h>
#include <crit/step_engine.h>
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

// crit/constraints.h
void check_constraints() {
    auto vocabulary = std::make_shared<const cc::ByteVocabulary>(std::vector<std::string>{"{", "}", "x", ""}, 3);
    cc::JsonObjectMatcher matcher(vocabulary);
    require(matcher.allowed() == std::vector<cc::Token>{0}, "constraints.h: JSON object start");
    matcher.append(0);
    matcher.append(1);
    require(matcher.allowed() == std::vector<cc::Token>{3}, "constraints.h: a finished object allows only the end");
}

// crit/sampling.h
void check_sampling() {
    const double draw = cc::uniform(7, 0);
    require(draw >= 0.0 && draw < 1.0, "sampling.h: uniform");
}

// crit/json.h
void check_json() {
    const auto document = cc::json::Document::parse(R"({"name":"crit","ids":[1,2]})", "inline");
    const auto* ids = document.root().find("ids");
    const auto* name = document.root().find("name");
    require(ids && name && ids->array().size() == 2 && ids->array()[1].u64() == 2 && name->string() == "crit",
            "json.h: parse");
}

// crit/file.h
void check_file(const std::filesystem::path& directory) {
    write_file(directory / "abc.bin", "abc");
    const auto file = cc::ReadOnlyFile::open(directory / "abc.bin");
    std::array<std::byte, 3> bytes{};
    file.read_exact(0, bytes);
    require(file.identity().size == 3 && bytes[0] == std::byte{0x61} && bytes[2] == std::byte{0x63}, "file.h: read");
}

// crit/native_engine.h
cc::NativeEngine refusing_factory(std::string_view, const void*) { throw std::runtime_error("consumer factory refused"); }

void check_native_engine() {
    cc::ReplayContinuation replay;
    replay.tokens = {1, 2, 3};
    require(replay.host_bytes() == 3 * sizeof(cc::Token) && cc::native_api::abi_version == 10, "native_engine.h: types");
    require(cc::native_api::open_bound(&refusing_factory, nullptr, "{}") == nullptr &&
                std::string_view(cc::native_api::last_error()) == "consumer factory refused",
            "native_engine.h: a failing factory");
}

// crit/step_engine.h
void check_step_engine() {
    cc::StepCallbacks callbacks;
    try {
        (void)cc::step_engine(R"({"engine":{"vocabulary":8,"context":16}})", callbacks);
        throw std::runtime_error("step_engine.h: a step engine without callbacks must be refused");
    } catch(const std::invalid_argument&) {
    }
}

}

int main() {
    try {
        check_resources();
        check_constraints();
        check_sampling();
        check_json();
        check_native_engine();
        check_step_engine();
        const TemporaryDirectory directory;
        check_file(directory.path());
        std::cout << "every installed header compiled and one call per component passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "installed headers consumer failed: " << error.what() << '\n';
        return 1;
    }
}
