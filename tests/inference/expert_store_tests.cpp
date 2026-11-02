// pin_store (device-expert-cache's expert_store.h) on the device: a store file pinned either as a registered
// read-only mapping or read into a pinned HostBlock, its device address reading the file's bytes, a file of the wrong
// size refused, and the registration released with its owner.
#include "expert_store.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dc = crit::techniques::device_cache;

namespace {
void require(bool c, const std::string& m) { if(!c) throw std::runtime_error(m); }
}

int main() {
    try {
        const auto path = std::filesystem::temp_directory_path() / "crit_expert_store_test.store";
        constexpr std::size_t bytes = std::size_t(3) << 20;   // three bundles of 1 MiB
        std::vector<char> data(bytes);
        for(std::size_t i = 0; i < bytes; ++i) data[i] = char((i * 131 + i / 4096) & 255);
        {
            std::ofstream out(path, std::ios::binary);
            out.write(data.data(), std::streamsize(bytes));
        }
        {
            const auto store = dc::pin_store(path, bytes, 4);
            require(store.host.size() == bytes && store.device, "pinned store");
            std::vector<char> back(bytes);
            if(cudaMemcpy(back.data(), store.device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("copy from the store's device address");
            require(std::memcmp(back.data(), data.data(), bytes) == 0, "the device address reads the file's bytes");
            std::cout << "expert store: " << (store.mapped_file ? "read-only file mapping registered" : "read into a pinned HostBlock")
                      << ", " << bytes << " bytes read back through the device address\n";
        }
        bool refused = false;
        try { dc::pin_store(path, bytes + 16); } catch(const std::runtime_error&) { refused = true; }
        require(refused, "a file of another size refused");
        std::filesystem::remove(path);
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
