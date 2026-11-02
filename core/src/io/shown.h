#pragma once
// How Core's readers quote a name or a value in an error message: printable, and cut at a byte limit without splitting
// a UTF-8 sequence. Shared by the JSON, config and safetensors readers.
#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace crit::core::detail {
inline std::string shown(std::string_view text, std::size_t limit) {
    std::size_t cut = std::min(text.size(), limit);
    while(cut < text.size() && cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
    std::string out = "'";
    for(char c:text.substr(0, cut)) {
        const auto b = static_cast<unsigned char>(c);
        if(b < 0x20 || b == 0x7F || c == '\'' || c == '\\') {
            static const char hex[] = "0123456789abcdef";
            out += "\\x"; out += hex[b >> 4]; out += hex[b & 15];
        } else out += c;
    }
    return out + (cut < text.size() ? "'..." : "'");
}
}
