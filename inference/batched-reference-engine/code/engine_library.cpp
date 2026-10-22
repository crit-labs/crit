// The batched synthetic engine as a native engine library (crit/native_engine.h) with concurrent requests, which the
// stock resident adapter (crit_core.native.resident_engine) serves through crit serve as chat-concurrent-v1.
//
// Configuration (JSON object; every key optional): vocab, hidden, layers, heads, kv_heads, head_dim, ffn, rope_theta,
// seed, page_tokens, format ("bf16" or "e4m3"), context (or the adapter's context_tokens), cache_tokens,
// max_concurrency, max_batch_tokens, prefill_chunk, prefix_cache, stop_tokens.
#include "engine.h"

#include <crit/json.h>
#include <crit/native_engine.h>

#include <memory>
#include <string>
#include <string_view>

namespace be = crit::batched_engine;
namespace cc = crit::core;

namespace {
be::Config parse(std::string_view text) {
    const auto document = cc::json::Document::parse(text, "batched engine configuration");
    const auto& root = document.root();
    be::Config c;
    c.shape = {512, 256, 2, 4, 2, 64, 512, 10000.f, 1};
    const auto integer = [&](const char* key, auto& out) {
        if(const auto* v = root.find(key)) out = static_cast<std::remove_reference_t<decltype(out)>>(v->i64());
    };
    integer("vocab", c.shape.vocab); integer("hidden", c.shape.hidden); integer("layers", c.shape.layers);
    integer("heads", c.shape.heads); integer("kv_heads", c.shape.kv_heads); integer("head_dim", c.shape.head_dim);
    integer("ffn", c.shape.ffn); integer("seed", c.shape.seed); integer("page_tokens", c.page_tokens);
    integer("context_tokens", c.context); integer("context", c.context); integer("cache_tokens", c.cache_tokens);
    integer("max_concurrency", c.max_concurrency); integer("max_batch_tokens", c.max_batch_tokens);
    integer("prefill_chunk", c.prefill_chunk);
    if(const auto* v = root.find("rope_theta")) c.shape.rope_theta = static_cast<float>(v->real());
    if(const auto* v = root.find("format")) {
        if(v->string() == "e4m3") c.format = crit::paged_attention::Format::e4m3;
        else if(v->string() != "bf16") throw std::invalid_argument("format must be bf16 or e4m3");
    }
    if(const auto* v = root.find("prefix_cache")) c.prefix_cache = v->boolean();
    if(const auto* v = root.find("stop_tokens"))
        for(const auto& t : v->array()) c.stop_tokens.push_back(static_cast<cc::Token>(t.i64()));
    if(!be::valid(c.shape)) throw std::invalid_argument("Invalid synthetic transformer shape");
    return c;
}

cc::NativeEngine open_engine(std::string_view configuration) {
    const auto config = parse(configuration);
    auto stats = std::make_shared<be::Stats>();
    cc::NativeEngine native;
    native.owner = stats;
    native.engine = be::make_engine(config, stats);
    native.concurrent_requests = true;
    native.info = [] { return std::string(R"({"batched": true, "paged_attention": true})"); };
    native.result = [](const cc::Result& result) {
        return "{\"reused_prompt_tokens\":" + std::to_string(result.timing.reused_prompt_tokens) + "}";
    };
    return native;
}
}  // namespace

CRIT_EXPORT_ENGINE(open_engine)
