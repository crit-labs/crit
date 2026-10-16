#include "crit/model_config.h"
#include "crit/file.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace crit::core {
namespace {
namespace fs = std::filesystem;
constexpr std::uint64_t u32_max = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t u64_max = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t max_layers = 1u << 16;
constexpr const char* length_keys[] = {"max_position_embeddings", "n_positions", "max_seq_len", "seq_length",
                                       "max_sequence_length", "model_max_length"};
constexpr const char* norm_eps_keys[] = {"rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"};
constexpr const char* expert_keys[] = {"num_experts", "num_local_experts", "n_routed_experts"};

std::string str(std::uint64_t value) { return std::to_string(value); }
// Short printable rendering of a string for diagnostics; never splits a UTF-8 sequence.
std::string shown(std::string_view text) {
    std::size_t cut = std::min<std::size_t>(text.size(), 48);
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
std::string shortest(double value) {
    char buffer[32];
    const auto done = std::to_chars(buffer, buffer + sizeof buffer, value);
    return std::string(buffer, done.ptr);
}
// Python's format(value, "g"): six significant digits, trailing zeros removed.
std::string general(double value) {
    char buffer[64];
    const auto done = std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::general, 6);
    return std::string(buffer, done.ptr);
}
std::optional<std::uint64_t> non_negative(const json::Value& value) {
    try { return value.u64(); } catch(const FormatError&) { return std::nullopt; }
}
std::string describe(const json::Value& value) {
    switch(value.kind()) {
    case json::Kind::null: return "null";
    case json::Kind::boolean: return value.boolean() ? "true" : "false";
    case json::Kind::integer:
        try { return str(value.u64()); } catch(const FormatError&) {}
        try { return std::to_string(value.i64()); } catch(const FormatError&) {}
        return "a very large integer";
    case json::Kind::real: return shortest(value.real());
    case json::Kind::string: return shown(value.string());
    case json::Kind::array: return "an array";
    case json::Kind::object: return "an object";
    }
    return "a value";
}
const json::Value* member(const json::Value* object, std::string_view key) {
    return object && object->kind() == json::Kind::object ? object->find(key) : nullptr;
}
bool absent(const json::Value* value) { return !value || value->is_null(); }

// Typed reads: absent keys and JSON null give nullopt or an empty vector; any other wrong value throws.
struct Reader {
    std::string source;
    [[noreturn]] void fail(std::string_view key, const std::string& reason) const {
        throw FormatError(source, "key '" + std::string(key) + "'", reason);
    }
    // Every typed field refuses an integer outside [-2^63, 2^64) the same way, whatever its own range.
    void within_64_bits(const json::Value* value, std::string_view key) const {
        if(value && value->kind() == json::Kind::integer && value->exceeds_64_bits()) fail(key, "integer exceeds 64 bits");
    }
    std::optional<std::uint64_t> unsigned_value(const json::Value* value, std::string_view key, std::uint64_t max) const {
        if(absent(value)) return std::nullopt;
        within_64_bits(value, key);
        const auto number = value->kind() == json::Kind::integer ? non_negative(*value) : std::nullopt;
        if(!number || *number > max) fail(key, "expected an integer in [0, " + str(max) + "], got " + describe(*value));
        return number;
    }
    std::optional<std::uint32_t> u32(const json::Value* value, std::string_view key) const {
        const auto number = unsigned_value(value, key, u32_max);
        return number ? std::optional<std::uint32_t>(std::uint32_t(*number)) : std::nullopt;
    }
    std::optional<Token> token(const json::Value* value, std::string_view key, bool allow_negative) const {
        if(absent(value)) return std::nullopt;
        within_64_bits(value, key);
        constexpr std::int64_t low = std::numeric_limits<Token>::min(), high = std::numeric_limits<Token>::max();
        const std::int64_t min = allow_negative ? low : 0;
        if(value->kind() == json::Kind::integer) {
            std::optional<std::int64_t> number;
            try { number = value->i64(); } catch(const FormatError&) {}
            if(number && *number >= min && *number <= high) return Token(*number);
        }
        fail(key, "expected an integer in [" + std::to_string(min) + ", " + std::to_string(high) + "], got " + describe(*value));
    }
    // An id, a list of ids or null.
    std::vector<Token> tokens(const json::Value* value, std::string_view key, bool allow_negative) const {
        std::vector<Token> out;
        if(absent(value)) return out;
        if(value->kind() != json::Kind::array) {
            out.push_back(*token(value, key, allow_negative));
            return out;
        }
        out.reserve(value->array().size());
        for(std::size_t i = 0; i < value->array().size(); ++i) {
            const auto& item = value->array()[i];
            const auto label = std::string(key) + "[" + str(i) + "]";
            if(item.is_null()) fail(label, "expected an integer, got null");
            out.push_back(*token(&item, label, allow_negative));
        }
        return out;
    }
    std::optional<double> real(const json::Value* value, std::string_view key) const {
        if(absent(value)) return std::nullopt;
        within_64_bits(value, key);
        if(value->kind() != json::Kind::integer && value->kind() != json::Kind::real) fail(key, "expected a finite number, got " + describe(*value));
        const double number = value->real();   // NaN and Infinity reach here from config files
        if(!std::isfinite(number)) fail(key, "expected a finite number, got " + describe(*value));
        return number;
    }
    std::optional<bool> boolean(const json::Value* value, std::string_view key) const {
        if(absent(value)) return std::nullopt;
        if(value->kind() != json::Kind::boolean) fail(key, "expected true or false, got " + describe(*value));
        return value->boolean();
    }
    std::optional<std::string> string(const json::Value* value, std::string_view key) const {
        if(absent(value)) return std::nullopt;
        if(value->kind() != json::Kind::string) fail(key, "expected a string, got " + describe(*value));
        return std::string(value->string());
    }
    const json::Value* object(const json::Value* value, std::string_view key) const {
        if(absent(value)) return nullptr;
        if(value->kind() != json::Kind::object) fail(key, "expected an object, got " + describe(*value));
        return value;
    }
    std::span<const json::Value> array(const json::Value* value, std::string_view key) const {
        if(absent(value)) return {};
        if(value->kind() != json::Kind::array) fail(key, "expected an array, got " + describe(*value));
        return value->array();
    }
    std::vector<std::string> strings(const json::Value* value, std::string_view key) const {
        std::vector<std::string> out;
        const auto items = array(value, key);
        out.reserve(items.size());
        for(std::size_t i = 0; i < items.size(); ++i) {
            if(items[i].kind() != json::Kind::string) fail(key, "entry " + str(i) + " is " + describe(items[i]) + ", not a string");
            out.emplace_back(items[i].string());
        }
        return out;
    }
    std::vector<std::uint32_t> u32s(const json::Value* value, std::string_view key) const {
        std::vector<std::uint32_t> out;
        const auto items = array(value, key);
        out.reserve(items.size());
        for(std::size_t i = 0; i < items.size(); ++i) {
            const auto label = std::string(key) + "[" + str(i) + "]";
            if(items[i].is_null()) fail(label, "expected an integer in [0, " + str(u32_max) + "], got null");
            out.push_back(*u32(&items[i], label));
        }
        return out;
    }
    std::vector<double> reals(const json::Value* value, std::string_view key) const {
        std::vector<double> out;
        const auto items = array(value, key);
        out.reserve(items.size());
        for(std::size_t i = 0; i < items.size(); ++i) {
            const auto label = std::string(key) + "[" + str(i) + "]";
            if(items[i].is_null()) fail(label, "expected a finite number, got null");
            out.push_back(*real(&items[i], label));
        }
        return out;
    }
};

// rope_type (else type) and the scaling fields of one rope object; theta and partial factor are the caller's.
void read_rope(const Reader& r, const json::Value& object, const std::string& prefix, RopeConfig& rope) {
    const auto at = [&](const char* name) { return object.find(name); };
    const auto label = [&](const char* name) { return prefix + "." + name; };
    auto type = r.string(at("rope_type"), label("rope_type"));
    if(!type || type->empty()) type = r.string(at("type"), label("type"));
    rope.type = type.value_or("");
    rope.factor = r.real(at("factor"), label("factor"));
    rope.attention_factor = r.real(at("attention_factor"), label("attention_factor"));
    rope.beta_fast = r.real(at("beta_fast"), label("beta_fast"));
    rope.beta_slow = r.real(at("beta_slow"), label("beta_slow"));
    rope.low_freq_factor = r.real(at("low_freq_factor"), label("low_freq_factor"));
    rope.high_freq_factor = r.real(at("high_freq_factor"), label("high_freq_factor"));
    rope.mscale = r.real(at("mscale"), label("mscale"));
    rope.mscale_all_dim = r.real(at("mscale_all_dim"), label("mscale_all_dim"));
    rope.original_max_position_embeddings = r.unsigned_value(at("original_max_position_embeddings"), label("original_max_position_embeddings"), u64_max);
    rope.short_factor = r.reals(at("short_factor"), label("short_factor"));
    rope.long_factor = r.reals(at("long_factor"), label("long_factor"));
    rope.mrope_section = r.u32s(at("mrope_section"), label("mrope_section"));
    rope.raw = &object;
}
bool keyed_by_layer_type(const json::Value& object) {
    const auto members = object.object();
    return !members.empty() && std::all_of(members.begin(), members.end(), [](const json::Member& m) { return m.value.kind() == json::Kind::object; });
}
void parse_rope(const Reader& r, ModelConfig& c) {
    const auto* scaling = r.object(c.get("rope_scaling"), "rope_scaling");
    const auto* parameters = r.object(c.get("rope_parameters"), "rope_parameters");
    const bool use_scaling = scaling && !scaling->object().empty();
    const json::Value* chosen = use_scaling ? scaling : parameters && !parameters->object().empty() ? parameters : nullptr;
    const std::string key = use_scaling ? "rope_scaling" : "rope_parameters";
    if(chosen) {
        read_rope(r, *chosen, key, c.rope);
        c.rope.key = key;
        if(keyed_by_layer_type(*chosen))
            for(const auto& entry:chosen->object()) {
                RopeLayer layer;
                layer.layer_type = entry.key;
                const auto prefix = key + "." + entry.key;
                read_rope(r, entry.value, prefix, layer.rope);
                layer.rope.theta = r.real(entry.value.find("rope_theta"), prefix + ".rope_theta");
                layer.rope.partial_rotary_factor = r.real(entry.value.find("partial_rotary_factor"), prefix + ".partial_rotary_factor");
                layer.rope.key = key;
                c.rope_by_layer_type.push_back(std::move(layer));
            }
    }
    c.rope.theta = r.real(c.get("rope_theta"), "rope_theta");
    if(!c.rope.theta) c.rope.theta = r.real(member(parameters, "rope_theta"), "rope_parameters.rope_theta");
    if(!c.rope.theta) c.rope.theta = r.real(member(scaling, "rope_theta"), "rope_scaling.rope_theta");
    c.rope.partial_rotary_factor = r.real(c.get("partial_rotary_factor"), "partial_rotary_factor");
    if(!c.rope.partial_rotary_factor && chosen) c.rope.partial_rotary_factor = r.real(chosen->find("partial_rotary_factor"), key + ".partial_rotary_factor");
    if(!c.rope.original_max_position_embeddings)
        c.rope.original_max_position_embeddings = r.unsigned_value(c.get("original_max_position_embeddings"), "original_max_position_embeddings", u64_max);
}
ModelConfig parse_model_config(json::Document document) {
    ModelConfig c;
    c.document = std::move(document);
    const auto& root = c.document.root();
    const Reader r{c.document.source()};
    if(root.kind() != json::Kind::object) throw FormatError(r.source, "root", "the config must be a JSON object, not " + describe(root));
    const auto* text = r.object(root.find("text_config"), "text_config");
    c.text = text ? text : &root;
    const auto get = [&](std::string_view key) { return c.get(key); };
    const auto u32 = [&](const char* key) { return r.u32(get(key), key); };
    c.architectures = r.strings(get("architectures"), "architectures");
    c.model_type = r.string(root.find("model_type"), "model_type").value_or("");
    c.text_model_type = r.string(get("model_type"), "model_type").value_or("");
    c.vocab_size = u32("vocab_size");
    c.hidden_size = u32("hidden_size");
    // An integer, or one per layer (Gemma 3n), which leaves intermediate_size absent.
    if(const auto* sizes = get("intermediate_size"); !absent(sizes) && sizes->kind() == json::Kind::array)
        c.intermediate_sizes = r.u32s(sizes, "intermediate_size");
    else if(absent(sizes) || sizes->kind() == json::Kind::integer) c.intermediate_size = u32("intermediate_size");
    else r.fail("intermediate_size", "expected an integer or an array of integers, got " + describe(*sizes));
    c.num_hidden_layers = u32("num_hidden_layers");
    c.num_attention_heads = u32("num_attention_heads");
    c.num_key_value_heads = u32("num_key_value_heads");
    c.head_dim = u32("head_dim");
    for(const char* key:length_keys) {   // eval's rule: the first positive integer; other values are skipped
        const auto* value = get(key);
        r.within_64_bits(value, key);     // except an integer beyond 64 bits, which is an error
        const auto number = value && value->kind() == json::Kind::integer ? non_negative(*value) : std::nullopt;
        if(number && *number > 0) {
            c.max_positions = number;
            c.max_positions_key = key;
            break;
        }
    }
    for(const char* key:norm_eps_keys)
        if(auto eps = r.real(get(key), key)) {
            c.rms_norm_eps = eps;
            c.norm_eps_key = key;
            break;
        }
    c.tie_word_embeddings = r.boolean(get("tie_word_embeddings"), "tie_word_embeddings");
    for(const char* key:{"dtype", "torch_dtype"})
        if(auto dtype = r.string(get(key), key)) {
            c.dtype = *dtype;
            break;
        }
    parse_rope(r, c);
    c.sliding_window = u32("sliding_window");
    c.full_attention_interval = u32("full_attention_interval");
    c.layer_types = r.strings(get("layer_types"), "layer_types");
    for(const char* key:expert_keys)
        if(auto experts = u32(key)) {
            if(*experts == 0) break;
            MoeConfig moe;
            moe.experts = *experts;
            moe.experts_key = key;
            moe.experts_per_token = u32("num_experts_per_tok").value_or(0);
            moe.shared_experts = u32("n_shared_experts");
            if(!moe.shared_experts) moe.shared_experts = u32("num_shared_experts");
            moe.moe_intermediate_size = u32("moe_intermediate_size");
            moe.shared_expert_intermediate_size = u32("shared_expert_intermediate_size");
            moe.first_k_dense_replace = u32("first_k_dense_replace");
            moe.decoder_sparse_step = u32("decoder_sparse_step");
            moe.n_group = u32("n_group");
            moe.topk_group = u32("topk_group");
            moe.mlp_only_layers = r.u32s(get("mlp_only_layers"), "mlp_only_layers");
            moe.norm_topk_prob = r.boolean(get("norm_topk_prob"), "norm_topk_prob");
            moe.routed_scaling_factor = r.real(get("routed_scaling_factor"), "routed_scaling_factor");
            moe.scoring_func = r.string(get("scoring_func"), "scoring_func").value_or("");
            c.moe = std::move(moe);
            break;
        }
    if(const auto rank = u32("kv_lora_rank"); rank && *rank) {
        MlaConfig mla;
        mla.kv_lora_rank = *rank;
        mla.qk_rope_head_dim = u32("qk_rope_head_dim").value_or(0);
        mla.qk_nope_head_dim = u32("qk_nope_head_dim").value_or(0);
        mla.v_head_dim = u32("v_head_dim").value_or(0);
        mla.q_lora_rank = u32("q_lora_rank");
        c.mla = mla;
    }
    LinearAttentionConfig linear;
    linear.num_key_heads = u32("linear_num_key_heads");
    linear.num_value_heads = u32("linear_num_value_heads");
    linear.key_head_dim = u32("linear_key_head_dim");
    linear.value_head_dim = u32("linear_value_head_dim");
    linear.conv_kernel_dim = u32("linear_conv_kernel_dim");
    if(linear.num_key_heads || linear.num_value_heads || linear.key_head_dim || linear.value_head_dim || linear.conv_kernel_dim) c.linear = linear;
    c.num_nextn_predict_layers = u32("num_nextn_predict_layers");
    c.eos_token_ids = r.tokens(get("eos_token_id"), "eos_token_id", true);
    c.bos_token_id = r.token(get("bos_token_id"), "bos_token_id", true);
    c.pad_token_id = r.token(get("pad_token_id"), "pad_token_id", true);
    if(const auto* quantization = r.object(get("quantization_config"), "quantization_config")) {
        QuantizationConfig q;
        q.method = r.string(quantization->find("quant_method"), "quantization_config.quant_method").value_or("");
        q.raw = quantization;
        c.quantization = std::move(q);
    }
    return c;
}
// config.json and generation_config.json are read as transformers writes them: NaN and Infinity load, and
// every typed real field still requires a finite number.
json::Limits metadata_limits(json::Limits limits) {
    limits.allow_nonfinite = true;
    return limits;
}
// The file a loader reads: the directory's file_name, or the path itself.
fs::path config_file(const fs::path& path, const char* file_name) {
    std::error_code ignored;
    return fs::is_directory(path, ignored) ? path / file_name : path;
}
// Every eos_token_id of a root object and its text_config, appended to ids (non-negative ids only).
void append_eos(const json::Document& document, std::vector<Token>& ids) {
    const auto& root = document.root();
    if(root.kind() != json::Kind::object) return;
    const Reader r{document.source()};
    const auto more = r.tokens(root.find("eos_token_id"), "eos_token_id", false);
    ids.insert(ids.end(), more.begin(), more.end());
    if(const auto* text = r.object(root.find("text_config"), "text_config")) {
        const auto nested = r.tokens(text->find("eos_token_id"), "text_config.eos_token_id", false);
        ids.insert(ids.end(), nested.begin(), nested.end());
    }
}
}

const json::Value* ModelConfig::get(std::string_view key) const {
    const auto& root = document.root();
    if(text && text != &root) if(const auto* value = member(text, key)) return value;
    return member(&root, key);
}
std::uint32_t ModelConfig::require_u32(std::string_view key) const {
    const Reader r{document.source()};
    const auto value = r.u32(get(key), key);
    if(!value) r.fail(key, "missing");
    if(*value == 0) r.fail(key, "expected a positive integer, got 0");
    return *value;
}
std::uint32_t ModelConfig::effective_head_dim() const {
    if(head_dim && *head_dim) return *head_dim;
    const Reader r{document.source()};
    if(!num_attention_heads || !*num_attention_heads) r.fail("num_attention_heads", num_attention_heads ? "expected a positive integer, got 0" : "missing (no head_dim)");
    if(!hidden_size || !*hidden_size) r.fail("hidden_size", hidden_size ? "expected a positive integer, got 0" : "missing (no head_dim)");
    if(*hidden_size % *num_attention_heads)
        r.fail("hidden_size", str(*hidden_size) + " is not divisible by num_attention_heads " + str(*num_attention_heads));
    return *hidden_size / *num_attention_heads;
}
std::uint32_t ModelConfig::effective_kv_heads() const {
    const Reader r{document.source()};
    if(!num_attention_heads || !*num_attention_heads) r.fail("num_attention_heads", num_attention_heads ? "expected a positive integer, got 0" : "missing");
    const std::uint32_t kv = num_key_value_heads && *num_key_value_heads ? *num_key_value_heads : *num_attention_heads;
    if(*num_attention_heads % kv) r.fail("num_key_value_heads", str(kv) + " does not divide num_attention_heads " + str(*num_attention_heads));
    return kv;
}
std::vector<LayerKind> ModelConfig::layer_kinds() const {
    const Reader r{document.source()};
    if(!num_hidden_layers || !*num_hidden_layers) r.fail("num_hidden_layers", num_hidden_layers ? "expected a positive integer, got 0" : "missing");
    const std::uint32_t layers = *num_hidden_layers;
    if(layers > max_layers) r.fail("num_hidden_layers", str(layers) + " layers exceed the " + str(max_layers) + "-layer limit");
    std::vector<LayerKind> kinds;
    kinds.reserve(layers);
    if(!layer_types.empty()) {
        if(layer_types.size() != layers) r.fail("layer_types", str(layer_types.size()) + " entries for " + str(layers) + " layers");
        for(const auto& type:layer_types)
            kinds.push_back(type == "full_attention" || type == "attention" ? LayerKind::full_attention
                            : type == "sliding_attention" ? LayerKind::sliding_attention
                            : type == "linear_attention" ? LayerKind::linear_attention : LayerKind::other);
    } else if(full_attention_interval && *full_attention_interval) {
        for(std::uint32_t i = 0; i < layers; ++i)
            kinds.push_back((i + 1) % *full_attention_interval == 0 ? LayerKind::full_attention : LayerKind::linear_attention);
    } else {
        kinds.assign(layers, LayerKind::full_attention);
    }
    return kinds;
}
std::optional<std::uint64_t> ModelConfig::position_limit(std::string* source) const {
    if(rope.type == "yarn") {
        std::optional<std::uint64_t> original;
        std::optional<double> factor;
        std::optional<std::uint64_t> whole;   // an integer factor multiplies exactly, as Python ints do
        if(rope.raw && rope.raw->kind() == json::Kind::object) {
            const auto* o = rope.raw->find("original_max_position_embeddings");
            const auto* f = rope.raw->find("factor");
            if(o && o->kind() == json::Kind::integer) original = non_negative(*o);
            if(f && f->kind() == json::Kind::integer) {
                whole = non_negative(*f);
                if(whole) factor = double(*whole);
            } else if(f && f->kind() == json::Kind::real) {
                factor = f->real();
            }
        } else {
            original = rope.original_max_position_embeddings;
            factor = rope.factor;
        }
        if(original && *original > 0 && factor && *factor > 0) {
            const Reader r{document.source()};
            const auto key = (rope.key.empty() ? std::string("rope_scaling") : rope.key) + ".factor";
            std::uint64_t limit = 0;
            if(whole) {
                if(*whole > u64_max / *original) r.fail(key, "yarn position limit " + str(*original) + " x " + str(*whole) + " exceeds 2^64 - 1");
                limit = *original * *whole;
            } else {
                const double product = double(*original) * *factor;
                if(!(product < 18446744073709551616.0)) r.fail(key, "yarn position limit " + str(*original) + " x " + general(*factor) + " exceeds 2^64 - 1");
                limit = std::uint64_t(product);
            }
            if(source) *source = "yarn rope scaling: " + str(*original) + " original positions x factor " + general(*factor);
            return limit;
        }
    }
    if(source) *source = max_positions ? max_positions_key : std::string();
    return max_positions;
}

ModelConfig load_model_config(const std::filesystem::path& dir_or_config_json, const json::Limits& limits) {
    return parse_model_config(json::Document::read_file(config_file(dir_or_config_json, "config.json"), metadata_limits(limits)));
}
std::optional<GenerationConfig> load_generation_config(const std::filesystem::path& dir_or_file, const json::Limits& limits) {
    std::error_code ignored;
    const auto path = config_file(dir_or_file, "generation_config.json");
    if(path != dir_or_file && fs::symlink_status(path, ignored).type() == fs::file_type::not_found) return std::nullopt;
    GenerationConfig g;
    g.document = json::Document::read_file(path, metadata_limits(limits));
    const auto& root = g.document.root();
    const Reader r{g.document.source()};
    if(root.kind() != json::Kind::object) throw FormatError(r.source, "root", "the generation config must be a JSON object, not " + describe(root));
    r.object(root.find("text_config"), "text_config");
    g.eos_token_ids = r.tokens(root.find("eos_token_id"), "eos_token_id", false);
    // Like config.json (from which transformers copies them), bos and pad may be any 32-bit id, e.g. pad -1.
    g.bos_token_id = r.token(root.find("bos_token_id"), "bos_token_id", true);
    g.pad_token_id = r.token(root.find("pad_token_id"), "pad_token_id", true);
    g.do_sample = r.boolean(root.find("do_sample"), "do_sample");
    const auto bounded = [&](const char* key, double low, double high, bool open_low, const char* expectation) {
        const auto value = r.real(root.find(key), key);
        if(value && (*value < low || *value > high || (open_low && *value == low)))
            r.fail(key, std::string("expected a number ") + expectation + ", got " + shortest(*value));
        return value;
    };
    const double huge = std::numeric_limits<double>::max();
    g.temperature = bounded("temperature", 0, huge, false, "at least 0");
    g.top_p = bounded("top_p", 0, 1, false, "between 0 and 1");
    g.min_p = bounded("min_p", 0, 1, false, "between 0 and 1");
    g.repetition_penalty = bounded("repetition_penalty", 0, huge, true, "greater than 0");
    g.top_k = r.u32(root.find("top_k"), "top_k");
    g.max_new_tokens = r.u32(root.find("max_new_tokens"), "max_new_tokens");
    g.max_length = r.u32(root.find("max_length"), "max_length");
    if(const auto* stops = root.find("stop_strings"); stops && stops->kind() == json::Kind::string) g.stop_strings.emplace_back(stops->string());
    else g.stop_strings = r.strings(stops, "stop_strings");
    for(const auto& stop:g.stop_strings) if(stop.empty()) r.fail("stop_strings", "expected non-empty strings, got ''");
    return g;
}
std::vector<Token> stop_token_ids(const ModelConfig& config, const GenerationConfig* generation) {
    std::vector<Token> ids;
    if(generation) append_eos(generation->document, ids);
    append_eos(config.document, ids);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}
}
