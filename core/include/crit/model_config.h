#pragma once
#include <crit/core.h>
#include <crit/json.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
enum class LayerKind : std::uint8_t { full_attention, sliding_attention, linear_attention, other };
// Rotary parameters from rope_scaling when it is a non-empty object, else rope_parameters (crit eval's order).
// theta: rope_theta, else rope_parameters.rope_theta, else rope_scaling.rope_theta; type: rope_type, else type.
struct RopeConfig {
    std::optional<double> theta = {}, partial_rotary_factor = {};
    std::string type = {};                                     // "" when none
    std::optional<double> factor = {}, attention_factor = {}, beta_fast = {}, beta_slow = {}, low_freq_factor = {},
                          high_freq_factor = {}, mscale = {}, mscale_all_dim = {};
    std::optional<std::uint64_t> original_max_position_embeddings = {};   // the rope object's, else the section's
    std::vector<double> short_factor = {}, long_factor = {};
    std::vector<std::uint32_t> mrope_section = {};
    const json::Value* raw = nullptr;                          // the object read; lives in the config's document
    std::string key = {};                                      // "rope_scaling", "rope_parameters" or ""
};
// One entry of a rope_parameters object keyed by layer type (every member an object), e.g. "sliding_attention".
struct RopeLayer { std::string layer_type = {}; RopeConfig rope = {}; };
// Present when num_experts, else num_local_experts, else n_routed_experts is positive; experts_key names it.
struct MoeConfig {
    std::uint32_t experts = 0;
    std::string experts_key = {};
    std::uint32_t experts_per_token = 0;                       // num_experts_per_tok; 0 when absent
    std::optional<std::uint32_t> shared_experts = {}, moe_intermediate_size = {}, shared_expert_intermediate_size = {},
                                 first_k_dense_replace = {}, decoder_sparse_step = {}, n_group = {}, topk_group = {};
    std::vector<std::uint32_t> mlp_only_layers = {};
    std::optional<bool> norm_topk_prob = {};
    std::optional<double> routed_scaling_factor = {};
    std::string scoring_func = {};
};
// Multi-head latent attention: present when kv_lora_rank is positive; absent sizes read as 0.
struct MlaConfig {
    std::uint32_t kv_lora_rank = 0, qk_rope_head_dim = 0, qk_nope_head_dim = 0, v_head_dim = 0;
    std::optional<std::uint32_t> q_lora_rank = {};
};
// The linear_* recurrence sizes; present when any of them is set.
struct LinearAttentionConfig {
    std::optional<std::uint32_t> num_key_heads = {}, num_value_heads = {}, key_head_dim = {}, value_head_dim = {}, conv_kernel_dim = {};
};
struct QuantizationConfig { std::string method = {}; const json::Value* raw = nullptr; };   // quant_method; never interpreted

// Typed view of config.json. Every field reads the text section: text_config keys over the root (a null
// value in text_config hides the root's, as in crit eval); model_type alone is the root's. JSON null reads
// as absent. Wrong types throw FormatError "<file>: key 'x': ..."; booleans never count as numbers. An
// integer outside [-2^63, 2^64) or a non-finite number may sit in keys no field reads; a typed field (the
// length keys included) throws "integer exceeds 64 bits" or "expected a finite number" for it.
// Pointers (text, raw) point into document and stay valid while any copy of document lives.
struct ModelConfig {
    json::Document document = {};
    const json::Value* text = nullptr;                         // the text_config object, or the root
    std::vector<std::string> architectures = {};
    std::string model_type = {}, text_model_type = {};          // the root's; the text section's
    std::optional<std::uint32_t> vocab_size = {}, hidden_size = {}, intermediate_size = {}, num_hidden_layers = {},
                                 num_attention_heads = {}, num_key_value_heads = {}, head_dim = {};
    std::optional<std::uint64_t> max_positions = {};           // first positive integer of eval's LENGTH_KEYS
    std::string max_positions_key = {};                        // others are skipped, as in eval
    std::optional<double> rms_norm_eps = {};
    std::string norm_eps_key = {};                             // rms_norm_eps | layer_norm_eps | layer_norm_epsilon
    std::optional<bool> tie_word_embeddings = {};              // absent stays absent
    std::string dtype = {};                                    // dtype, else torch_dtype
    RopeConfig rope = {};
    std::optional<std::uint32_t> sliding_window = {}, full_attention_interval = {};
    std::vector<std::string> layer_types = {};                 // declared strings, verbatim
    std::optional<MoeConfig> moe = {};
    std::optional<MlaConfig> mla = {};
    std::optional<LinearAttentionConfig> linear = {};
    std::optional<std::uint32_t> num_nextn_predict_layers = {};
    std::vector<Token> eos_token_ids = {};                     // an id or a list of ids, in [-2^31, 2^31)
    std::optional<Token> bos_token_id = {}, pad_token_id = {};
    std::optional<QuantizationConfig> quantization = {};
    std::vector<RopeLayer> rope_by_layer_type = {};            // rope_parameters keyed by layer type, else empty
    std::vector<std::uint32_t> intermediate_sizes = {};        // a per-layer intermediate_size array (Gemma 3n),
                                                               // which leaves intermediate_size absent

    const json::Value* get(std::string_view key) const;        // the text section's value (null included), else nullptr
    std::uint32_t require_u32(std::string_view key) const;     // FormatError "missing" or "expected a positive integer"
    std::uint32_t effective_head_dim() const;                  // head_dim, else hidden_size / num_attention_heads, exactly
    std::uint32_t effective_kv_heads() const;                  // num_key_value_heads, else heads; must divide heads
    // layer_types mapped ("attention" is full, unknown names other; exactly num_hidden_layers entries), else
    // every full_attention_interval-th layer full and the rest linear_attention (the layer_types Hugging Face
    // derives, e.g. for Qwen3-Next), else all full. num_hidden_layers must be in [1, 65536]. The full count
    // equals eval's attention_shape full layers.
    std::vector<LayerKind> layer_kinds() const;
    // crit eval's context.model_limit: max_positions, unless rope.type is "yarn" with a positive original and
    // factor in rope.raw (the typed fields when raw is unset): then int(original x factor). source gets the key
    // or "yarn rope scaling: <original> original positions x factor <factor:g>", or "" when there is no limit.
    std::optional<std::uint64_t> position_limit(std::string* source = nullptr) const;
};
// config.json in a directory, or the named file; strict JSON (crit/json.h) within limits, except that NaN,
// Infinity and -Infinity load as transformers writes them (allow_nonfinite is set whatever limits says).
ModelConfig load_model_config(const std::filesystem::path& dir_or_config_json,
                              const json::Limits& limits = {.max_bytes = std::size_t(16) << 20});

// generation_config.json as written (no fallback to config.json). eos ids are non-negative, bos and pad any
// 32-bit id (pad -1 is common); temperature >= 0, top_p and min_p in [0, 1], repetition_penalty > 0;
// stop_strings is one non-empty string or a list. It mirrors the file, not SamplingParams: an adapter
// translates it, and do_sample false or top_p 0 means greedy (temperature 0).
struct GenerationConfig {
    json::Document document = {};
    std::vector<Token> eos_token_ids = {};
    std::optional<Token> bos_token_id = {}, pad_token_id = {};
    std::optional<bool> do_sample = {};
    std::optional<double> temperature = {}, top_p = {}, min_p = {}, repetition_penalty = {};
    std::optional<std::uint32_t> top_k = {}, max_new_tokens = {}, max_length = {};
    std::vector<std::string> stop_strings = {};
};
// A directory's generation_config.json (nullopt when the directory has none), or the named file; read like
// config.json (NaN and Infinity load; typed fields require finite numbers).
std::optional<GenerationConfig> load_generation_config(const std::filesystem::path& dir_or_file,
                                                       const json::Limits& limits = {.max_bytes = std::size_t(16) << 20});
// crit eval's prepare.stop_ids: the sorted union of eos_token_id at the root and in text_config of the
// generation config (when given) and the model config. Ids must be non-negative (FormatError otherwise).
std::vector<Token> stop_token_ids(const ModelConfig& config, const GenerationConfig* generation);
}
