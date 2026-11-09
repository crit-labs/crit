#pragma once
// Step engines: a ConcurrentModel whose model code lives behind C callbacks, for engines written in another language
// (crit_core.native.StepEngine drives one from Python through ctypes). Core's runtime keeps everything else: the
// scheduler with mixed prefill/decode batches, admission, request options, stop tokens, cancellation, retirement and
// the scoring barrier. The engine implements a step: given one row per scheduled request, it computes them together
// and reports each row's progress and tokens. The callbacks run on the engine's one worker thread, never at once.
//
// Shared library: Core builds crit_core_steps, which exports the native engine ABI (crit/native_engine.h) plus
//   void* crit_engine_open_steps(const char* configuration_json, const crit::core::StepCallbacks* callbacks)
//   void* crit_engine_open_steps_rounds(const char* configuration_json, const crit::core::StepCallbacks* callbacks,
//                                       const crit::core::StepRounds* rounds)
// returning an engine handle the other crit_engine_* functions take. The callbacks and their user pointer must stay
// valid until crit_engine_close returns.
#include <crit/native_engine.h>
#include <cstdint>
#include <string_view>

namespace crit::core {
// The structures are standard layout with fixed-width fields, so other languages can mirror them.
// One request: its key, prompt and output limit, and its options as a JSON object: the sampling fields it set
// ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty", "frequency_penalty", "presence_penalty",
// "logit_bias" as {"id": bias}, a banned token's bias as -Infinity (Python's json spelling), "min_tokens"),
// "stop_tokens" (the effective list), "logprobs" (alternatives per token),
// "replay" (a forced continuation, ReplayContinuation), "plain" (decode without drafting, PlainDecoding, and every
// constrained request) and "attachment" (the id of host-side state for the request, HostAttachment) with
// "constrained" (every token masked by Core's host side; one token per row, never in a native round).
struct StepRequest {
    std::uint64_t request = 0, generation = 0;
    std::uint32_t lane = 0, prompt_tokens = 0;
    const std::int32_t* prompt = nullptr;
    std::uint32_t output_tokens = 0, reserved = 0;
    const char* options_json = nullptr;
};
// One scheduled row: the session the engine's open returned and the row's WorkBudget.
struct StepRow {
    std::uint64_t session = 0;
    std::uint32_t prefill_tokens = 0, output_tokens = 0, token_work = 0;
    std::int32_t prefill_complete = 0;
};
// What the engine reports per row (Core's Progress): prompt tokens consumed this step (reused ones included), the
// reused prefix, the token work actually computed (mixed batches; zero takes the prompt tokens computed, or one per
// token), the tokens produced, whether prefill is complete and, for a decode row, the drafts its round verified plus
// one (0: not reported). The tokens go to tokens[row * max_row_tokens ...]; with token log-probabilities, each token's
// into logprobs[row * max_row_tokens + i] and its alternatives into top_ids/top_logprobs[(row * max_row_tokens + i) *
// max_top_logprobs ...], ranked as Core's TokenLogprob rows are.
struct StepProgress {
    std::uint32_t prompt_tokens = 0, reused_prompt_tokens = 0, token_work = 0, tokens = 0;
    std::int32_t prefill_complete = 0, drafted = 0;
};
// Core's CommitDecision for one row of the last step: the tokens Core kept (stops and the output limit trim them),
// whether the request ends or was cancelled (or failed). `discard` is always 0 for a step engine: Core discards only a
// descendant of a step still in flight when its predecessor ended, and a step engine runs one step at a time (it
// declares no async batches, so its profile runs none in flight), so every step it ran is kept, or its request ends
// and retire() drops the sequence. Core fails the engine if a discard ever reached one: a model need not undo a step
// (a replayed prompt chunk's state written in place, crit_plugins.prompt_graphs).
struct StepDecision {
    std::uint64_t session = 0;
    std::uint32_t accepted = 0;
    std::int32_t terminal = 0, cancelled = 0, discard = 0;
};
// Native decode rounds (optional: step_engine's `rounds`, crit_engine_open_steps_rounds). A decode step whose rows' sessions all sit in the round's
// slots runs without the step callback: Core launches the engine's round for those slots (a captured device graph
// that drafts, verifies, accepts and copies its result block to host memory), waits for the block and reports each
// row's tokens, so nothing in the engine's language runs while its requests decode. The engine keeps its decoding rows
// packed in slots 0 .. rows - 1 and names their sessions in slot_session (0 for a slot Core must not run natively);
// every other step (prompt chunks, a row outside the slots, a constrained request, a request asking for more
// alternatives than the rounds rank) goes to `step` as before. After a native round Core calls `commit` only when a
// decision trims, ends, cancels or discards a row, with last = {rows, width, 1} naming that round (last[2] is 0 after
// a `step`).
struct StepRounds {
    void* user = nullptr;
    // Launches the round of slots 0 .. rows - 1 at `width` drafts; returns without waiting. 0 on success. Each round
    // has two forms: `logprobs` 1 launches its log-probability form, which writes the log-probability buffers below
    // with its block and runs the rows that follow a continuation of their own (a replay's), 0 its plain form, which
    // does neither and costs less. Core launches the log-probability form when a row's request asks for
    // log-probabilities or replays a continuation, else the plain one.
    int (*launch)(void* user, std::uint32_t rows, std::uint32_t width, std::uint32_t logprobs) = nullptr;
    // Returns once the launched round's block is in host memory. 0 on success.
    int (*wait)(void* user) = nullptr;
    // The result block: max_rows rows of row_stride entries of entry_bytes (4 or 8) each, per slot: [0] tokens
    // emitted, [1] drafts matched, [2] frontier after the round, [3 ..] the emitted tokens and, from entry
    // `telemetry` (0: none; past the widest round's tokens), the row's words: its drafts accepted, then `counters`
    // counters of the engine's own, which the configuration's round_counters name.
    const void* block = nullptr;
    std::uint32_t row_stride = 0, entry_bytes = 8, max_rows = 0, telemetry = 0;
    // widest: the widest round the engine runs (open checks its settings hold it for every row count given a width).
    std::uint32_t counters = 0, widest = 0;
    // The round's width for r rows at widths[r - 1], read before each round; a negative entry runs r rows through
    // `step`. The engine may change the entries between steps (a draft-width policy: each round's own copy of the
    // next round's widths into this array among them), each within `widest`; a row count negative at open stays so.
    const std::int32_t* widths = nullptr;
    // [max_rows]: each slot's session handle (0: empty), written by the engine between steps.
    const std::uint64_t* slot_session = nullptr;
    // [3], written by Core: rows and width of the last step's round, and 1 when it ran natively.
    std::uint32_t* last = nullptr;
    // Optional [3], written by Core: native rounds run, the rows they ran, and how many of those rounds ran the
    // log-probability form.
    std::uint64_t* runs = nullptr;
    // Optional, written by each log-probability round with its block (null: a request asking for log-probabilities
    // decodes through `step`): each slot's log-probability of the round's emitted tokens, [max_rows][widest + 1]
    // floats, and `top` ranked alternatives of each, top_ids [max_rows][widest + 1][top] (int64) and top_logprobs
    // (floats), ranked as Core's TokenLogprob rows are. Core gives a request asking for log-probabilities its row's (a
    // request asking for more than `top` alternatives decodes through `step`).
    const float* logprobs = nullptr;
    const std::int64_t* top_ids = nullptr;
    const float* top_logprobs = nullptr;
    std::uint32_t top = 0;
};

// Every callback but release returns 0 on success; any other value fails the engine, with error()'s message. Optional
// callbacks may be null.
struct StepCallbacks {
    void* user = nullptr;
    // Optional (null: always ready): 0 ready, 1 deferred, 2 infeasible (the request fails alone). Inspection only.
    int (*admission)(void* user, const StepRequest*) = nullptr;
    // A new session for the request; *session is the engine's handle for it in every later call.
    int (*open)(void* user, const StepRequest*, std::uint64_t* session) = nullptr;
    // One step over `count` rows (one per scheduled request, in any order). Buffers are sized from the configuration's
    // max_row_tokens and max_top_logprobs; the logprob buffers are null without token_logprobs.
    int (*step)(void* user, const StepRow* rows, std::uint32_t count, StepProgress* progress, std::int32_t* tokens,
                float* logprobs, std::int32_t* top_ids, float* top_logprobs) = nullptr;
    // Optional: Core's decisions for every row of the last step, before it publishes their tokens.
    int (*commit)(void* user, const StepDecision* decisions, std::uint32_t count) = nullptr;
    // Optional: the request ended (Finish as int: 0 length, 1 stop, 2 cancelled, 3 failed) and its session retires.
    int (*retire)(void* user, std::uint64_t session, int finish) = nullptr;
    // Optional: the session is destroyed; the engine drops it. Must not fail.
    void (*release)(void* user, std::uint64_t session) = nullptr;
    // Optional scoring (configuration "scoring": true): native_api::score's arguments and buffers for one sequence.
    int (*score)(void* user, const std::int32_t* ids, std::uint32_t count, std::uint32_t first_target, const char* path,
                 const std::int32_t* queries, std::uint32_t per_target, std::uint32_t top_k, float* target, float* query,
                 std::int32_t* argmax, std::int32_t* has_argmax, std::int32_t* top_ids, float* top_logprobs) = nullptr;
    // The last failure's message into out (at most capacity bytes with the terminating zero); returns its length.
    int (*error)(void* user, char* out, int capacity) = nullptr;
    // Optional: the facts of a request whose session retires with a result (length or stop), before `retire`: given
    // Core's own (core_facts, a JSON object of its decode rounds, with `_native`, each round's kind), the JSON object
    // the request's result carries, written into out (at most capacity bytes with the terminating zero); returns its
    // length (capacity or more: called again with room for it, when it writes what it wrote the first time), negative
    // to keep Core's facts alone.
    int (*facts)(void* user, std::uint64_t session, const char* core_facts, char* out, int capacity) = nullptr;
};

// StepRounds launchers over captured CUDA graphs, exported by crit_core_steps (the CUDA driver API is loaded at run
// time, libcuda.so.1 or nvcuda.dll, so Core builds without a CUDA toolkit):
//   void* crit_rounds_cuda_open(uint64_t stream, const uint64_t* graphs, const uint64_t* logprob_graphs,
//                               uint32_t max_rows, uint32_t max_width)
//   int crit_rounds_cuda_launch(void* user, uint32_t rows, uint32_t width, uint32_t logprobs)   // a StepRounds::launch
//   int crit_rounds_cuda_wait(void* user)                                    // a StepRounds::wait
//   void crit_rounds_cuda_close(void* user)
//   int crit_rounds_cuda_error(char* out, int capacity)                      // the last failure's message
// graphs[(rows - 1) * (max_width + 1) + width] is a plain round's CUgraphExec (0: none), logprob_graphs the same table
// of the log-probability rounds (null: none; an engine whose rounds have one form gives its graphs twice), and stream a
// CUstream, as a framework reports them (torch.cuda.CUDAGraph's raw_cuda_graph_exec(), torch.cuda.Stream's
// cuda_stream). launch replays the form's graph on the stream; wait synchronizes the stream. open returns null when
// the driver or a symbol is missing.

// A NativeEngine (concurrent_requests set) around a step model. The configuration JSON holds "engine", Core's
// settings: "implementation", "vocabulary" and "context" (required), "valid_vocabulary", "prefill_chunk" (default
// min(512, context)), "stop_tokens", "max_concurrency" (default 8), "max_pending" (default max_concurrency),
// "max_batch_tokens" (mixed batches; default prefill_chunk + max_concurrency * max_row_tokens, 0 separates prefill and
// decode steps), "batch_policy" ("fair" or "decode_first"), "max_row_tokens" (most tokens one row reports, default 1),
// "decode_token_work" (default 1), "prefill_work_quantum" (default 1: prompt work granted in its multiples, so the
// prefill chunk as the quantum keeps a prompt's chunk boundaries whatever shares its steps), "sampling" (the fields the
// engine applies, as ModelInfo::sampling), "token_logprobs" and "max_top_logprobs", "scoring" and "score_paths",
// "host_limit" (bytes, default 1 GiB), "round_counters" (the names of the counters a native round's words carry); and
// "facts", a JSON object merged into crit_engine_info. invalid_argument for a missing or invalid setting, a missing open
// or step callback, and native rounds the settings cannot hold (a width's columns past max_row_tokens,
// decode_token_work or, with mixed batches, max_batch_tokens; words inside the widest round's tokens; counters other
// than round_counters names). `rounds` (optional, valid until crit_engine_close returns) runs decode steps natively
// (StepRounds).
//
// Each generation that ends with a result carries the facts of its decode rounds (StepCallbacks::facts adds the
// engine's): "rounds" (decode rows after prefill, one target pass each), and when it drafted "accepted" (per round: a
// native round's word, else the row's tokens less one) and "drafted" (per round: a native round's width, else what the
// engine reported in StepProgress::drafted; only when every round's is known), and each of round_counters summed over
// its native rounds' words when it ran one.
NativeEngine step_engine(std::string_view configuration_json, const StepCallbacks&, const StepRounds* rounds = nullptr);
}
