// Gated-delta and causal-convolution kernels over a POOL of sequences: the decode and verify rows of many sequences in
// one launch, each row reading and writing its own sequence's state, and the device commit of a speculative round's
// accepted prefix (recurrent-state-recovery's strategies) without a host read.
//
// The arithmetic is the column scan of recurrent_kernels.h (one block per value head and 32 value columns, a warp per 4
// columns, the state in registers, k . S and q . S reduced together), with three additions: key and query L2
// normalization inside the scan (x / sqrt(sum x^2 + epsilon), from the same reduction), 16-bit or FP32 inputs, and any
// key width up to 256 and any value width. Every path applies one update expression (S' = k delta + S e, fmaf), so a
// token decoded alone, inside a verify row, replayed from a stash or recomputed at commit gives the same bits, and a
// row's bits do not depend on the rows beside it.
//
// A pool keeps, per layer and sequence:
//   states  [layers][sequences][v_heads][dk][dv]                 FP32, the live (committed) state
//   stash   [layers][sequences][2][pool_delta_half_floats]       two halves, each capacity columns of
//                                                                key [capacity][k_heads][dk] (normalized, as used),
//                                                                decay [capacity][v_heads] (e^l), delta [capacity][v_heads][dv]
//   control [layers][sequences][2 * pool_delta_blocks]           a pending count per block, then a stash parity per block
// and for the convolution
//   window  [layers][sequences][channels][width - 1]             FP32, the live window (oldest input first)
//   stash   [layers][sequences][capacity][channels]              the raw inputs of the last rows
//   pending [layers][sequences][pool_conv_blocks]                a pending count per block
//
// Commit strategies for a row of up to `capacity` tokens (a decode token or a verify window):
//   kStash (deferred): the row reads the live state after applying its sequence's pending count from the stash (and
//     writes the live state then), verifies its tokens without writing the state, and writes the stash; the commit is
//     the caller's: it writes the accepted count into the pending counts (control[l][sequence][0 .. blocks - 1]), which
//     the next row applies. A flush (kFlush rows, or kFlushPending) applies it without a row.
//   kCheckpoints: the row writes the state before its first token and after each of its tokens into the scratch
//     [slot][capacity + 1][S]; the caller's commit copies the one after the accepted count into the live state.
//   kRecompute: the row writes the state before its first token into the scratch [slot][S] and the stash; the commit
//     (kReplay) replays the accepted count from that start into the live state.
// The commit's bookkeeping (writing counts, selecting a checkpoint) is recurrent-state-recovery's, not these kernels';
// they hold the arithmetic: the scan, the replay of stashed steps and the convolution. Every commit can run again with a
// smaller count before the next row of its sequence (a host truncation). A row of kFinal kind scans its tokens and
// writes the state after the last one into the scratch [slot][S], the live state untouched (a prompt row, committed
// whole by the caller).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "recurrent_kernels.h"

namespace crit::recurrent_kernels {

enum class Scalar : int { f32 = 0, bf16 = 1, f16 = 2 };
enum RowKind : int { kStash = 0, kCheckpoints = 1, kRecompute = 2, kFinal = 3, kFlush = 4 };
enum CommitMode : int { kReplay = 2, kFlushPending = 3 };

// Which tokens and which sequence each row of a launch reads: a step's rows from a table the host wrote, or a captured
// round's rows from device tensors the round driver owns (row r is tokens starts[r] .. starts[r + 1] - 1, sequence
// table[slots[r]]; -1 skips the row: its outputs are zero, no state or window is read or written).
struct RowSource {
    const std::int32_t* meta = nullptr;     // [rows][5]: first token, tokens, sequence (-1: skip), kind, scratch slot (-1:
                                            // none; a FINAL, CHECKPOINTS or RECOMPUTE row without one writes zeros)
    const std::int32_t* starts = nullptr;   // [rows + 1]
    const std::int64_t* slots = nullptr;    // [rows]
    const std::int32_t* table = nullptr;    // [round slots] -> sequence
    int kind = kStash;                      // every round row's kind
    int rows = 0;
};

struct PoolDelta {
    float* states = nullptr;
    std::int64_t state_sequence = 0, state_layer = 0;       // floats
    float* stash = nullptr;
    std::int64_t stash_sequence = 0, stash_layer = 0;       // floats
    int capacity = 0;                                       // stash columns per half (>= 1)
    std::int32_t* control = nullptr;
    std::int64_t control_sequence = 0, control_layer = 0;   // ints
    float* scratch = nullptr;                               // checkpoints, start or final states by row slot
    std::int64_t scratch_slot = 0, scratch_layer = 0;       // floats
};

struct RowInputs {
    const void* q = nullptr;                // row t, key head kh at q + t * q_row + kh * q_head (dk contiguous)
    const void* k = nullptr;
    const void* v = nullptr;                // row t, value head h at v + t * v_row + h * v_head (dv contiguous)
    Scalar dtype = Scalar::f32;             // of q, k and v
    std::int64_t q_row = 0, q_head = 0, k_row = 0, k_head = 0, v_row = 0, v_head = 0;   // elements
    const float* log_decay = nullptr;       // [t * gate_row + h], <= 0
    const float* beta = nullptr;            // [t * beta_row + h]
    std::int64_t gate_row = 0, beta_row = 0;
    float epsilon = 1e-6f, scale = 1.f;     // out = scale S'^T q
    bool normalize_keys = true, normalize_queries = true;
    float* out = nullptr;                   // [t * out_row + h * dv + v], FP32
    std::int64_t out_row = 0;
};

// dk 1 .. 256, dv >= 1, v_heads a multiple of k_heads.
bool pool_delta_supported(const DeltaShape&);
int pool_delta_blocks(const DeltaShape&);                           // v_heads * ceil(dv / 32)
std::size_t pool_delta_half_floats(const DeltaShape&, int capacity);
// One layer's rows (the pool's pointers at that layer). A stash row holds at most `capacity` tokens. pipelined: the
// pool-delta-pipe structure (the same bits, another schedule: staged state tiles, a block running several slots).
cudaError_t enqueue_pool_delta(const DeltaShape&, const RowInputs&, const PoolDelta&, const RowSource&, cudaStream_t,
                               bool pipelined = false);
// The replay of every row of `source` over `layers` layers (the pool's pointers at layer 0, layer strides set; a block
// walks the layers, so any number of rows and layers fits one launch):
// kReplay replays accepted[r] (clamped to 0 .. columns) stashed steps from the start state in the scratch into the live
// state, kFlushPending applies and consumes each sequence's pending count (accepted unused).
cudaError_t enqueue_pool_delta_replay(const DeltaShape&, const PoolDelta&, const RowSource&, int layers,
                                      const std::int64_t* accepted, int columns, CommitMode, cudaStream_t);

// ---------------------------------------------------------------------------------------------- convolution
struct PoolConv {
    float* window = nullptr;
    std::int64_t window_sequence = 0, window_layer = 0;     // floats
    float* stash = nullptr;
    std::int64_t stash_sequence = 0, stash_layer = 0;       // floats
    int capacity = 0;
    std::int32_t* pending = nullptr;
    std::int64_t pending_sequence = 0, pending_layer = 0;   // ints
    float* scratch = nullptr;                               // final windows [slot][channels][width - 1]
    std::int64_t scratch_slot = 0;
};
struct ConvInputs {
    const void* x = nullptr;                // row t at x + t * x_row (channels contiguous)
    void* out = nullptr;                    // row t at out + t * out_row, x's dtype; may be x
    Scalar dtype = Scalar::f32;
    std::int64_t x_row = 0, out_row = 0;
    const float* weight = nullptr;          // [channels][width]
    const float* bias = nullptr;            // [channels] or null
    int channels = 0, width = 0;            // taps (width) and their dilation: pool_conv_supported(width, dilation)
    int dilation = 1;                       // a window holds the last (width - 1) * dilation inputs
    bool silu = true;
};
int pool_conv_blocks(int channels);
// The convolutions the kernels run: 2 .. 8 taps undilated, 2 .. 4 taps at dilation 2 .. 4.
bool pool_conv_supported(int width, int dilation);
// Rows of kStash (deferred window: apply the pending count, convolve from the window without writing it, stash the
// inputs), kFinal (convolve, the window after the last input into the scratch) or kFlush, over `layers` layers (a block
// walks them). out must not be x.
cudaError_t enqueue_pool_conv(const ConvInputs&, const PoolConv&, const RowSource&, int layers, cudaStream_t);
// One sequence's prompt chunk padded to `rows` tokens (a captured graph's bucket), its count and sequence read from
// meta [5] (first, count, sequence, kind, slot; first, kind and slot unused) on the device: the first count rows
// convolve from the sequence's window after its pending count (applied and consumed), the rest output zeros, and the
// window after the count real inputs is written in place; a sequence below 0 outputs zeros and touches no window. The
// pool's pointers at the layer. The bits of the row kernel. out must not overlap x.
cudaError_t enqueue_pool_conv_prompt(const ConvInputs&, const PoolConv&, const std::int32_t* meta, int rows, cudaStream_t);
// One long row (a prompt chunk) out of place from window_in, writing window_out: a thread per (channel, 32 rows), the
// same bits as the row kernel. out must not overlap x.
cudaError_t enqueue_conv_segments(const ConvInputs&, const float* window_in, float* window_out, int rows, cudaStream_t);

}  // namespace crit::recurrent_kernels
