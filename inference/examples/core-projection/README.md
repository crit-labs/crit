# Connect Inference arithmetic to a Core engine

This example connects the native [grouped-integer projection](../../quantized-projections/code/grouped_int.h)
to Core's Model/Session/Backend interfaces. It needs CMake 3.28+ and a C++20
compiler and runs on the CPU. The model has four tokens, `A` through `D`, and a
hand-authored signed-int4 G32 weight matrix that predicts the next letter
cyclically. It is a teaching model: no pretrained weights or benchmark results.

From the repository root:

```sh
cmake -S inference/examples/core-projection -B build/core-projection -DCMAKE_BUILD_TYPE=Release
cmake --build build/core-projection --config Release
ctest --test-dir build/core-projection -C Release --output-on-failure
```

Run `build/core-projection/crit_core_projection` (with Visual Studio,
`build/core-projection/Release/crit_core_projection.exe`) to see:

```text
ABCDAB -> CDABCD
D -> ABC
```

## Follow the integration

Read [main.cpp](main.cpp) in this order:

1. **`ProjectionModel` owns the representation.** `make_weights()` uses Inference's
   `quantize()`; the immutable `PackedRows` outlives every session and its payload
   is reserved in Core's resource ledger.
2. **`ProjectionSession` owns request state.** Prefill consumes the prompt within
   Core's chunk budget. Decode calls the native projection and picks greedily:

   ```cpp
   const auto logits = crit::quant::project(weights_.view(), activations, 1);
   pending_token_ = static_cast<Token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
   ```

3. **Progress stays provisional until `commit()`.** The session returns a borrowed
   view of `pending_token_`; Core decides how much to accept, calls `commit()` to
   advance private state, then publishes. `close()` releases the prompt view and
   reservation.
4. **`HostBackend` establishes completion.** The projection is synchronous, so
   `drain()` has nothing to settle. An asynchronous backend must supply real
   readiness and drain behavior.

The test covers several prefill chunks, multiple decode/commit rounds and a second
request sharing the retained weights. A real engine supplies its own layers,
weights, tokenizer, sampling and backend behind the same Core boundary.
