# Dense layer offload

Serve a dense model whose layer weights exceed the device memory left for them by
placing each layer, as a unit, on the device, in a streaming slot or on the host. A
planner predicts each placement's decode step from measured rates (the device's
bandwidth from `crit eval probe`, the pinned upload bandwidth and the host's GEMV
rate on this machine) and the engine times the leading candidates for a few real steps
and keeps the measured fastest. Host layers run on the CPU with AVX2 or AVX-512
BF16 GEMVs over all cores; streamed layers are uploaded into two device staging slots
ahead of their use while the layers before them compute.

## When to use it

When a dense model's weights plus KV cache and activations do not fit the device but
fit host memory, and a single sequence (or a few rows) is decoded per step. Not for
expert-routed models (their misses are per expert, see
[heterogeneous-expert-dispatch](../../experts/heterogeneous-expert-dispatch/README.md)), when the
weights fit (keep them resident), or when a quantized form of the weights fits the
device within the engine's quality budget and measures faster than the offload plans.

## How it works

**Placements.** A layer's weights are one contiguous byte range (here QKV, O,
gate/up and down in BF16). A *resident* layer keeps them on the device. A *streamed*
layer is uploaded every step from pinned host memory into one of two device slots: the
upload of streamed layer `k + 2` waits for layer `k` to release its slot and then
overlaps every layer in between (copy stream and events, the host never blocks). A
*host* layer runs on the CPU from host memory with its KV cache in host memory:
before a run of host layers the activations are copied down and the compute stream
synchronized once, and afterwards copied back.

**Prediction.** With measured rates in GB/s (bytes per ns): a resident layer costs
`bytes / device_gbs`; a streamed layer `max(bytes / device_gbs, bytes / upload_gbs)`
(its upload overlaps the layers before it); a run of host layers its
`bytes / host_gbs` plus one activation round trip. Candidates are all resident when
that fits, host execution of what does not fit, and streaming of it with two slots of
the largest streamed layer taken from the budget; resident layers are chosen by time
saved per byte. The fastest prediction ranks them and `select_measured` times every
candidate within a slack of it and keeps the measured fastest, so a wrong rate can
reorder candidates but never picks a slower plan than one that was measured.

## Worked example

From `code/`:

```python
from planner import DEVICE, HOST, STREAM, candidates, predict

# Eight 10-byte layers, room for 50 bytes: device 1, upload 0.5 and host 0.25 bytes per ns, 5 ns per host run.
m = {"device_gbs": 1.0, "upload_gbs": 0.5, "host_gbs": 0.25, "host_layer_ns": 5.0}
plans = candidates([10] * 8, 50, m)
assert [p[0] for p in plans] == ["stream", "host"]
assert plans[0][3] == 3 * 10 + 5 * 20       # 3 resident, 5 streamed at max(10, 20) each, two slots of 10 bytes
assert plans[1][3] == 5 * 10 + 3 * 40 + 5   # 5 resident, one run of 3 host layers
m["upload_gbs"] = 0.05                       # a slow link: host execution ranks first
assert candidates([10] * 8, 50, m)[0][0] == "host"
assert predict([10, 10], [HOST, HOST], m) == 85 and predict([10], [DEVICE], m) == 10
```

## Invariants

- A streamed layer is read only after its upload's event and its slot is reused only
  after the layer released it; uploads stay on their own stream.
- Host layers read activations only after the copy down has completed, and the device
  reads them only after the copy back.
- A layer's KV cache lives where the layer runs; changing a layer's placement starts
  a new sequence (or moves its cache with it).
- Plans are ranked by prediction and chosen by measurement.

## Costs and validation

`test_reusable_techniques.py` checks the planner's candidates and predictions.

Measured with an earlier fixture's synthetic model, on one machine (RTX 5090, 12 host
threads with AVX-512), with 16 layers of 436 MB (7 GB of BF16 weights) and the device budget
set to part of the weights to exercise the planner: device 1,131 GB/s
(from an all-resident run), upload 28.8 GB/s, host GEMV 35 GB/s, activation round trip
21 us. With a quarter of the weights resident, host execution of 12 layers was predicted
at 147 ms per decode step and measured at 171 ms, streaming 14 layers predicted at
214 ms and measured at 212 ms; with 60% resident, 91 against 94 ms and 139 against
137 ms. Measured selection chose host execution both times: on this machine the host
reads its memory faster than the link uploads it. These numbers come from one machine
(with another process using the CPU) and are not settings to copy.
