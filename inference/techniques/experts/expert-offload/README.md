# Offload routed experts

Serve a mixture of experts whose routed experts do not fit on the device, with one plugin in the experts slot. The
model plugin calls it exactly as it calls experts that fit; the plugin decides where every expert lives, who computes
each route and how prompt chunks run, sizes itself on the device's free memory while the engine serves, and says what
it decided.

## When to use it

Use it whenever the routed experts might not fit beside the model's weights and its KV cache at the required context:
recon's placement scenario `sparse_device_host` (experts in host memory) or `sparse_device_host_disk` (not even that).
It also serves experts that do fit (the device tier), so an engine need not know which case it is in.
Experts gated by GELU (Gemma 4's) are served as SwiGLU ones are: the model plugin calls `activate("gelu_tanh")`
before `open()` (the fact `experts:activations`), the device rows take the gate in the grouped-experts kernel, and
every miss goes over the link, since the CPU tier computes SiLU-gated NVFP4 and INT4 experts.

Don't use it for a model without routed experts.

## How it works

The model plugin makes three calls, and a fourth where it has memory to lend:

```text
experts = parts["experts"]                           # expert-offload; expert_source: expert-tensors
experts.begin(layers, experts_per_layer, "nvfp4", hidden, inter, device_bytes=kv_and_state_bytes)
for layer, expert in every_expert:                   # skipped when experts.complete (a store file reopened)
    experts.load(layer, expert, gate_up, down, "nvfp4")
experts.open()                                       # after the model's own pools (else the first forward() opens)
out = experts.forward(layer, x, ids, weights, rows=batch.rows_t, prompt=is_prompt_chunk)
```

`layers` counts every layer whose experts load, a draft layer's included. Nothing else is the model's to call: Core
tells the plugin of every step it ends (`end_step`: `crit_plugins.device.STEP_FRAMES`), frames every step with the
device's memory and tells that memory when the load is over, and captured
rounds find the plugin in its slot (its `start_round` and `count_round` run inside the round).

**The decision tree.**

| Question | Answer | Decided |
| --- | --- | --- |
| Where do the experts live? | `device` when they fit beside `device_bytes` and `reserve_mib`; `ram` when host memory holds them beside `ram_headroom_mib`; else `disk` | `begin()`, from the device's free memory and the host's available memory |
| How big is the device cache? | Every byte the device has free while the engine serves (below) | `open()`, then Core's step frames |
| Who computes a decode miss? | `cpu` where it builds: the CPU tier ([host-expert-tier](../host-expert-tier/README.md)) computes every miss in the device's bits beside the device, while the link copies the missed expert into the cache beside the round; `link`: every miss copied into a cache slot before its layer reads it | `open()`: the CPU path needs nvfp4 W4A4 or int4 experts, the native tier and the in-stream join built here, and a mapped store |
| Which experts stay? | Those routed last (`policy: recent`): a missed expert serves from the next round, in the slot of the expert routed longest ago | every decode round |
| Which arithmetic? | Wherever the CPU computes misses, the device's decode rows take the CPU's integer order, so who computes an expert never changes a token; with misses over the link every row is grouped-experts' tensor arithmetic (in quality mode `speed` the experts' own tensor cores where the kernel has them here, as grouped-experts' `arithmetic: auto`: `fp4` for W4A4 NVFP4 experts, `fp8` for FP8 ones, and for NVFP4 experts without input scales `fp4-row` on prompt rows where the engine asks them no more than `row-exact`, speed's level; the facts' `decode_arithmetic` says which). Prompt chunks never go to the CPU: `prompt_arithmetic: tensor` runs them on tensor cores where the engine asks `repeatable` or `row-exact` prompt rows | the miss path, `prompt_arithmetic` |
| Prompt chunks | The layer's non-resident experts staged through the cache's ring by the copy engine, one layer ahead; nothing admitted. A short chunk borrows as many ring slots as it has routes. A staged layer serves every chunk that reads it, across steps (below) | every `forward(prompt=True)` |
| A short prompt chunk's misses | Where the CPU tier computes decode misses and prompt rows keep its order (`prompt_arithmetic` declared), a chunk of fewer routes than a layer has experts sends its missed routes to the CPU tier when they cost less there than their experts over the link, by the two times `open()` measured (`miss_times_us`; under four fifths, as for decode misses): nothing copied, no ring borrowed, the same bits. Else, and in the tensor arithmetic, its routed misses are copied into the ring | every short `forward(prompt=True)`, from the times measured at `open()` |
| What does the cache give back? | An elastic cache is a giver of the device's ledger: a prompt step's room, and the pages a KV pool backs as a sequence grows, come out of its coldest slots, down to its ring and, without a CPU tier, a layer's experts (over the link a miss needs a slot). A cache that keeps its size gives nothing back, so at `open()` it leaves the model's elastic pools the bytes they have reserved and not backed (`reasons.pools`) | the ledger's `need()` |
| Layers stored otherwise | An NVFP4 checkpoint may store some layers' experts weight-only (W4A16: no input scales; a draft layer's) beside W4A4 ones. The CPU tier computes W4A4 experts only, so where it serves the other layers' misses those layers are the cache's link layers under the recent policy: in the cache with the rest, each miss admitted in its own plan from a reserve of the round's victims and copied over the link before the layer reads it, their rows the kernel's tensor arithmetic (the bits they had held whole). The reserve is what one decode round routes in a layer, `begin(decode_rows=, top=)` (a model's requests at once times every row a captured round runs through the layer, times its top k; a layer's experts where the model does not say), so a tight cache keeps its prompt ring: FP4-Local Qwen3.8-Flash-Next's 512-expert draft layer takes 240 slots beside a ring's 512, where held whole it took 512 and left too few for the ring. A captured round plans the draft layer first (the drafter before the verify); a step decoding eagerly plans the decoder's layers first, whose hits and CPU-tier copies may take the round's every victim, so there a link layer's plan starts a round of its own. `reasons.link_layers` names them and the reserve. Under the ranked policy they stay whole on the device, outside the cache and the tier; where every miss goes over the link (misses `link`, a file tier, a store with compressed scales) they are cached as the others | `open()` |

**Prompts: a layer's experts once a span.** A prompt chunk of a hundred tokens or more routes nearly every expert of
a layer, so chunk by chunk every chunk brings every layer's non-resident experts over the link: on
Qwen3.8-Flash-Next (49 expert layers, 69.7 GB of experts of which the cache holds about 20) that is some 48 GB and
2.2 to 2.3 s a chunk whatever its size, 250 to 270 tokens a second of prefill with every step past crit serve's
one-second cancel grace. Neither the link nor the CPU tier changes that: both read the same bytes from host memory at
about the same rate. What changes it is the order of the model's loops. `stages_prompts()` tells the model plugin
that prompt chunks stage so, and a model plugin that then runs a prompt's spans layer by layer (model-template's
passes: every chunk of a span through one layer before the next layer) calls `forward(layer, ..., prompt=True)`
once a chunk for the same layer: the cache serves the calls after the first from what the first staged, so each
layer's experts cross the link once for the span, with the next layer's staged under the current one. Such a span
takes several of Core's steps: while a prompt goes on (`end_step(prompts_open=True)`) the ring stays borrowed with
the layer staged in it, decode rows of other requests plan beside it, and the next step stages only what their plans
pushed out of the cache; when no prompt is in flight, or the request that held the ring is cancelled, the ring goes
back. Nothing in it depends on the model: a layer id, its routes and the order of the calls. Measured through
model-template on an RTX 5090 with every option here at its default: Qwen3.8-Flash-Next 2,131 tokens a second at a
16,384-token prompt where chunk by chunk gave 555, Gemma 4 26B (46 GB of BF16 experts, GELU-gated) 4,727 where chunk
by chunk gave 1,050, every step under 0.62 s and the same tokens either way ([the
table](../../../../models/core/model-template/README.md#the-serving-contracts)).

So a prompt's experts are served three ways, each chosen here and none by the engine's author: a span of long
chunks stages each layer once (the link, at the device's arithmetic rate for every chunk after); a short chunk
copies its routed misses into the ring, or, where the CPU tier serves decode misses in the rows' own order and
was measured the sooner, has the CPU tier compute them; and decode rows miss to the link or the CPU tier by the same
measurement (`misses: auto`). `facts()`'s `prompts` and `reasons.misses` say which with the numbers. On a machine
whose link and CPU tier read host memory at about the same rate (an RTX 5090 on PCIe 5.0 x16 beside a Ryzen 5
9600: 110 to 140 us a missed expert either way) the short chunk's two ways cost the same and the link keeps it; the
CPU's way is for a slower link, where the same measurement already sends decode misses to the CPU tier.

**The cache's memory.** Under WDDM (Windows, WSL) an allocation past the device's memory is not refused: it moves to
system memory and everything that reads it slows down. So the cache never takes memory it was not shown to be free:

- `open()` leaves `reserve_mib` free for what the model still makes (graphs, workspaces, its first prompts).
- The cache is a giver of the device's memory (`crit_plugins.memory`), and nothing more: it gives back its
  coldest slots when asked (`release`) and grows into free memory when told (`regrow`). Once the engine's warm
  requests ran, Core tells that memory the load is over (`serve()`) and the cache grows into every free byte the
  device does not keep. Its slots live in an elastic range (address space for every expert, backed as it grows), so
  growing and shrinking moves no slot and breaks no captured round.
- What stays free is the device's to say, not the cache's ([the device's memory](../../../core/python/crit_plugins/memory.py)):
  the run's VRAM headroom at every moment (Core's loader sets it from the configuration's `vram_headroom_bytes`,
  which crit run writes into every candidate; nothing here sizes or tunes it) and the step arena, where a step's
  temporaries live: Core allocates every byte a step takes and the arena is the most a step took while the engine
  loaded. The cache is not asked during a step: it gives back only where something it holds is needed outside the
  arena (a KV pool backing pages for a longer context, a kept prompt's room), and grows back once the device is
  quiet. `telemetry()`'s `elastic` section repeats what the device keeps (`headroom_bytes`, `arena_bytes`). Until
  Core's fill the load's steps live on `reserve_mib` and the cache keeps its size.
- A KV pool allocated for the whole context holds memory its sequences rarely use. With attention-state's `elastic`
  option the pool backs only the pages sequences hold and asks the device's givers before it grows: the cache
  holds the rest and gives it back as KV grows (never out of the room of the prompt step in flight), with no call
  from the model plugin. A recurrence plugin's `elastic` option does the same for recurrent states: only the
  sequences that are open are backed (on Qwen3.8-Flash-Next 0.14 GiB instead of 1.0 for a pool of 8 sequences,
  about 300 more cache slots and 6% fewer misses).
- Where the pool cannot be elastic, the model can lend it: `pool = kv_cache.pool(...,
  spare=experts.spare_bytes(pool_bytes))` and `experts.open(lend=pool)` put the cache's own slots in the pool's spare
  bytes and its further slots in the pages no sequence holds, taken back before KV lands there. Such a cache keeps
  its size (it is not elastic).

The cache is [device-expert-cache](../device-expert-cache/README.md)'s ExpertCache, the host store expert-tensors'
BundleStore (host memory of its own, registered, read in place by the CPU tier; a file under `store_dir` is built
once per checkpoint and read into that memory at later loads), the disk tier's store a file read on the host (its misses cannot be
captured, so it decodes through `step()`), the CPU tier behind
[heterogeneous-expert-dispatch](../heterogeneous-expert-dispatch/README.md)'s in-stream post and join, and every
device row grouped-experts' kernel. The plugin adds no arithmetic of its own.

**The store between loads.** Building the store reads every routed expert from the checkpoint and packs it: minutes
for tens of gigabytes, at every load. A store file is built once and reopened (`complete`: the model skips its expert
loads). `store_dir` names a folder for it. Without one, the plugin uses the host's plugin cache (`context.cache`, the
`CRIT_PLUGIN_CACHE` folder: crit run sets it to `.plugin-cache` in the run's folder, so a candidate's first load
builds the store and every later load of the run reopens it; `crit run stop` removes the folder). There the file is keyed by the checkpoint, the layout
and the code of the package's plugins (another model plugin or other packing is another store), the cache holds one
store a checkpoint (the others of that checkpoint are removed when a new one is built), and a new file is written
only on a local disk and where the machine's plugin cache ledger (`crit_plugins.disk_cache`) makes room for it: every
run's kept stores together stay under `CRIT_PLUGIN_CACHE_GIB` (default 100) and the disk keeps `CRIT_DISK_FLOOR_GIB`
(default 50) free after the write, the least recently opened stores of any run, a run never stopped included, removed
first. Where it cannot, `reasons.store` says why and the store is built in host memory as before. Set `CRIT_PLUGIN_CACHE` to another folder to share stores between runs,
or to nothing for no cache.

**The store between engines.** The ram tier's store is kept in host memory for the next engine built for the model in
the same process (expert-tensors' `open_store(kept=)`, Core's `LoadContext.keep`): crit serve's model pool keeps it
while the model is loaded, so an ablation's variants and a measurement's lower context reuse it, and crit eval's
process keeps it for its next engine (`crit eval ab --package`'s rounds). It is keyed by the checkpoint, the experts'
geometry and format, the layout and the package's plugin code, as the cache file is; the next engine gets it sealed,
as the last engine left it (the bundles it left out for a pinned core stay out), and skips its loads (`complete`).
Its bytes count as free host memory when the next engine chooses its tier, and an engine whose experts go to another
tier closes it first.

**The disk tier keeps each expert once.** Where host memory does not hold the experts, what it has free
(`host_tier_mib`, by default all of it beside `ram_headroom_mib`) is a host tier between the file and the device
cache. The cache hands an expert it evicts back to it (one copy down the link, far cheaper than the file's read),
and an expert that goes up into a cache slot leaves it, so the device's slots and the host's add up: a 16 GB card
and 24 GB of free host memory keep 40 GB of experts off the file, where a copy on both sides would keep 24. The
file is read past the OS's page cache, since the host tier is the copy, and a layer's misses are read all at once
by Core's reader threads (`crit_plugins.host`'s reads, 8 at a time), so the drive works at a queue depth instead
of one page fault after another. A prompt's ring slots are no residence: an expert read for them stays in the
host tier. `telemetry()`'s `file_store` counts the host tier's hits, the file's reads, the experts handed back and
the ones that left for them. Not measured on a model larger than host memory yet.

**The ram tier can hold its hottest experts once (`pinned`).** In the ram tier host memory holds every expert
(the store, registered host memory) and the device holds the resident ones a second time. Handing an evicted
expert back, as the disk tier does, has no place here: the store is one registered mapping the device reads at a
fixed address, and taking a bundle's pages out of it and back costs a registration each way (measured on one
machine: 470 to 540 us to register an expert's pages, 240 to 280 us to unregister them, beside 166 us for the
copy itself), tens of milliseconds a round at a hundred misses. What costs nothing while serving is a core that
never leaves. `pinned` is that core's share of the cache's slots at `open()`: the experts the `profile` file
ranks hottest are read once from the store file straight into slots no plan lists as a victim, no shrink gives
back and no preload or prompt ring takes, and the store maps a filler over their pages before it is registered,
so host memory never holds them (the layers kept whole on the device, W4A16 beside W4A4 under the ranked policy,
leave it the same way). The slots left follow the routing as before, and every reply keeps its bits: a pinned expert is the same
bundle in a slot, and the miss path is the CPU tier's as before. The core comes from `profile` alone, never from
`profile_save`, which follows whatever was served last: a core is a bet on what every workload routes, so it is
made from routing recorded on general text.

What a core costs is hits where the workload routes differently from the profile, and `crit plugins offload
--trace T --profile P` replays that before an engine is loaded. Measured on Qwen3.8-Flash-Next (69.7 GB of
experts, about 8,660 slots, the profile recorded on general text, the workload the probe's chat prompts): a core
of 3,991 experts (`pinned: 0.5`, 10.3 GiB the store does not hold) hits 89.3% where the plain rule hits 90.7%
(the replay said 88.6% against 89.9%), 15% more misses for the CPU tier, and greedy decode goes from 68.8 to
68.3 tok/s with every reply's tokens unchanged; the replay gives 89.4% for a core of 1,996 and 86.5% for one of
5,747. On the profile's own kind of text a core of half the slots costs nothing (a trace's first half as the
profile, its second half replayed: 89.5% against 89.7%). What it buys depends on the host. On a PC whose 84 GB
Linux VM leaves Windows 10 GB, the engine without a core pushed Windows to 0.1 GiB available and 313,000 pages a
second out to the page file while it loaded, and the first reply after the load ran at 22 tok/s; with the core
Windows kept 10.6 GiB or more available, wrote no page out, and the first reply ran at 64 tok/s (the load took
403 s against 515 s). So `pinned` is off by default: on a host with memory to spare it only costs hits, and the
host that needs it cannot be told from inside (the VM above sees 77 GiB available either way).

**What it says.** `facts()["expert_offload"]` names the tier, the miss path, the slots, the policy and the reason for
each decision, and `prompts`: whether prompt chunks stage their layers' experts, the link's rate as measured at
`open()` (`link_us` a bundle, `link_gib_s`) and a sweep, the bytes and seconds of bringing every expert the cache
does not hold to the device once (`sweep_bytes`, `sweep_seconds`): what a long chunk run through every layer pays,
and a span run layer by layer pays once for all its chunks. `telemetry()["expert_offload"]` carries the cache's and the CPU tier's counters, the elastic cache's
size and what prompt chunks took, who holds the device's memory (`device`: the cache, the model's weights, pools
and workspaces in torch's allocator, what is free), and `diagnosis`, one line: the hit rate over the slots, where
the misses went, those shares, and the next lever when the hit rate is low.

**Experts a GGUF checkpoint stores (format `gguf`).** A GGUF file quantizes each layer's expert stacks in a block
format of its own (Unsloth's UD-IQ1_M Qwen3.8-Flash-Next: gate and up projections IQ2_XXS, IQ2_XS or IQ2_S by layer,
down projections Q2_0, two layers' IQ4_NL), so `begin(..., "gguf", ..., formats={layer: (gate_up, down)})` takes each
layer's formats and a bundle is an expert's stored rows as the file has them: its gate and up rows, then at the next
64-byte boundary its down rows ([code/gguf_experts.py](code/gguf_experts.py)). Bundles are cached, staged and
streamed over the link as any other (every store, tier, cache and prompt path above). The slot is the bundle most
layers need: the smallest layer bundle past which the wider layers hold at most a sixteenth of the experts' bytes.
A narrower layer's bundle is padded to it; a wider layer's (a few layers a file stores in a wider format: two IQ4_NL
layers of Qwen3.8-Flash-Next's UD-IQ1_M) is stored in pieces of a slot (extra store layers after the model's) and
kept whole on the device at its own size, outside the cache (`reasons.outsized`), so no slot is padded to it. Their rows are crit's kernel gguf-rows' arithmetic, read where they lie in the cache's slots
(a layer's gate-up and down stacks are views of the slot array at its stride): a decode row's routes on gguf-rows'
gathered gemv (int8 activations a 32 values, each stored sub-block's exact integer dot, FP32 in the kernel's declared
order), a prompt chunk's routes grouped by slot and multiplied on each expert by gguf-rows' matmul, which has the same
bits, so a prompt row has its decode row's bits. The CPU tier computes GGUF misses in torch on the host, in gguf-rows'
declared order with the splits the kernel takes on this device (`GgufCpuTier` over Core's `declared_rows`, the bundles
read in place in the store): a route it serves has the bits it has in a slot. It is not native, so its rows join the
device's after a synchronization and a round with it is not captured (`round_problems` says so); on `misses` auto it
serves misses where the cache is too small for link misses, and where it is timed faster than the link. The model
plugin hands GGUF experts as stored only to an experts plugin that provides
`experts:gguf`, as this one does.

The decisions are plain functions of sizes and measured times ([code/offload.py](code/offload.py)), so they run without
a device:

```python
from offload import GIB, cache_slots, choose_misses, choose_tier

# 65.8 GB of routed experts, a 32 GB device with 6 GiB of the model's pools still to come, 80 GiB of host memory free.
experts, free, pools, host = 65_800_000_000, 30 * GIB, 6 * GIB, 80 * GIB
assert choose_tier(experts, free, pools, host, margin=3 * GIB, ram_headroom=2 * GIB) == "ram"
assert choose_tier(experts, free, pools, 48 * GIB, margin=3 * GIB, ram_headroom=2 * GIB) == "disk"
assert choose_tier(20 * GIB, free, pools, host, margin=3 * GIB, ram_headroom=2 * GIB) == "device"
assert cache_slots(24 * GIB, 3 * GIB, 2_780_176) == 8109    # every free byte past the margin, a scratch slot beside

# Who serves a miss: the CPU tier only where it takes under four fifths of the link's time.
assert choose_misses("nvfp4", True, measured={"cpu": 464.0, "link": 102.0}) == "link"
assert choose_misses("nvfp4", True, measured={"cpu": 60.0, "link": 102.0}) == "cpu"
assert choose_misses("bf16", True) == "link"                # a format the CPU tier does not compute
```

## Tuning

Nothing needs setting. To see what a setting is worth before changing an engine:

1. Serve with `record_routes: 250000` and `trace_path: routes.json`, and send a few requests of the kind the engine
   will serve (a general corpus, never Eval's prompts).
2. `crit plugins offload --model M --device-file DEVICE --context TOKENS --trace routes.json --out offload.json`
   replays the trace through the cache's residency at several sizes and copy budgets (two seconds each) and writes
   the options; put `offload.json` in the package and set `settings: offload.json`.
3. [code/bench.py](code/bench.py) runs the plugin alone on the engine's store file and the trace, a captured graph a
   round as an engine's rounds run it: `python bench.py --model M --store experts.bin --trace routes.json --code
   ../../device-expert-cache/code --variant 'fill_blocks=8' --variant 'swaps=128'` prints, for the options and each
   variant, the round's time, the CPU tier's share and time per missed expert, the hit rate, and the misses and
   copies a round. A setting is judged in two minutes, without an engine build or an Eval. `--hold-gib G` holds
   device memory beside the plugin, as an engine's weights and pools do, and `--model-us U` puts U microseconds of
   device work after every layer's experts, as a model's own layers do: with both set to an engine's numbers the
   bench's round is that engine's (Qwen3.8-Flash-Next: 10.5 GiB and 450 us give 50 ms a round, the engine's 40 to
   56). The emulated work is a kernel that only waits: the bench judges what a setting costs the experts and the
   CPU tier, not what device work running beside a model's own kernels costs those kernels (copies launched after
   a layer's CPU work won 3% here and lost 20% in the engine, where they ran beside the model's kernels instead of
   while the device waited for the CPU).

**The cache is what the model leaves.** More slots are worth more than any setting here. On Qwen3.8-Flash-Next's
recorded routing, replayed, 7,200 slots miss 144 experts a round, 8,200 miss 115, 9,100 miss 94 and 10,000 miss 79:
each GiB given to the cache is 390 slots and about 7% fewer misses. The engine measured below held 8.8 GiB beside
its experts: 6.0 of projection weights (1.2 of them the output head in BF16), 1.2 for the token embedding table,
0.9 of recurrent states for 48 state slots of which one request uses 4, and 0.5 of workspaces. A table read from
the host through the embeddings slot, pools sized for what the engine serves and weights held as they are stored
give the cache what no option can. Measured there: the token table opened from the host (`open()` of the
embeddings slot with the table's tensor: its rows in registered host memory, a `tables.` group of the placement)
gave the cache 1.1 GiB, the replies bit for bit the same, and a round's lookup of 4 rows costs what it costs on
the device (80 us either way; the registration takes no device memory); elastic recurrent states gave 0.9; a
VRAM headroom of 0.25 GiB instead of 1 GiB (the device's headroom, then this plugin's `margin_mib`) gave 0.7; and the round driver's `share_pool` (one
memory pool for every captured round; the replies stayed the same bit for bit) gave 0.2. Together the cache went
from 18.9 to 21.8 GiB (8,408 slots). What is left beside it and the weights is well under 1 GiB: the engine's
scratch (Core's, `context.scratch`: the experts' route rows, the split-KV reader's partials, the sampler's launch
scratch and the projections' tile activations and partial sums are taken from it by each call, 0.16 after
1,024-token chunks, the route rows' size, where those plugins' own workspaces added up to 0.32 and to 0.38 with
the split-KV reader's default; the cache gained 0.17); the projections' other shared buffers, made at first use
(0.09 where a matrix decodes a chunk of rows or pins a GEMM, none here); the recurrence's stashes, backed with
their sequence, 0.03 of 0.15; the captured graphs' pools 0.28: there, 0.25 of temporaries the engine's own model
plugin makes in a round, a whole sequence's indexer keys gathered for every attention layer; and what a prompt step takes from torch's
allocator: 4,700 tensors a step, a few alive at once, 6 to 92 MiB for a short prompt and 250 for a 1,024-token
chunk since the recurrence keeps a prompt row's final states in its pool (they were 108 MiB of every step). A
long prompt's blocks are another matter: kept in the allocator they were
counted as every later prompt's need, and after one 6,316-token prompt the cache stayed 1.0 to 1.3 GiB smaller
(20.4 to 20.75 GiB); since a prompt's blocks go back to the device when it ends, it stays at 21.7.
`telemetry()`'s `device` and the diagnosis give the shares, the three plugins that hold most beside the cache
and the three largest tensors by shape (a vocabulary-sized table shows there), from Core's census of the device
(the engine's `"memory"` telemetry has all of it).

| Option | Default | Meaning |
| --- | --- | --- |
| `tier` | `auto` | Pin `device`, `ram` or `disk` |
| `host_tier_mib` | -1 | The disk tier's host tier: -1 takes the host memory free beside `ram_headroom_mib`, 0 none (the page cache holds what was read) |
| `misses` | `auto` | Pin `cpu` or `link`. `auto` times a missed expert on the CPU tier and over the link at load as the engine will serve each, and takes the CPU tier only where it needs under four fifths of the link's time (`telemetry()`'s `miss_times_us`, every timed batch in `miss_trials_us`). The link: batches of the same bundles until its rate has stopped doubling (a second at most), the fastest counted, timed before the CPU tier exists (a store just built or reopened pays an expert's first use several times over; a device left idle while the engine loaded has its link at its lowest speed and brings it up only after a fraction of a second under load, so its first batches agree with each other at a seventh of the rate it serves misses at; and the tier's threads take the processor the copies' host side runs on: timed any other way the link read 700 to 750 us here where it serves at 100, every miss went to the CPU tier, and with it every prompt row to the declared order, at under half the prefill rate). The CPU tier: computing the expert (`compute`) and the way of its layer's rows to the host and back (`handoff`), which a decode layer with a miss there pays and a copy never does; the choice changes the device rows' arithmetic, so it is a [measured decision](../../../../docs/PLUGINS.md#measured-decisions): a package whose record holds it (`crit plugins decisions`) takes the same path in every process (`miss_times_from`: `recorded`), and without one each load measures: beside the CPU tier every device row runs the declared order, and the tier starts slow after a pause. Qwen3.8-Flash-Next on an RTX 5090 and a Ryzen 5 9600 under WSL: the link (100 us against the tier's 80 to 125 to compute an expert and 180 to 240 for the handoff), 73 sampled and 78 greedy tok/s on fresh prompts where the CPU tier gave 45 and 73 |
| `keep_routes` | 0 | An approximation, off by default: N keeps each token's first N routes of the router's order and leaves the rest out, not renormalized. Fewer expert rows and fewer misses for a changed model, the same for a row whatever the cache holds (so replies stay repeatable and independent of the rows beside them, which a skip of only the missed experts would not be). On Qwen3.8-Flash-Next's recorded routing (10 routes a token, 8,408 slots, replayed) 9 routes miss 119 experts a round where 10 miss 143, 8 miss 98 and 7 miss 80. Measured on that model on an RTX 5090 (the cache in 8,662 slots, six greedy replies of 256 tokens, the all-routes replies' tokens scored again): greedy decode 59.6 tok/s with all routes, 68.9 with 9 and 72.8 with 8; over 1,536 positions 9 routes keep the most likely token at 95.2% and move a mean total variation of 0.039 over the top 20 tokens, 8 routes 95.1% and 0.045, and every greedy reply leaves the all-routes reply within its first 80 tokens (a second all-routes run repeats the first bit for bit). A changed model, not a preserved one: Eval's drift budget judges it in quality mode `speed` |
| `policy` | `recent` | The cache's residency beside the CPU tier: `recent`, or `ranked` (decayed route counts ranked every `swap_period` rounds) |
| `swaps` | 512 | The most misses a round copies in |
| `fill_blocks` | 32 | Thread blocks of the copies' fill kernel: too few and a round waits for its copies |
| `prompt_arithmetic` | `auto` | `tensor`: prompt chunks on tensor cores; `declared`: the decode rows' order. `auto` is `tensor` where the engine asks `repeatable` or `row-exact` prompt rows, else `declared` (the default level, `exact`, in both quality modes) |
| `reserve_mib` | 3072 | Device memory left free at `open()`, until the warm requests ran |
| `elastic` | on | The cache grows and shrinks with the device's free memory; off, it keeps its size from `open()` |
| `store_dir`, `store_path` | empty | Where the bundle store file is kept, or an existing complete store. With neither, the store goes to the host's plugin cache where there is one (below), else it is built in host memory at every load |
| `profile`, `profile_save` | empty | A routing profile the cache starts from, and where the plugin writes the routing it counted |
| `pinned` | 0 | The ram tier's pinned core: this share (up to 0.9) of the cache's slots at `open()` holds the experts `profile` ranks hottest for good, and host memory does not hold them. For a host short of memory; needs `misses: cpu`, the recent policy, a profile and a store file |
| `settings` | empty | A tuned settings file in the package |
| `record_routes`, `trace_path` | 0, empty | Record the decode routes for `--trace` and the bench |
| `workers`, `pin` | -1, on | The CPU tier's threads |

## Costs and validation

Measured on Qwen3.8-Flash-Next FP4-Local (49 expert layers of 512 experts, 2.78 MB a bundle, 68 GB in host memory),
one RTX 5090 and a Ryzen 5 9600 under WSL2.

The plugin alone (bench.py, 1,232 recorded decode rounds of about 1,250 distinct expert reads each, about 11,000
cache slots, nothing else on the device):

| Residency | Hits | Misses and copies a round | Round | of which the CPU tier |
| --- | --- | --- | --- | --- |
| `ranked`, every 3 rounds, 96 swaps | 85.4% | 183, 34 | 40.5 ms | 26.3 ms |
| `recent`, `swaps` 96 | 90.5% | 119, 61 | 22.1 ms | 15.2 ms |
| `recent`, `swaps` 256 | 93.6% | 80, 72 | 17.0 ms | 10.8 ms |
| `recent`, `swaps` 512 | 94.1% | 74, 74 | 16.4 ms | 10.2 ms |

`fill_blocks` 4, 16, 32, 64: 18.6, 17.0, 16.7 and 16.8 ms a round; the CPU tier took 133 to 137 us a missed expert
whatever the fill's rate (host memory reads 49 GB/s here, the tier and the copies together use about half).

In an engine (ten fresh prompts of 320 tokens, five sampled and five greedy, after a 6,316-token prompt), where the
model's weights and pools leave the cache less:

| Engine | Cache slots, hits | tok/s sampled | tok/s greedy |
| --- | --- | --- | --- |
| The run engine whose experts the plugin replaced | | 24.8 | 32.5 |
| `ranked`, in the KV pool's spare bytes and idle pages | 8,070, 74% | 36.6 | 41.5 |
| `recent`, an elastic cache beside whole KV pools | 6,050, 82% | 39.5 | 49.7 |
| `recent`, in the KV pool's spare bytes and idle pages | 8,166, 87% | 45.8 | 59.5 |
| `recent`, an elastic cache beside elastic KV pools, nothing lent | 7,250, 87% | 53.5 | 60.0 |

A round of the last engine takes 40 to 56 ms: the CPU tier's misses 11 to 36 ms (110 to 150 us a missed expert),
the experts' device rows about 6 ms, and the rest is the model's own device work. A 35-token prompt's first token
takes 0.5 to 0.6 s there, most of it its missed experts staged through the ring.

With a headroom of 1 GiB that engine's least free device memory at a prompt step's peak (Core's step probe, what the
run's full-context check reads) was 1,137 MiB over the 6,316-token prompt and the ten prompts after it, and 1,162
to 1,275 MiB between requests; the long prompt had the cache give slots back once. With 0.25 GiB the
device has 330 to 430 MiB free between requests.

GGUF experts, measured on Unsloth's UD-IQ1_M Qwen3.8-Flash-Next (48 expert layers of 512, 34.8 GB stored) through
model-template with every option here at its default, one RTX 5090 beside a Ryzen 5 9600 under WSL2, 131,072 tokens of
context: the ram tier, every miss over the link (the CPU tier computes a GGUF expert in torch in 83 ms, the link
copies it in 54 us), a slot of 1.51 MB (layers 4 and 27, IQ4_NL down projections, kept whole on the device: 1.88 GiB)
and a store of 38.7 GB (48.4 GB when every slot was the widest layer's 1.97 MB), an elastic cache of 13,589 to 15,863
slots (22.3 GiB; 12,792 before); the model's weights and pools 5.8 GiB (gguf-projections 3.6, these whole layers 1.9;
the token table read in place off the device), 0.7 GiB free. Decode after a 2,000-token prompt 63.7 tok/s sampled and
65.8 greedy (no drafter), 95% and 99% of their expert reads hits; prefill run layer by layer 210, 835 and 1,377 tok/s
at 1,024, 4,096 and 16,384 prompt tokens (254, 406 and 411 with a prompt chunk's experts multiplied one slot at a
time), a cancel inside the 16,384-token prompt settled 0.49 s after it was asked. The link moved bundles at 25.9 GiB/s
at open() and 26 to 29 GB/s from the engine's own store after its first reply and after 10 s idle (PCIe 4.0 x16, as
a pinned buffer moves them), also from a store reopened from disk; the 3.5 GiB/s measured earlier came from a run
that shared the device with another process. The first 24-token prompt took 1.80 s (every routed expert staged into
a cold cache), the same prompt again 0.39 s.

Host tests check the host form against the reference bit for bit, the decision tree on given sizes and what a prompt
step's memory follows. CUDA tests (`tests/test_device.py`) check that link misses give the all-device bits for decode
rows and prompt chunks, that CPU misses and the experts copied in beside the rounds give the declared all-device
bits, that the disk tier reads its file and that recorded routes are the decode routes.
