"""Small engines of a model plugin, for the tools and tests that load one over a tiny checkpoint (crit models diff,
crit_models.testing): one definition of their options and of the plain plugins of the routed experts' slots."""
# The options of a small engine: a short context, a small KV margin and prompt chunk, a small activation plan, four
# lanes.
SMALL_OPTIONS = {"context": 1024, "kv_margin": 256, "prefill_chunk": 64, "activations_mib": 64, "concurrency": 4}
# The plain plugins of the routed experts' slots.
ROUTED = {"router": "topk-router", "experts": "experts-reference", "expert_source": "expert-tensors"}
