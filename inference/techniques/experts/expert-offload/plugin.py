"""expert-offload in the experts slot: routed experts of any size against the device, placed and computed by a decision
tree instead of an engine's wiring (code/offload.py). Host form: the reference block over the expert_source's
matrices. Device form: begin(), load() per expert, open() after the model's pools, forward() per routed layer."""
from importlib import import_module
from pathlib import Path

from .offload import offload

HERE = Path(__file__).resolve().parent


def create(context, options, parts):
    from . import arithmetic, cache, dispatch, grouped, prefill, quantized, tier
    from .offload import gguf_experts
    code = {"cache": import_module(f"{cache.__name__}.expert_cache"),
            "store": import_module(f"{cache.__name__}.expert_store"),
            "dispatch": import_module(f"{dispatch.__name__}.device_dispatch"),
            "cpu_tier": import_module(f"{tier.__name__}.cpu_tier"),
            "tier_bundles": import_module(f"{tier.__name__}.bundles"),
            "grouped_bundles": import_module(f"{grouped.__name__}.bundles"),
            "reference": import_module(f"{prefill.__name__}.reference"),
            "gguf": gguf_experts, "arithmetic": arithmetic, "quantized": Path(quantized.__path__[0]), "folder": HERE}
    return offload.ExpertOffload(parts["expert_source"], parts.get("placement"), options, code, context)
