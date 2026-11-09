"""A folder of fixture plugins for crit_plugins' tests: KV formats, attention that reads them, a fused pair, a drafter.
Each factory appends its id to context["created"] and returns a small object whose `value(x)` the tests compare, with
every slot's device form as stubs unless its spec says "device_form": False."""
import json
from pathlib import Path

PLUGIN = '''
MEMORY = {memory}

class Plugin:
    def __init__(self, context, options, parts):
        self.options, self.parts = options, parts
        self.scale = {scale}
        self.kv = parts.get("kv_cache") if {reads_kv} else None
        if context is not None:
            context.setdefault("created", []).append("{id}")

    def value(self, x):
        base = self.kv.value(x) if self.kv is not None else x
        return base * self.scale + self.options.get("bias", 0)

    def close(self):
        if self.options.get("fail_close"):
            raise RuntimeError("{id} failed to close")
{device_form}

def create(context, options, parts):
    if options.get("fail_create"):
        raise RuntimeError("{id} failed to create")
    return Plugin(context, options, parts)


def memory(context, options):
    return MEMORY
'''

# Stubs of every slot's device form (crit_plugins.device), which assembling on a device asks of each plugin.
DEVICE_FORM = """
    def load(self, tensors, format):
        return tensors

    def matmul(self, x, matrix, out_dtype=None):
        return x

    def pool(self, *args, **kwargs):
        return None

    def forward(self, *args, **kwargs):
        return None

    def convolve(self, *args, **kwargs):
        return None

    def sample(self, logits, params):
        return []
"""

PLUGINS = {
    "kv-bf16": {"fills": ["kv_cache"], "provides": ["kv:bf16"], "memory": {"device": 2 << 30}, "scale": 1.0},
    "kv-fp8": {"fills": ["kv_cache"], "provides": ["kv:fp8"], "memory": {"device": 1 << 30}, "scale": 1.0},
    "attn-plain": {"fills": ["attention"], "needs": [["kv:bf16", "kv:fp8"]], "reads_kv": True, "scale": 2.0,
                   "options": {"block": {"type": "int", "default": 64, "min": 16, "max": 256},
                               "bias": {"type": "float", "default": 0.0},
                               "fail_create": {"type": "bool", "default": False},
                               "fail_close": {"type": "bool", "default": False}}},
    "attn-fp8": {"fills": ["attention"], "needs": ["kv:fp8"], "reads_kv": True, "scale": 2.0},
    "attn-kv-fused": {"fills": ["attention", "kv_cache"], "provides": ["kv:fp8"], "scale": 2.0,
                      "reference": ["attn-fp8", "kv-fp8"], "memory": {"device": 1 << 30}},
    "draft-eagle": {"fills": ["drafter"], "needs": ["taps:hidden"], "conflicts": ["sampler:host"], "scale": 1.0,
                    "options": {"width": {"type": "int", "default": 6, "min": 1, "max": 16}}},
    "sampler-host": {"fills": ["sampler"], "provides": ["sampler:host"], "scale": 1.0},
}


def write_plugins(root, plugins=None, tests=None):
    """Write the fixture plugins (or `plugins`: id -> spec) under root; tests: id -> {file: text}."""
    root = Path(root)
    for plugin_id, spec in (plugins or PLUGINS).items():
        folder = root / plugin_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "crit-plugin-v1", "id": plugin_id, "title": f"Fixture {plugin_id}",
                    "fills": spec["fills"], "entry": "plugin.py:create",
                    **{key: spec[key] for key in ("provides", "needs", "needs_when", "conflicts", "options", "reference",
                                                  "requires", "devices")
                       if key in spec}}
        (folder / "plugin.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (folder / "plugin.py").write_text(PLUGIN.format(id=plugin_id, memory=spec.get("memory", {}),
                                                        scale=spec.get("scale", 1.0),
                                                        reads_kv=spec.get("reads_kv", False),
                                                        device_form=DEVICE_FORM if spec.get("device_form", True) else ""),
                                            encoding="utf-8")
        for name, text in ((tests or {}).get(plugin_id) or {}).items():
            (folder / "tests").mkdir(exist_ok=True)
            (folder / "tests" / name).write_text(text, encoding="utf-8")
    return root
