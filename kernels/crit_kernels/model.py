"""What a sweep needs to know about a checkpoint: its config.json and every tensor's name, dtype and shape, read from the
safetensors headers, or a GGUF checkpoint's tensor table (crit_core.model.inspect_gguf_model), alone (no weights are
loaded, no model code runs)."""
import json
import struct
from pathlib import Path


def loaded_matrices(path):
    """The matrix shapes an engine loaded, from a JSON file: a list of {N, K, format}, an engine's facts ({"matrices":
    [...]}, crit serve's /v1/models engine_facts) or crit run serve --check's report (logs/serve-check-attempt-*.json),
    whose draft_bytes check keeps the served engine's facts."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and "checks" in value:
        value = next((c["engine_facts"] for c in value["checks"] if isinstance(c, dict) and c.get("engine_facts")), {})
    if isinstance(value, dict):
        value = (value.get("engine_facts") or value).get("matrices")
    if not isinstance(value, list) or not value or not all(
            isinstance(m, dict) and isinstance(m.get("N"), int) and isinstance(m.get("K"), int) and m.get("format")
            for m in value):
        raise ValueError(f"{path} holds no loaded matrices: a list of {{N, K, format}}, an engine's facts with "
                         "matrices, or a serve --check report of an engine whose projections record them")
    return [{"name": f"loaded.{i}", "N": m["N"], "K": m["K"], "format": m["format"]} for i, m in enumerate(value)]


class Model:
    def __init__(self, folder, matrices=None):
        """matrices: the shapes an engine loaded (loaded_matrices), which matrices() returns instead of reading
        the checkpoint's tensors and guessing its stacks."""
        self.loaded = matrices
        self.folder = Path(folder)
        config = self.folder / "config.json"
        self.config = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        self.tensors = {}   # name -> (dtype, shape)
        for path in sorted(self.folder.glob("*.safetensors")):
            with open(path, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            for name, info in header.items():
                if name != "__metadata__":
                    self.tensors[name] = (info["dtype"], tuple(info["shape"]))
        self.gguf = not self.tensors and any(self.folder.glob("*.gguf"))
        if self.gguf:   # a GGUF checkpoint: its tensors' GGUF types and element shapes
            from crit_core.model import inspect_gguf_model
            for name, info in inspect_gguf_model(self.folder).tensors.items():
                self.tensors[name] = (info.dtype, tuple(info.shape))
        if matrices is None and not self.config and not self.tensors:
            # A folder with no checkpoint in it (such as a model's folder whose files lie in a revision subfolder)
            # would give no shapes, and a sweep or the gates would then check nothing.
            raise FileNotFoundError(f"{self.folder} holds no checkpoint (no config.json, .safetensors or .gguf files): "
                                    "give the folder that holds them")
        quant = self.config.get("quantization_config") or {}
        self.quantization = quant
        # A multimodal checkpoint keeps its language model's config under text_config or llm_config.
        self.text = self.config.get("text_config") or self.config.get("llm_config") or self.config

    def matrices(self):
        """Every linear weight as {name, N, K, format}: NVFP4 (packed E2M1 codes beside E4M3 block scales), FP8 (E4M3
        codes), BF16 or FP16, by the tensors stored for it, and EXL3 (a trellis [K / 16][N / 16][16 bits], its "bits"
        too). N is the output rows, K the inputs. Then the stacked matrices engines load (_stacked), which name their parts.
        A GGUF checkpoint's matrices in a GGUF block format are {name, N, K, format: the GGUF type} (an expert stack
        [E][N][K] with "experts" E), unstacked: GGUF names do not say which matrices an engine stacks
        (--matrices gives the engine's own). Given the shapes an engine loaded, those alone."""
        if self.loaded is not None:
            return [dict(m) for m in self.loaded]
        found = []
        if getattr(self, "gguf", False):
            from crit_core.model.gguf_blocks import TYPES
            for name, (dtype, shape) in self.tensors.items():
                if dtype in TYPES and len(shape) in (2, 3):
                    found.append({"name": name.removesuffix(".weight"), "N": shape[-2], "K": shape[-1], "format": dtype,
                                  **({"experts": shape[0]} if len(shape) == 3 else {})})
            return found
        for name, (dtype, shape) in self.tensors.items():
            if len(shape) == 3 and name.endswith(".trellis") and dtype == "I16":
                found.append({"name": name[:-8], "N": 16 * shape[1], "K": 16 * shape[0], "format": "exl3",
                              "bits": shape[2] / 16})
                continue
            if len(shape) != 2:
                continue
            if name.endswith(".weight_packed") or (name.endswith(".weight") and dtype == "U8"
                                                    and name[:-len("weight")] + "weight_scale" in self.tensors):
                base = name.rsplit(".", 1)[0]
                found.append({"name": base, "N": shape[0], "K": shape[1] * 2, "format": "nvfp4"})
            elif name.endswith(".weight") and dtype == "F8_E4M3" and "embed" not in name:   # FP8 lookup tables too
                found.append({"name": name[:-7], "N": shape[0], "K": shape[1], "format": "fp8"})
            elif name.endswith(".weight") and dtype in ("BF16", "F16") and "norm" not in name and "embed" not in name:
                found.append({"name": name[:-7], "N": shape[0], "K": shape[1], "format": dtype.lower()})
        return found + self._stacked(found)

    @staticmethod
    def _stacked(matrices):
        """Engines stack the matrices of one module that read the same input (q, k and v; gate and up) into one
        matrix of their rows together, so a sweep meets those shapes too: one {name, N, K, format, stacked} per group
        of two or more siblings with the same K and format (EXL3 trellises stay apart)."""
        groups = {}
        for m in matrices:
            parent, _, leaf = m["name"].rpartition(".")
            if parent and m["format"] != "exl3":
                groups.setdefault((parent, m["K"], m["format"]), []).append(m)
        return [{"name": f"{parent}.{'|'.join(m['name'].rpartition('.')[2] for m in group)}",
                 "N": sum(m["N"] for m in group), "K": K, "format": fmt, "stacked": [m["name"] for m in group]}
                for (parent, K, fmt), group in groups.items() if len(group) > 1]

    def get(self, *names, default=None):
        """The first of names in the text model's config."""
        for name in names:
            if name in self.text:
                return self.text[name]
        return default


__all__ = ["Model"]
