"""What a sweep needs to know about a checkpoint: its config.json and every tensor's name, dtype and shape, read from the
safetensors headers alone (no weights are loaded, no model code runs)."""
import json
import struct
from pathlib import Path


class Model:
    def __init__(self, folder):
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
        quant = self.config.get("quantization_config") or {}
        self.quantization = quant
        self.text = self.config.get("text_config") or self.config

    def matrices(self):
        """Every linear weight as {name, N, K, format}: NVFP4 (packed E2M1 codes beside E4M3 block scales), FP8 (E4M3
        codes), BF16 or FP16, by the tensors stored for it. N is the output rows, K the inputs."""
        found = []
        for name, (dtype, shape) in self.tensors.items():
            if len(shape) != 2:
                continue
            if name.endswith(".weight_packed") or (name.endswith(".weight") and dtype == "U8"
                                                    and name[:-len("weight")] + "weight_scale" in self.tensors):
                base = name.rsplit(".", 1)[0]
                found.append({"name": base, "N": shape[0], "K": shape[1] * 2, "format": "nvfp4"})
            elif name.endswith(".weight") and dtype == "F8_E4M3":
                found.append({"name": name[:-7], "N": shape[0], "K": shape[1], "format": "fp8"})
            elif name.endswith(".weight") and dtype in ("BF16", "F16") and "norm" not in name and "embed" not in name:
                found.append({"name": name[:-7], "N": shape[0], "K": shape[1], "format": dtype.lower()})
        return found

    def get(self, *names, default=None):
        """The first of names in the text model's config."""
        for name in names:
            if name in self.text:
                return self.text[name]
        return default


__all__ = ["Model"]
