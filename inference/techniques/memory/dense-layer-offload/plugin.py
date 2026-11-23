"""dense-layer-offload in the placement slot: whole decoder layers on the device or in host memory, by its planner
(code/planner.py) over measured rates, or the leading layers first without them."""
import re

from .offload import planner

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_of(name):
    found = LAYER.search(name)
    return int(found[1]) if found else None


class LayerOffload:
    def __init__(self, options):
        self.rates = {"device_gbs": options["device_gbs"], "upload_gbs": options["upload_gbs"],
                      "host_gbs": options["host_gbs"], "host_layer_ns": options["host_layer_ns"]}

    def layers(self, tensors, budget):
        """{layer: device, stream or host} and the device bytes left for the layers."""
        if not isinstance(tensors, dict) or not all(type(v) is int and v >= 0 for v in tensors.values()):
            raise ValueError("tensors maps names to their bytes")
        free = budget.get("device")
        if type(free) is not int:
            raise ValueError("the budget names the device's free bytes")
        fixed = sum(size for name, size in tensors.items() if layer_of(name) is None)
        if fixed > free:
            raise MemoryError(f"the tensors outside the layers need {fixed / 2**30:.2f} GiB on the device, "
                              f"{free / 2**30:.2f} GiB is free")
        sizes = {}
        for name, size in tensors.items():
            layer = layer_of(name)
            if layer is not None:
                sizes[layer] = sizes.get(layer, 0) + size
        order = sorted(sizes)
        room = free - fixed
        if not order:
            return {}, room
        if all(self.rates[key] > 0 for key in ("device_gbs", "upload_gbs", "host_gbs")):
            _, places, _, _ = planner.candidates([sizes[layer] for layer in order], room, self.rates)[0]
            return dict(zip(order, places)), room
        places, used = {}, 0
        for layer in order:   # unmeasured: the leading layers stay resident while they fit
            fits = used + sizes[layer] <= room
            places[layer], used = (planner.DEVICE, used + sizes[layer]) if fits else (planner.HOST, used)
        return places, room

    def plan(self, tensors, budget):
        places, _ = self.layers(tensors, budget)
        host = budget.get("host")
        offloaded = sum(size for name, size in tensors.items()
                        if layer_of(name) is not None and places[layer_of(name)] != planner.DEVICE)
        if type(host) is int and offloaded > host:
            raise MemoryError(f"the offloaded layers need {offloaded / 2**30:.2f} GiB of host memory, "
                              f"{host / 2**30:.2f} GiB is free")
        return {name: "device" if layer_of(name) is None or places[layer_of(name)] == planner.DEVICE else "host"
                for name in tensors}


def create(context, options, parts):
    return LayerOffload(options)
