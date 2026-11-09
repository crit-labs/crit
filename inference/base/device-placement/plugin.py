"""The plain placement in the placement slot: every tensor on the device, or a refusal before anything loads.

`reserve_mib` of the device's free memory stays out of the plan: what no tensor of the plan holds but the process
still takes (the CUDA context, library workspaces, graph pools, the allocator's rounding) and, on a GPU that also draws
a desktop, what the desktop takes. Under the Windows display driver (native Windows, WSL) a plan that fills the device
to its last byte does not fail: the driver pages device memory to host memory and every kernel slows down.

`off_device` names groups that live off the device (a model whose routed experts or lookup tables are larger than the
device: they reach it through a device expert cache and a host table, device-expert-cache and lookup-table-rows): the
name prefixes, comma-separated and in order of preference ("experts.,tables.": routed experts' bundles first, then
lookup tables). The groups of one prefix live in one place (a device expert cache reads one store): host memory in the
prefixes' order while the budget's host bytes hold all of them, else files (or a refusal with files false); every
other group stays on the device and must fit. slots() sizes the device expert cache: the bundles that fill the device
memory the device groups leave, less the cache's scratch slot."""


class DevicePlacement:
    def __init__(self, reserve=0, off_device="", files=True):
        self.reserve = reserve
        self.prefixes = [p.strip() for p in off_device.split(",") if p.strip()]
        self.files = files

    def _off(self, name):
        """The rank of the first prefix `name` starts with (its preference), or None for a device group."""
        return next((i for i, p in enumerate(self.prefixes) if name.startswith(p)), None)

    @staticmethod
    def _check(tensors, budget):
        if not isinstance(tensors, dict) or not all(type(v) is int and v >= 0 for v in tensors.values()):
            raise ValueError("tensors maps names to their bytes")
        if type(budget.get("device")) is not int:
            raise MemoryError("the budget names the device's free bytes")

    def plan(self, tensors, budget):
        self._check(tensors, budget)
        free = budget["device"]
        need = sum(size for name, size in tensors.items() if self._off(name) is None)
        if need + self.reserve > free:
            kept = f" and {self.reserve / 2**30:.2f} GiB kept free (reserve_mib)" if self.reserve else ""
            raise MemoryError(f"the tensors need {need / 2**30:.2f} GiB on the device{kept}, "
                              f"{free / 2**30:.2f} GiB is free")
        places = {name: "device" for name in tensors if self._off(name) is None}
        groups = {}
        for name in tensors:
            rank = self._off(name)
            if rank is not None:
                groups.setdefault(rank, []).append(name)
        if groups and type(budget.get("host")) is not int:
            raise MemoryError("the budget names no host bytes, and groups live off the device (off_device)")
        room = budget.get("host", 0)
        for rank in sorted(groups):
            need = sum(tensors[name] for name in groups[rank])
            if need <= room:
                place, room = "host", room - need
            elif self.files:
                place = "files"
            else:
                raise MemoryError(f"the {self.prefixes[rank]}* groups need {need / 2**30:.2f} GiB of host memory, "
                                  f"{room / 2**30:.2f} GiB is left and files are off")
            places.update(dict.fromkeys(groups[rank], place))
        return {name: places[name] for name in tensors}

    def slots(self, tensors, budget, bundle_bytes):
        """The device expert cache's own slots: the bundles of `bundle_bytes` that fill the device bytes the device
        groups of `tensors` and the reserve leave, less one for the cache's scratch slot (a group named expert_slots, if
        given, is not counted)."""
        self._check(tensors, budget)
        if type(bundle_bytes) is not int or bundle_bytes < 1:
            raise ValueError("a bundle has bytes")
        need = sum(size for name, size in tensors.items() if self._off(name) is None and name != "expert_slots")
        return max(0, (budget["device"] - self.reserve - need) // bundle_bytes - 1)


def create(context, options, parts):
    return DevicePlacement(int(options["reserve_mib"]) << 20, options["off_device"], options["files"])
