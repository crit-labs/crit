"""A leased expert cache in the expert_residency slot: expert-residency's ExpertCache (implementation.py)."""
from .residency import implementation as residency


class LeasedExpertCache:
    def __init__(self, options):
        self.nbytes, self.revision, self.encoding = options["expert_bytes"], options["revision"], options["encoding"]
        self.cache = residency.ExpertCache(options["slots"] * self.nbytes, self._release)
        self.loaded = self.released = 0

    def _load(self, key, nbytes):
        self.loaded += 1
        return ("device", key.layer, key.expert)

    def _release(self, owner):
        self.released += 1

    def plan(self, layer, experts):
        experts = list(experts)
        if len(set(experts)) != len(experts):
            raise ValueError("the experts of one plan must be distinct")
        leases, plan = [], []
        try:
            for expert in experts:
                key = residency.ExpertKey(self.revision, layer, expert, self.encoding)
                try:
                    leases.append(self.cache.acquire(key, self.nbytes, self._load))
                    plan.append(True)
                except ValueError as exc:
                    if "pinned" not in str(exc):
                        raise
                    plan.append(False)
        finally:
            for lease in leases:
                lease.close()
        return plan

    def resident(self, layer):
        return {key.expert for key in self.cache.entries if key.layer == layer}


def create(context, options, parts):
    return LeasedExpertCache(options)
