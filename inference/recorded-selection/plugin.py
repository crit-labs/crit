"""Graph replay chosen from measured offers in the graphs slot: recorded-selection's select_replay (replay.py)."""
from .replay import replay


def shape_of(shape):
    return replay.ExecutionShape(shape["phase"], shape["tokens"], shape["rows"], tuple(shape["identity"]))


class RecordedGraphs:
    def __init__(self, modes, allow_padding):
        self.modes = frozenset(mode.strip() for mode in modes.split(",") if mode.strip())
        self.allow_padding = allow_padding
        self.chosen = {}

    def choose(self, shape, offers, free_bytes, uses=1):
        offers = self.same_bits(shape, offers, free_bytes)
        offers = [replay.ReplayOffer(offer["name"], offer["mode"], shape_of(offer["shape"]), offer["run_ns"],
                                     offer.get("workspace_bytes", 0), offer.get("resident_bytes", 0),
                                     offer.get("setup_ns", 0), offer.get("setup_peak_bytes", 0),
                                     offer.get("ready", False)) for offer in offers]
        choice = replay.select_replay(shape_of(shape), offers, modes=self.modes, free_bytes=free_bytes,
                                      expected_uses=uses, allow_padding=self.allow_padding)
        self.chosen[(shape["phase"], shape["tokens"], shape["rows"])] = choice.offer.name
        return choice.offer.name


    def same_bits(self, shape, offers, free_bytes):
        """The offers a timing may choose among. Where the engine names each offer's arithmetic ("bits": the eager
        path, or a bucket padded to its size, which differ in rounding unless every prompt form keeps a row's bits
        whatever the padding), the arithmetic is fixed by the options, never by a time: of the offers this plugin may
        take (its modes, and memory as select_replay counts it), with allow_padding the smallest graph that holds the
        shape (one of exactly its size included), else, or where no graph fits, the offer of exactly the shape;
        then the measured times choose among the offers that compute that arithmetic. Offers that name no bits are
        all one arithmetic (a graph of exactly the eager path's shape replays its calls)."""
        if not any("bits" in offer for offer in offers):
            return offers
        size = lambda o: (o["shape"]["tokens"], o["shape"]["rows"])
        want = (shape["tokens"], shape["rows"])

        def admissible(o):
            ready = o.get("ready", False)
            memory = o.get("workspace_bytes", 0) if ready else max(o.get("setup_peak_bytes", 0),
                                                                    o.get("resident_bytes", 0) + o.get("workspace_bytes", 0))
            return o["mode"] in self.modes and memory <= free_bytes
        usable = [o for o in offers if admissible(o)]
        chosen = None
        if self.allow_padding:
            graphs = [o for o in usable if o["mode"] != "eager" and o["shape"]["tokens"] >= want[0]
                      and o["shape"]["rows"] >= want[1]]
            chosen = min(graphs, key=size) if graphs else None
        if chosen is None:
            chosen = next((o for o in usable if size(o) == want), None)
        if chosen is None:
            return offers   # nothing this plugin may take: select_replay says why
        return [o for o in usable if o.get("bits") == chosen.get("bits")]


def create(context, options, parts):
    return RecordedGraphs(options["modes"], options["allow_padding"])
