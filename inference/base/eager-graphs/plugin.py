"""Eager execution in the graphs slot: the first eager offer of the exact shape."""


class EagerGraphs:
    def choose(self, shape, offers, free_bytes, uses=1):
        for offer in offers:
            offered = offer["shape"]
            if offer["mode"] == "eager" and offered["phase"] == shape["phase"] and \
                    tuple(offered["identity"]) == tuple(shape["identity"]) and \
                    (offered["tokens"], offered["rows"]) == (shape["tokens"], shape["rows"]) and \
                    offer.get("workspace_bytes", 0) <= free_bytes:
                return offer["name"]
        raise ValueError("no eager offer of this shape fits")


def create(context, options, parts):
    return EagerGraphs()
