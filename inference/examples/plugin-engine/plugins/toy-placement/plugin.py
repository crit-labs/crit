"""Every toy tensor on the device."""


class Placement:
    def plan(self, tensors, budget):
        return {name: "device" for name in tensors}


def create(context, options, parts):
    return Placement()
