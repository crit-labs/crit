"""Host model of the layer placement planner (offload_plan.h).

A layer is resident on the device, streamed through one of two device staging slots, or executed on the host. With
measured rates (GB/s = bytes per ns): a resident layer costs bytes / device_gbs; a streamed layer costs
max(bytes / device_gbs, bytes / upload_gbs), because the next streamed layer uploads while the layers before it
compute; a run of consecutive host layers costs its bytes / host_gbs plus one host_layer_ns for
moving activations down and back. Plans: all resident when it fits; host execution of what does not fit; streaming of
it with two slots of the largest streamed layer taken from the budget. Resident layers are chosen by time saved per
byte. The fastest prediction ranks the candidates; engines time the leading ones and keep the measured fastest.
"""

DEVICE, STREAM, HOST = "device", "stream", "host"


def predict(sizes, places, m):
    ns = 0.0
    for i, (b, p) in enumerate(zip(sizes, places)):
        if p == DEVICE:
            ns += b / m["device_gbs"]
        elif p == STREAM:
            ns += max(b / m["device_gbs"], b / m["upload_gbs"])
        else:
            ns += b / m["host_gbs"] + (m.get("host_layer_ns", 0.0) if i == 0 or places[i - 1] != HOST else 0.0)
    return ns


def _fill(sizes, budget, other, m):
    def saved(i):
        return (predict([sizes[i]], [other], m) - predict([sizes[i]], [DEVICE], m)) / sizes[i]
    places, used = [other] * len(sizes), 0
    for i in sorted(range(len(sizes)), key=saved, reverse=True):
        if used + sizes[i] <= budget:
            places[i], used = DEVICE, used + sizes[i]
    return places


def candidates(sizes, budget, m):
    """[(name, places, device_bytes, predicted_ns)], fastest prediction first."""
    if not sizes or min(m["device_gbs"], m["upload_gbs"], m["host_gbs"]) <= 0:
        raise ValueError("planning needs layers and positive measured rates")
    total, plans = sum(sizes), []
    if total <= budget:
        places = [DEVICE] * len(sizes)
        plans.append(("resident", places, total, predict(sizes, places, m)))
    else:
        places = _fill(sizes, budget, HOST, m)
        resident = sum(b for b, p in zip(sizes, places) if p == DEVICE)
        plans.append(("host", places, resident, predict(sizes, places, m)))
    slot = 0
    for _ in range(64):
        if 2 * slot > budget:
            break
        places = _fill(sizes, budget - 2 * slot, STREAM, m)
        largest = max([b for b, p in zip(sizes, places) if p == STREAM], default=0)
        if largest > slot:
            slot = largest
            continue
        if largest:
            resident = sum(b for b, p in zip(sizes, places) if p == DEVICE) + 2 * largest
            plans.append(("stream", places, resident, predict(sizes, places, m)))
        break
    return sorted(plans, key=lambda plan: plan[3])
