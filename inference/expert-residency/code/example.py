"""Run a host POC: bank sizing, exact staging, complete publication and read-only reuse."""
from concurrent.futures import Future

from implementation import BundleComponent, BundleDirectory, BundleReceipt, ExpertKey
from planning import BankComponent, LayerExperts, cache_admission, copy_runs, observe_cache, size_banks


def main():
    components = (BankComponent("expand", "demo-u8-v1", (2,), 2, alignment=4, guard_bytes=4),
                  BankComponent("contract", "demo-u8-v1", (1,), 1, alignment=4))
    bank = size_banks((LayerExperts("model-r1", 0, 4, components, minimum_slots=2),), 12).banks[0]
    assert bank.slots == 2 and bank.nbytes == 12
    backing = bytearray(bank.nbytes)
    storage = {span.component.name: memoryview(backing)[span.offset:span.offset+span.nbytes] for span in bank.spans}
    directory = BundleDirectory(bank.slots,
        tuple(BundleComponent(c.name, c.encoding, c.expert_bytes) for c in components),
        (backing,))
    stock = {"expand": bytes((1, 2, 3, 4, 5, 6, 7, 8)), "contract": bytes((10, 20, 30, 40))}
    key = lambda expert: ExpertKey("model-r1", 0, expert, "demo-bundle-v1")
    admission = cache_admission("decode", 1, 2, 4, bank.slots)
    routes = (1, 2)  # occurrence identity/weight belongs to the projection
    tickets = {expert: directory.begin_upload(key(expert)) for expert in sorted(set(routes))}
    completions, uploaded, calls = {}, 0, 0
    for component in components:
        pairs = [(expert, ticket.slot) for expert, ticket in tickets.items()]
        for run in copy_runs(pairs, component, expert_count=4, slots=bank.slots):
            completion = Future()  # one fresh receipt covers this physical batch
            for expert in range(run.first_expert, run.first_expert + run.count):
                ticket = tickets[expert]
                directory.submit(ticket, component.name, stock[component.name],
                    lambda t, c, owner, f=completion: BundleReceipt(t, c.name, c.encoding, c.nbytes, f))
            start, end = run.source_offset, run.source_offset + run.nbytes
            storage[component.name][run.destination_offset:run.destination_offset + run.nbytes] = stock[component.name][start:end]
            completions[(component.name, run.first_expert)] = completion
            uploaded += run.nbytes
            calls += 1
    assert not directory.ready_slots()  # submitting does not publish READY
    for completion in completions.values():
        completion.set_result(None)
    for ticket in tickets.values():
        directory.publish(ticket)
    observation = observe_cache(admission, routes, (), uploaded_bytes=uploaded, transfer_calls=calls)
    assert observation.route_misses == 2 and observation.unique_misses == 2

    def exact_read(expert, policy):
        use = (None if policy.mode == "bypass" else
               directory.acquire(key(expert), touch=policy.mode == "populate"))
        if use is None:
            return tuple(stock[c.name][expert*c.expert_bytes:(expert+1)*c.expert_bytes] for c in components)
        value = tuple(bytes(storage[c.name][use.slot*c.expert_bytes:(use.slot+1)*c.expert_bytes]) for c in components)
        finished = Future(); finished.set_result(None)
        directory.settle(use, finished)
        return value

    ages = tuple(slot.tick for slot in directory.slots)
    prefill = cache_admission("prefill", 64, 2, 4, bank.slots)
    for expert in (0, 1, 2, 3, 1):
        expected = tuple(stock[c.name][expert*c.expert_bytes:(expert+1)*c.expert_bytes] for c in components)
        assert exact_read(expert, prefill) == expected
    assert tuple(slot.tick for slot in directory.slots) == ages
    assert {k.expert for k in directory.ready_slots()} == {1, 2}
    directory.close()
    print("Expert cache POC passed: bounded banks, complete coalesced uploads, exact fallback and read-only prefill.")


if __name__ == "__main__":
    main()
