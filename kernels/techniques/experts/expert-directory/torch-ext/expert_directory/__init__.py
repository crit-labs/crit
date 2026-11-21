"""device-expert-cache's directory on the device (crit's kernel expert-directory, which crit's device-expert-cache
plugin loads for its device form, built on this GPU): the plan of one routed layer over a step's routes (hits pinned,
misses admitted by the clock or the recent policy, or sent to the CPU tier; a prompt chunk's layer into ring slots)
and the fills that copy the admitted bundles from the device-readable store into the slots, by SM loads; fill_program
rebuilds them from a store in another layout (CSF scales). One launch each, capture-safe, no host synchronization; the
rule is device_cache.LayerDirectory's, bit for bit. An empty tensor stands for an absent buffer (directory_ops.cu).

    plan(key_slot, slot_key, slot_ref, slot_pin, state, counters, seen, route_counts, quota, admit_quota, layer,
         experts, ids, top, rows, row_counters, ring, out, jobs, job_count, slot_last, victims, recent, pending,
         range, link)
    fill(source, store, jobs, job_count, blocks, vector, job_first)
    fill_program(source, store, jobs, job_count, program, shared, blocks)
"""
from ._ops import ops

__all__ = ["fill", "fill_program", "plan"]


def plan(key_slot, slot_key, slot_ref, slot_pin, state, counters, seen, route_counts, quota, admit_quota, layer,
         experts, ids, top, rows, row_counters, ring, out, jobs, job_count, slot_last, victims, recent, pending, range_,
         link):
    ops.plan(key_slot, slot_key, slot_ref, slot_pin, state, counters, seen, route_counts, quota, int(admit_quota),
             int(layer), int(experts), ids, int(top), rows, row_counters, ring, out, jobs, job_count, slot_last,
             victims, recent, pending, range_, bool(link))


def fill(source, store, jobs, job_count, blocks, vector, job_first):
    ops.fill(source, store, jobs, job_count, int(blocks), int(vector), job_first)


def fill_program(source, store, jobs, job_count, program, shared, blocks):
    ops.fill_program(source, store, jobs, job_count, program, int(shared), int(blocks))
