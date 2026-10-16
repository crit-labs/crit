"""Finite FP64 host formula; no learned predictor, kernels or device work."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    keys: int
    values: int
    epsilon: float = 1e-6
    query_scale: float = 1.0
    normalize_keys: bool = True
    normalize_queries: bool = True


@dataclass(frozen=True)
class Step:
    query: tuple
    key: tuple
    value: tuple
    log_decay: float
    beta: float


@dataclass(frozen=True)
class Node:
    parent: int | None
    step: Step


@dataclass(frozen=True)
class Snapshot:
    after_tokens: int
    matrix: tuple


def checkpoint_node(prefix_after, accepted_nodes, boundary_after):
    if (type(prefix_after) is not int or type(boundary_after) is not int
            or not 0 <= prefix_after <= boundary_after
            or boundary_after - prefix_after > len(accepted_nodes)):
        raise ValueError("checkpoint must name a materialized after-token boundary")
    return None if boundary_after == prefix_after else accepted_nodes[boundary_after - prefix_after - 1]


def _finite(values):
    if any(not math.isfinite(x) or abs(x) > 1e12 for x in values):
        raise ValueError("finite bounded recurrent values required")


def _normal(values, epsilon, enabled):
    if not enabled:
        return values
    norm = math.hypot(math.sqrt(epsilon), *values)
    return tuple(x / norm for x in values)


def _advance(matrix, step, geometry):
    key = _normal(step.key, geometry.epsilon, geometry.normalize_keys)
    query = _normal(step.query, geometry.epsilon, geometry.normalize_queries)
    gated = tuple(x * math.exp(step.log_decay) for x in matrix)
    residual = tuple(step.beta * (step.value[v] - math.fsum(
        key[k] * gated[k * geometry.values + v] for k in range(geometry.keys)))
        for v in range(geometry.values))
    result = tuple(gated[k * geometry.values + v] + key[k] * residual[v]
                   for k in range(geometry.keys) for v in range(geometry.values))
    output = tuple(geometry.query_scale * math.fsum(
        query[k] * result[k * geometry.values + v] for k in range(geometry.keys))
        for v in range(geometry.values))
    _finite((*result, *output))
    return result, output


@dataclass(frozen=True)
class VerifiedChain:
    initial: Snapshot
    geometry: Geometry
    nodes: tuple
    path: tuple
    outputs: tuple

    def recover(self, accepted):
        if type(accepted) is not int or not 0 <= accepted <= len(self.path):
            raise ValueError("accepted count is outside the verified path")
        matrix = self.initial.matrix
        for physical in self.path[:accepted]:
            matrix, _ = _advance(matrix, self.nodes[physical].step, self.geometry)
        return Snapshot(self.initial.after_tokens + accepted, matrix)


def verify_chain(matrix, nodes, path, geometry, *, prefix_after=0):
    if (type(geometry.keys) is not int or type(geometry.values) is not int
            or not 0 < geometry.keys * geometry.values <= 65536
            or geometry.keys <= 0 or geometry.values <= 0
            or not math.isfinite(geometry.epsilon) or geometry.epsilon <= 0
            or not math.isfinite(geometry.query_scale) or geometry.query_scale <= 0):
        raise ValueError("explicit bounded gated-delta geometry required")
    nodes, path, matrix = tuple(nodes), tuple(path), tuple(matrix)
    if (type(prefix_after) is not int or not 0 <= prefix_after <= 1 << 32
            or len(nodes) > 1024 or len(path) > 64
            or len(path) * geometry.values > 65536
            or prefix_after + len(path) > 1 << 32
            or len(matrix) != geometry.keys * geometry.values):
        raise ValueError("recurrent geometry, path or position bounds exceeded")
    _finite(matrix)
    frozen = []
    for physical, node in enumerate(nodes):
        if node.parent is not None and (type(node.parent) is not int or not 0 <= node.parent < physical):
            raise ValueError("physical nodes require earlier parents")
        step = Step(tuple(node.step.query), tuple(node.step.key), tuple(node.step.value),
                    node.step.log_decay, node.step.beta)
        if (len(step.query) != geometry.keys or len(step.key) != geometry.keys
                or len(step.value) != geometry.values or step.log_decay > 0
                or not 0 <= step.beta <= 1):
            raise ValueError("step geometry or gating differs")
        _finite((*step.query, *step.key, *step.value, step.log_decay, step.beta))
        frozen.append(Node(node.parent, step))
    parent = None
    working, outputs = matrix, []
    for physical in path:
        if type(physical) is not int or not 0 <= physical < len(frozen) or frozen[physical].parent != parent:
            raise ValueError("accepted physical nodes must form one rooted chain")
        working, output = _advance(working, frozen[physical].step, geometry)
        outputs.append(output)
        parent = physical
    return VerifiedChain(Snapshot(prefix_after, matrix), geometry, tuple(frozen), path, tuple(outputs))
