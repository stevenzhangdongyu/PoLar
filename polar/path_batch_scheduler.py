"""Path-aware layer batching plans and cost-model simulation.

This module does not execute model kernels. It creates deterministic schedules
that can be replayed by a model runner and estimates their cost from measured
per-layer batch latencies when supplied.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class LayerBatch:
    layer: int
    request_ids: tuple[str, ...]
    phase: int


@dataclass
class ScheduleResult:
    strategy: str
    batches: list[LayerBatch]

    @property
    def layer_executions(self) -> int:
        return sum(len(batch.request_ids) for batch in self.batches)

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def max_batch_size(self) -> int:
        return max((len(batch.request_ids) for batch in self.batches), default=0)

    def metrics(self, batch_latency_ms: Callable[[int], float]) -> dict:
        latencies = [batch_latency_ms(len(batch.request_ids)) for batch in self.batches]
        duration_ms = sum(latencies)
        count = len({rid for batch in self.batches for rid in batch.request_ids})
        sizes = [len(batch.request_ids) for batch in self.batches]
        return {
            "strategy": self.strategy,
            "requests": count,
            "layer_executions": self.layer_executions,
            "layer_batch_invocations": self.batch_count,
            "mean_batch_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max_batch_size": max(sizes, default=0),
            "estimated_duration_ms": duration_ms,
            "requests_per_cost_unit": count / duration_ms if duration_ms else 0.0,
            "layer_executions_per_cost_unit": self.layer_executions / duration_ms if duration_ms else 0.0,
            "batches": [asdict(batch) for batch in self.batches],
        }


def validate_paths(paths: Iterable[Sequence[int]], depth: int | None = None) -> list[tuple[int, ...]]:
    result = []
    for request_idx, path in enumerate(paths):
        if not isinstance(path, (list, tuple)) or not path:
            raise ValueError(f"path {request_idx} must be a non-empty array")
        if any(type(layer) is not int for layer in path):
            raise ValueError(f"path {request_idx} must contain integer layer indices")
        if depth is not None and any(layer < 0 or layer >= depth for layer in path):
            raise ValueError(f"path {request_idx} contains a layer outside [0, {depth})")
        result.append(tuple(path))
    if not result:
        raise ValueError("no paths supplied")
    return result


def serial_schedule(paths: Sequence[tuple[int, ...]]) -> ScheduleResult:
    batches = []
    for idx, path in enumerate(paths):
        rid = str(idx)
        for phase, layer in enumerate(path):
            batches.append(LayerBatch(layer, (rid,), phase))
    return ScheduleResult("serial", batches)


def identical_path_schedule(paths: Sequence[tuple[int, ...]]) -> ScheduleResult:
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for idx, path in enumerate(paths):
        groups[path].append(str(idx))
    batches = []
    # Deterministic first-seen ordering avoids changing schedules across runs.
    for path, request_ids in groups.items():
        for phase, layer in enumerate(path):
            batches.append(LayerBatch(layer, tuple(request_ids), phase))
    return ScheduleResult("identical_path_batch", batches)


def dynamic_layer_schedule(paths: Sequence[tuple[int, ...]]) -> ScheduleResult:
    """Greedily batch all currently-ready requests needing the same layer.

    Requests advance independently. A request's next layer becomes ready only
    after its previous scheduled layer batch has completed.
    """
    positions = [0] * len(paths)
    batches = []
    while True:
        ready: dict[int, list[int]] = defaultdict(list)
        for request_idx, path in enumerate(paths):
            pos = positions[request_idx]
            if pos < len(path):
                ready[path[pos]].append(request_idx)
        if not ready:
            break
        # Largest ready batch first; layer index is a stable tie-breaker.
        layer = min(ready, key=lambda item: (-len(ready[item]), item))
        request_indices = ready[layer]
        phase = max(positions[idx] for idx in request_indices)
        batches.append(LayerBatch(layer, tuple(str(idx) for idx in request_indices), phase))
        for idx in request_indices:
            positions[idx] += 1
    return ScheduleResult("dynamic_layer_batch", batches)


def build_schedules(paths: Sequence[tuple[int, ...]]) -> list[ScheduleResult]:
    return [
        serial_schedule(paths),
        identical_path_schedule(paths),
        dynamic_layer_schedule(paths),
    ]
