# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random


class _MultiTaskDatasetInfo:
    """Lightweight dataset wrapper for aggregate length reporting."""

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets.values())


class _SamplerProxy:
    """Proxy sampler to propagate set_epoch in distributed training."""

    def __init__(self, loaders: dict):
        self.loaders = loaders

    def set_epoch(self, epoch: int):
        for loader in self.loaders.values():
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)


class MultiTaskLoader:
    """Round-robin weighted loader that yields one task batch per step."""

    def __init__(self, loaders: dict, weights: dict, cls_offsets: dict):
        self.loaders = loaders
        self.cls_offsets = cls_offsets
        self.task_names = list(loaders.keys())
        self.weights = [max(1, int(weights.get(k, 1))) for k in self.task_names]
        self.iters = {k: iter(loaders[k]) for k in self.task_names}
        self.sampler = _SamplerProxy(loaders)
        self.num_batches = sum(len(loader) for loader in loaders.values())
        self.dataset = _MultiTaskDatasetInfo({k: l.dataset for k, l in loaders.items()})
        self.num_workers = max((getattr(l, "num_workers", 0) for l in loaders.values()), default=0)

    def __len__(self) -> int:
        return self.num_batches

    def reset(self):
        for loader in self.loaders.values():
            if hasattr(loader, "reset"):
                loader.reset()
        self.iters = {k: iter(self.loaders[k]) for k in self.task_names}

    def __iter__(self):
        for _ in range(self.num_batches):
            task = random.choices(self.task_names, weights=self.weights, k=1)[0]
            batch = self._next(task)
            offset = self.cls_offsets.get(task, 0)
            if offset:
                batch["cls"] = batch["cls"] + offset
            batch["task"] = task
            yield batch

    def _next(self, task: str):
        loader = self.loaders[task]
        it = self.iters[task]
        try:
            return next(it)
        except StopIteration:
            if hasattr(loader, "reset"):
                loader.reset()
            it = iter(loader)
            self.iters[task] = it
            return next(it)
