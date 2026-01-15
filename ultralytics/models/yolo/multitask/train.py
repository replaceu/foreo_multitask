# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.multitask import MultiTaskLoader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER, RANK, YAML


class MultiTaskTrainer(DetectionTrainer):
    """Trainer for a multi-task detect/segment/pose model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self) -> dict[str, Any]:
        data = YAML.load(self.args.data, append_filename=True)
        if "tasks" not in data:
            raise RuntimeError("multitask data.yaml must define a 'tasks' mapping.")

        if "names" not in data:
            raise RuntimeError("multitask data.yaml must define 'names'.")
        data["names"] = check_class_names(data["names"])
        data["nc"] = len(data["names"])
        data["channels"] = data.get("channels", 3)

        base = Path(data.get("path") or Path(data.get("yaml_file", "")).parent)
        if not base.is_absolute():
            base = (DATASETS_DIR / base).resolve()
        data["path"] = base

        tasks = data["tasks"]
        for task_name in ("detect", "segment", "pose"):
            if task_name not in tasks:
                raise RuntimeError(f"multitask data.yaml missing '{task_name}' task definition.")
            task_cfg = tasks[task_name]
            for split in ("train", "val", "test"):
                if task_cfg.get(split):
                    path = Path(task_cfg[split])
                    if not path.is_absolute():
                        path = (base / path).resolve()
                    task_cfg[split] = str(path)
            if "cls" not in task_cfg:
                task_cfg["cls"] = 0
        data["tasks"] = tasks

        data["train"] = tasks["detect"]["train"]
        data["val"] = tasks["detect"].get("val") or tasks["detect"].get("test")

        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
            for task_cfg in data["tasks"].values():
                task_cfg["cls"] = 0

        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None, task: str = "detect"):
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        task_args = copy(self.args)
        task_args.task = task
        return build_yolo_dataset(task_args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}
        if mode == "train":
            loaders = {}
            weights = {}
            cls_offsets = {}
            for task_name, task_cfg in self.data["tasks"].items():
                img_path = task_cfg["train"]
                dataset = self.build_dataset(img_path, mode=mode, batch=batch_size, task=task_name)
                loader = build_dataloader(
                    dataset,
                    batch=batch_size,
                    workers=self.args.workers,
                    shuffle=True,
                    rank=rank,
                    drop_last=self.args.compile,
                )
                loaders[task_name] = loader
                weights[task_name] = len(dataset)
                cls_offsets[task_name] = int(task_cfg.get("cls", 0))
            return MultiTaskLoader(loaders, weights, cls_offsets)

        detect_cfg = self.data["tasks"]["detect"]
        img_path = detect_cfg.get("val") or detect_cfg.get("test")
        dataset = self.build_dataset(img_path, mode=mode, batch=batch_size, task="detect")
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers * 2,
            shuffle=False,
            rank=rank,
            drop_last=False,
        )

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = MultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data.get("kpt_shape", (None, None)),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_validator(self):
        from .val import MultiTaskValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "seg_loss", "pose_loss", "kobj_loss"
        return MultiTaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_labels(self):
        return
