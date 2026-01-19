# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

import torch

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOCAL_RANK, LOGGER, ops


def _select_task_preds(preds, task):
    if isinstance(preds, dict):
        return preds.get(task)
    return preds


def _split_pred_proto(preds):
    proto = None
    pred = preds
    if isinstance(preds, (list, tuple)):
        if preds and isinstance(preds[0], tuple):
            pred, proto = preds[0]
        else:
            pred = preds[0] if preds else preds
            if len(preds) > 1 and isinstance(preds[-1], torch.Tensor):
                proto = preds[-1]
    return pred, proto


class _MultiTaskDetectValidator(DetectionValidator):
    def postprocess(self, preds):
        preds = _select_task_preds(preds, "detect")
        if isinstance(preds, (list, tuple)):
            preds0 = preds[0]
            if isinstance(preds0, (list, tuple)):
                preds0 = preds0[0]
            preds = preds0
        return super().postprocess(preds)


class _MultiTaskSegmentationValidator(SegmentationValidator):
    def postprocess(self, preds):
        preds = _select_task_preds(preds, "segment")
        pred, proto = _split_pred_proto(preds)
        if pred is None or proto is None:
            raise ValueError("Missing segment predictions for multitask validation.")

        preds = DetectionValidator.postprocess(self, pred)
        nm = proto.shape[1]
        imgsz = [4 * x for x in proto.shape[2:]]
        for i, pred in enumerate(preds):
            extra = pred.pop("extra")
            coeff = extra[:, :nm] if nm else extra[:, :0]
            pred["masks"] = (
                self.process(proto[i], coeff, pred["bboxes"], shape=imgsz)
                if coeff.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds


class _MultiTaskPoseValidator(PoseValidator):
    def postprocess(self, preds):
        preds = _select_task_preds(preds, "pose")
        pred, _ = _split_pred_proto(preds)
        if pred is None:
            raise ValueError("Missing pose predictions for multitask validation.")

        preds = DetectionValidator.postprocess(self, pred)
        nk = self.kpt_shape[0] * self.kpt_shape[1] if self.kpt_shape else 0
        for pred in preds:
            extra = pred.pop("extra")
            if nk:
                kpts_raw = extra[:, -nk:]
                pred["keypoints"] = kpts_raw.view(-1, *self.kpt_shape)
            else:
                pred["keypoints"] = extra.new_zeros((extra.shape[0], 0, 0))
        return preds


class MultiTaskValidator(DetectionValidator):
    """Validation for multi-task models (detect/segment/pose)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "multitask"

    def _get_task_loader(self, trainer, task_name, task_cfg):
        # Accept per-task loaders provided by the trainer.
        if isinstance(self.dataloader, dict):
            loader = self.dataloader.get(task_name)
            if loader is not None:
                return loader
        elif task_name == "detect" and self.dataloader is not None:
            return self.dataloader

        split = task_cfg.get(self.args.split) or task_cfg.get("val") or task_cfg.get("test")
        if not split:
            LOGGER.warning("multitask val: '%s' task has no %s split, skipping.", task_name, self.args.split)
            return None

        dataset = trainer.build_dataset(split, mode="val", batch=trainer.args.batch, task=task_name)
        return build_dataloader(
            dataset,
            batch=trainer.args.batch,
            workers=trainer.args.workers * 2,
            shuffle=False,
            rank=LOCAL_RANK,
            drop_last=False,
        )

    def __call__(self, trainer=None, model=None):
        if trainer is None or "tasks" not in getattr(trainer, "data", {}):
            return super().__call__(trainer, model)

        task_validators = {
            "detect": _MultiTaskDetectValidator,
            "segment": _MultiTaskSegmentationValidator,
            "pose": _MultiTaskPoseValidator,
        }

        results = {}
        fitness = None
        for task_name, validator_cls in task_validators.items():
            task_cfg = trainer.data["tasks"].get(task_name)
            if not task_cfg:
                continue
            loader = self._get_task_loader(trainer, task_name, task_cfg)
            if loader is None:
                continue

            validator = validator_cls(
                dataloader=loader,
                save_dir=self.save_dir / task_name,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

            task_results = validator(trainer=trainer)
            if not task_results:
                continue

            if task_name == "detect":
                self.metrics = validator.metrics
                fitness = task_results.get("fitness")
                results.update(task_results)
            else:
                for key, value in task_results.items():
                    if key == "fitness":
                        continue
                    results[f"{task_name}/{key}"] = value

        if fitness is not None:
            results["fitness"] = fitness

        return results

