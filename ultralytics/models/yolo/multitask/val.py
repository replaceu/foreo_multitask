# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator


class MultiTaskValidator(DetectionValidator):
    """Validation for multi-task models (box metrics only)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "multitask"
