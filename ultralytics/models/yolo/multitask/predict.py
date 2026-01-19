# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops


class MultiTaskPredictor(DetectionPredictor):
    """Predictor for multi-task detect/segment/pose outputs."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "multitask"

    def _get_head_meta(self):
        if hasattr(self, "_forced_head_meta") and self._forced_head_meta is not None:
            return self._forced_head_meta
        head = getattr(self.model, "model", None)
        if hasattr(head, "model"):
            head = head.model
        if hasattr(head, "__getitem__"):
            head = head[-1]
        nm = getattr(head, "nm", 0) if head is not None else 0
        nk = getattr(head, "nk", 0) if head is not None else 0
        kpt_shape = getattr(head, "kpt_shape", None)
        return nm, nk, kpt_shape

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        proto = None
        pred = preds
        if isinstance(preds, dict):
            pred = preds.get("detect")
        if isinstance(pred, (list, tuple)):
            if isinstance(pred[0], tuple):
                pred, proto = pred[0]
            else:
                pred = pred[0]

        self._forced_head_meta = None
        if isinstance(preds, dict) and proto is None:
            self._forced_head_meta = (0, 0, None)

        if pred is None:
            raise ValueError("Missing detect predictions for multitask inference.")

        pred = nms.non_max_suppression(
            pred,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=False,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        return self.construct_results(pred, img, orig_imgs, proto)

    def construct_results(self, preds, img, orig_imgs, proto):
        return [
            self.construct_result(pred, img, orig_img, img_path, proto[i] if proto is not None else None)
            for pred, orig_img, img_path, i in zip(preds, orig_imgs, self.batch[0], range(len(preds)))
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        nm, nk, kpt_shape = self._get_head_meta()

        if pred.shape[0] == 0:
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

        mask_coeff = pred[:, 6 : 6 + nm] if nm else None
        kpts_raw = pred[:, 6 + nm :] if nk else None

        masks = None
        if proto is not None and mask_coeff is not None and mask_coeff.numel():
            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto, mask_coeff, pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto, mask_coeff, pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        else:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        keypoints = None
        if kpt_shape is not None and kpts_raw is not None and kpts_raw.numel():
            keypoints = kpts_raw.view(kpts_raw.shape[0], *kpt_shape)
            keypoints = ops.scale_coords(img.shape[2:], keypoints, orig_img.shape)

        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, keypoints=keypoints)
