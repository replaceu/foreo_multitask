# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOCAL_RANK, LOGGER, ops


def _select_task_preds(preds, task):
    """
    辅助函数:从多任务模型的预测结果字典中提取指定任务的预测值,如果预测结果是字典(通常是这种情况),则按照任务名获取,否则直接返回
    """
    if isinstance(preds, dict):
        return preds.get(task)
    return preds


def _split_pred_proto(preds):
    """
    辅助函数:用于分割预测结果(pred)和原型掩码(proto),这主要用于分割任务,因为分割任务的输出包含检测框预测和用于生成掩码的原型向量
    """
    proto = None
    pred = preds
    # 检测预测结果是否为列表或元组
    if isinstance(preds, (list, tuple)):
        # 如果第一个元素也是元组,说明结构可能是((pred, proto),...)
        if preds and isinstance(preds[0], tuple):
            pred, proto = preds[0]
        else:
            # 否则通常结构式[pred, proto]
            pred = preds[0] if preds else preds
            # 如果列表长度大于1且最后一个元素是Tensor,则认为是原型掩码
            if len(preds) > 1 and isinstance(preds[-1], torch.Tensor):
                proto = preds[-1]
    return pred, proto


class _MultiTaskDetectValidator(DetectionValidator):
    """
    多任务场景下的检测任务验证器,继承自DetectionValidator
    """
    def postprocess(self, preds):
        """
        重写后处理方法

        先从多任务的复杂输出中提取出'detect'任务的预测部分,然后再调用父类的postprocess进行标准的检测后处理(NMS等)
        """

        # 1.提取检测任务的预测
        preds = _select_task_preds(preds, "detect")

        # 2.处理可能得嵌套结构,确保格式符合父类要求
        if isinstance(preds, (list, tuple)):
            preds0 = preds[0]
            if isinstance(preds0, (list, tuple)):
                preds0 = preds0[0]
            preds = preds0
        
        # 3.调用父类DetectionValidator的后处理
        return super().postprocess(preds)


class _MultiTaskSegmentationValidator(SegmentationValidator):
    """
    多任务场景下的分割任务验证器,继承自SegmentationValidator
    """
    def postprocess(self, preds):
        """
        重写后处理方法

        需要处理掩码原型(proto)和检测框系数的结合
        """

        # 1.提取分割任务的预测
        preds = _select_task_preds(preds, "segment")
        # 2.分离预测框/系数(pred)和原型掩码(proto)
        pred, proto = _split_pred_proto(preds)
        if pred is None or proto is None:
            raise ValueError("Missing segment predictions for multitask validation.")
        
        # 3.首先使用检测验证器的方法处理边界框(如NMS)
        preds = DetectionValidator.postprocess(self, pred)
        # 4.以下逻辑用于将预测的掩码系数与原型相乘,生成最终的二进制掩码
        nm = proto.shape[1] # 原型掩码的通道数
        imgsz = [4 * x for x in proto.shape[2:]]
        mask_size = imgsz if self.process is ops.process_mask_native else [s // 4 for s in imgsz] # 计算掩码图像尺寸(通常原型图是原图的1/4大小)
        for i, pred in enumerate(preds):
            # 从预测中提取额外的掩码系数(extra)
            extra = pred.pop("extra")
            coeff = extra[:, :nm] if nm else extra[:, :0]
            # 生成最终掩码
            pred["masks"] = (
                self.process(proto[i], coeff, pred["bboxes"], shape=imgsz)
                if coeff.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
            if pred["masks"].shape[-2:] != tuple(mask_size):
                pred["masks"] = (
                    F.interpolate(pred["masks"].float().unsqueeze(0), size=mask_size, mode="nearest")[0].byte()
                    if pred["masks"].numel()
                    else pred["masks"].new_zeros((0, *mask_size))
                )
        return preds

    def _process_batch(self, preds, batch):
        """
        Ensure predicted mask sizes match prepared ground-truth mask sizes before IoU.
        """
        if preds["masks"].shape[-2:] != batch["masks"].shape[-2:]:
            target_size = batch["masks"].shape[-2:]
            preds["masks"] = (
                F.interpolate(preds["masks"].float().unsqueeze(0), size=target_size, mode="nearest")[0].byte()
                if preds["masks"].numel()
                else preds["masks"].new_zeros((0, *target_size))
            )
        return super()._process_batch(preds, batch)


class _MultiTaskPoseValidator(PoseValidator):
    """
    多任务场景下的姿态估计(关键点)验证器,继承自PoseValidator
    """
    def postprocess(self, preds):
        """
        重写后处理方法

        需要从预测结果中解析出关键点坐标
        """

        # 1.提取姿态任务的预测
        preds = _select_task_preds(preds, "pose")
        # 2.分离预测部分,姿态任务通常不需要proto,所以忽略第二个返回值
        pred, _ = _split_pred_proto(preds)
        if pred is None:
            raise ValueError("Missing pose predictions for multitask validation.")
        # 3.使用检测验证器处理边界框
        preds = DetectionValidator.postprocess(self, pred)
        # 4.解析关键点数据
        # nk:关键点总数 = 关键点个数 * 维度(通常是x,y,visibility或者x,y)
        nk = self.kpt_shape[0] * self.kpt_shape[1] if self.kpt_shape else 0
        for pred in preds:
            extra = pred.pop("extra") # 提取包含关键点信息的额外数据
            if nk:
                # 获取最后nk个值作为关键点数据
                kpts_raw = extra[:, -nk:]
                # 重塑为[N,关键点数,维度]
                pred["keypoints"] = kpts_raw.view(-1, *self.kpt_shape)
            else:
                pred["keypoints"] = extra.new_zeros((extra.shape[0], 0, 0))
        return preds


class MultiTaskValidator(DetectionValidator):
    """
    多任务验证器的主入口类,它负责协调Detect、Segment、Pose三个任务的验证过程
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        # 标记当前任务类型为多任务
        self.args.task = "multitask"

    def _get_task_loader(self, trainer, task_name, task_cfg):
        """
        获取特定任务的数据加载器(DataLoader),因为多任务训练时,不同任务的数据集可能不同
        """

        # 1.如果传入的dataloader已经是字典(包含多个加载器),直接按任务名获取
        if isinstance(self.dataloader, dict):
            loader = self.dataloader.get(task_name)
            if loader is not None:
                return loader
        # 2.如果是'detect'任务且有一个通用的dataloader,直接使用
        elif task_name == "detect" and self.dataloader is not None:
            return self.dataloader
        
        # 3.如果没有现成的dataloader,则根据配置构建新的dataset和loader
        # 确定使用哪个数据集分割(val/test)
        split = task_cfg.get(self.args.split) or task_cfg.get("val") or task_cfg.get("test")
        if not split:
            LOGGER.warning("multitask val: '%s' task has no %s split, skipping.", task_name, self.args.split)
            return None
        # 构建数据集
        dataset = trainer.build_dataset(split, mode="val", batch=trainer.args.batch, task=task_name)
        # 构建数据加载器
        return build_dataloader(
            dataset,
            batch=trainer.args.batch,
            workers=trainer.args.workers * 2,
            shuffle=False,
            rank=LOCAL_RANK,
            drop_last=False,
        )

    def __call__(self, trainer=None, model=None):
        """
        执行验证的主函数.会遍历所有任务,分别调用对应的子验证器进行评估
        """

        # 如果trainer中没有定义任务配置,回退到普通的父类验证逻辑
        if trainer is None and isinstance(self.dataloader, dict):
            task_validators = {
                "detect": _MultiTaskDetectValidator,
                "segment": _MultiTaskSegmentationValidator,
                "pose": _MultiTaskPoseValidator,
            }
            results = {}
            fitness = None
            for task_name, validator_cls in task_validators.items():
                loader = self.dataloader.get(task_name)
                if loader is None:
                    continue
                validator = validator_cls(
                    dataloader=loader,
                    save_dir=self.save_dir / task_name,
                    args=copy(self.args),
                    _callbacks=self.callbacks,
                )
                task_results = validator(trainer=None, model=model)
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

        if trainer is None or "tasks" not in getattr(trainer, "data", {}):
            return super().__call__(trainer, model)
        
        # 定义任务名称与对应验证器类的映射
        task_validators = {
            "detect": _MultiTaskDetectValidator,
            "segment": _MultiTaskSegmentationValidator,
            "pose": _MultiTaskPoseValidator,
        }
        
        # 遍历每个任务进行验证
        results = {}
        fitness = None
        for task_name, validator_cls in task_validators.items():
            # 获取该任务的配置
            task_cfg = trainer.data["tasks"].get(task_name)
            if not task_cfg:
                continue

            # 获取该任务的数据加载器
            loader = self._get_task_loader(trainer, task_name, task_cfg)
            if loader is None:
                continue

            # 实例化该任务的验证器,注意:传入了save_dir/task_name,以便将结果保存到不同子目录
            validator = validator_cls(
                dataloader=loader,
                save_dir=self.save_dir / task_name,
                # 复制参数防止相互污染
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

            # 执行验证，传入trainer以便获取模型和状态
            task_results = validator(trainer=trainer)
            if not task_results:
                continue

            # 聚合结果
            if task_name == "detect":
                # 检测任务通常作为主要指标来源
                self.metrics = validator.metrics
                fitness = task_results.get("fitness")
                results.update(task_results)
            else:
                # 其他任务的指标加上前缀(如 segment/map50)放入结果字典
                for key, value in task_results.items():
                    if key == "fitness":
                        continue
                    results[f"{task_name}/{key}"] = value
        # 如果计算出了fitness(主要基于检测任务)，也放入最终结果
        if fitness is not None:
            results["fitness"] = fitness

        return results

