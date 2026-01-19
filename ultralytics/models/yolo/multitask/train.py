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
    """多任务训练器类,用于同时训练目标检测、实例分割和姿态估计任务。继承自DetectionTrainer."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化训练器,调用父类构造函数"""
        if overrides is None:
            overrides = {}
        # Keep task consistent with multitask heads and datasets.
        overrides.setdefault("task", "multitask")
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self) -> dict[str, Any]:

        """
        加载并处理多任务数据集配置文件(data.yaml)

        这个方法不仅读取路径,还会检验是否包含了'task'字段,并解析detect/segment/pose三个任务各自的train/cal/test路径
        """

        #加载yaml文件
        data = YAML.load(self.args.data, append_filename=True)
        # 核心检查：多任务配置必须包含'task'键
        if "tasks" not in data:
            raise RuntimeError("multitask data.yaml must define a 'tasks' mapping.")
        # 设置默认通道数为3
        data["channels"] = data.get("channels", 3)
        # 解析数据集的根目录路径
        base = Path(data.get("path") or Path(data.get("yaml_file", "")).parent)
        if not base.is_absolute():
            base = (DATASETS_DIR / base).resolve()
        data["path"] = base

        # 遍历处理三个任务,检测、分割、姿态估计
        tasks = data["tasks"]
        for task_name in ("detect", "segment", "pose"):
            if task_name not in tasks:
                raise RuntimeError(f"multitask data.yaml missing '{task_name}' task definition.")
            task_cfg = tasks[task_name]
            # 解析并转换该任务下train,val,test的绝对路径
            for split in ("train", "val", "test"):
                if task_cfg.get(split):
                    path = Path(task_cfg[split])
                    if not path.is_absolute():
                        path = (base / path).resolve()
                    task_cfg[split] = str(path)
            # 如果任务没定义'cls'(类别映射),默认为0
            if "cls" not in task_cfg:
                task_cfg["cls"] = 0
        data["tasks"] = tasks

        # Prefer pose kpt_shape if the top-level config does not define one.
        pose_cfg = tasks.get("pose")
        if pose_cfg and pose_cfg.get("kpt_shape") and not data.get("kpt_shape"):
            data["kpt_shape"] = pose_cfg["kpt_shape"]

        # 设置类别名称和数量,以Detect任务为基准
        detect_cfg = tasks["detect"]
        detect_nc = detect_cfg.get("nc")
        # 优先使用全局names,否则使用detect任务的names
        names = data.get("names") or detect_cfg.get("names")

        # 如果没有定义名称,自动生成数字索引名称['0','1',...]
        if names is None:
            names = [str(i) for i in range(int(detect_nc or 1))]
        data["names"] = check_class_names(names)
        # 确定类别数量nc
        if detect_nc is None:
            data["nc"] = len(data["names"])
        else:
            data["nc"] = int(detect_nc)
            # 检验names长度与nc是否一致
            if len(data["names"]) != data["nc"]:
                LOGGER.warning("multitask data.yaml names length does not match detect nc, using detect task names.")
                names = detect_cfg.get("names") or [str(i) for i in range(data["nc"])]
                data["names"] = check_class_names(names)
        # 设置主训练和验证路径(默认使用detect任务作为主路径)
        data["train"] = tasks["detect"]["train"]
        data["val"] = tasks["detect"].get("val") or tasks["detect"].get("test")

        # 处理单类别训练模式(Single Class Mode)
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
            # 将所有任务的类别索引强制设为0
            for task_cfg in data["tasks"].values():
                task_cfg["cls"] = 0

        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None, task: str = "detect"):
        """
        构建特定任务的YOLO数据集

        Args:
            task (str):当前构建的是哪个任务的数据集(detect/segment/pose)

        """

        # 计算网络步长(grid size),最小为32
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        # 复制参数并临时修改当前任务类型,以便build_yolo_dataset知道如何处理标签
        task_args = copy(self.args)
        task_args.task = task
        return build_yolo_dataset(task_args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        构建数据加载器(DataLoader)

        关键点:训练模式下会返回一个MultiTaskLoader,包含所有任务的数据流
        """

        assert mode in {"train", "val"}

        # 训练模式:构建多任务混合加载器
        if mode == "train":
            loaders = {}     # 存储每个任务的DataLoader
            weights = {}     # 存储每个任务的权重(基于数据集大小)
            cls_offsets = {} # 类别偏移量

            # 遍历所有任务(detect,segment,pose)
            for task_name, task_cfg in self.data["tasks"].items():
                img_path = task_cfg["train"]
                # 为该任务构建数据集
                dataset = self.build_dataset(img_path, mode=mode, batch=batch_size, task=task_name)
                # 构建该任务的数据加载器
                loader = build_dataloader(
                    dataset,
                    batch=batch_size,
                    workers=self.args.workers,
                    shuffle=True,
                    rank=rank,
                    drop_last=self.args.compile,
                )
                loaders[task_name] = loader
                weights[task_name] = len(dataset) # 权重通常取决于数据量大小
                cls_offsets[task_name] = 0
            # 返回自定义的MultiTaskLoader,它负责在训练循环中从不同任务的loader中采样(抽取数据)
            return MultiTaskLoader(loaders, weights, cls_offsets)
        
        # 验证模式:仅使用Detect数据集
        # 目前看来验证阶段主要评估检测性能,或者使用检测数据集进行基础验证
        # Validation uses per-task loaders so each head is evaluated on its own data.
        loaders = {}
        for task_name, task_cfg in self.data["tasks"].items():
            img_path = task_cfg.get(self.args.split) or task_cfg.get("val") or task_cfg.get("test")
            if not img_path:
                LOGGER.warning("multitask val: '%s' task has no %s split, skipping.", task_name, self.args.split)
                continue
            dataset = self.build_dataset(img_path, mode=mode, batch=batch_size, task=task_name)
            loaders[task_name] = build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers * 2,
                shuffle=False,
                rank=rank,
                drop_last=False,
            )
        return loaders

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        初始化多任务模型 (MultiTaskModel)
        """

        # 创建MultiTaskModel实例
        model = MultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            # 传入姿态估计所需的关键点形状信息(kpt_shape)
            data_kpt_shape=self.data.get("kpt_shape", (None, None)),
            verbose=verbose and RANK == -1,
        )
        # 如果有预训练权重,则加载
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """设置模型的属性，如类别数和类别名称"""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_validator(self):
        """
        返回多任务验证器 (MultiTaskValidator)。
        """
        from .val import MultiTaskValidator

        self.loss_names = {
            "det": ["box_loss", "cls_loss", "dfl_loss"],
            "seg": ["Tv_loss", "FL_loss"],
            "pose": ["pose_loss", "kobj_loss"],
        }
        return MultiTaskValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _flatten_loss_names(self):
        loss_names = []
        for task_name in ("det", "seg", "pose"):
            loss_names.extend(self.loss_names.get(task_name, []))
        return loss_names

    def _loss_items_to_list(self, loss_items):
        values = loss_items.tolist() if hasattr(loss_items, "tolist") else list(loss_items)
        det = (values + [0.0] * 3)[:3]
        # v8MultiTaskLoss provides a single seg loss; pad if more seg names are configured.
        seg_values = values[3:4]
        seg = seg_values + [0.0] * max(0, len(self.loss_names.get("seg", [])) - len(seg_values))
        pose_values = values[4:6]
        pose = pose_values + [0.0] * max(0, len(self.loss_names.get("pose", [])) - len(pose_values))
        return det + seg + pose

    def label_loss_items(self, loss_items=None, prefix="train"):
        loss_names = self._flatten_loss_names()
        keys = [f"{prefix}/{x}" for x in loss_names]
        if loss_items is None:
            return keys
        loss_values = [round(float(x), 5) for x in self._loss_items_to_list(loss_items)]
        return dict(zip(keys, loss_values))

    def progress_string(self):
        loss_names = self._flatten_loss_names()
        return ("\n" + "%11s" * (4 + len(loss_names))) % ("Epoch", "GPU_mem", *loss_names, "Instances", "Size")

    def plot_training_labels(self):
        """
        绘制训练标签可视化图。
        此处直接 return,意味着多任务训练时跳过了默认的标签可视化步骤(可能是因为标签类型混合太复杂，难以统一绘制)
        """
        return
