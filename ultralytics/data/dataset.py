"""双流数据集实现。

这个文件的目标不是完整复刻官方 Ultralytics 的所有 dataset 能力，
而是先把当前仓库最核心的双流训练链补齐：

1. 读取 RGB 与深度/红外两路图像
2. 按文件名一一配对
3. 解析 YOLO 标签
4. 转成当前增强链能消费的 Instances 结构
5. 输出 trainer / validator 直接可用的 batch 字段
"""

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import img2label_paths, verify_image_label
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments


class YOLODataset(BaseDataset):
    """双流 YOLO 数据集。

    这版实现是围绕当前仓库的双流约定写的：

    - `img_path_rgb` 对应 RGB 图像目录
    - `img_path_depth` 对应深度图或红外图目录
    - 两路图像通过文件名 stem 配对
    - 标签路径仍然按 RGB 路径推导

    最终单个样本会输出给当前仓库的训练/验证链：

    - `img_rgb`
    - `img_depth`
    - `cls`
    - `bboxes`
    - `masks` / `keypoints`（按任务决定）
    - `batch_idx`
    """

    def __init__(
        self,
        *args,
        data=None,
        task="detect",
        img_path=None,
        img_path_rgb=None,
        img_path_depth=None,
        **kwargs,
    ):
        # 兼容两种调用方式：
        # 1. 这个仓库里的双流 builder，会显式传入 img_path_rgb / img_path_depth
        # 2. 某些保留了上游风格的调用，会只传一个 img_path 或位置参数
        if args:
            if len(args) == 1 and img_path is None and img_path_rgb is None:
                img_path = args[0]
                args = ()
            elif len(args) >= 2 and img_path_rgb is None and img_path_depth is None:
                img_path_rgb, img_path_depth = args[:2]
                args = args[2:]

        if img_path is not None:
            img_path_rgb = img_path_rgb or img_path
            img_path_depth = img_path_depth or img_path

        if img_path_rgb is None or img_path_depth is None:
            raise ValueError("YOLODataset requires both 'img_path_rgb' and 'img_path_depth'.")

        self.data = data or {}
        self.task = task
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.kpt_shape = self.data.get("kpt_shape", [0, 0])
        super().__init__(*args, img_path_rgb=img_path_rgb, img_path_depth=img_path_depth, **kwargs)

        # 这里保留一组单流别名，是为了兼容仓库里还没彻底改完的上游工具代码。
        # 例如某些统计、导出、探索工具仍会访问 im_files / ims / npy_files 这些旧字段。
        self.im_files = self.im_files_rgb
        self.npy_files = self.npy_files_rgb
        self.ims = self.ims_rgb
        self.im_hw0 = self.im_hw0_rgb
        self.im_hw = self.im_hw_rgb

    def _pair_modal_files(self):
        """按 RGB 顺序重排第二路图像。

        当前双流训练的核心前提是：相同 index 下的两张图必须是同一个样本。
        所以这里不直接假设目录遍历出来的顺序天然一致，而是显式按 stem 配对。
        """
        if len(self.im_files_rgb) != len(self.im_files_depth):
            raise ValueError(
                f"{self.prefix}RGB/depth image count mismatch: {len(self.im_files_rgb)} vs {len(self.im_files_depth)}"
            )

        depth_by_stem = {}
        for path in self.im_files_depth:
            stem = Path(path).stem
            # 如果第二路里同名文件出现重复，后面就无法唯一配对，所以这里直接失败。
            if stem in depth_by_stem:
                raise ValueError(
                    f"{self.prefix}Duplicate depth/IR filename stem '{stem}'. "
                    "Please ensure paired files have unique basenames."
                )
            depth_by_stem[stem] = path

        reordered_depth = []
        missing = []
        for rgb_path in self.im_files_rgb:
            stem = Path(rgb_path).stem
            depth_path = depth_by_stem.get(stem)
            if depth_path is None:
                missing.append(stem)
                continue
            reordered_depth.append(depth_path)

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"{self.prefix}Missing paired depth/IR images for RGB stems: {preview}")

        # 后续 load_image(index) 会同时访问 self.im_files_rgb[index] / self.im_files_depth[index]，
        # 所以这里必须把第二路重排成和 RGB 完全一致的顺序。
        self.im_files_depth = reordered_depth

    def get_labels(self):
        """扫描并解析标签。

        这里有几个关键约定：

        - 标签路径由 RGB 图像路径推导
        - 标签解析复用仓库现有的 verify_image_label()
        - 如果某张图或标签损坏，会把这个样本整体剔除
        - 剔除坏样本时，RGB 文件列表、第二路文件列表、labels 列表必须同步收缩
        """
        self._pair_modal_files()
        paired_files = list(zip(self.im_files_rgb, self.im_files_depth))
        label_files = img2label_paths(self.im_files_rgb)
        nkpt, ndim = self.kpt_shape if self.use_keypoints else (0, 0)
        nc = int(self.data.get("nc", len(self.data.get("names", [])) or 0))
        if nc <= 0:
            raise ValueError(f"{self.prefix}Dataset 'nc' is missing or invalid.")

        labels = []
        kept_rgb = []
        kept_depth = []
        nm = nf = ne = nc_corrupt = 0
        messages = []
        total = len(paired_files)
        desc = f"{self.prefix}Scanning labels"
        verify_args = (
            (im_file, lb_file, self.prefix, self.use_keypoints, nc, nkpt, ndim)
            for im_file, lb_file in zip(self.im_files_rgb, label_files)
        )

        with ThreadPool(NUM_THREADS) as pool:
            # verify_image_label 会同时校验图像与标签，并返回：
            # - 解析后的 bbox / segment / keypoint
            # - 各种统计计数
            # - 报警信息
            results = pool.imap(verify_image_label, verify_args)
            pbar = TQDM(results, total=total, desc=desc, disable=LOCAL_RANK > 0)
            for i, (im_file, lb, shape, segments, keypoints, nm_f, nf_f, ne_f, nc_f, msg) in enumerate(pbar):
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc_corrupt += nc_f
                if msg:
                    messages.append(msg)
                if im_file is None:
                    continue

                # 只保留通过校验的样本，并保持 RGB / Depth / Label 三者顺序完全一致。
                kept_rgb.append(paired_files[i][0])
                kept_depth.append(paired_files[i][1])
                labels.append(
                    dict(
                        im_file=im_file,
                        shape=shape,
                        cls=lb[:, :1].astype(np.float32),
                        bboxes=lb[:, 1:].astype(np.float32),
                        segments=segments,
                        keypoints=keypoints,
                        normalized=True,
                        bbox_format="xywh",
                    )
                )
            pbar.close()

        if messages:
            for msg in messages:
                LOGGER.info(msg)
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No label files were found. Training will run as unlabeled data.")
        if nc_corrupt:
            LOGGER.warning(f"{self.prefix}Ignored {nc_corrupt} corrupt image/label pairs.")
        if len(labels) != total:
            LOGGER.warning(f"{self.prefix}Usable image/label pairs: {len(labels)}/{total}.")

        # 这里必须回写过滤后的文件列表，否则后面 __getitem__ 用 index 取图像时会和 labels 错位。
        self.im_files_rgb = kept_rgb
        self.im_files_depth = kept_depth
        return labels

    def update_labels_info(self, label):
        """把原始标签字典封装成 Instances。

        当前仓库的大部分增强逻辑都不是直接处理 bboxes / segments 原始数组，
        而是依赖 Instances 提供的统一接口，例如：

        - convert_bbox()
        - normalize() / denormalize()
        - clip()
        - flipud() / fliplr()

        所以在样本真正送入 transforms 之前，需要先做一次格式统一。
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format", "xywh")
        normalized = label.pop("normalized", True)

        # 分割和 OBB 分支后面都默认 segments 是一个规则张量，而不是长度不一的 list。
        # 所以这里把 polygon 重采样到固定 1000 点，方便后续：
        # - 仿射变换
        # - mask 生成
        # - OBB 相关转换
        if (self.use_segments or self.use_obb) and len(segments):
            segments = np.stack(resample_segments(segments), axis=0).astype(np.float32)
        else:
            # 即使没有 segment，也给一个形状稳定的空张量，避免后面的 len()/索引/拼接分支报错。
            segments = np.zeros((0, 1000, 2), dtype=np.float32)

        label["instances"] = Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=normalized,
        )
        return label

    def build_transforms(self, hyp=None):
        """按 train / val 构建增强流水线。"""
        if self.augment:
            # 训练阶段直接走当前仓库已有的 v8 双流增强链。
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            # 验证阶段保持几何关系稳定，不做随机增强，只做 letterbox 与格式整理。
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        # 最后统一接 Format，把 numpy / Instances 整理成训练器真正需要的字段：
        # - img_rgb / img_depth
        # - cls / bboxes
        # - masks / keypoints
        # - batch_idx
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=getattr(hyp, "mask_ratio", 4),
                mask_overlap=getattr(hyp, "overlap_mask", True),
                bgr=getattr(hyp, "bgr", 0.0),
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch):
        """自定义 batch 拼接逻辑。

        当前仓库的 trainer / validator 不是在 dataloader 里就把图像堆成 tensor，
        而是在 preprocess() 里再做一次 stack，并可能继续做多尺度 resize。
        所以这里：

        - 图像字段保留成 list
        - 标注字段直接 concat
        - batch_idx 按样本编号偏移
        """
        out = {}
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            if key in {"img_rgb", "img_depth", "im_file", "ori_shape", "resized_shape", "ratio_pad"}:
                # 图像和元信息保留列表形式，后面 preprocess() 会自己决定什么时候 stack。
                out[key] = values
            elif key == "batch_idx":
                # 每个单样本内部的 batch_idx 都从 0 开始，拼 batch 后要加上样本偏移量。
                out[key] = torch.cat([value + i for i, value in enumerate(values)], 0)
            elif key in {"cls", "bboxes", "masks", "keypoints"}:
                # 这些字段在 loss 计算前需要是一个大张量，所以这里直接拼接。
                out[key] = torch.cat(values, 0) if values else torch.empty(0)
            else:
                out[key] = values
        return out


class YOLOMultiModalDataset(YOLODataset):
    """兼容别名。

    build_yolo_dataset() 里保留了 `YOLOMultiModalDataset` 这个名字，
    这里直接继承 `YOLODataset`，避免外部逻辑继续改动。
    """


class YOLOConcatDataset(ConcatDataset):
    """拼接多个 YOLO 数据集时使用的包装类。"""

    @staticmethod
    def collate_fn(batch):
        return YOLODataset.collate_fn(batch)


class GroundingDataset(YOLODataset):
    """占位类。

    当前仓库的主要目标是双流 detect / segment 训练，
    grounding 数据集链路并不是本次恢复重点，所以显式标记为未实现。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GroundingDataset is not implemented in this dual-stream fork.")


class ClassificationDataset(Dataset):
    """最小可用分类数据集。

    这个类的目的不是构建真正的双流分类系统，
    只是为了让分类相关 import 不在当前仓库里直接崩掉。
    """

    def __init__(self, root, args, augment=False, prefix="train"):
        from torchvision.datasets import ImageFolder

        self.base = ImageFolder(root=root)
        self.samples = self.base.samples
        self.classes = self.base.classes
        # 分类链路仍沿用 torchvision + 上游的 classify transforms。
        self.torch_transforms = classify_augmentations(size=args.imgsz) if augment else classify_transforms(size=args.imgsz)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        # ImageFolder 返回 PIL.Image 和类别 id。
        image, cls = self.base[index]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        image = self.torch_transforms(image)
        # 当前仓库某些可视化代码默认还会访问第二路图像，这里先用 RGB 占位，保证接口不崩。
        return {"img": image, "cls": torch.tensor(cls, dtype=torch.long), "depth": image}


class SemanticDataset(Dataset):
    """占位类，当前未实现。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SemanticDataset is not implemented in this dual-stream fork.")
