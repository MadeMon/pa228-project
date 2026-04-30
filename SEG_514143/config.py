from utils import device
from typing import Any
from pathlib import Path
from network import MobilnetASPP
from label_dict import label_dict

CONFIG: dict[str, Any] = {
    "runtime_amp": True,
    "runtime_grad_scaler": True,
    "runtime_num_workers": 0,
    "runtime_pin_memory": True,
    "runtime_persistent_workers": True,
    "runtime_prefetch_factor": 2,
    "runtime_cudnn_benchmark": True,
    "runtime_non_blocking_transfer": True,
    "runtime_compile_model": True,
    "transforms_random_crop_size": 512,  # 512, 256, None
    "transforms_rare_crop_prob": 0.5,
    "transforms_normalize_mean": (0.485, 0.456, 0.406),
    "transforms_normalize_std": (0.229, 0.224, 0.225),
    "transforms_rare_classes": [
        3,  # object
        6,  # human
    ],
    "num_classes": len(label_dict),
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 3e-4,  # 1e-3
    "class_ignore_index": 0,  # 0 is the "void" class
    "model_checkpoint_path": Path("models"),
    "debug_dir_rare_crops": None,  # "debug_rare_crops", None
    "preload_samples": True,
    "network": MobilnetASPP(len(label_dict)),
    "warmup_ratio": 0.04,
    "optimizer_weight_decay": 1e-4,
}
