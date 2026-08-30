from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config

_BACKBONES = {"sd3", "flux", "z-image"}
_TASKS = {"train-register", "sample", "train-rgopd"}
@dataclass(frozen=True)
class WorkflowSpec:
    path: Path
    task: str
    backbone: str
    config: dict[str, Any]

    def plan(self) -> dict[str, Any]:
        return {
            "config": str(self.path),
            "task": self.task,
            "backbone": self.backbone,
        }


def _validate_register(config: dict[str, Any]) -> None:
    backbone = config.get("backbone")
    if not isinstance(backbone, dict):
        raise ValueError("Register workflow requires a backbone mapping")
    if not backbone.get("model_name_or_path"):
        raise ValueError("Register workflow requires backbone.model_name_or_path")
    register = config.get("register")
    if not isinstance(register, dict) or not register.get("head_names"):
        raise ValueError("Register workflow requires register.head_names")
    if not register.get("feature_layers"):
        raise ValueError("Register workflow requires register.feature_layers")
    train = config.get("train")
    if not isinstance(train, dict) or int(train.get("epochs", 0)) < 1:
        raise ValueError("Register workflow requires train.epochs >= 1")


def _validate_rgs(config: dict[str, Any]) -> None:
    if int(config.get("steps", 0)) < 2:
        raise ValueError("RGS workflow requires steps >= 2")
    if not config.get("heads"):
        raise ValueError("RGS workflow requires heads")
    guidance = config.get("guidance")
    if not isinstance(guidance, dict) or not guidance.get("bands"):
        raise ValueError("RGS workflow requires guidance.bands")


def _validate_rgopd(config: dict[str, Any]) -> None:
    if int(config.get("rollout_steps", 0)) < 2:
        raise ValueError("RG-OPD workflow requires rollout_steps >= 2")
    student = config.get("student")
    if not isinstance(student, dict) or str(student.get("adaptation", "")).lower() != "lora":
        raise ValueError("RG-OPD student adaptation must be LoRA")
    if not config.get("reward_heads"):
        raise ValueError("RG-OPD workflow requires reward_heads")


def load_workflow(path: str | Path) -> WorkflowSpec:
    workflow_path = Path(path)
    config = load_config(workflow_path)
    task = str(config.get("task", "")).strip().lower()
    if task not in _TASKS:
        raise ValueError(f"Unsupported workflow task: {task or '<missing>'}")
    raw_backbone = config.get("backbone")
    if isinstance(raw_backbone, dict):
        backbone = str(raw_backbone.get("name", "")).strip().lower()
    else:
        backbone = str(raw_backbone or "").strip().lower()
    if backbone not in _BACKBONES:
        raise ValueError(f"Unsupported workflow backbone: {backbone or '<missing>'}")
    if task == "train-register":
        _validate_register(config)
    elif task == "sample":
        _validate_rgs(config)
    else:
        _validate_rgopd(config)
    return WorkflowSpec(workflow_path, task, backbone, config)
