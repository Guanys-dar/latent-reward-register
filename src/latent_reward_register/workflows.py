from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config


_BACKBONES = {"sd3", "flux", "z-image"}
# Module backing each task. "workspace" is always the release package: a plan must
# never name an internal experiment workspace, since a public reader cannot run it.
_TASK_PACKAGES = {
    "train-register": "latent_reward_register.training",
    "sample": "latent_reward_register.sampling",
    "train-rgopd": "latent_reward_register.rollout",
}
_CONFIG_GLOBS = ("configs/register/*/*.yaml", "configs/rgs/*/*.yaml", "configs/rgopd/*/*.yaml")


@dataclass(frozen=True)
class WorkflowSpec:
    path: Path
    task: str
    backbone: str
    config: dict[str, Any]
    implementation: dict[str, str]

    @property
    def required_inputs(self) -> list[str]:
        if self.task == "train-register":
            return ["dataset_manifest", "output_directory"]
        if self.task == "sample":
            return ["register_checkpoint", "prompt_file", "output_directory"]
        return ["register_checkpoint", "training_manifest", "output_directory"]

    def plan(self) -> dict[str, Any]:
        return {
            "config": str(self.path),
            "task": self.task,
            "backbone": self.backbone,
            "implementation": self.implementation,
            "required_inputs": self.required_inputs,
            "checkpoint_and_data": "deferred",
        }


def _implementation(task: str) -> dict[str, str]:
    return {
        "package": _TASK_PACKAGES[task],
        "entry_point": f"latent_reward_register.cli {task}",
        "workspace": "release-package",
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
    if task not in _TASK_PACKAGES:
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
    return WorkflowSpec(workflow_path, task, backbone, config, _implementation(task))


def validate_release(root: str | Path) -> dict[str, Any]:
    release_root = Path(root)
    paths = sorted(path for pattern in _CONFIG_GLOBS for path in release_root.glob(pattern))
    workflows = [load_workflow(path) for path in paths]
    capabilities = {
        "register_training": sorted(spec.backbone for spec in workflows if spec.task == "train-register"),
        "reward_guided_sampling": sorted(spec.backbone for spec in workflows if spec.task == "sample"),
        "rgopd_training": sorted(spec.backbone for spec in workflows if spec.task == "train-rgopd"),
    }
    return {
        "valid": True,
        "release_ready": False,
        "model_execution_ready": False,
        "root": str(release_root),
        "workflow_count": len(workflows),
        "capabilities": capabilities,
        "deferred": ["checkpoints", "training_data", "paper_test_sets"],
        "blockers": [
            "sample and train-rgopd as executable CLI subcommands",
            "benchmark generation and scoring pipeline",
        ],
    }
