"""Latent Reward Registers: reward models that ride a frozen diffusion transformer.

Where to start, by task:

- **Score latents with a published register.** ``load_legacy_register`` returns a
  ``CheckpointRewardRegister``; ``score``, ``score_groups``, and
  ``score_and_grad`` are its interface.
- **Build an untrained register from a paper config.** ``build_register`` /
  ``build_register_from_config`` in ``latent_reward_register.backbones``.
- **Reward-guided sampling.** ``reward_guided_sample``.
- **RG-OPD.** ``train_rgopd_rollout`` is the paper path (on-policy). ``train_rgopd``
  is the off-policy single-step trainer kept for ablations; see ``rgopd.py``.

``ReferenceRewardRegister`` is a weight-free test scaffold; real model-backed
registers use ``CheckpointRewardRegister``.
"""

from .checkpoint import CheckpointManifest, load_register_checkpoint, save_register_checkpoint
from .flowmatch import euler_step, make_reference_step, make_student_policy
from .guidance import GuidanceSchedule, RewardGradientGuidance
from .implementations import (
    CheckpointRewardRegister,
    latent_gradient_enabled,
    load_legacy_register,
)
from .preference import PreferenceMetrics, PreferencePairBatch, evaluate_preference_pairs
from .register import ReferenceRegisterConfig, ReferenceRewardRegister
from .rgopd import RGOPDBatch, RGOPDTrainConfig, RGOPDTrainMetrics, train_rgopd
from .rollout import RolloutConfig, RolloutTrace, train_rgopd_rollout
from .sampling import SamplingTrace, reward_guided_sample
from .table1 import evaluate_table1, position_bias, read_pair_file, shuffled
from .teacher import RewardGradientTeacher, TeacherStep
from .velocity import FluxVelocityModel, SD3VelocityModel, attach_lora_student

__all__ = [
    # Registers and checkpoints
    "CheckpointManifest",
    "CheckpointRewardRegister",
    "ReferenceRegisterConfig",
    "ReferenceRewardRegister",
    "load_legacy_register",
    "load_register_checkpoint",
    "save_register_checkpoint",
    "latent_gradient_enabled",
    # Guidance, shared by RGS and RG-OPD
    "GuidanceSchedule",
    "RewardGradientGuidance",
    "RewardGradientTeacher",
    "TeacherStep",
    # Backbone transitions
    "FluxVelocityModel",
    "SD3VelocityModel",
    "attach_lora_student",
    "euler_step",
    "make_reference_step",
    "make_student_policy",
    # Reward-guided sampling
    "SamplingTrace",
    "reward_guided_sample",
    # RG-OPD: train_rgopd_rollout is the paper path, train_rgopd the ablation one
    "RolloutConfig",
    "RolloutTrace",
    "train_rgopd_rollout",
    "RGOPDBatch",
    "RGOPDTrainConfig",
    "RGOPDTrainMetrics",
    "train_rgopd",
    # Evaluation
    "PreferenceMetrics",
    "PreferencePairBatch",
    "evaluate_preference_pairs",
    "evaluate_table1",
    "position_bias",
    "read_pair_file",
    "shuffled",
]
