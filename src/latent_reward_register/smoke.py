from __future__ import annotations

from tempfile import TemporaryDirectory

import torch

from .backbones.base import BackboneAdapter, BackboneFeatures
from .checkpoint import CheckpointManifest
from .guidance import GuidanceSchedule, RewardGradientGuidance
from .preference import PreferencePairBatch, evaluate_preference_pairs
from .register import RewardRegister, RewardRegisterConfig
from .rgopd import RGOPDBatch, RGOPDTrainConfig, train_rgopd
from .sampling import reward_guided_sample
from .training import GroupBatch, TrainConfig, train_register
from .types import RegisterCondition


class _SyntheticAdapter(BackboneAdapter):
    name = "synthetic"
    hidden_size = 4

    def extract_features(self, latents, condition, sigma, *, reward_tokens, feature_layers):
        del condition, feature_layers
        batch_size = latents.shape[0]
        signal = latents.flatten(1).mean(dim=1) + sigma
        tokens = reward_tokens.unsqueeze(0).expand(batch_size, -1, -1) + signal[:, None, None]
        return BackboneFeatures(tokens, (), (), signal[:, None])

    def reference_step(self, latents, condition, sigma, next_sigma, **kwargs):
        del condition, kwargs
        delta = (sigma - next_sigma).reshape(-1, *([1] * (latents.ndim - 1)))
        return latents - delta


class _SyntheticStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, latents, sigma, next_sigma):
        del sigma, next_sigma
        return latents + self.offset


def _register() -> RewardRegister:
    return RewardRegister(
        _SyntheticAdapter(),
        RewardRegisterConfig(
            backbone="synthetic",
            head_names=("preference",),
            feature_layers=(0,),
            num_register_tokens=2,
        ),
    )


def run_release_smoke() -> dict[str, str]:
    torch.manual_seed(0)
    register = _register()
    condition = RegisterCondition(prompt_embeds=torch.zeros(2, 1, 1))
    sigma = torch.ones(2)
    # Two prompt groups of two images each: the preferred item scores higher.
    group_latents = torch.stack([torch.ones(2, 1, 2, 2), torch.zeros(2, 1, 2, 2)], dim=1)
    group_batch = GroupBatch(
        latents=group_latents,
        condition=condition,
        sigma=sigma,
        targets={"preference": torch.tensor([[1.0, -1.0], [1.0, -1.0]])},
    )
    manifest = CheckpointManifest(1, "synthetic", "test", "synthetic", ("preference",), (0,), "sigma")
    with TemporaryDirectory() as output_dir:
        train_register(
            model=register,
            batches=[group_batch],
            config=TrainConfig(epochs=1),
            output_dir=output_dir,
            manifest=manifest,
        )

    metrics = evaluate_preference_pairs(
        register,
        [
            PreferencePairBatch(
                first_latents=group_latents[:, 0],
                second_latents=group_latents[:, 1],
                preferred=torch.zeros(2, dtype=torch.long),
                condition=condition,
                sigma=sigma,
            )
        ],
        head="preference",
    )
    if metrics.total != 2:
        raise RuntimeError("Preference smoke did not evaluate both pairs")

    guidance = RewardGradientGuidance(min_gradient_rms=0.0)
    sampled = reward_guided_sample(
        register=register,
        latents=torch.zeros(2, 1, 2, 2),
        condition=condition,
        sigmas=(1.0, 0.5, 0.0),
        heads=("preference",),
        schedule=GuidanceSchedule(((0.5, 0.2),)),
        guidance=guidance,
    )
    if not torch.isfinite(sampled).all():
        raise RuntimeError("RGS smoke produced non-finite latents")

    student = _SyntheticStudent()
    rgopd_metrics = train_rgopd(
        student=student,
        batches=[
            RGOPDBatch(
                latents=torch.zeros(2, 1),
                sigma=torch.ones(2),
                next_sigma=torch.zeros(2),
                reference_next=torch.ones(2, 1),
                reward_gradient=torch.ones(2, 1),
                transition_std=torch.ones(2),
            )
        ],
        config=RGOPDTrainConfig(reward_scale=0.4),
        guidance=guidance,
    )
    if rgopd_metrics.steps != 1:
        raise RuntimeError("RG-OPD smoke did not optimize the student")
    return {
        "register_training": "ok",
        "preference_scoring": "ok",
        "reward_guided_sampling": "ok",
        "rgopd_training": "ok",
    }
