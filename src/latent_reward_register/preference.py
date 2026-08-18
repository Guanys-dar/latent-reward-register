from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import torch

from .types import RegisterCondition


class PreferenceRegister(Protocol):
    def score(
        self, latents: torch.Tensor, condition: RegisterCondition, sigma: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class PreferencePairBatch:
    first_latents: torch.Tensor
    second_latents: torch.Tensor
    preferred: torch.Tensor
    condition: RegisterCondition
    sigma: torch.Tensor


@dataclass(frozen=True)
class PreferenceMetrics:
    correct: int
    total: int
    ties: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@torch.no_grad()
def evaluate_preference_pairs(
    register: PreferenceRegister,
    batches: Iterable[PreferencePairBatch],
    *,
    head: str,
) -> PreferenceMetrics:
    correct = 0
    total = 0
    ties = 0
    for batch in batches:
        labels = batch.preferred.to(dtype=torch.long)
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Preference labels must be 0 or 1")
        first_scores = register.score(batch.first_latents, batch.condition, batch.sigma)
        second_scores = register.score(batch.second_latents, batch.condition, batch.sigma)
        if head not in first_scores or head not in second_scores:
            raise ValueError(f"Register does not provide requested reward head: {head}")
        first = first_scores[head]
        second = second_scores[head]
        if first.shape != labels.shape or second.shape != labels.shape:
            raise ValueError(
                "Preference scores and labels must have identical shapes; "
                f"got first={tuple(first.shape)}, second={tuple(second.shape)}, labels={tuple(labels.shape)}"
            )
        predictions = (second > first).to(dtype=torch.long)
        tied = first == second
        correct += int(((predictions == labels) & ~tied).sum().item())
        ties += int(tied.sum().item())
        total += int(labels.numel())
    return PreferenceMetrics(correct=correct, total=total, ties=ties)
