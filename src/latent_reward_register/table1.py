"""Table 1: pairwise preference accuracy over the released pair file.

The evaluator in `preference.py` compares latents; the released pair file names
images. This module bridges the two, and reports accuracy per dataset because a
single pooled number hides which benchmark a register is weak on.

The published pair file records each image as a dataset-relative path, so the
image roots are supplied at call time (see `scripts/fetch_table1_images.py`).

**Position bias.** In the released file `image1` is always the human-preferred
image, so every `preferred` label is 0. A scorer that simply always answers
"first" would report 100% accuracy. `shuffled` randomizes which side the
preferred image lands on, and `position_bias` measures how much a register's
answer depends on order; use one of them before trusting a number.
"""
from __future__ import annotations

import collections
import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .preference import PreferenceMetrics, PreferencePairBatch, evaluate_preference_pairs
from .types import RegisterCondition

# Every released pair carries these; anything else is provenance.
REQUIRED_FIELDS = ("dataset", "image1", "image2", "preferred", "prompt")


@dataclass(frozen=True)
class PreferencePair:
    """One released comparison. ``preferred`` is 0 for image1, 1 for image2."""

    pair_id: str
    dataset: str
    prompt: str
    image1: str
    image2: str
    preferred: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PreferencePair:
        missing = [key for key in REQUIRED_FIELDS if key not in payload]
        if missing:
            raise ValueError(f"Pair record missing fields: {', '.join(missing)}")
        preferred = int(payload["preferred"])  # type: ignore[arg-type]
        if preferred not in (0, 1):
            raise ValueError(f"preferred must be 0 or 1, got {preferred!r}")
        return cls(
            pair_id=str(payload.get("pair_id", "")),
            dataset=str(payload["dataset"]),
            prompt=str(payload["prompt"]),
            image1=str(payload["image1"]),
            image2=str(payload["image2"]),
            preferred=preferred,
        )

    def paths(self, roots: Mapping[str, Path]) -> tuple[Path, Path]:
        try:
            root = Path(roots[self.dataset])
        except KeyError:
            raise KeyError(
                f"No image root configured for dataset {self.dataset!r}; "
                f"have {sorted(roots)}"
            ) from None
        return root / self.image1, root / self.image2


def read_pair_file(path: str | Path) -> Iterator[PreferencePair]:
    """Stream the released pair file, naming the offending line on error."""
    pair_path = Path(path)
    with pair_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield PreferencePair.from_mapping(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid pair at {pair_path}:{line_number}: {error}") from error


def _exchange_sides(pair: PreferencePair) -> PreferencePair:
    """One pair with its images exchanged and its label moved to match.

    Both must move together: exchanging the images while leaving ``preferred``
    alone would relabel the pair rather than reorder it.
    """
    return PreferencePair(
        pair_id=pair.pair_id,
        dataset=pair.dataset,
        prompt=pair.prompt,
        image1=pair.image2,
        image2=pair.image1,
        preferred=1 - pair.preferred,
    )


def shuffled(pairs: Sequence[PreferencePair], *, seed: int = 0) -> list[PreferencePair]:
    """Randomize which side holds the preferred image.

    The released order is always preferred-first, so evaluating it directly
    cannot distinguish a real register from a constant "first" answer.
    """
    generator = random.Random(seed)
    return [pair if generator.random() < 0.5 else _exchange_sides(pair) for pair in pairs]


def swapped(pairs: Sequence[PreferencePair]) -> list[PreferencePair]:
    """Every pair with its two sides exchanged."""
    return [_exchange_sides(pair) for pair in pairs]


def missing_images(pairs: Sequence[PreferencePair], roots: Mapping[str, Path]) -> list[str]:
    """Paths the pair file references but that are absent.

    Worth calling before evaluating: a partially fetched root would otherwise
    yield an accuracy silently covering only part of the benchmark.
    """
    absent = []
    for pair in pairs:
        for path in pair.paths(roots):
            if not path.exists():
                absent.append(str(path))
    return absent


@dataclass
class _Tally:
    """Running (correct, total, ties) for one dataset."""

    correct: int = 0
    total: int = 0
    ties: int = 0

    def add(self, metrics: PreferenceMetrics) -> None:
        self.correct += metrics.correct
        self.total += metrics.total
        self.ties += metrics.ties

    def as_tuple(self) -> tuple[int, int, int]:
        return self.correct, self.total, self.ties


@dataclass(frozen=True)
class Table1Report:
    """Overall and per-dataset accuracy. Report both."""

    head: str
    overall_correct: int
    overall_total: int
    overall_ties: int
    per_dataset: Mapping[str, tuple[int, int, int]]

    @property
    def accuracy(self) -> float:
        return self.overall_correct / self.overall_total if self.overall_total else 0.0

    def dataset_accuracy(self, dataset: str) -> float:
        correct, total, _ = self.per_dataset[dataset]
        return correct / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "accuracy": self.accuracy,
            "correct": self.overall_correct,
            "total": self.overall_total,
            "ties": self.overall_ties,
            "per_dataset": {
                name: {
                    "accuracy": correct / total if total else 0.0,
                    "correct": correct,
                    "total": total,
                    "ties": ties,
                }
                for name, (correct, total, ties) in sorted(self.per_dataset.items())
            },
        }


def evaluate_table1(
    register,
    pairs: Sequence[PreferencePair],
    *,
    head: str,
    encode: Callable[[PreferencePair], tuple[torch.Tensor, torch.Tensor, RegisterCondition, torch.Tensor]],
    batch_size: int = 8,
) -> Table1Report:
    """Score every pair and report accuracy overall and per dataset.

    ``encode`` turns one pair into
    ``(first_latents, second_latents, condition, sigma)``: it owns VAE encoding
    and prompt encoding, which are backbone-specific and need model weights.
    Keeping it a parameter is what lets this run on CPU in tests.
    """
    if not pairs:
        raise ValueError("No pairs to evaluate")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    tallies: dict[str, _Tally] = collections.defaultdict(_Tally)
    overall = _Tally()

    by_dataset: dict[str, list[PreferencePair]] = collections.defaultdict(list)
    for pair in pairs:
        by_dataset[pair.dataset].append(pair)

    for dataset, group in by_dataset.items():
        for start in range(0, len(group), batch_size):
            batches = []
            for pair in group[start : start + batch_size]:
                first, second, condition, sigma = encode(pair)
                batches.append(
                    PreferencePairBatch(
                        first_latents=first,
                        second_latents=second,
                        preferred=torch.tensor([pair.preferred], dtype=torch.long),
                        condition=condition,
                        sigma=sigma,
                    )
                )
            metrics = evaluate_preference_pairs(register, batches, head=head)
            tallies[dataset].add(metrics)
            overall.add(metrics)

    return Table1Report(
        head=head,
        overall_correct=overall.correct,
        overall_total=overall.total,
        overall_ties=overall.ties,
        per_dataset={name: tally.as_tuple() for name, tally in tallies.items()},
    )


def position_bias(
    register,
    pairs: Sequence[PreferencePair],
    *,
    head: str,
    encode,
    batch_size: int = 8,
) -> dict:
    """How much the register's answer depends on presentation order.

    Evaluates the pairs as given and with both sides exchanged. An
    order-independent register scores the same both ways, so a large gap means
    the reported accuracy is partly measuring position, not preference.
    """
    forward = evaluate_table1(register, pairs, head=head, encode=encode, batch_size=batch_size)
    reverse = evaluate_table1(
        register, swapped(list(pairs)), head=head, encode=encode, batch_size=batch_size
    )
    return {
        "accuracy_as_given": forward.accuracy,
        "accuracy_swapped": reverse.accuracy,
        "gap": abs(forward.accuracy - reverse.accuracy),
        "mean": (forward.accuracy + reverse.accuracy) / 2.0,
    }
