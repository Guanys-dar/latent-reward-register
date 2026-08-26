from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A builder takes the backbone name plus config keyword arguments and returns a
# scoring register. It is not a BackboneAdapter: the research implementations own
# their own backbone traversal rather than sitting behind a feature-extraction
# interface.
RegisterBuilder = Callable[..., Any]
_BUILDERS: dict[str, RegisterBuilder] = {}


def register_backbone(name: str, builder: RegisterBuilder) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Backbone name cannot be empty")
    if normalized in _BUILDERS:
        raise ValueError(f"Backbone already registered: {normalized}")
    _BUILDERS[normalized] = builder


def create_backbone(name: str, **kwargs: Any) -> Any:
    """Build a model-backed register for ``name``. Requires model weights."""
    normalized = name.strip().lower()
    try:
        builder = _BUILDERS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_BUILDERS)) or "none"
        raise ValueError(f"Unsupported backbone {name!r}; available: {supported}") from error
    return builder(normalized, **kwargs)


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))
