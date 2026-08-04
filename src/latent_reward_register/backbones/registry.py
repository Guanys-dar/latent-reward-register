from __future__ import annotations

from collections.abc import Callable

from .base import BackboneAdapter

AdapterFactory = Callable[..., BackboneAdapter]
_ADAPTERS: dict[str, AdapterFactory] = {}


def register_backbone(name: str, factory: AdapterFactory) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Backbone name cannot be empty")
    if normalized in _ADAPTERS:
        raise ValueError(f"Backbone adapter already registered: {normalized}")
    _ADAPTERS[normalized] = factory


def create_backbone(name: str, **kwargs) -> BackboneAdapter:
    normalized = name.strip().lower()
    try:
        factory = _ADAPTERS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported backbone {name!r}; available: {supported}") from error
    return factory(**kwargs)


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))

