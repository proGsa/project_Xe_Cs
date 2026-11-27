"""Temperature limits of the supplied Xe collision tables; no extrapolation."""
from __future__ import annotations

import numpy as np

from .settings import RunSettings
from .xe_dataset import PropertyDomainError


class CollisionDomainError(PropertyDomainError):
    """A required collision integral is outside the supplied table."""


def temperature_interval(ratio: float) -> tuple[float, float]:
    """Both translational temperatures must lie in [300, 30000] K."""
    return max(300., 300./ratio), min(30000., 30000./ratio)


def temperature_grid(settings: RunSettings, element: str) -> np.ndarray:
    values = settings.temperatures()
    if element != "Xe" or not settings.xe_properties_enabled:
        return values
    ratio = settings.fixed_temperature_ratio if settings.sweep_mode == "fixed_ratio" else 1.
    lower, upper = temperature_interval(ratio)
    start = max(settings.temperature_start_k, lower)
    stop = min(settings.temperature_stop_k, upper)
    if start > stop:
        raise ValueError(f"Для свойств Xe при gamma={ratio:g} нет допустимых температур: "
                         f"требуется {lower:g} <= Th <= {upper:g} K.")
    # Retain the requested interior grid; include the exact permitted endpoints.
    inside = values[(values >= start) & (values <= stop)]
    return np.unique(np.r_[start, inside, stop])


def ratio_grid(settings: RunSettings, element: str) -> np.ndarray:
    values = settings.ratios()
    if element != "Xe" or not settings.xe_properties_enabled:
        return values
    th = settings.heavy_temperature_k
    if not 300 <= th <= 30000:
        raise ValueError("Для свойств Xe фиксированная Th должна быть в [300, 30000] K.")
    return values[(values*th >= 300) & (values*th <= 30000)]
