from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RunSettings:
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    elements: tuple[str, ...] = ("Xe", "Cs")
    models: tuple[str, ...] = ("ideal", "nonideal")
    pressure_pa: float = 100_000.0
    temperature_start_k: float = 3000.0
    temperature_stop_k: float = 10000.0
    temperature_step_k: float = 250.0
    max_charge: int = 1
    reference_density_m3: float = 1e24
    default_energy_unit: str = "eV"
    bound_levels_only: bool = True
    relative_tolerance: float = 1e-8
    gamma_tolerance: float = 1e-12
    max_iterations: int = 150
    max_backtracks: int = 24
    max_log_step: float = 2.0
    continuation_depth: int = 4
    minimum_temperature_step_k: float = 10.0
    show_plots: bool = False
    plot_groups: tuple[int, ...] = (1, 2, 3, 4)
    plot_all_species: bool = False

    def validate(self) -> None:
        for name in (
            "pressure_pa", "temperature_start_k", "temperature_stop_k",
            "temperature_step_k", "reference_density_m3", "relative_tolerance",
            "gamma_tolerance", "max_log_step", "minimum_temperature_step_k",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: требуется положительное конечное число.")
        if self.temperature_stop_k < self.temperature_start_k:
            raise ValueError("Конечная температура меньше начальной.")
        if (self.temperature_stop_k - self.temperature_start_k) / self.temperature_step_k > 100_000:
            raise ValueError("Температурная сетка содержит более 100000 интервалов.")
        if not isinstance(self.max_charge, int) or isinstance(self.max_charge, bool) or not 1 <= self.max_charge <= 4:
            raise ValueError("max_charge должен быть целым числом от 1 до 4.")
        if not self.elements or any(x not in ("Xe", "Cs") for x in self.elements):
            raise ValueError("elements должен содержать Xe и/или Cs.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("elements содержит повторения.")
        if not self.models or any(x not in ("ideal", "nonideal") for x in self.models):
            raise ValueError("models должен содержать ideal и/или nonideal.")
        if len(set(self.models)) != len(self.models):
            raise ValueError("models содержит повторения.")
        for name in ("max_iterations", "max_backtracks", "continuation_depth"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "continuation_depth" else 1):
                raise ValueError(f"Некорректное значение {name}.")
        if self.relative_tolerance >= 1 or self.gamma_tolerance >= 1:
            raise ValueError("Допуски должны быть меньше единицы.")
        if any(not isinstance(g, int) or isinstance(g, bool) or g not in (1, 2, 3, 4)
               for g in self.plot_groups):
            raise ValueError("plot_groups должен содержать номера групп от 1 до 4.")
        if len(set(self.plot_groups)) != len(self.plot_groups):
            raise ValueError("plot_groups содержит повторения.")
        if not self.plot_all_species and not any(g <= self.max_charge for g in self.plot_groups):
            raise ValueError("Нет доступных групп графиков: увеличь max_charge или выбери другую plot_groups.")

    def temperatures(self) -> np.ndarray:
        self.validate()
        start, stop, step = self.temperature_start_k, self.temperature_stop_k, self.temperature_step_k
        count = int(np.floor((stop - start) / step + 1e-12))
        values = start + step * np.arange(count + 1, dtype=float)
        if stop - values[-1] > 1e-9 * max(1.0, abs(stop)):
            values = np.append(values, stop)
        else:
            values[-1] = stop
        return values
