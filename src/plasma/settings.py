from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def format_temperature_ratio(ratio: float) -> str:
    """Краткая запись без потери значащих цифр для подписей и имён файлов."""
    return str(float(ratio)).removesuffix(".0")


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
    sweep_mode: str = "temperature"  # temperature; temperature_ratio (T_h=const); fixed_ratio (gamma=const).
    heavy_temperature_k: float = 4000.0
    temperature_ratios: tuple[float, ...] = tuple(1.0 + 0.25 * i for i in range(17))
    fixed_temperature_ratio: float = 2.0
    xe_properties_enabled: bool = False
    xe_lte_data_path: Path | None = None
    xe_collision_branch: str = "traditional"
    xe_high_ion_neutral_model: str = "langevin_isotropic_capture"
    xe_heat_capacity_relative_step: float = 1e-3
    xe_heat_capacity_tolerance: float = 5e-4

    def validate(self) -> None:
        for name in (
            "pressure_pa", "temperature_start_k", "temperature_stop_k",
            "temperature_step_k", "reference_density_m3", "relative_tolerance",
            "gamma_tolerance", "max_log_step", "minimum_temperature_step_k",
            "heavy_temperature_k",
            "fixed_temperature_ratio",
            "xe_heat_capacity_relative_step", "xe_heat_capacity_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: требуется положительное конечное число.")
        if self.temperature_stop_k < self.temperature_start_k:
            raise ValueError("Конечная температура меньше начальной.")
        if not isinstance(self.xe_properties_enabled, bool):
            raise ValueError("xe_properties_enabled должен быть bool.")
        if self.xe_collision_branch not in ("traditional", "phenomenological"):
            raise ValueError("xe_collision_branch: traditional или phenomenological.")
        if self.xe_high_ion_neutral_model not in ("langevin_isotropic_capture", "error"):
            raise ValueError("xe_high_ion_neutral_model: langevin_isotropic_capture или error.")
        if self.xe_heat_capacity_relative_step > .05 or self.xe_heat_capacity_tolerance >= 1:
            raise ValueError("Шаг производной должен быть <= 0.05, допуск теплоёмкости < 1.")
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
        if self.sweep_mode not in ("temperature", "temperature_ratio", "fixed_ratio"):
            raise ValueError("sweep_mode должен быть temperature, temperature_ratio или fixed_ratio.")
        if self.sweep_mode == "fixed_ratio":
            if self.fixed_temperature_ratio != 1.0 and "nonideal" in self.models:
                raise ValueError(
                    "При фиксированном gamma != 1 поддерживается только models=('ideal',). "
                    "Для nonideal при T_e != T_h формулы поправок ещё не заданы."
                )
            with np.errstate(over="ignore", under="ignore"):
                endpoints = np.asarray([self.temperature_start_k, self.temperature_stop_k]) * self.fixed_temperature_ratio
            if not np.all(np.isfinite(endpoints)) or np.any(endpoints <= 0):
                raise ValueError("Произведение T_h * fixed_temperature_ratio вне диапазона температур float64.")
        if self.sweep_mode == "temperature_ratio":
            if self.models != ("ideal",):
                raise ValueError(
                    "Двухтемпературная прогонка поддерживает только models=('ideal',). "
                    "Для nonideal при T_e != T_h формулы поправок ещё не заданы."
                )
            ratios = np.asarray(self.temperature_ratios, dtype=float)
            if ratios.ndim != 1 or ratios.size == 0 or ratios.size > 100_000:
                raise ValueError("temperature_ratios: требуется от 1 до 100000 значений.")
            if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
                raise ValueError("temperature_ratios должны быть положительными конечными числами.")
            if np.any(np.diff(ratios) <= 0):
                raise ValueError("temperature_ratios должны строго возрастать без повторений.")
            with np.errstate(over="ignore", under="ignore"):
                electron_temperatures = ratios * self.heavy_temperature_k
            if not np.all(np.isfinite(electron_temperatures)) or np.any(electron_temperatures <= 0):
                raise ValueError("Произведение T_h * temperature_ratios вне диапазона температур float64.")
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

    def ratios(self) -> np.ndarray:
        """Контрольная точка T_e/T_h=1 включается в вывод автоматически."""
        self.validate()
        return np.unique(np.r_[1.0, self.temperature_ratios])
