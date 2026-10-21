"""Однотемпературные формулы семинарского листочка в согласованных единицах."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import constants
from scipy.special import logsumexp

from .data import AtomicData
from .settings import RunSettings


class InvalidState(ValueError):
    pass


@dataclass(frozen=True)
class PhysicalConstants:
    reference_density_m3: float
    kb: float
    kb_ev: float
    a_star: float
    c_gamma: float
    c_alpha: float

    @classmethod
    def create(cls, reference_density_m3: float) -> "PhysicalConstants":
        b = constants.e**2 / (4 * np.pi * constants.epsilon_0 * constants.k)
        c_gamma = 4 * np.pi * b**3 * reference_density_m3
        return cls(
            reference_density_m3, constants.k, constants.k / constants.e,
            (2 * np.pi * constants.m_e * constants.k / constants.h**2)**1.5 / reference_density_m3,
            c_gamma, 1 / (6 * c_gamma),
        )


@dataclass(frozen=True)
class PreparedState:
    data: AtomicData
    temperature_k: float
    pressure_pa: float
    b_pressure: float
    charges: np.ndarray
    log_partitions: np.ndarray
    log_k_ideal: np.ndarray
    physical: PhysicalConstants


@dataclass(frozen=True)
class Errors:
    ionization: float
    charge: float
    pressure: float
    gamma: float

    @property
    def maximum(self) -> float:
        return max(self.ionization, self.charge, self.pressure)


@dataclass(frozen=True)
class Evaluation:
    y: np.ndarray
    density_star: np.ndarray  # [a_e, a_0, ..., a_Z]
    gamma: float
    alpha: float
    delta_ionization_ev: np.ndarray
    log_k: np.ndarray
    errors: Errors


def prepare_state(data: AtomicData, temperature_k: float, settings: RunSettings) -> PreparedState:
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Температура должна быть положительной и конечной.")
    physical = PhysicalConstants.create(settings.reference_density_m3)
    thermal_ev = physical.kb_ev * temperature_k
    log_q = np.asarray([
        logsumexp(np.log(level.weights) - level.energies_ev / thermal_ev)
        for level in data.levels
    ])
    if not np.all(np.isfinite(log_q)):
        raise ValueError("Некорректные статистические суммы.")
    log_k = (
        np.log(2 * physical.a_star * temperature_k**1.5)
        + np.diff(log_q) - data.ionization_ev[:data.max_charge] / thermal_ev
    )
    return PreparedState(
        data, float(temperature_k), float(settings.pressure_pa),
        settings.pressure_pa / (physical.kb * temperature_k * physical.reference_density_m3),
        np.arange(len(data.levels), dtype=float), log_q, log_k, physical,
    )


def analytic_initial_guess(prepared: PreparedState) -> np.ndarray:
    """Идеальные нейтрал/ион + логарифмическая рекурсия для старших зарядов.

    u = B / (1 + sqrt(1+B/K)); все действия выполняются в логарифмах.
    """
    log_b = np.log(prepared.b_pressure)
    log_k0 = prepared.log_k_ideal[0]
    log_u = log_b - np.logaddexp(0.0, 0.5 * np.logaddexp(0.0, log_b - log_k0))
    y = np.empty(len(prepared.charges) + 1)
    y[0] = log_u
    y[1] = 2 * log_u - log_k0
    for i, log_k in enumerate(prepared.log_k_ideal):
        y[i + 2] = y[i + 1] + log_k - log_u
    return y


def solve_gamma(
    prepared: PreparedState, density_star: np.ndarray, tolerance: float,
) -> tuple[float, float]:
    """Половинное деление по q=Gamma/Gamma_D на [0,1]."""
    ae, ions = density_star[0], density_star[1:]
    z2 = prepared.charges**2
    weighted = z2 * ions
    total = ae + float(np.sum(weighted))
    if total <= 0:
        return 0.0, 0.0
    gamma_d = np.sqrt(prepared.physical.c_gamma * total / prepared.temperature_k**3)

    def function(q: float) -> float:
        gamma = q * gamma_d
        rhs = ae / (1 + gamma / 2) + np.sum(weighted / (1 + z2 * gamma / 2))
        return float(q*q - rhs / total)

    if not np.isfinite(gamma_d):
        raise InvalidState("Переполнение при вычислении Gamma_D.")
    if abs(function(1.0)) <= tolerance:
        return float(gamma_d), abs(function(1.0))
    lo, hi = 0.0, 1.0
    for _ in range(160):
        mid = (lo + hi) / 2
        value = function(mid)
        if value < 0:
            lo = mid
        else:
            hi = mid
        if abs(value) <= tolerance and hi - lo <= tolerance:
            return float(mid * gamma_d), abs(value)
        if mid == lo == hi:
            break
    q = (lo + hi) / 2
    error = abs(function(q))
    if error > 10 * tolerance:
        raise InvalidState(f"Половинное деление Gamma не достигло точности: {error:g}.")
    return float(q * gamma_d), error


def evaluate_state(
    y: np.ndarray, prepared: PreparedState, mode: str, settings: RunSettings,
) -> Evaluation:
    y = np.asarray(y, dtype=float)
    if y.shape != (len(prepared.charges) + 1,) or not np.all(np.isfinite(y)):
        raise InvalidState("Некорректный вектор логарифмов концентраций.")
    if np.min(y) < -700 or np.max(y) > 600:
        raise InvalidState("Логарифмы вне диапазона float64; требуется другое приближение или набор зарядов.")
    with np.errstate(over="raise", invalid="raise"):
        density = np.exp(y)
        if mode == "ideal":
            gamma, gamma_error, alpha = 0.0, 0.0, 0.0
            delta = np.zeros(len(prepared.log_k_ideal))
        elif mode == "nonideal":
            gamma, gamma_error = solve_gamma(prepared, density, settings.gamma_tolerance)
            alpha = prepared.physical.c_alpha * (gamma * prepared.temperature_k)**3
            z = prepared.charges
            correction_log = (
                np.log1p(z[1:]**2 * gamma / 2) + np.log1p(gamma / 2)
                - np.log1p(z[:-1]**2 * gamma / 2)
            )
            delta = prepared.physical.kb_ev * prepared.temperature_k * correction_log
        else:
            raise ValueError(f"Неизвестный режим: {mode}.")
        log_k = prepared.log_k_ideal + delta / (prepared.physical.kb_ev * prepared.temperature_k)
        ae, ions = density[0], density[1:]
        charge = float(np.sum(prepared.charges * ions))
        errors = Errors(
            float(np.max(np.abs(y[0] + y[2:] - y[1:-1] - log_k))),
            abs(ae - charge) / (ae + charge),
            abs(float(np.sum(density)) - alpha - prepared.b_pressure) / prepared.b_pressure,
            gamma_error,
        )
    if not all(np.isfinite(x) for x in (alpha, gamma, errors.maximum, errors.gamma)):
        raise InvalidState("Неконечное значение поправок или ошибок.")
    return Evaluation(y.copy(), density, gamma, float(alpha), delta, log_k, errors)


def assemble_system(
    evaluation: Evaluation, prepared: PreparedState, scale_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Матрица A y_new=b при фиксированных L_i и alpha.

    Столбцы: v, x_1,...,x_N. Первые N-1 строк — ионизация,
    предпоследняя — заряд, последняя — давление.
    """
    y, a = evaluation.y, evaluation.density_star
    ae, ions = a[0], a[1:]
    n = len(prepared.charges)
    matrix = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    for i, log_k in enumerate(evaluation.log_k):
        matrix[i, 0] = 1.0
        matrix[i, i + 1] = -1.0
        matrix[i, i + 2] = 1.0
        rhs[i] = log_k
    charged = prepared.charges * ions
    matrix[-2, 0] = ae
    matrix[-2, 1:] = -charged
    rhs[-2] = ae * (y[0] - 1) - np.sum(charged * (y[1:] - 1))
    matrix[-1, 0] = ae
    matrix[-1, 1:] = ions
    rhs[-1] = (
        prepared.b_pressure + evaluation.alpha + ae * (y[0] - 1)
        + np.sum(ions * (y[1:] - 1))
    )
    if scale_rows:
        charge_scale = ae + np.sum(charged)
        pressure_scale = prepared.b_pressure + evaluation.alpha
        matrix[-2] /= charge_scale
        rhs[-2] /= charge_scale
        matrix[-1] /= pressure_scale
        rhs[-1] /= pressure_scale
    return matrix, rhs
