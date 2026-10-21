from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (
    Evaluation, InvalidState, PreparedState, analytic_initial_guess,
    assemble_system, evaluate_state,
)
from .settings import RunSettings


@dataclass(frozen=True)
class Solution:
    temperature_k: float
    pressure_pa: float
    element: str
    mode: str
    converged: bool
    iterations: int
    state: Evaluation | None
    message: str
    continuation_steps: int = 0


def solve_state(
    prepared: PreparedState, settings: RunSettings, mode: str,
    initial: np.ndarray | None = None,
) -> Solution:
    y = analytic_initial_guess(prepared) if initial is None else np.asarray(initial, dtype=float).copy()
    current = None

    def result(ok: bool, iterations: int, message: str) -> Solution:
        return Solution(
            prepared.temperature_k, prepared.pressure_pa, prepared.data.element,
            mode, ok, iterations, current, message,
        )

    try:
        current = evaluate_state(y, prepared, mode, settings)
    except (InvalidState, FloatingPointError) as exc:
        return result(False, 0, str(exc))

    for iteration in range(settings.max_iterations + 1):
        if current.errors.maximum <= settings.relative_tolerance and current.errors.gamma <= settings.gamma_tolerance:
            return result(True, iteration, "Уравнения выполнены с заданной точностью.")
        if iteration == settings.max_iterations:
            break
        matrix, rhs = assemble_system(current, prepared)
        try:
            # LAPACK gesv: LU с выбором главного элемента (реализация Гаусса).
            y_linear = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return result(False, iteration, "Вырожденная матрица линейной системы.")
        direction = y_linear - current.y
        if not np.all(np.isfinite(direction)):
            return result(False, iteration, "Линейный решатель вернул неконечный шаг.")
        size = float(np.max(np.abs(direction)))
        eta = min(1.0, settings.max_log_step / size) if size > 0 else 1.0
        accepted = None
        for _ in range(settings.max_backtracks):
            try:
                trial = evaluate_state(current.y + eta * direction, prepared, mode, settings)
            except (InvalidState, FloatingPointError):
                trial = None
            if trial is not None and (
                trial.errors.maximum < current.errors.maximum
                or trial.errors.maximum <= settings.relative_tolerance
            ):
                accepted = trial
                break
            eta /= 2
        if accepted is None:
            return result(False, iteration + 1, "Не найден уменьшающий ошибки шаг; попробуй меньший шаг температуры или другое приближение.")
        current = accepted
    return result(False, settings.max_iterations, "Достигнут предел итераций.")
