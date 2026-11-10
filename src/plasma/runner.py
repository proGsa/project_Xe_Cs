from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from .data import AtomicData, DataError, load_atomic_data
from .model import prepare_state
from .output import plot_results, print_table, write_csv, write_metadata
from .settings import RunSettings
from .solver import Solution, solve_state


@dataclass(frozen=True)
class RunSummary:
    solutions: tuple[Solution, ...]
    validation_path: Path
    csv_path: Path | None = None
    plot_paths: tuple[Path, ...] = ()
    metadata_path: Path | None = None

    @property
    def failed_points(self) -> int:
        return sum(not s.converged for s in self.solutions)


def validate_inputs(settings: RunSettings) -> tuple[dict[str, AtomicData], Path]:
    settings.validate()
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    atoms, reports, errors = {}, {}, []
    for element in settings.elements:
        try:
            atom = load_atomic_data(element, settings)
        except DataError as exc:
            reports[element] = {"status": "failed", "message": str(exc)}
            errors.append(str(exc))
        else:
            atoms[element] = atom
            reports[element] = {"status": "ok", "files": [asdict(x) for x in atom.reports]}
    path = Path(settings.output_dir) / "data_validation.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise DataError("\n".join(errors) + f"\nОтчёт: {path}")
    return atoms, path


def _continue_to(
    atom: AtomicData, previous: Solution, target_t: float, mode: str,
    settings: RunSettings, depth: int = 0,
) -> Solution:
    assert previous.state is not None and previous.converged
    prepared = prepare_state(atom, target_t, settings)
    target = solve_state(prepared, settings, mode, previous.state.y)
    gap = abs(target_t - previous.temperature_k)
    if target.converged or depth >= settings.continuation_depth or gap / 2 < settings.minimum_temperature_step_k:
        return target
    middle_t = (previous.temperature_k + target_t) / 2
    middle = _continue_to(atom, previous, middle_t, mode, settings, depth + 1)
    if not middle.converged:
        return target
    retry = _continue_to(atom, middle, target_t, mode, settings, depth + 1)
    return replace(retry, continuation_steps=middle.continuation_steps + retry.continuation_steps + 1)


def temperature_sweep(
    data: dict[str, AtomicData], settings: RunSettings,
    progress: Callable[[str], None] | None = None,
) -> list[Solution]:
    solutions = []
    temperatures = settings.temperatures()
    for element in settings.elements:
        atom = data[element]
        previous: dict[str, Solution] = {}
        for temperature in temperatures:
            prepared = prepare_state(atom, float(temperature), settings)
            ideal_here = None
            for mode in ("ideal", "nonideal"):
                selected = mode in settings.models
                warmup = mode == "ideal" and "nonideal" in settings.models and "nonideal" not in previous
                if not selected and not warmup:
                    continue
                if mode in previous:
                    solution = _continue_to(atom, previous[mode], float(temperature), mode, settings)
                else:
                    seed = ideal_here.state.y if mode == "nonideal" and ideal_here is not None and ideal_here.converged else None
                    solution = solve_state(prepared, settings, mode, seed)
                if mode == "ideal":
                    ideal_here = solution
                elif not solution.converged and ideal_here is not None and ideal_here.converged:
                    retry = solve_state(prepared, settings, mode, ideal_here.state.y)
                    if retry.converged:
                        solution = retry
                if solution.converged:
                    previous[mode] = solution
                if selected:
                    solutions.append(solution)
                    if progress:
                        error = solution.state.errors.maximum if solution.state is not None else float("inf")
                        progress(f"{element:2} {mode:8} T={temperature:8.2f} K: "
                                 f"{'ok' if solution.converged else 'FAILED'}, "
                                 f"итераций={solution.iterations}, ошибка={error:.3e}")
    return solutions


def run(settings: RunSettings, validate_only: bool = False, quiet: bool = False) -> RunSummary:
    atoms, validation = validate_inputs(settings)
    if not quiet:
        print("Модель: Tₑ = T; давление в Па; выходные концентрации в м⁻³.")
        skipped_groups = [g for g in settings.plot_groups if g > settings.max_charge]
        if skipped_groups:
            print(f"Группы графиков {skipped_groups} пропущены: для них нужен больший max_charge.")
        for name, atom in atoms.items():
            print(f"{name}: уровней по зарядам: {[len(x.energies_ev) for x in atom.levels]}")
            for report in atom.reports:
                for warning in report.warnings:
                    print(f"  {Path(report.path).name}: {warning}")
        print(f"Отчёт входных данных: {validation}")
    if validate_only:
        return RunSummary((), validation)
    solutions = temperature_sweep(atoms, settings, None if quiet else print)
    csv_path = write_csv(solutions, settings)
    plots = plot_results(solutions, settings)
    metadata = write_metadata(settings, atoms, sum(not x.converged for x in solutions))
    if not quiet:
        print()
        print_table(solutions, settings)
        print(f"\nCSV: {csv_path}")
        print(f"Графики: {Path(settings.output_dir) / 'plots'}")
        for solution in solutions:
            if not solution.converged:
                print(f"Не сошлось: {solution.element}, {solution.mode}, {solution.temperature_k:g} K: {solution.message}")
    return RunSummary(tuple(solutions), validation, csv_path, tuple(plots), metadata)
