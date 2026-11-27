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
from .xe_dataset import XeDataset
from .xe_domain import ratio_grid, temperature_grid
from .xe_properties import run_xe_properties


@dataclass(frozen=True)
class RunSummary:
    solutions: tuple[Solution, ...]
    validation_path: Path
    csv_path: Path | None = None
    plot_paths: tuple[Path, ...] = ()
    metadata_path: Path | None = None
    xe_properties_path: Path | None = None
    xe_property_failed_points: int = 0

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
    *, electron_temperature_k: float | None = None,
) -> Solution:
    assert previous.state is not None and previous.converged
    target_te = target_t if electron_temperature_k is None else electron_temperature_k
    prepared = prepare_state(atom, target_t, settings, electron_temperature_k=target_te)
    target = solve_state(prepared, settings, mode, previous.state.y)
    previous_te = (previous.temperature_k if previous.electron_temperature_k is None
                   else previous.electron_temperature_k)
    gap = max(abs(target_t - previous.temperature_k), abs(target_te - previous_te))
    if target.converged or depth >= settings.continuation_depth or gap / 2 < settings.minimum_temperature_step_k:
        return target
    middle_t = (previous.temperature_k + target_t) / 2
    middle_te = (previous_te + target_te) / 2
    middle = _continue_to(
        atom, previous, middle_t, mode, settings, depth + 1,
        electron_temperature_k=middle_te,
    )
    if not middle.converged:
        return target
    retry = _continue_to(
        atom, middle, target_t, mode, settings, depth + 1,
        electron_temperature_k=target_te,
    )
    return replace(retry, continuation_steps=middle.continuation_steps + retry.continuation_steps + 1)


def temperature_sweep(
    data: dict[str, AtomicData], settings: RunSettings,
    progress: Callable[[str], None] | None = None,
) -> list[Solution]:
    """Рост T_h при gamma=1 либо при заданном постоянном T_e/T_h."""
    solutions = []
    fixed_ratio = settings.sweep_mode == "fixed_ratio"
    ratio = settings.fixed_temperature_ratio if fixed_ratio else 1.0
    for element in settings.elements:
        atom = data[element]
        temperatures = temperature_grid(settings,element)
        previous: dict[str, Solution] = {}
        for temperature in temperatures:
            te = float(ratio * temperature)
            prepared = prepare_state(atom, float(temperature), settings, electron_temperature_k=te)
            ideal_here = None
            for mode in ("ideal", "nonideal"):
                selected = mode in settings.models
                warmup = mode == "ideal" and "nonideal" in settings.models and "nonideal" not in previous
                if not selected and not warmup:
                    continue
                if mode in previous:
                    solution = _continue_to(
                        atom, previous[mode], float(temperature), mode, settings,
                        electron_temperature_k=te,
                    )
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
                        condition = (f"T_h={temperature:8.2f} K, T_e={te:8.2f} K, gamma={ratio:g}"
                                     if fixed_ratio else f"T={temperature:8.2f} K")
                        progress(f"{element:2} {mode:8} {condition}: "
                                 f"{'ok' if solution.converged else 'FAILED'}, "
                                 f"итераций={solution.iterations}, ошибка={error:.3e}")
    return solutions


def temperature_ratio_sweep(
    data: dict[str, AtomicData], settings: RunSettings,
    progress: Callable[[str], None] | None = None,
) -> list[Solution]:
    """Фиксированные p,T_h; базовое решение при gamma=1, затем протяжка по gamma."""
    settings.validate()
    if settings.sweep_mode != "temperature_ratio":
        raise ValueError("Для прогонки отношения температур требуется sweep_mode='temperature_ratio'.")
    solutions = []
    th = float(settings.heavy_temperature_k)
    for element in settings.elements:
        atom = data[element]
        ratios = ratio_grid(settings,element)
        baseline = solve_state(prepare_state(atom, th, settings), settings, "ideal")
        previous = baseline if baseline.converged else None
        for ratio in ratios:
            te = float(ratio * th)
            if ratio == 1.0:
                solution = baseline
            elif previous is not None:
                solution = _continue_to(
                    atom, previous, th, "ideal", settings, electron_temperature_k=te,
                )
            else:
                solution = solve_state(
                    prepare_state(atom, th, settings, electron_temperature_k=te),
                    settings, "ideal",
                )
            if solution.converged:
                previous = solution
            solutions.append(solution)
            if progress:
                error = solution.state.errors.maximum if solution.state is not None else float("inf")
                progress(f"{element:2} ideal T_h={th:g} K, T_e={te:g} K, "
                         f"gamma={ratio:g}: {'ok' if solution.converged else 'FAILED'}, "
                         f"итераций={solution.iterations}, ошибка={error:.3e}")
    return solutions


def run(settings: RunSettings, validate_only: bool = False, quiet: bool = False) -> RunSummary:
    atoms, validation = validate_inputs(settings)
    dataset = None
    if settings.xe_properties_enabled and "Xe" in atoms:
        data_path = settings.xe_lte_data_path or Path(settings.input_dir)/"xe_lte_data.zip"
        dataset = XeDataset(data_path)
    if not quiet:
        if settings.sweep_mode == "temperature_ratio":
            print(f"Идеальная двухтемпературная модель: T_h={settings.heavy_temperature_k:g} K, "
                  f"p={settings.pressure_pa:g} Па; прогонка по gamma=T_e/T_h; концентрации в м⁻³.")
        elif settings.sweep_mode == "fixed_ratio":
            print(f"Прогонка по T_h: gamma=T_e/T_h={settings.fixed_temperature_ratio:g} фиксирована, "
                  f"T_e=gamma*T_h, p={settings.pressure_pa:g} Па; концентрации в м⁻³.")
        else:
            print("Модель: Tₑ = T_h = T; давление в Па; выходные концентрации в м⁻³.")
        skipped_groups = [g for g in settings.plot_groups if g > settings.max_charge]
        if skipped_groups:
            print(f"Группы графиков {skipped_groups} пропущены: для них нужен больший max_charge.")
        for name, atom in atoms.items():
            print(f"{name}: уровней по зарядам: {[len(x.energies_ev) for x in atom.levels]}")
            for report in atom.reports:
                for warning in report.warnings:
                    print(f"  {Path(report.path).name}: {warning}")
        print(f"Отчёт входных данных: {validation}")
        if dataset is not None:
            print(f"Свойства Xe: LTE либо ideal при Te!=Th; ветвь {settings.xe_collision_branch}.")
            print(f"Модель отсутствующих пар Xe–Xe2+,3+,4+: {settings.xe_high_ion_neutral_model}.")
            if settings.sweep_mode == "temperature_ratio":
                grid=ratio_grid(settings,"Xe")
                print(f"Сетка Xe ограничена таблицами: gamma={grid[0]:g}..{grid[-1]:g}, точек {len(grid)}; Th фиксирована.")
            else:
                grid=temperature_grid(settings,"Xe")
                print(f"Сетка Xe ограничена таблицами: Th={grid[0]:g}..{grid[-1]:g} K, точек {len(grid)}.")
            print("Cp,Cv: производные по Th при gamma=const; lambda относится к градиенту Th.")
    if validate_only:
        return RunSummary((), validation)
    sweep = temperature_ratio_sweep if settings.sweep_mode == "temperature_ratio" else temperature_sweep
    solutions = sweep(atoms, settings, None if quiet else print)
    properties = (run_xe_properties(solutions,atoms["Xe"],dataset,settings,None if quiet else print)
                  if dataset is not None else None)
    csv_path = write_csv(solutions, settings, properties.rows if properties else None)
    plots = plot_results(solutions, settings)
    if properties:
        plots.extend(properties.plot_paths)
    metadata = write_metadata(settings, atoms, sum(not x.converged for x in solutions),
                              properties.metadata if properties else None)
    if not quiet:
        print()
        print_table(solutions, settings)
        if properties:
            print("\nСвойства Xe: u,h в Дж/кг; Cp,Cv в Дж/(кг·К); sigma в См/м; eta в Па·с; lambda в Вт/(м·К).")
            keys=("model","T_h_K","T_e_K","gamma_Te_Th","u_J_kg","h_J_kg","cp_J_kg_K","cv_J_kg_K","sigma_S_m","eta_Pa_s","lambda_W_m_K","properties_status")
            print(" | ".join(keys))
            for row in properties.rows:
                print(" | ".join(f"{row[k]:.6e}" if isinstance(row[k],float) else str(row[k]) or "—" for k in keys))
            print(f"Свойства и отдельные эталоны: {properties.csv_path}")
        print(f"\nCSV: {csv_path}")
        print(f"Графики: {Path(settings.output_dir) / 'plots'}")
        for solution in solutions:
            if not solution.converged:
                print(f"Не сошлось: {solution.element}, {solution.mode}, "
                      f"T_h={solution.temperature_k:g} K, gamma={solution.temperature_ratio:g}: {solution.message}")
    return RunSummary(tuple(solutions), validation, csv_path, tuple(plots), metadata,
                      properties.csv_path if properties else None, properties.failed_points if properties else 0)
