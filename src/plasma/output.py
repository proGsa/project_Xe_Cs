from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import AtomicData
from .settings import RunSettings
from .solver import Solution


def solution_row(solution: Solution, settings: RunSettings) -> dict:
    row = {
        "element": solution.element, "model": solution.mode,
        "T_K": solution.temperature_k, "p_Pa": solution.pressure_pa,
        "Z_max": settings.max_charge,
        "n_e_m3": "",
        **{f"n_{z}_m3": "" for z in range(settings.max_charge + 1)},
        "Gamma": "", "alpha": "", "deltaI_max_eV": "",
        "iterations": solution.iterations, "continuation_steps": solution.continuation_steps,
        "ion_error": "", "charge_error": "", "pressure_error": "", "gamma_error": "",
        "status": "ok" if solution.converged else "failed", "message": solution.message,
    }
    state = solution.state
    if state is not None:
        row.update(
            ion_error=state.errors.ionization, charge_error=state.errors.charge,
            pressure_error=state.errors.pressure, gamma_error=state.errors.gamma,
        )
    # Не публикуем пробные концентрации несошедшейся точки как результат.
    if solution.converged and state is not None:
        physical = state.density_star * settings.reference_density_m3
        row["n_e_m3"] = float(physical[0])
        for z, density in enumerate(physical[1:]):
            row[f"n_{z}_m3"] = float(density)
        row.update(Gamma=state.gamma, alpha=state.alpha, deltaI_max_eV=float(np.max(state.delta_ionization_ev)))
    return row


def write_csv(solutions: list[Solution], settings: RunSettings) -> Path:
    path = Path(settings.output_dir) / "concentrations.csv"
    rows = [solution_row(x, settings) for x in solutions]
    if not rows:
        raise ValueError("Нет результатов для сохранения.")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(solutions: list[Solution], settings: RunSettings) -> None:
    columns = ["element", "model", "T_K", "p_Pa", "Z_max", "n_e_m3"]
    columns += [f"n_{z}_m3" for z in range(settings.max_charge + 1)]
    columns += ["Gamma", "iterations", "status"]

    def formatted(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.5e}"
        return str(value) if value != "" else "—"

    rows = [[formatted(solution_row(x, settings)[key]) for key in columns] for x in solutions]
    widths = [max(len(key), *(len(row[i]) for row in rows)) for i, key in enumerate(columns)]
    print(" | ".join(key.ljust(width) for key, width in zip(columns, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def plot_results(solutions: list[Solution], settings: RunSettings) -> list[Path]:
    import matplotlib
    if not settings.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(settings.output_dir) / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for element in settings.elements:
        for mode in settings.models:
            selected = sorted(
                (x for x in solutions if x.element == element and x.mode == mode),
                key=lambda x: x.temperature_k,
            )
            if not selected:
                continue
            temperature = np.asarray([x.temperature_k for x in selected])
            # NaN прерывает кривую в несошедшейся точке.
            density = np.full((len(selected), settings.max_charge + 2), np.nan)
            for i, solution in enumerate(selected):
                if solution.converged and solution.state is not None:
                    density[i] = solution.state.density_star * settings.reference_density_m3
            fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
            labels = ["Электроны", f"{element}, нейтралы"]
            labels += [f"{element}, заряд +{z}" for z in range(1, settings.max_charge + 1)]
            for i, label in enumerate(labels):
                ax.plot(
                    temperature, density[:, i], marker=".", markersize=4, label=label,
                    linestyle="--" if i == 0 else "-", linewidth=1.6,
                    zorder=5 if i == 0 else 2,
                )
            ax.set_yscale("log")
            ax.set_xlabel("Температура T, K")
            ax.set_ylabel("Концентрация, м⁻³")
            ax.set_title(f"{element}: {mode}; Tₑ = T; p = {settings.pressure_pa:g} Па")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            failed = sum(not x.converged for x in selected)
            if failed:
                ax.text(0.02, 0.02, f"Несошедшиеся точки: {failed}; показаны пропусками",
                        transform=ax.transAxes, fontsize=8)
            if not np.isfinite(density).any():
                ax.set_ylim(1, 10)
                ax.text(0.5, 0.5, "Нет сошедшихся точек", ha="center", transform=ax.transAxes)
            path = folder / f"{element}_{mode}.png"
            fig.savefig(path, dpi=180)
            paths.append(path)
            if not settings.show_plots:
                plt.close(fig)
    if settings.show_plots:
        plt.show()
        plt.close("all")
    return paths


def write_metadata(settings: RunSettings, data: dict[str, AtomicData], failed: int) -> Path:
    values = asdict(settings)
    values["input_dir"] = str(Path(settings.input_dir).resolve())
    values["output_dir"] = str(Path(settings.output_dir).resolve())
    payload = {
        "settings": values,
        "physical_model": "one-temperature; fixed finite bound-level tables; seminar frozen-correction Newton",
        "density_unit": "m^-3", "pressure_unit": "Pa", "energy_unit_after_loading": "eV",
        "failed_points": failed,
        "loaded_levels": {name: [len(x.energies_ev) for x in atom.levels] for name, atom in data.items()},
        "notes": [
            "Gamma=0 in ideal mode is a disabled correction, not a separate physical estimate.",
            "Atomic-level cutoff does not depend on DeltaI in this version.",
            "iterations counts the final point solve; continuation_steps counts intermediate temperatures.",
        ],
    }
    path = Path(settings.output_dir) / "run_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
