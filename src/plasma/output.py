from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import AtomicData
from .settings import RunSettings, format_temperature_ratio
from .solver import Solution


# Индексы столбцов density: электрон=0, нейтрал=1, заряд z -> z+1.
_PLOT_GROUPS = {
    1: ("group1_neutral_ion1_e", (1, 2, 0)),
    2: ("group2_ion1_e_ion2", (2, 0, 3)),
    3: ("group3_ion2_ion3", (3, 4)),
    4: ("group4_ion3_ion4", (4, 5)),
}
_SPECIES_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")


def solution_row(solution: Solution, settings: RunSettings) -> dict:
    row = {
        "element": solution.element, "model": solution.mode,
        "T_K": solution.temperature_k, "p_Pa": solution.pressure_pa,
        "T_h_K": solution.temperature_k,
        "T_e_K": (solution.temperature_k if solution.electron_temperature_k is None
                  else solution.electron_temperature_k),
        "gamma_Te_Th": solution.temperature_ratio,
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


def write_csv(solutions: list[Solution], settings: RunSettings,
              xe_properties: tuple[dict, ...] | None = None) -> Path:
    path = Path(settings.output_dir) / "concentrations.csv"
    rows = [solution_row(x, settings) for x in solutions]
    if xe_properties:
        from .xe_properties import _VALUES
        keys=(*_VALUES,"thermodynamic_model","thermodynamics_status","heat_capacity_status",
              "transport_status","properties_status","high_ion_neutral_model",
              "heat_capacity_temperature","heat_capacity_constraint","lambda_temperature_gradient",
              "reactive_transport_model")
        lookup={(r["model"],r["T_h_K"],r["gamma_Te_Th"]):r for r in xe_properties}
        for row in rows:
            extra=lookup.get((row["model"],row["T_h_K"],row["gamma_Te_Th"])) if row["element"]=="Xe" else None
            row.update({key:extra[key] if extra else "" for key in keys})
            row["properties_message"]=extra["message"] if extra else ""
            if extra is None:
                row["properties_status"]="not_applicable"
    if not rows:
        raise ValueError("Нет результатов для сохранения.")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(solutions: list[Solution], settings: RunSettings) -> None:
    columns = ["element", "model", "T_h_K", "T_e_K", "gamma_Te_Th", "p_Pa", "Z_max", "n_e_m3"]
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
            ratio_sweep = settings.sweep_mode == "temperature_ratio"
            fixed_ratio = settings.sweep_mode == "fixed_ratio"
            ratio_label = format_temperature_ratio(settings.fixed_temperature_ratio)
            if ratio_sweep:
                selected.sort(key=lambda x: x.temperature_ratio)
            abscissa = np.asarray([
                x.temperature_ratio if ratio_sweep else x.temperature_k for x in selected
            ])
            # NaN прерывает кривую в несошедшейся точке.
            density = np.full((len(selected), settings.max_charge + 2), np.nan)
            for i, solution in enumerate(selected):
                if solution.converged and solution.state is not None:
                    density[i] = solution.state.density_star * settings.reference_density_m3
            labels = ["Электроны", f"{element}, нейтралы"]
            labels += [f"{element}, заряд +{z}" for z in range(1, settings.max_charge + 1)]
            specifications = [
                (*_PLOT_GROUPS[g], f"группа {g}")
                for g in settings.plot_groups if g <= settings.max_charge
            ]
            if settings.plot_all_species:
                specifications.append(("", tuple(range(len(labels))), "все частицы"))
            for suffix, indices, description in specifications:
                fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
                for i in indices:
                    ax.plot(
                        abscissa, density[:, i], marker=".", markersize=4, label=labels[i],
                        color=_SPECIES_COLORS[i], linestyle="--" if i == 0 else "-",
                        linewidth=1.6, zorder=5 if i == 0 else 2,
                    )
                ax.set_yscale("log")
                # Обрезка!!!
                values = density[:, indices]
                positive = values[np.isfinite(values) & (values > 0)]

                if positive.size:
                    lower = max(
                        1e16,
                        10.0 ** np.floor(np.log10(positive.min())),
                    )
                    upper = 10.0 ** np.ceil(
                        np.log10(max(positive.max() * 1.1, lower * 10))
                    )
                    ax.set_ylim(lower, upper)
                    if positive.max() < lower:
                        ax.text(0.02, 0.95, f"Все концентрации ниже {lower:.0e} м⁻³",
                                va="top", transform=ax.transAxes, fontsize=9)

                xlabel = ("Отношение температур γ = Tₑ/Tₕ" if ratio_sweep else
                          "Температура тяжёлых частиц Tₕ, K" if fixed_ratio else "Температура T, K")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Концентрация, м⁻³")
                condition = (f"Tₕ = {settings.heavy_temperature_k:g} K" if ratio_sweep else
                             f"γ = {ratio_label}; Tₑ = γTₕ" if fixed_ratio else "Tₑ = Tₕ = T")
                ax.set_title(f"{element}: {mode}; {description}; {condition}; p = {settings.pressure_pa:g} Па")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend()
                failed = sum(not x.converged for x in selected)
                if failed:
                    ax.text(0.02, 0.02, f"Несошедшиеся точки: {failed}; показаны пропусками",
                            transform=ax.transAxes, fontsize=8)
                if not np.isfinite(density[:, indices]).any():
                    ax.set_ylim(1, 10)
                    ax.text(0.5, 0.5, "Нет сошедшихся точек", ha="center", transform=ax.transAxes)
                prefix = f"{element}_{mode}" + ("_two_temperature" if ratio_sweep else "")
                if fixed_ratio:
                    prefix += f"_fixed_gamma_{ratio_label}"
                filename = prefix + (f"_{suffix}" if suffix else "") + ".png"
                path = folder / filename
                fig.savefig(path, dpi=180)
                paths.append(path)
                if not settings.show_plots:
                    plt.close(fig)
    if settings.show_plots:
        plt.show()
        plt.close("all")
    return paths


def write_metadata(settings: RunSettings, data: dict[str, AtomicData], failed: int,
                   xe_properties: dict | None = None) -> Path:
    values = asdict(settings)
    values["input_dir"] = str(Path(settings.input_dir).resolve())
    values["output_dir"] = str(Path(settings.output_dir).resolve())
    values["xe_lte_data_path"] = str(Path(settings.xe_lte_data_path).resolve()) if settings.xe_lte_data_path is not None else None
    if settings.sweep_mode == "temperature_ratio":
        physical_model = (
            "two-temperature ideal; Saha and electronic excitation at T_e; fixed p,T_h; "
            "fixed finite bound-level tables; seminar Newton"
        )
    elif settings.sweep_mode == "fixed_ratio":
        physical_model = (
            "temperature sweep at fixed p and T_e/T_h; Saha and electronic excitation at T_e; "
            "nonideal corrections allowed only for T_e=T_h; fixed finite bound-level tables; seminar Newton"
        )
    else:
        physical_model = "one-temperature; fixed finite bound-level tables; seminar frozen-correction Newton"
    payload = {
        "settings": values,
        "physical_model": physical_model,
        "density_unit": "m^-3", "pressure_unit": "Pa", "energy_unit_after_loading": "eV",
        "temperature_unit": "K", "T_K_alias": "T_h_K",
        "sweep_variable": ("gamma_Te_Th" if settings.sweep_mode == "temperature_ratio" else
                           "T_h_K" if settings.sweep_mode == "fixed_ratio" else "T_K"),
        "temperature_ratio_grid": settings.ratios().tolist() if settings.sweep_mode == "temperature_ratio" else None,
        "fixed_temperature_ratio": settings.fixed_temperature_ratio if settings.sweep_mode == "fixed_ratio" else None,
        "failed_points": failed,
        "xe_properties": xe_properties,
        "skipped_plot_groups": [g for g in settings.plot_groups if g > settings.max_charge],
        "loaded_levels": {name: [len(x.energies_ev) for x in atom.levels] for name, atom in data.items()},
        "notes": [
            "Gamma=0 in ideal mode is a disabled correction, not a separate physical estimate.",
            "gamma_Te_Th is T_e/T_h; Gamma is a distinct nonideality correction parameter.",
            "For T_e != T_h only ideal mode is implemented; nonideal corrections require T_e=T_h.",
            "Both temperatures are prescribed; no time-dependent temperature or energy evolution is solved.",
            "Atomic-level cutoff does not depend on DeltaI in this version.",
            "iterations counts the final point solve; continuation_steps counts intermediate states.",
        ],
    }
    path = Path(settings.output_dir) / "run_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
