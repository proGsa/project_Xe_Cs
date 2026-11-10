"""Начальные параметры. Пути отсчитываются от папки этого файла."""

from pathlib import Path

from plasma.settings import RunSettings

BASE_DIR = Path(__file__).resolve().parent

SETTINGS = RunSettings(
    input_dir=BASE_DIR / "input",
    output_dir=BASE_DIR / "results",
    elements=("Xe", "Cs"),
    models=("ideal", "nonideal"),
    pressure_pa=100_000.0,       # 1 бар; ввод всегда в Па.
    temperature_start_k=3000.0,
    temperature_stop_k=30000.0,
    temperature_step_k=250.0,
    max_charge=4,               # Совместный расчёт нейтрала и ионов +1, +2, +3, +4.
    reference_density_m3=1e24,  # 10^18 см^-3 — масштаб из семинарского листочка.
    default_energy_unit="eV", # Только для CSV, в заголовке которых нет единицы.
    bound_levels_only=True,    # Фиксированный набор уровней ниже исходного порога.
    relative_tolerance=1e-8,
    gamma_tolerance=1e-12,
    max_iterations=150,
    max_backtracks=24,
    max_log_step=2.0,
    continuation_depth=4,       # Дополнительные промежуточные температуры при сбое.
    minimum_temperature_step_k=10.0,
    show_plots=False,           # PNG сохраняются в любом случае.
    plot_groups=(1, 2, 3, 4),   # 1: 0,+1,e; 2: +1,e,+2; 3: +2,+3; 4: +3,+4.
    plot_all_species=True,     # Дополнительно сохранять общий график всех частиц.
)
