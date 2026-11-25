"""Начальные параметры. Пути отсчитываются от папки этого файла."""

from pathlib import Path

from plasma.settings import RunSettings

BASE_DIR = Path(__file__).resolve().parent

SETTINGS = RunSettings(
    input_dir=BASE_DIR / "input",
    output_dir=BASE_DIR / "results",
    elements=("Xe", "Cs"),
    models=("ideal", "nonideal"),
    sweep_mode="temperature",  # Прежняя прогонка: T_e=T_h=T.
    # Для нового режима запускай: uv run main.py --two-temperature
    # Или задай sweep_mode="temperature_ratio", models=("ideal",)
    # и output_dir=BASE_DIR / "results_two_temperature".
    heavy_temperature_k=4000.0,  # T_h фиксирована в двухтемпературной прогонке.
    temperature_ratios=tuple(1.0 + 0.25 * i for i in range(17)),  # gamma: 1..5.
    # Рост T_h при постоянном gamma: uv run main.py --fixed-gamma 2
    # Для запуска без флага: sweep_mode="fixed_ratio", models=("ideal",),
    # output_dir=BASE_DIR / "results_fixed_gamma_2".
    fixed_temperature_ratio=2.0,  # Только для sweep_mode="fixed_ratio": T_e=gamma*T_h.
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
    continuation_depth=4,       # Дополнительные промежуточные точки при сбое.
    minimum_temperature_step_k=10.0,
    show_plots=False,           # PNG сохраняются в любом случае.
    plot_groups=(1, 2, 3, 4),   # 1: 0,+1,e; 2: +1,e,+2; 3: +2,+3; 4: +3,+4.
    plot_all_species=True,     # Дополнительно сохранять общий график всех частиц.
)
