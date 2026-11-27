from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .data import DataError
from .runner import run
from .settings import RunSettings, format_temperature_ratio


def main(settings: RunSettings | None = None, argv: list[str] | None = None) -> int:
    if settings is None:
        from config import SETTINGS
        settings = SETTINGS
    parser = argparse.ArgumentParser(description="Состав плазмы Xe/Cs и свойства Xe при одной или двух температурах.")
    parser.add_argument("--validate-only", action="store_true", help="Проверить CSV без расчёта состава.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--element", choices=("Xe", "Cs"), help="Рассчитать только один элемент.")
    parser.add_argument("--model", choices=("ideal", "nonideal"), help="Рассчитать только один режим.")
    parser.add_argument("--max-charge", type=int)
    sweep_options = parser.add_mutually_exclusive_group()
    sweep_options.add_argument("--two-temperature", action="store_true",
                               help="Идеальная прогонка по T_e/T_h при фиксированных p,T_h; отдельная папка результатов.")
    sweep_options.add_argument("--fixed-gamma", type=float,
                               help="Рост T_h при фиксированных gamma=T_e/T_h и p. При gamma!=1 только ideal.")
    parser.add_argument("--heavy-temperature-k", type=float,
                        help="Фиксированная T_h в K для двухтемпературного режима.")
    parser.add_argument("--temperature-ratios", type=float, nargs="+",
                        help="Возрастающая сетка gamma=T_e/T_h; точка 1 добавляется автоматически.")
    parser.add_argument("--pressure-pa", type=float, help="Фиксированное давление в Па.")
    parser.add_argument("--temperature-start-k", type=float, help="Начальная T_h в K для температурной прогонки.")
    parser.add_argument("--temperature-stop-k", type=float, help="Конечная T_h в K для температурной прогонки.")
    parser.add_argument("--temperature-step-k", type=float, help="Шаг T_h в K для температурной прогонки.")
    parser.add_argument("--plot-groups", type=int, nargs="+", choices=(1, 2, 3, 4),
                        help="Группы графиков, например --plot-groups 1 3 4.")
    parser.add_argument("--plot-all", action="store_true",
                        help="Дополнительно сохранить общий график всех частиц.")
    properties_options=parser.add_mutually_exclusive_group()
    properties_options.add_argument("--xe-properties",dest="xe_properties_enabled",action="store_true",default=None,
                                    help="Свойства Xe: Te=Th или ideal при Te!=Th. Сетка Xe ограничивается таблицами.")
    properties_options.add_argument("--no-xe-properties",dest="xe_properties_enabled",action="store_false",
                                    help="Отключить дополнительные свойства Xe.")
    parser.add_argument("--xe-data",dest="xe_lte_data_path",type=Path,
                        help="Путь к xe_lte_data.zip или распакованной папке; включает свойства Xe.")
    parser.add_argument("--xe-collision-branch",choices=("traditional","phenomenological"))
    parser.add_argument("--xe-high-ion-model",choices=("langevin_isotropic_capture","error"),
                        help="Модель отсутствующих интегралов Xe–Xe2+,3+,4+; error запрещает приближение.")
    parser.add_argument("--quiet", action="store_true", help="Не печатать промежуточные результаты и таблицу.")
    args = parser.parse_args(argv)
    if args.fixed_gamma is not None and (args.heavy_temperature_k is not None or args.temperature_ratios is not None):
        parser.error("--fixed-gamma нельзя сочетать с --heavy-temperature-k или --temperature-ratios: "
                     "в этом режиме растёт T_h, а отношение температур фиксировано.")
    updates = {}
    for key in ("input_dir", "output_dir", "max_charge", "pressure_pa", "heavy_temperature_k",
                "temperature_start_k", "temperature_stop_k", "temperature_step_k"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if args.two_temperature or args.heavy_temperature_k is not None or args.temperature_ratios is not None:
        updates["sweep_mode"] = "temperature_ratio"
        updates["models"] = ("ideal",)
        if settings.sweep_mode != "temperature_ratio" and args.output_dir is None:
            folder = Path(settings.output_dir)
            updates["output_dir"] = folder.with_name(folder.name + "_two_temperature")
    if args.temperature_ratios is not None:
        updates["temperature_ratios"] = tuple(args.temperature_ratios)
    if args.fixed_gamma is not None:
        updates["sweep_mode"] = "fixed_ratio"
        updates["fixed_temperature_ratio"] = args.fixed_gamma
        if args.fixed_gamma != 1.0:
            updates["models"] = ("ideal",)
        if args.output_dir is None:
            folder = Path(settings.output_dir)
            label = format_temperature_ratio(args.fixed_gamma)
            updates["output_dir"] = folder.with_name(folder.name + f"_fixed_gamma_{label}")
    if args.element:
        updates["elements"] = (args.element,)
    if args.model:
        updates["models"] = (args.model,)
    if args.plot_groups is not None:
        updates["plot_groups"] = tuple(args.plot_groups)
    if args.plot_all:
        updates["plot_all_species"] = True
    if args.xe_lte_data_path is not None:
        updates.update(xe_lte_data_path=args.xe_lte_data_path,xe_properties_enabled=True)
    if args.xe_properties_enabled is not None:
        updates["xe_properties_enabled"]=args.xe_properties_enabled
    if args.xe_collision_branch is not None:
        updates["xe_collision_branch"]=args.xe_collision_branch
    if args.xe_high_ion_model is not None:
        updates["xe_high_ion_neutral_model"]=args.xe_high_ion_model
    settings = replace(settings, **updates)
    try:
        summary = run(settings, validate_only=args.validate_only, quiet=args.quiet)
    except (DataError, ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    if summary.failed_points:
        print(f"Несошедшихся точек: {summary.failed_points}. См. status/message в CSV.", file=sys.stderr)
        return 1
    if summary.xe_property_failed_points:
        print(f"Неполные свойства Xe (ошибка или выход за область интегралов): точек {summary.xe_property_failed_points}. "
              "См. статусы и message в xe_properties.csv; состав сохранён.",file=sys.stderr)
        return 1
    return 0
