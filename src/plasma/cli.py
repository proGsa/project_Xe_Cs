from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .data import DataError
from .runner import run
from .settings import RunSettings


def main(settings: RunSettings | None = None, argv: list[str] | None = None) -> int:
    if settings is None:
        from config import SETTINGS
        settings = SETTINGS
    parser = argparse.ArgumentParser(description="Семинарский расчёт однотемпературной плазмы Xe и Cs.")
    parser.add_argument("--validate-only", action="store_true", help="Проверить CSV без расчёта состава.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--element", choices=("Xe", "Cs"), help="Рассчитать только один элемент.")
    parser.add_argument("--model", choices=("ideal", "nonideal"), help="Рассчитать только один режим.")
    parser.add_argument("--max-charge", type=int)
    parser.add_argument("--plot-groups", type=int, nargs="+", choices=(1, 2, 3, 4),
                        help="Группы графиков, например --plot-groups 1 3 4.")
    parser.add_argument("--plot-all", action="store_true",
                        help="Дополнительно сохранить общий график всех частиц.")
    parser.add_argument("--quiet", action="store_true", help="Не печатать промежуточные результаты и таблицу.")
    args = parser.parse_args(argv)
    updates = {}
    for key in ("input_dir", "output_dir", "max_charge"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if args.element:
        updates["elements"] = (args.element,)
    if args.model:
        updates["models"] = (args.model,)
    if args.plot_groups is not None:
        updates["plot_groups"] = tuple(args.plot_groups)
    if args.plot_all:
        updates["plot_all_species"] = True
    settings = replace(settings, **updates)
    try:
        summary = run(settings, validate_only=args.validate_only, quiet=args.quiet)
    except (DataError, ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    if summary.failed_points:
        print(f"Несошедшихся точек: {summary.failed_points}. См. status/message в CSV.", file=sys.stderr)
        return 1
    return 0
