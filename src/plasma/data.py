"""Чтение исходных NIST CSV и простых нормализованных CSV.

Не выполняет формулы Excel: обёртка ="..." снимается как текст.
Сохраняет в отчёте причины исключения строк; отсутствующие числа не заменяет 0.
"""

from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import constants

from .settings import RunSettings


class DataError(ValueError):
    """Входные данные не задают требуемую модель."""


@dataclass
class FileReport:
    path: str
    kind: str
    unit: str = ""
    total_rows: int = 0
    accepted_rows: int = 0
    estimated_rows: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    skipped_examples: dict[str, list[dict]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def skip(self, reason: str, line: int, headers: list[str], row: list[str]) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1
        examples = self.skipped_examples.setdefault(reason, [])
        if len(examples) < 3:
            examples.append({"line": line, "values": {
                header: clean_cell(value) for header, value in zip(headers, row) if header
            }})


@dataclass(frozen=True)
class LevelData:
    energies_ev: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class AtomicData:
    element: str
    levels: tuple[LevelData, ...]
    ionization_ev: np.ndarray
    reports: tuple[FileReport, ...] = ()

    @property
    def max_charge(self) -> int:
        return len(self.levels) - 1


def clean_cell(value: str | None) -> str:
    text = (value or "").strip().lstrip("\ufeff")
    match = re.fullmatch(r'=\s*"((?:[^"]|"")*)"', text)
    if match:
        text = match.group(1).replace('""', '"').strip()
    return text


def normalize_header(value: str) -> str:
    text = unicodedata.normalize("NFKD", clean_cell(value)).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def parse_number(value: str) -> float:
    text = clean_cell(value).replace("−", "-").replace("–", "-")
    if not text or "?" in text or re.search(r"\+\s*[xyz]\b", text, re.I):
        raise ValueError("пустое/неопределённое число")
    # NIST: [33.195], (43.0), 12.12984(15). Последняя скобка — неопределённость.
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1].strip()
    text = re.sub(r"(?<=[0-9])\([0-9]+\)$", "", text)
    text = re.sub(r"\s+", "", text).replace("D", "E").replace("d", "e")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("число не является конечным")
    return result


def _level_energy_column(headers: list[str]) -> int | None:
    for i, header in enumerate(headers):
        key = normalize_header(header)
        if key.startswith(("excitationenergy", "energyev", "energycm")) or key == "energy":
            return i
        if key.startswith("level") and not any(x in key for x in ("id", "splitting", "uncertainty")):
            return i
    return None


def _ion_energy_column(headers: list[str]) -> int | None:
    return next((i for i, h in enumerate(headers) if normalize_header(h).startswith("ionizationenergy")), None)


def _column(headers: list[str], *names: str) -> int | None:
    return next((i for i, h in enumerate(headers) if normalize_header(h) in names), None)


def _cell(row: list[str], index: int | None) -> str:
    return clean_cell(row[index]) if index is not None and index < len(row) else ""


def _read_table(path: Path, kind: str) -> tuple[list[str], list[tuple[int, list[str]]]]:
    if not path.is_file():
        raise DataError(f"Нет файла {path}. Помести сюда соответствующий CSV NIST.")
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    if text is None:
        raise DataError(f"Не удалось определить кодировку {path}.")
    if re.search(r"<(?:!doctype\s+html|html)\b", text[:1000], re.I):
        raise DataError(f"{path}: сохранён HTML. В NIST выбери CSV (text) или Tab-delimited.")
    candidates = []
    for delimiter in (",", "\t", ";"):
        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
            rows = []
            start_line = 1
            for row in reader:
                rows.append((start_line, row))
                start_line = reader.line_num + 1
        except csv.Error:
            continue
        for index, (_, row) in enumerate(rows[:100]):
            headers = [clean_cell(x) for x in row]
            if kind == "levels":
                valid = _level_energy_column(headers) is not None and _column(
                    headers, "j", "g", "statisticalweight", "degeneracy"
                ) is not None
            else:
                valid = _ion_energy_column(headers) is not None and _column(
                    headers, "charge", "ioncharge", "spectrum", "spectrumname", "spname"
                ) is not None
            if valid:
                candidates.append((len(headers), -index, headers, rows[index + 1:]))
                break
    if not candidates:
        required = "энергия уровня и J/g" if kind == "levels" else "Ion charge и Ionization energy"
        raise DataError(f"{path}: не найдена таблица ({required}). Проверь заголовки и формат CSV.")
    _, _, headers, rows = max(candidates, key=lambda x: (x[0], x[1]))
    return headers, rows


def energy_factor(header: str, fallback: str) -> tuple[float, str, bool]:
    key = normalize_header(header)
    inferred = True
    if "hartree" in key:
        unit = "Hartree"
    elif "ryd" in key or key.endswith("ry"):
        unit = "Rydberg"
    elif "ghz" in key:
        unit = "GHz"
    elif "cm1" in key:
        unit = "cm-1"
    elif "ev" in key:
        unit = "eV"
    else:
        unit = fallback
        inferred = False
    normalized = normalize_header(unit)
    factors = {
        "ev": 1.0,
        "cm1": constants.h * constants.c * 100 / constants.e,
        "rydberg": constants.physical_constants["Rydberg constant times hc in eV"][0],
        "ryd": constants.physical_constants["Rydberg constant times hc in eV"][0],
        "hartree": constants.physical_constants["Hartree energy in eV"][0],
        "ghz": constants.h * 1e9 / constants.e,
    }
    if normalized not in factors:
        raise DataError(f"Неизвестная единица энергии: {unit!r}.")
    return factors[normalized], unit, inferred


def _weight_from_j(text: str) -> float:
    text = clean_cell(text)
    if not text or "?" in text or re.search(r"\bor\b", text, re.I):
        raise ValueError("неоднозначное или отсутствующее J")
    # По NIST, перечисление через запятую может обозначать неразрешённые уровни.
    # Суммируем их кратности; альтернативы через 'or' не суммируем.
    result = 0.0
    for part in text.split(","):
        j = Fraction(part.strip().strip("[]()"))
        if j < 0 or (2 * j).denominator != 1:
            raise ValueError("J должен быть неотрицательным целым/полуцелым")
        result += float(2 * j + 1)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("некорректный статистический вес")
    return result


def read_levels(path: Path, settings: RunSettings, threshold_ev: float | None) -> tuple[LevelData, FileReport]:
    headers, rows = _read_table(path, "levels")
    e_col = _level_energy_column(headers)
    assert e_col is not None
    j_col = _column(headers, "j")
    g_col = _column(headers, "g", "statisticalweight", "degeneracy")
    prefix_col = _column(headers, "prefix")
    suffix_col = _column(headers, "suffix")
    factor, unit, explicit = energy_factor(headers[e_col], settings.default_energy_unit)
    report = FileReport(str(path), "levels", unit)
    if not explicit:
        report.warnings.append(f"В заголовке нет единицы энергии: использована config.default_energy_unit={unit}.")
    if settings.bound_levels_only and (threshold_ev is None or not np.isfinite(threshold_ev)):
        raise DataError(f"{path}: для отбора связанных уровней нужен порог этого иона. Сохрани полную таблицу ионизации NIST.")
    energies, weights = [], []
    seen = Counter()
    for line, row in rows:
        if not any(clean_cell(x) for x in row):
            continue
        report.total_rows += 1
        if any(re.search(r"\blimit\b", clean_cell(x), re.I) for x in row):
            report.skip("ionization_limit", line, headers, row)
            continue
        suffix = _cell(row, suffix_col)
        if suffix.lower() == "a":
            report.skip("autoionizing_level", line, headers, row)
            continue
        try:
            energy_text = _cell(row, prefix_col) + _cell(row, e_col) + suffix
            energy = parse_number(energy_text) * factor
        except ValueError:
            report.skip("missing_or_ambiguous_energy", line, headers, row)
            continue
        if energy < -1e-10:
            report.skip("negative_excitation_energy", line, headers, row)
            continue
        if settings.bound_levels_only and energy >= float(threshold_ev):
            report.skip("above_bare_ionization_threshold", line, headers, row)
            continue
        try:
            if g_col is not None and _cell(row, g_col):
                weight = parse_number(_cell(row, g_col))
                if weight <= 0 or not math.isclose(weight, round(weight), abs_tol=1e-8):
                    raise ValueError("некорректный вес")
                if j_col is not None and _cell(row, j_col):
                    try:
                        j_weight = _weight_from_j(_cell(row, j_col))
                    except ValueError:
                        j_weight = weight
                    if not math.isclose(weight, j_weight, rel_tol=1e-8):
                        raise DataError(f"{path}: g={weight} не соответствует J={_cell(row, j_col)}. Проверь, что это статистический вес, а не Landé-g.")
            else:
                weight = _weight_from_j(_cell(row, j_col))
        except DataError:
            raise
        except (ValueError, ZeroDivisionError):
            report.skip("missing_or_ambiguous_statistical_weight", line, headers, row)
            continue
        energies.append(max(0.0, energy))
        weights.append(weight)
        report.estimated_rows += int(any(c in energy_text for c in "[]()"))
        seen[tuple(clean_cell(x) for x in row)] += 1
    if not energies:
        raise DataError(f"{path}: не найдено пригодных уровней; исключённые строки: {report.skipped}.")
    if not any(abs(x) < 1e-9 for x in energies):
        raise DataError(f"{path}: отсутствует основной уровень с энергией 0; нули автоматически не добавляются.")
    report.accepted_rows = len(energies)
    duplicates = sum(n - 1 for n in seen.values())
    if duplicates:
        report.warnings.append(f"Найдено {duplicates} точных повторов строк. Они сохранены; проверь отсутствие двойного учёта уровней.")
    if report.skipped.get("missing_or_ambiguous_energy", 0) or report.skipped.get("missing_or_ambiguous_statistical_weight", 0):
        report.warnings.append("Есть строки без однозначных E/J/g: проверь отчёт исключений. Они не заменены нулями.")
    if report.estimated_rows:
        report.warnings.append(f"Принято уровней с энергией в скобках NIST: {report.estimated_rows}; скобки не означают нулевую энергию.")
    order = np.argsort(energies, kind="stable")
    return LevelData(np.asarray(energies)[order], np.asarray(weights)[order]), report


def _charge_from_spectrum(text: str) -> int:
    match = re.fullmatch(r"(?:Xe|Cs)\s+([IVXLCDM]+)", clean_cell(text), re.I)
    if not match:
        raise ValueError("неизвестный спектр")
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result, previous = 0, 0
    for letter in reversed(match.group(1).upper()):
        value = values[letter]
        result += -value if value < previous else value
        previous = value
    return result - 1


def read_ionization(path: Path, element: str, settings: RunSettings) -> tuple[dict[int, float], FileReport]:
    headers, rows = _read_table(path, "ionization")
    e_col = _ion_energy_column(headers)
    assert e_col is not None
    charge_col = _column(headers, "charge", "ioncharge")
    spectrum_col = _column(headers, "spectrum", "spectrumname", "spname")
    element_col = _column(headers, "element", "elementname", "elname")
    prefix_col = _column(headers, "prefix")
    suffix_col = _column(headers, "suffix")
    factor, unit, explicit = energy_factor(headers[e_col], settings.default_energy_unit)
    report = FileReport(str(path), "ionization", unit)
    if not explicit:
        report.warnings.append(f"В заголовке нет единицы энергии: использована {unit}.")
    result: dict[int, float] = {}
    estimated = []
    aliases = {"Xe": {"xe", "xenon"}, "Cs": {"cs", "cesium", "caesium"}}
    for line, row in rows:
        if not any(clean_cell(x) for x in row):
            continue
        report.total_rows += 1
        if element_col is not None:
            name = _cell(row, element_col).lower()
            if name and name not in aliases[element]:
                raise DataError(f"{path}: строка содержит другой элемент: {name}.")
        spectrum = _cell(row, spectrum_col)
        spectrum_charge = None
        if spectrum:
            try:
                spectrum_charge = _charge_from_spectrum(spectrum)
            except ValueError as exc:
                raise DataError(f"{path}, строка {line}: неизвестный спектр {spectrum!r}.") from exc
            if spectrum.split()[0].lower() != element.lower():
                raise DataError(f"{path}, строка {line}: спектр {spectrum!r} содержит другой элемент.")
        try:
            if charge_col is not None and _cell(row, charge_col):
                raw_charge = parse_number(_cell(row, charge_col))
                if raw_charge < 0 or not raw_charge.is_integer():
                    raise ValueError("заряд не целый")
                charge = int(raw_charge)
            else:
                if spectrum_charge is None:
                    raise ValueError("нет заряда и спектра")
                charge = spectrum_charge
            energy_text = _cell(row, prefix_col) + _cell(row, e_col) + _cell(row, suffix_col)
            energy = parse_number(energy_text) * factor
            if energy <= 0:
                raise ValueError("порог должен быть положительным")
        except ValueError:
            report.skip("missing_or_ambiguous_charge_or_energy", line, headers, row)
            continue
        if spectrum_charge is not None and charge != spectrum_charge:
            raise DataError(f"{path}, строка {line}: Ion Charge={charge} не соответствует Sp. Name={spectrum!r}.")
        if charge in result and not math.isclose(result[charge], energy, rel_tol=1e-10):
            raise DataError(f"{path}: разные пороги для заряда {charge}.")
        result[charge] = energy
        report.accepted_rows += 1
        if any(c in energy_text for c in "[]()"):
            estimated.append(charge)
            report.estimated_rows += 1
    if estimated:
        report.warnings.append(f"Есть значения в скобках NIST для зарядов {estimated}; учти их неопределённость при интерпретации.")
    return result, report


def load_atomic_data(element: str, settings: RunSettings) -> AtomicData:
    root = Path(settings.input_dir)
    symbol = element.lower()
    thresholds, ion_report = read_ionization(root / "ionization_energies" / f"{symbol}.csv", element, settings)
    last_required = settings.max_charge if settings.bound_levels_only else settings.max_charge - 1
    missing = [z for z in range(last_required + 1) if z not in thresholds]
    if missing:
        raise DataError(f"{element}: нет порогов для зарядов {missing}. При bound_levels_only=True нужен также порог последнего иона.")
    levels, reports = [], [ion_report]
    for charge in range(settings.max_charge + 1):
        path = root / "energy_levels" / f"{symbol}_{charge + 1}.csv"
        table, report = read_levels(path, settings, thresholds.get(charge))
        levels.append(table)
        reports.append(report)
    ionization = np.asarray([thresholds.get(z, np.nan) for z in range(settings.max_charge + 1)], dtype=float)
    return AtomicData(element, tuple(levels), ionization, tuple(reports))
