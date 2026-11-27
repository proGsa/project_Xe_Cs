"""Inputs from xe_lte_data: SI area integrals, constants and separate references."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.special import gammaln

from .data import DataError


class PropertyDomainError(ValueError):
    """A state is outside the chosen physical/data model, not a zero coefficient."""


class XeDataset:
    """Read a directory or the original ZIP without extracting or executing it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise DataError(f"Нет данных Xe: {self.path}. Укажи --xe-data путь к xe_lte_data.zip или папке.")
        self.is_zip = self.path.is_file()
        self.prefix = ""
        if self.is_zip:
            try:
                with ZipFile(self.path) as z:
                    names = z.namelist()
                    if "constants.json" not in names:
                        self.prefix = "xe_lte_data/"
                    if self.prefix + "constants.json" not in names:
                        raise DataError("Архив не содержит xe_lte_data/constants.json.")
            except BadZipFile as exc:
                raise DataError(f"Повреждён архив Xe: {self.path}") from exc
        self.constants = self.json("constants.json")
        self.models = self.json("models.json")
        self.catalog = self.json("reference/catalog.json")
        self.sources = self.json("sources.json")
        self.manifest = self.json("manifest_sha256.json")
        self.verified_files: set[str] = set()
        for key in ("k_B", "e", "h", "m_e", "epsilon_0"):
            if not np.isfinite(self.constant(key)) or self.constant(key) <= 0:
                raise DataError(f"Некорректная константа {key} в архиве Xe.")
        species = self.csv("atomic/species.csv")
        masses = {r["species"]: float(r["mass_kg"]) for r in species}
        try:
            self.masses = np.array([masses["e"], *[masses[f"Xe{z}"] for z in range(5)]])
        except KeyError as exc:
            raise DataError("В архиве Xe отсутствуют массы нужных частиц.") from exc
        if not np.all(np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise DataError("Некорректные массы Xe.")
        self.c4 = {int(p["charge"]): float(p["C4_J_m4"])
                   for p in self.models["optional_polarization_model"]["parameters"]}
        if set(self.c4) != {2, 3, 4} or any(not np.isfinite(v) or v <= 0 for v in self.c4.values()):
            raise DataError("Некорректные C4 для Xe–Xe2+,3+,4+.")
        # Verify used model/constant inputs as well as tables, not the executable examples.
        for name in ("constants.json", "models.json", "atomic/species.csv", "sources.json", "reference/catalog.json"):
            self.read(name)
        for pair, branch in (("Xe0_e", "shared"), ("Xe0_Xe0", "traditional"),
                             ("Xe0_Xe0", "phenomenological"), ("Xe0_Xe1", "traditional"),
                             ("Xe0_Xe1", "phenomenological")):
            self.table(f"collision_integrals/{pair}_{branch}.csv", "temperature_K")
        self.table("collision_integrals/coulomb_reduced.csv", "reduced_temperature_Tstar")

    def read(self, name: str) -> bytes:
        try:
            if self.is_zip:
                with ZipFile(self.path) as z:
                    raw = z.read(self.prefix + name)
            else:
                raw = (self.path / name).read_bytes()
        except (OSError, KeyError, BadZipFile) as exc:
            raise DataError(f"Не удалось прочитать данные Xe: {name}: {exc}") from exc
        if hasattr(self, "manifest"):
            expected = self.manifest.get(name)
            if expected is None or hashlib.sha256(raw).hexdigest() != expected:
                raise DataError(f"Нарушена контрольная сумма данных Xe: {name}.")
            self.verified_files.add(name)
        return raw

    def json(self, name: str):
        try:
            return json.loads(self.read(name))
        except DataError:
            raise
        except (ValueError, UnicodeError) as exc:
            raise DataError(f"Некорректный JSON Xe: {name}") from exc

    def csv(self, name: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"))))

    def constant(self, key: str) -> float:
        return float(self.constants[key]["value"])

    @lru_cache(maxsize=32)
    def table(self, name: str, x_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        try:
            rows = self.csv(name)
            columns = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
            grid = columns.pop(x_name)
        except DataError:
            raise
        except (ValueError, KeyError, IndexError) as exc:
            raise DataError(f"Некорректная таблица Xe: {name}") from exc
        if len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise DataError(f"Некорректная сетка Xe: {name}")
        if any(not np.all(np.isfinite(v)) for v in columns.values()):
            raise DataError(f"Неконечные значения Xe: {name}")
        if name.startswith("collision_integrals/") and any(np.any(v <= 0) for v in columns.values()):
            raise DataError(f"Неположительные интегралы Xe: {name}")
        return grid, columns

    @staticmethod
    def interpolate(grid: np.ndarray, values: np.ndarray, x: float, *, log: bool = False) -> float:
        if not np.isfinite(x) or not grid[0] <= x <= grid[-1]:
            raise PropertyDomainError(f"Аргумент {x:g} вне [{grid[0]:g}, {grid[-1]:g}]; экстраполяция запрещена.")
        # Exact knot reproduction also avoids tiny endpoint round-off.
        index = int(np.searchsorted(grid, x))
        if grid[index] == x:
            return float(values[index])
        if log:
            return float(np.exp(np.interp(np.log(x), np.log(grid), np.log(values))))
        return float(np.interp(x, grid, values))

    def area(self, T: float, i: int, j: int, l: int, s: int,
             debye: float, branch: str, high_ion_model: str) -> float:
        """i=0 electron; i=z+1 Xe^z. Return pi*Omega^(l,s), m²."""
        i, j = sorted((i, j))
        if i != 1 and j != 1:  # Both charged, including identical particles.
            zi, zj = (-1 if i == 0 else i-1), (-1 if j == 0 else j-1)
            tstar = (4*np.pi*self.constant("epsilon_0")*self.constant("k_B")*T*debye
                     / (abs(zi*zj)*self.constant("e")**2))
            sign = "attractive" if zi*zj < 0 else "repulsive"
            grid, columns = self.table("collision_integrals/coulomb_reduced.csv", "reduced_temperature_Tstar")
            f = self.interpolate(grid, columns[f"F_{sign}_l{l}_s{s}"], tstar, log=True)
            return float(np.pi*debye**2*f/tstar**2)
        if i == 0:
            pair, label = "Xe0_e", "shared"
        elif j <= 2:
            pair, label = ("Xe0_Xe0" if j == 1 else "Xe0_Xe1"), branch
        else:
            if high_ion_model == "error":
                raise PropertyDomainError(f"Нет таблицы Xe–Xe{j-1}+. Для расчёта выбери явную модель недостающей пары.")
            return self.langevin_area(T, self.c4[j-1], s)
        grid, columns = self.table(f"collision_integrals/{pair}_{label}.csv", "temperature_K")
        return self.interpolate(grid, columns[f"pi_omega_l{l}_s{s}_m2"], T, log=True)

    def langevin_area(self, T: float, c4: float, s: int) -> float:
        """Chosen closure, NOT recovered MDPI data: isotropic capture scattering.

        V=-C4/r^4; sigma_capture(E)=2*pi*sqrt(C4/E). Taking a uniform
        outgoing direction, Devoto's normalized area is independent of l:
        Q^(l,s)=2*pi*sqrt(C4/kT)*Gamma(s+3/2)/Gamma(s+2).
        Glancing collisions and detailed short-range/charge-transfer channels
        are not included. C4 is supplied by models.json in the data archive.
        """
        return float(2*np.pi*np.sqrt(c4/(self.constant("k_B")*T))*np.exp(gammaln(s+1.5)-gammaln(s+2)))

    def reference(self, T: float, pressure: float, branch: str) -> tuple[dict[str, float], str]:
        entry = next((r for r in self.catalog if r["branch"] == branch and r["pressure_Pa"] == pressure), None)
        if entry is None:
            return {}, "pressure_not_tabulated"
        grid, columns = self.table("reference/" + entry["file"], "T_h_K")
        if not grid[0] <= T <= grid[-1]:
            return {}, "temperature_outside_reference"
        index = int(np.searchsorted(grid, T))
        used = [index] if grid[index] == T else [index-1, index]
        bad_lambda = any(columns["lambda_total_W_m_K"][k] <= 0 for k in used)
        result = {k: self.interpolate(grid, v, T) for k, v in columns.items()}
        if bad_lambda:
            result = {k: v for k, v in result.items() if not k.startswith("lambda_")}
        return result, "source_lambda_invalid" if bad_lambda else "ok"

    def provenance(self) -> dict:
        return {"path": str(self.path.resolve()), "archive_sha256":
                hashlib.sha256(self.path.read_bytes()).hexdigest() if self.is_zip else None,
                "verified_files": sorted(self.verified_files), "sources": self.sources,
                "interpolation": "references: linear T; area integrals: log-log; no extrapolation"}
