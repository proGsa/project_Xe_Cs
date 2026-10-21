from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from plasma.data import DataError, load_atomic_data, parse_number, read_ionization, read_levels
from plasma.settings import RunSettings


def write_input(root: Path, max_charge: int = 1) -> None:
    """СИНТЕТИЧЕСКИЕ данные для проверки программы, не таблицы реальных Xe/Cs."""
    (root / "energy_levels").mkdir(parents=True, exist_ok=True)
    (root / "ionization_energies").mkdir(parents=True, exist_ok=True)
    for element in ("xe", "cs"):
        lines = ["Ion Charge,Ionization Energy (eV),Uncertainty (eV)"]
        for z, energy in enumerate((4.0, 15.0, 30.0, 45.0, 60.0)):
            lines.append(f'{z},"=""{energy}""",0.01')
        (root / "ionization_energies" / f"{element}.csv").write_text("\n".join(lines), encoding="utf-8")
        for z in range(max_charge + 1):
            (root / "energy_levels" / f"{element}_{z + 1}.csv").write_text(
                'Configuration,Term,J,Level (eV),Landé-g\n'
                'synthetic,s,0,"=""0.0""",0\n'
                'synthetic,p,1,"=""1.0""",1.5\n'
                'next,Limit,---,99.0,\n', encoding="utf-8",
            )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = RunSettings(input_dir=self.root)

    def tearDown(self):
        self.temp.cleanup()

    def file(self, text: str) -> Path:
        path = self.root / "data.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_numeric_wrappers_and_estimates(self):
        for text, expected in [('="3.5"', 3.5), ('[33.195]', 33.195), ('(43.0)', 43.0),
                               ('12.12984(15)', 12.12984), ('12 345.6', 12345.6)]:
            self.assertEqual(parse_number(text), expected)
        for text in ('', '1+x', '12?', '=1+2', 'nan', 'inf'):
            with self.assertRaises(ValueError):
                parse_number(text)

    def test_nist_csv_and_lande_factor(self):
        path = self.file('Configuration,Term,J,Level (eV),Landé-g\n'
                         'a,s,"=""1/2""","=""0.0""",2.0023\n'
                         'b,p,3/2,1.0,1.333\n'
                         'c,Limit,---,4.0,\n')
        levels, report = read_levels(path, self.settings, 4.0)
        np.testing.assert_allclose(levels.energies_ev, [0, 1])
        np.testing.assert_allclose(levels.weights, [2, 4])
        self.assertEqual(report.skipped['ionization_limit'], 1)

    def test_wavenumber_conversion(self):
        path = self.file('Level (cm-1),J\n0,0\n8065.543937,1\n')
        levels, _ = read_levels(path, self.settings, 4.0)
        self.assertAlmostEqual(levels.energies_ev[1], 1.0, places=8)

    def test_tab_with_preamble_and_missing_energy(self):
        path = self.file('NIST exported table\nJ\tLevel (eV)\n0\t0\n1\t\n2\t1.5\n')
        levels, report = read_levels(path, self.settings, 4.0)
        self.assertEqual(len(levels.energies_ev), 2)
        self.assertEqual(report.skipped['missing_or_ambiguous_energy'], 1)

    def test_unresolved_and_ambiguous_j(self):
        path = self.file('Level (eV),J\n0,0\n1,"1/2,3/2"\n2,"1/2 or 3/2"\n')
        levels, report = read_levels(path, self.settings, 4.0)
        np.testing.assert_allclose(levels.weights, [1, 6])
        self.assertEqual(report.skipped['missing_or_ambiguous_statistical_weight'], 1)

    def test_bound_level_filter(self):
        path = self.file('energy_eV,g\n0,1\n3,3\n4,5\n5,7\n')
        levels, report = read_levels(path, self.settings, 4.0)
        np.testing.assert_allclose(levels.energies_ev, [0, 3])
        self.assertEqual(report.skipped['above_bare_ionization_threshold'], 2)

    def test_nist_separate_prefix_suffix_columns(self):
        path = self.file('J,Prefix,Level (eV),Suffix\n0,,0,\n1,[,1,]\n2,,2,?\n3,,3,a\n')
        levels, report = read_levels(path, self.settings, 4.0)
        np.testing.assert_allclose(levels.energies_ev, [0, 1])
        self.assertEqual(report.skipped['missing_or_ambiguous_energy'], 1)
        self.assertEqual(report.skipped['autoionizing_level'], 1)
        self.assertEqual(report.estimated_rows, 1)

    def test_ionization_prefix_suffix_columns(self):
        path = self.file('charge,Prefix,Ionization energy (eV),Suffix\n0,(,4,)\n1,,15,\n')
        values, report = read_ionization(path, 'Xe', self.settings)
        self.assertEqual(values, {0: 4., 1: 15.})
        self.assertTrue(report.warnings)

    def test_missing_ground_is_error(self):
        path = self.file('Level (eV),J\n1,0\n')
        with self.assertRaisesRegex(DataError, 'основной уровень'):
            read_levels(path, self.settings, 4.0)

    def test_wrong_g_is_not_lande(self):
        path = self.file('Level (eV),J,g\n0,1/2,1\n')
        with self.assertRaisesRegex(DataError, 'не соответствует'):
            read_levels(path, self.settings, 4.0)

    def test_ionization_estimated_values(self):
        path = self.file('Element name,Ion charge,Ionization Energy (eV),Uncertainty\n'
                         'Cesium,0,"=""3.8939""",0.0001\n'
                         'Cesium,+1,[23.15],0.1\nCesium,+2,(33.0),1\n')
        values, report = read_ionization(path, 'Cs', self.settings)
        self.assertEqual(values, {0: 3.8939, 1: 23.15, 2: 33.0})
        self.assertTrue(report.warnings)

    def test_ionization_spectrum_fallback(self):
        path = self.file('Spectrum name,Ionization energy (eV)\nXe I,12\nXe II,21\n')
        values, _ = read_ionization(path, 'Xe', self.settings)
        self.assertEqual(values, {0: 12.0, 1: 21.0})

    def test_nist_abbreviated_spectrum_header_and_empty_charge(self):
        path = self.file('Sp. Name,Ion Charge,El. Name,Prefix,Ionization Energy (eV),Suffix,\n'
                         'Cs I,,Cesium,,3.89390572743,,\n'
                         'Cs II,+1,Cesium,,23.15745,,\n'
                         'Cs III,+2,Cesium,[,33.195,],\n')
        values, report = read_ionization(path, 'Cs', self.settings)
        self.assertEqual(values, {0: 3.89390572743, 1: 23.15745, 2: 33.195})
        self.assertEqual(report.estimated_rows, 1)
        path = self.file('Sp. Name,Ionization Energy (eV)\nXe I,12\nXe II,21\n')
        values, _ = read_ionization(path, 'Xe', self.settings)
        self.assertEqual(values, {0: 12., 1: 21.})

    def test_spectrum_and_charge_must_agree(self):
        path = self.file('Sp. Name,Ion Charge,Ionization Energy (eV)\nXe II,0,12\n')
        with self.assertRaisesRegex(DataError, 'не соответствует'):
            read_ionization(path, 'Xe', self.settings)

    def test_spectrum_must_match_requested_element(self):
        path = self.file('Sp. Name,Ionization Energy (eV)\nCs I,4\n')
        with self.assertRaisesRegex(DataError, 'другой элемент'):
            read_ionization(path, 'Xe', self.settings)

    def test_skip_examples_keep_source_lines_including_multiline_cells(self):
        path = self.file('NIST table\nConfiguration,Level (eV),J\n'
                         'ground,0,0\n"two\nlines",5,1\nmissing,,2\n')
        _, report = read_levels(path, self.settings, 4.0)
        above = report.skipped_examples['above_bare_ionization_threshold'][0]
        missing = report.skipped_examples['missing_or_ambiguous_energy'][0]
        self.assertEqual(above['line'], 4)
        self.assertEqual(above['values']['Level (eV)'], '5')
        self.assertEqual(missing['line'], 6)

    def test_different_element_is_error(self):
        path = self.file('Element,Ion charge,Ionization energy (eV)\nCesium,0,4\n')
        with self.assertRaisesRegex(DataError, 'другой элемент'):
            read_ionization(path, 'Xe', self.settings)

    def test_missing_file_and_html_are_errors(self):
        with self.assertRaisesRegex(DataError, 'Нет файла'):
            read_levels(self.root / 'absent.csv', self.settings, 4)
        path = self.file('<html><body>not a csv</body></html>')
        with self.assertRaisesRegex(DataError, 'HTML'):
            read_levels(path, self.settings, 4)

    def test_load_tree_file_index_is_not_charge(self):
        write_input(self.root)
        atom = load_atomic_data('Xe', self.settings)
        self.assertEqual(atom.max_charge, 1)
        self.assertEqual(len(atom.reports), 3)
        self.assertEqual(atom.ionization_ev[0], 4)
        self.assertEqual(len(atom.levels[0].energies_ev), 2)

    def test_missing_last_threshold_with_bound_filter(self):
        write_input(self.root)
        (self.root / 'ionization_energies' / 'xe.csv').write_text(
            'charge,ionization_energy_eV\n0,4\n', encoding='utf-8')
        with self.assertRaisesRegex(DataError, 'последнего иона'):
            load_atomic_data('Xe', self.settings)
        atom = load_atomic_data('Xe', replace(self.settings, bound_levels_only=False))
        self.assertEqual(atom.max_charge, 1)


if __name__ == '__main__':
    unittest.main()
