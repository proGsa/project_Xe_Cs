"""Регрессия на исходных NIST CSV, приложенных пользователем.

Числа ниже относятся именно к включённому в проект набору таблиц.
При замене атомных данных эти контрольные значения нужно пересмотреть.
"""

from pathlib import Path
import unittest

import numpy as np

from plasma.data import load_atomic_data
from plasma.settings import RunSettings


class SuppliedNistTests(unittest.TestCase):
    def setUp(self):
        self.settings = RunSettings(
            input_dir=Path(__file__).resolve().parents[1] / 'input', max_charge=4,
        )

    def test_all_ten_level_tables_and_ground_weights(self):
        expected = {
            'Xe': ([342, 161, 157, 94, 53], [1, 4, 5, 4, 1]),
            'Cs': ([174, 303, 173, 115, 49], [2, 1, 4, 5, 4]),
        }
        for element, (counts, weights) in expected.items():
            with self.subTest(element=element):
                atom = load_atomic_data(element, self.settings)
                self.assertEqual([len(t.energies_ev) for t in atom.levels], counts)
                self.assertEqual([t.weights[0] for t in atom.levels], weights)
                for charge, levels in enumerate(atom.levels):
                    self.assertEqual(levels.energies_ev[0], 0.)
                    self.assertTrue(np.all(levels.energies_ev < atom.ionization_ev[charge]))
                self.assertTrue(all(not any('нет единицы' in w for w in r.warnings) for r in atom.reports))

    def test_actual_ionization_prefixes_suffixes_and_trailing_column(self):
        atom = load_atomic_data('Cs', self.settings)
        np.testing.assert_allclose(atom.ionization_ev,
                                   [3.89390572743, 23.15745, 33.195, 43., 56.], rtol=1e-12)
        self.assertEqual(atom.reports[0].total_rows, 55)
        self.assertEqual(atom.reports[0].accepted_rows, 55)
        self.assertEqual(atom.reports[0].skipped, {})

    def test_actual_xenon_limit_and_above_threshold_rows_are_reported(self):
        atom = load_atomic_data('Xe', self.settings)
        report = atom.reports[1]
        self.assertEqual(report.total_rows, 445)
        self.assertEqual(report.skipped['ionization_limit'], 2)
        self.assertEqual(report.skipped['above_bare_ionization_threshold'], 101)
        examples = report.skipped_examples['above_bare_ionization_threshold']
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(x['line'] > 1 for x in examples))


if __name__ == '__main__':
    unittest.main()
