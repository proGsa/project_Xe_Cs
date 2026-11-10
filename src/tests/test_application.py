from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from plasma.cli import main
from plasma.output import solution_row
from plasma.runner import run
from plasma.settings import RunSettings
from plasma.solver import Solution
from test_data import write_input


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = RunSettings(input_dir=root/'input', output_dir=root/'results',
                                    temperature_start_k=3000, temperature_stop_k=6000, temperature_step_k=1000)

    def tearDown(self):
        self.temp.cleanup()

    def test_end_to_end_csv_and_png(self):
        write_input(self.settings.input_dir)
        result = run(self.settings, quiet=True)
        self.assertEqual(len(result.solutions), 16)
        self.assertEqual(result.failed_points, 0)
        self.assertEqual(len(result.plot_paths), 4)
        for path in result.plot_paths:
            self.assertGreater(path.stat().st_size, 5000)
        with result.csv_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row['status'], 'ok')
            self.assertGreater(float(row['n_e_m3']), 0)
            self.assertLess(float(row['pressure_error']), 1e-8)
            if row['model'] == 'ideal':
                self.assertEqual(float(row['Gamma']), 0)
            else:
                self.assertGreater(float(row['Gamma']), 0)
        metadata = json.loads(result.metadata_path.read_text())
        self.assertEqual(metadata['failed_points'], 0)

    def test_validate_only(self):
        write_input(self.settings.input_dir)
        result = run(self.settings, validate_only=True, quiet=True)
        self.assertEqual(result.solutions, ())
        self.assertIsNone(result.csv_path)
        self.assertTrue(result.validation_path.is_file())

    def test_grouped_plots_use_the_corresponding_csv_concentrations(self):
        settings = replace(self.settings, elements=('Xe',), models=('ideal',), max_charge=4,
                           temperature_stop_k=4000, plot_all_species=True)
        write_input(settings.input_dir, max_charge=4)
        figures = {}

        def capture(figure, path, **kwargs):
            ax = figure.axes[0]
            self.assertEqual(ax.get_yscale(), 'log')
            figures[Path(path).name] = {
                line.get_label(): (line.get_xdata().copy(), line.get_ydata().copy(), line.get_color())
                for line in ax.lines
            }

        with patch('matplotlib.figure.Figure.savefig', autospec=True, side_effect=capture):
            result = run(settings, quiet=True)
        with result.csv_path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        expected = {
            'Xe_ideal_group1_neutral_ion1_e.png': ('Xe, нейтралы', 'Xe, заряд +1', 'Электроны'),
            'Xe_ideal_group2_ion1_e_ion2.png': ('Xe, заряд +1', 'Электроны', 'Xe, заряд +2'),
            'Xe_ideal_group3_ion2_ion3.png': ('Xe, заряд +2', 'Xe, заряд +3'),
            'Xe_ideal_group4_ion3_ion4.png': ('Xe, заряд +3', 'Xe, заряд +4'),
            'Xe_ideal.png': ('Электроны', 'Xe, нейтралы', 'Xe, заряд +1', 'Xe, заряд +2',
                             'Xe, заряд +3', 'Xe, заряд +4'),
        }
        self.assertEqual(set(figures), set(expected))
        columns = {'Электроны': 'n_e_m3', 'Xe, нейтралы': 'n_0_m3',
                   **{f'Xe, заряд +{z}': f'n_{z}_m3' for z in range(1, 5)}}
        colors = {}
        for filename, labels in expected.items():
            self.assertEqual(tuple(figures[filename]), labels)
            for label, (x, y, color) in figures[filename].items():
                np.testing.assert_allclose(x, [float(r['T_K']) for r in rows])
                np.testing.assert_allclose(y, [float(r[columns[label]]) for r in rows], rtol=1e-12)
                self.assertEqual(color, colors.setdefault(label, color))

    def test_missing_csv_readable_cli_failure(self):
        stream = io.StringIO()
        with redirect_stderr(stream), redirect_stdout(io.StringIO()):
            code = main(self.settings, ['--quiet'])
        self.assertEqual(code, 2)
        self.assertIn('Нет файла', stream.getvalue())
        self.assertNotIn('Traceback', stream.getvalue())

    def test_failed_solution_does_not_export_trial_densities(self):
        solution = Solution(3000, 1e5, 'Xe', 'nonideal', False, 1, None, 'failed')
        row = solution_row(solution, self.settings)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['n_e_m3'], '')
        self.assertEqual(row['Gamma'], '')

    def test_cli_single_element_and_mode(self):
        write_input(self.settings.input_dir)
        with redirect_stdout(io.StringIO()):
            code = main(self.settings, ['--element','Cs','--model','ideal','--quiet'])
        self.assertEqual(code, 0)
        with (self.settings.output_dir/'concentrations.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r['element']=='Cs' and r['model']=='ideal' for r in rows))


if __name__ == '__main__':
    unittest.main()
