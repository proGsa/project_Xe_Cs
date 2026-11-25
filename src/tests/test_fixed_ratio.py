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
from plasma.runner import run, temperature_sweep
from plasma.settings import RunSettings
from plasma.solver import solve_state
from test_data import write_input
from test_solver import synthetic_atom


class FixedRatioSolverTests(unittest.TestCase):
    def setUp(self):
        self.atom = synthetic_atom(4)
        self.data = {"Xe": self.atom}
        self.settings = RunSettings(elements=("Xe",), max_charge=4, models=("ideal",),
                                    sweep_mode="fixed_ratio", fixed_temperature_ratio=2.,
                                    temperature_start_k=3000., temperature_stop_k=6000.,
                                    temperature_step_k=1000.)

    def test_gamma_one_recovers_all_previous_temperature_points_and_modes(self):
        original = replace(self.settings, sweep_mode="temperature", models=("ideal", "nonideal"))
        fixed = replace(original, sweep_mode="fixed_ratio", fixed_temperature_ratio=1.)
        old = temperature_sweep(self.data, original)
        new = temperature_sweep(self.data, fixed)
        self.assertEqual(len(old), 8)
        self.assertEqual(len(new), len(old))
        for first, second in zip(old, new):
            self.assertTrue(first.converged and second.converged)
            self.assertEqual((first.mode, first.temperature_k), (second.mode, second.temperature_k))
            self.assertEqual(second.electron_temperature_k, second.temperature_k)
            np.testing.assert_array_equal(first.state.density_star, second.state.density_star)
            self.assertEqual(first.state.gamma, second.state.gamma)
            self.assertEqual(first.state.alpha, second.state.alpha)

    def test_fixed_gamma_pressure_and_charge_at_each_heavy_temperature(self):
        for ratio in (.5, 2., 2.5):
            with self.subTest(ratio=ratio):
                settings = replace(self.settings, fixed_temperature_ratio=ratio)
                results = temperature_sweep(self.data, settings)
                self.assertEqual([s.temperature_k for s in results], [3000., 4000., 5000., 6000.])
                for s in results:
                    self.assertTrue(s.converged, s.message)
                    self.assertEqual(s.electron_temperature_k, ratio * s.temperature_k)
                    a = s.state.density_star
                    kb = 1.380649e-23
                    physical_pressure = kb * settings.reference_density_m3 * (
                        s.electron_temperature_k * a[0] + s.temperature_k * a[1:].sum())
                    self.assertLess(abs(physical_pressure - settings.pressure_pa) / settings.pressure_pa, 1e-8)
                    charge = np.dot(np.arange(5), a[1:])
                    self.assertLess(abs(a[0] - charge) / (a[0] + charge), 1e-8)

    def test_adaptive_intermediate_points_preserve_fixed_gamma(self):
        settings = replace(self.settings, temperature_step_k=3000.)
        calls = []
        forced_failure = False

        def fail_first_target(prepared, settings, mode, initial=None):
            nonlocal forced_failure
            calls.append((prepared.temperature_k, prepared.electron_temperature_k))
            result = solve_state(prepared, settings, mode, initial)
            if prepared.temperature_k == 6000. and not forced_failure:
                forced_failure = True
                return replace(result, converged=False)
            return result

        with patch("plasma.runner.solve_state", side_effect=fail_first_target):
            results = temperature_sweep(self.data, settings)
        self.assertTrue(all(s.converged for s in results))
        self.assertEqual(calls, [(3000., 6000.), (6000., 12000.), (4500., 9000.), (6000., 12000.)])
        self.assertEqual(results[-1].continuation_steps, 1)

    def test_invalid_ratio_and_unsupported_nonideal_are_rejected(self):
        for ratio in (0., -1., float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                replace(self.settings, fixed_temperature_ratio=ratio).validate()
        with self.assertRaisesRegex(ValueError, "gamma != 1"):
            replace(self.settings, models=("nonideal",)).validate()
        replace(self.settings, fixed_temperature_ratio=1., models=("ideal", "nonideal")).validate()


class FixedRatioApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = RunSettings(input_dir=root / "input", output_dir=root / "results",
                                    elements=("Xe",), max_charge=4, models=("ideal",),
                                    sweep_mode="fixed_ratio", fixed_temperature_ratio=2.,
                                    temperature_start_k=3000., temperature_stop_k=6000.,
                                    temperature_step_k=1000., plot_all_species=True)
        write_input(self.settings.input_dir, max_charge=4)

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_plots_and_metadata_record_fixed_ratio(self):
        figures = []

        def capture(figure, path, **kwargs):
            ax = figure.axes[0]
            figures.append((Path(path).name, ax.get_xlabel(), ax.get_title(),
                            {line.get_label(): (line.get_xdata(), line.get_ydata()) for line in ax.lines}))

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=capture), redirect_stdout(io.StringIO()) as printed:
            result = run(self.settings)
        self.assertEqual(result.failed_points, 0)
        self.assertEqual(len(figures), 5)
        self.assertIn("gamma=T_e/T_h=2", printed.getvalue())
        with result.csv_path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(r["T_h_K"]) for r in rows], [3000., 4000., 5000., 6000.])
        self.assertEqual([float(r["T_e_K"]) for r in rows], [6000., 8000., 10000., 12000.])
        self.assertTrue(all(float(r["gamma_Te_Th"]) == 2. for r in rows))
        columns = {"Электроны": "n_e_m3", "Xe, нейтралы": "n_0_m3",
                   **{f"Xe, заряд +{z}": f"n_{z}_m3" for z in range(1, 5)}}
        for name, xlabel, title, lines in figures:
            self.assertIn("fixed_gamma_2", name)
            self.assertIn("Tₕ", xlabel)
            self.assertIn("γ = 2", title)
            for label, (x, y) in lines.items():
                np.testing.assert_array_equal(x, [float(r["T_h_K"]) for r in rows])
                np.testing.assert_allclose(y, [float(r[columns[label]]) for r in rows], rtol=1e-12)
        metadata = json.loads(result.metadata_path.read_text())
        self.assertEqual(metadata["sweep_variable"], "T_h_K")
        self.assertEqual(metadata["fixed_temperature_ratio"], 2.)

    def test_cli_gamma_one_and_two_use_separate_directories_and_requested_grid(self):
        original = replace(self.settings, sweep_mode="temperature", models=("ideal", "nonideal"))
        for gamma in (1, 2, 2.000001):
            with patch("matplotlib.figure.Figure.savefig"), redirect_stdout(io.StringIO()):
                code = main(original, ["--fixed-gamma", str(gamma), "--temperature-start-k", "3000",
                                       "--temperature-stop-k", "4000", "--temperature-step-k", "1000", "--quiet"])
            self.assertEqual(code, 0)
            path = original.output_dir.with_name(f"results_fixed_gamma_{gamma}") / "concentrations.csv"
            with path.open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["model"] for r in rows}, {"ideal", "nonideal"} if gamma == 1 else {"ideal"})
            self.assertEqual({float(r["T_h_K"]) for r in rows}, {3000., 4000.})
            self.assertTrue(all(float(r["T_e_K"]) == gamma * float(r["T_h_K"]) for r in rows))
        self.assertFalse(original.output_dir.exists())

    def test_cli_nonideal_gamma_two_fails_before_writing(self):
        errors = io.StringIO()
        with redirect_stderr(errors), redirect_stdout(io.StringIO()):
            code = main(self.settings, ["--fixed-gamma", "2", "--model", "nonideal", "--quiet"])
        self.assertEqual(code, 2)
        self.assertIn("gamma != 1", errors.getvalue())
        self.assertFalse(self.settings.output_dir.exists())

    def test_conflicting_sweep_arguments_fail(self):
        for args in (["--fixed-gamma", "2", "--two-temperature"],
                     ["--fixed-gamma", "2", "--heavy-temperature-k", "4000"],
                     ["--fixed-gamma", "2", "--temperature-ratios", "1", "2"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(self.settings, args)
            self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.settings.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
