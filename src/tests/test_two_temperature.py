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
from scipy.optimize import brentq
from scipy.special import logsumexp

from plasma.cli import main
from plasma.model import analytic_initial_guess, assemble_system, evaluate_state, prepare_state
from plasma.runner import _continue_to, run
from plasma.settings import RunSettings
from plasma.solver import solve_state
from test_data import write_input
from test_solver import synthetic_atom


class TwoTemperatureSolverTests(unittest.TestCase):
    def test_ratio_one_recovers_one_temperature_solution(self):
        for max_charge in (1, 4):
            atom = synthetic_atom(max_charge)
            settings = RunSettings(max_charge=max_charge, models=("ideal",))
            old = solve_state(prepare_state(atom, 6000., settings), settings, "ideal")
            new = solve_state(prepare_state(atom, 6000., settings, electron_temperature_k=6000.),
                              settings, "ideal")
            self.assertTrue(old.converged and new.converged)
            np.testing.assert_array_equal(old.state.density_star, new.state.density_star)

    def test_partitions_and_saha_depend_on_electron_temperature(self):
        atom = synthetic_atom(4)
        settings = RunSettings(max_charge=4)
        hot = prepare_state(atom, 10000., settings)
        two = prepare_state(atom, 4000., settings, electron_temperature_k=10000.)
        np.testing.assert_array_equal(hot.log_partitions, two.log_partitions)
        np.testing.assert_array_equal(hot.log_k_ideal, two.log_k_ideal)
        self.assertAlmostEqual(two.b_pressure / hot.b_pressure, 2.5)

    def test_analytic_two_state_solution_and_newton_at_both_temperature_orders(self):
        settings = RunSettings(models=("ideal",))
        for ratio in (0.25, 1., 2., 5.):
            with self.subTest(ratio=ratio):
                prepared = prepare_state(synthetic_atom(), 6000., settings,
                                         electron_temperature_k=ratio * 6000.)
                exact = analytic_initial_guess(prepared)
                self.assertLess(evaluate_state(exact, prepared, "ideal", settings).errors.maximum, 1e-12)
                result = solve_state(prepared, settings, "ideal", exact + np.array([.3, -.2, .15]))
                self.assertTrue(result.converged, result.message)
                np.testing.assert_allclose(result.state.y, exact, rtol=0, atol=3e-8)

    def test_multicharge_against_independent_scalar_reduction(self):
        settings = RunSettings(max_charge=4, models=("ideal",))
        for ratio in (.5, 1., 2.5, 5.):
            with self.subTest(ratio=ratio):
                prepared = prepare_state(synthetic_atom(4), 6000., settings,
                                         electron_temperature_k=6000. * ratio)
                b, charges = prepared.b_pressure, prepared.charges
                cumulative = np.r_[0., np.cumsum(prepared.log_k_ideal)]

                def fractions(log_ae):
                    logs = cumulative - charges * log_ae
                    return np.exp(logs - logsumexp(logs))

                def balance(log_ae):
                    mean_charge = np.dot(charges, fractions(log_ae))
                    return np.exp(log_ae) - b * mean_charge / (1 + ratio * mean_charge)

                root = brentq(balance, -500., np.log(b / ratio), xtol=1e-13)
                f = fractions(root)
                heavy = b / (1 + ratio * np.dot(charges, f))
                expected = np.r_[np.exp(root), heavy * f]
                result = solve_state(prepared, settings, "ideal")
                self.assertTrue(result.converged, result.message)
                np.testing.assert_allclose(result.state.density_star, expected, rtol=4e-8, atol=0)

    def test_jacobian_and_direct_new_logarithm_system(self):
        settings = RunSettings(max_charge=2, models=("ideal",))
        prepared = prepare_state(synthetic_atom(2), 4000., settings, electron_temperature_k=10000.)
        y = np.array([-2., 0., -2., -6.])
        state = evaluate_state(y, prepared, "ideal", settings)
        matrix, rhs = assemble_system(state, prepared, scale_rows=False)

        def equations(arg):
            a = np.exp(arg)
            return np.r_[arg[0] + arg[2:] - arg[1:-1] - prepared.log_k_ideal,
                         a[0] - np.dot(prepared.charges, a[1:]),
                         2.5 * a[0] + a[1:].sum() - prepared.b_pressure]

        h = 1e-5
        numeric = np.column_stack([(equations(y+h*e)-equations(y-h*e))/(2*h) for e in np.eye(len(y))])
        np.testing.assert_allclose(matrix, numeric, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(matrix @ y - rhs, equations(y), atol=1e-12)

    def test_same_electron_temperature_with_reconstructed_pressure(self):
        atom = synthetic_atom(4)
        settings = RunSettings(max_charge=4, models=("ideal",))
        old_prepared = prepare_state(atom, 8000., settings)
        old = solve_state(old_prepared, settings, "ideal")
        ae, ions = old.state.density_star[0], old.state.density_star[1:]
        pressure = old_prepared.physical.kb * settings.reference_density_m3 * (8000. * ae + 4000. * ions.sum())
        new_settings = replace(settings, pressure_pa=pressure)
        prepared = prepare_state(atom, 4000., new_settings, electron_temperature_k=8000.)
        new = solve_state(prepared, new_settings, "ideal")
        self.assertTrue(new.converged, new.message)
        np.testing.assert_allclose(new.state.density_star, old.state.density_star, rtol=4e-8, atol=0)

    def test_nonideal_is_rejected_for_unequal_temperatures(self):
        settings = RunSettings()
        prepared = prepare_state(synthetic_atom(), 4000., settings, electron_temperature_k=8000.)
        with self.assertRaisesRegex(ValueError, "только для T_e=T_h"):
            solve_state(prepared, settings, "nonideal")

    def test_continuation_subdivides_electron_temperature_with_fixed_heavy_temperature(self):
        settings = RunSettings(models=("ideal",))
        atom = synthetic_atom()
        baseline = solve_state(prepare_state(atom, 4000., settings), settings, "ideal")
        calls = []

        def fail_first(prepared, settings, mode, initial=None):
            calls.append((prepared.temperature_k, prepared.electron_temperature_k))
            result = solve_state(prepared, settings, mode, initial)
            return replace(result, converged=False) if len(calls) == 1 else result

        with patch("plasma.runner.solve_state", side_effect=fail_first):
            result = _continue_to(atom, baseline, 4000., "ideal", settings, electron_temperature_k=8000.)
        self.assertTrue(result.converged, result.message)
        self.assertEqual(calls, [(4000., 8000.), (4000., 6000.), (4000., 8000.)])
        self.assertEqual(result.continuation_steps, 1)
        self.assertEqual(result.temperature_ratio, 2.)

    def test_invalid_parameters_and_automatic_baseline(self):
        settings = RunSettings(sweep_mode="temperature_ratio", models=("ideal",))
        for values in ((), (0., 1.), (1., float("nan")), (1., float("inf")), (2., 1.), (1., 1.)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(settings, temperature_ratios=values).validate()
        with self.assertRaises(ValueError):
            replace(settings, models=("nonideal",)).validate()
        np.testing.assert_array_equal(replace(settings, temperature_ratios=(.5, 2.)).ratios(), [.5, 1., 2.])
        for te in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare_state(synthetic_atom(), 4000., settings, electron_temperature_k=te)


class TwoTemperatureApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = RunSettings(input_dir=root / "input", output_dir=root / "results",
                                    elements=("Xe",), models=("ideal",), max_charge=4,
                                    sweep_mode="temperature_ratio", heavy_temperature_k=4000.,
                                    temperature_ratios=(1., 2., 3.), plot_all_species=True)
        write_input(self.settings.input_dir, max_charge=4)

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_plot_abscissa_and_metadata(self):
        figures = []

        def capture(figure, path, **kwargs):
            ax = figure.axes[0]
            figures.append((Path(path).name, ax.get_xlabel(), ax.get_title(),
                            {line.get_label(): (line.get_xdata(), line.get_ydata()) for line in ax.lines}))

        with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=capture):
            result = run(self.settings, quiet=True)
        self.assertEqual(result.failed_points, 0)
        self.assertEqual(len(result.solutions), 3)
        self.assertEqual(len(figures), 5)
        with result.csv_path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        columns = {"Электроны": "n_e_m3", "Xe, нейтралы": "n_0_m3",
                   **{f"Xe, заряд +{z}": f"n_{z}_m3" for z in range(1, 5)}}
        for row in rows:
            ratio = float(row["gamma_Te_Th"])
            self.assertEqual(float(row["T_h_K"]), 4000.)
            self.assertEqual(float(row["T_K"]), 4000.)
            self.assertEqual(float(row["T_e_K"]), 4000. * ratio)
            self.assertEqual(float(row["Gamma"]), 0.)
            self.assertEqual(float(row["alpha"]), 0.)
            self.assertEqual(float(row["p_Pa"]), 100000.)
        for name, xlabel, title, lines in figures:
            self.assertIn("two_temperature", name)
            self.assertIn("Tₑ/Tₕ", xlabel)
            self.assertIn("Tₕ = 4000", title)
            for label, (x, y) in lines.items():
                np.testing.assert_allclose(x, [float(r["gamma_Te_Th"]) for r in rows])
                np.testing.assert_allclose(y, [float(r[columns[label]]) for r in rows], rtol=1e-12)
        metadata = json.loads(result.metadata_path.read_text())
        self.assertEqual(metadata["sweep_variable"], "gamma_Te_Th")
        self.assertEqual(metadata["temperature_ratio_grid"], [1., 2., 3.])

    def test_cli_enables_ideal_mode_and_keeps_outputs_separate(self):
        old = replace(self.settings, sweep_mode="temperature", models=("ideal", "nonideal"))
        with patch("matplotlib.figure.Figure.savefig"), redirect_stdout(io.StringIO()):
            code = main(old, ["--two-temperature", "--heavy-temperature-k", "5000",
                              "--temperature-ratios", "2", "3", "--quiet"])
        self.assertEqual(code, 0)
        self.assertFalse(old.output_dir.exists())
        path = old.output_dir.with_name(old.output_dir.name + "_two_temperature") / "concentrations.csv"
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(r["gamma_Te_Th"]) for r in rows], [1., 2., 3.])
        self.assertTrue(all(r["model"] == "ideal" and float(r["T_h_K"]) == 5000. for r in rows))

    def test_cli_rejects_two_temperature_nonideal_before_writing_results(self):
        stream = io.StringIO()
        with redirect_stderr(stream), redirect_stdout(io.StringIO()):
            code = main(self.settings, ["--two-temperature", "--model", "nonideal", "--quiet"])
        self.assertEqual(code, 2)
        self.assertIn("только models=('ideal',)", stream.getvalue())
        self.assertFalse(self.settings.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
