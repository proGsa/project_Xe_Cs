from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from plasma.data import AtomicData, LevelData
from plasma.model import (
    PhysicalConstants, analytic_initial_guess, assemble_system, evaluate_state,
    prepare_state, solve_gamma,
)
from plasma.settings import RunSettings
from plasma.solver import solve_state


def synthetic_atom(max_charge=1):
    levels = tuple(LevelData(np.array([0., 1.]), np.array([1., 3.])) for _ in range(max_charge + 1))
    return AtomicData('Xe', levels, np.array([4., 15., 30., 45., 60.])[:max_charge + 1])


class SolverTests(unittest.TestCase):
    def test_constants_match_sheet_and_scale_consistency(self):
        c = PhysicalConstants.create(1e24)
        self.assertAlmostEqual(c.a_star, 2.415e-3, delta=5e-7)
        self.assertAlmostEqual(c.c_gamma / 5.87e10, 1, delta=.002)
        self.assertAlmostEqual(c.c_gamma * c.c_alpha, 1/6)

    def test_analytic_guess_satisfies_two_state_ideal_problem(self):
        settings = RunSettings()
        for T in (3000., 6000., 15000.):
            prepared = prepare_state(synthetic_atom(), T, settings)
            state = evaluate_state(analytic_initial_guess(prepared), prepared, 'ideal', settings)
            self.assertLess(state.errors.maximum, 1e-12)

    def test_newton_two_states_from_perturbed_start(self):
        settings = RunSettings()
        for T in (3000., 6000., 15000.):
            prepared = prepare_state(synthetic_atom(), T, settings)
            exact = analytic_initial_guess(prepared)
            result = solve_state(prepared, settings, 'ideal', exact + np.array([.3, -.2, .15]))
            self.assertTrue(result.converged, result.message)
            self.assertGreater(result.iterations, 0)
            np.testing.assert_allclose(np.exp(result.state.y), np.exp(exact), rtol=2e-8)

    def test_ideal_multicharge_against_independent_scalar_reduction(self):
        # Независимый скалярный решатель используется только в тесте.
        for charge in (2, 4):
            settings = RunSettings(max_charge=charge)
            prepared = prepare_state(synthetic_atom(charge), 15000., settings)
            B, z, L = prepared.b_pressure, prepared.charges, prepared.log_k_ideal
            cumulative = np.r_[0., np.cumsum(L)]

            def reference(log_ae):
                log_weights = cumulative - z * log_ae
                fractions = np.exp(log_weights - logsumexp(log_weights))
                mean_z = np.dot(z, fractions)
                return np.exp(log_ae) - B * mean_z / (1 + mean_z)

            log_ae = brentq(reference, -500, np.log(B))
            log_weights = cumulative - z * log_ae
            fractions = np.exp(log_weights - logsumexp(log_weights))
            total = B / (1 + np.dot(z, fractions))
            expected = np.r_[np.exp(log_ae), total * fractions]
            result = solve_state(prepared, settings, 'ideal')
            self.assertTrue(result.converged, result.message)
            np.testing.assert_allclose(result.state.density_star, expected, rtol=3e-8)

    def test_frozen_matrix_against_finite_differences(self):
        settings = RunSettings(max_charge=2)
        prepared = prepare_state(synthetic_atom(2), 10000., settings)
        y = np.array([-2., 0., -2., -6.])
        state = evaluate_state(y, prepared, 'nonideal', settings)
        A, b = assemble_system(state, prepared, scale_rows=False)

        def equations(arg):
            a = np.exp(arg)
            return np.r_[arg[0] + arg[2:] - arg[1:-1] - state.log_k,
                         a[0] - np.dot(prepared.charges, a[1:]),
                         a.sum() - state.alpha - prepared.b_pressure]

        h = 1e-5
        J = np.column_stack([(equations(y+h*e)-equations(y-h*e))/(2*h) for e in np.eye(len(y))])
        np.testing.assert_allclose(A, J, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(A@y-b, equations(y), atol=1e-12)
        np.testing.assert_allclose(np.linalg.solve(A,b), y+np.linalg.solve(A,-equations(y)), atol=1e-10)

    def test_gamma_bracket_and_weak_limit(self):
        settings = RunSettings()
        prepared = prepare_state(synthetic_atom(), 10000., settings)
        for scale in (1e-18, 1e-6, 1):
            a = np.array([scale, 2., scale])
            gamma, error = solve_gamma(prepared, a, 1e-13)
            GD = np.sqrt(prepared.physical.c_gamma*(a[0]+a[2])/prepared.temperature_k**3)
            self.assertGreater(gamma, 0)
            self.assertLessEqual(gamma, GD)
            self.assertLess(error, 1e-12)
            if scale == 1e-18:
                self.assertAlmostEqual(gamma/GD, 1, places=8)

    def test_nonideal_checks_original_equations(self):
        settings = RunSettings()
        prepared = prepare_state(synthetic_atom(), 6000., settings)
        result = solve_state(prepared, settings, 'nonideal')
        self.assertTrue(result.converged, result.message)
        a = result.state.density_star
        self.assertGreater(result.state.gamma, 0)
        self.assertGreater(result.state.alpha, 0)
        self.assertLess(abs(a[0]-a[2])/(a[0]+a[2]), 1e-8)
        self.assertLess(abs(a.sum()-result.state.alpha-prepared.b_pressure)/prepared.b_pressure, 1e-8)
        lhs = np.log(a[0]*a[2]/a[1])
        self.assertAlmostEqual(lhs, result.state.log_k[0], delta=1e-8)

    def test_nonideal_approaches_ideal_at_low_density(self):
        settings = RunSettings(pressure_pa=1e-8)
        prepared = prepare_state(synthetic_atom(), 10000., settings)
        ideal = solve_state(prepared, settings, 'ideal')
        nonideal = solve_state(prepared, settings, 'nonideal', ideal.state.y)
        self.assertTrue(nonideal.converged, nonideal.message)
        np.testing.assert_allclose(nonideal.state.density_star, ideal.state.density_star, rtol=1e-5)

    def test_iteration_limit_is_failure(self):
        settings = RunSettings(max_iterations=1)
        prepared = prepare_state(synthetic_atom(), 6000., settings)
        result = solve_state(prepared, settings, 'ideal', analytic_initial_guess(prepared)+2)
        self.assertFalse(result.converged)

    def test_invalid_settings(self):
        for updates in ({'pressure_pa': 0}, {'temperature_step_k': 0}, {'max_charge': 0},
                        {'temperature_stop_k': 1000}, {'gamma_tolerance': float('nan')},
                        {'models': ('wrong',)}):
            with self.assertRaises(ValueError):
                replace(RunSettings(), **updates).validate()

    def test_grid_includes_stop_without_overshoot(self):
        settings = RunSettings(temperature_start_k=3000, temperature_stop_k=3400, temperature_step_k=250)
        np.testing.assert_allclose(settings.temperatures(), [3000, 3250, 3400])


if __name__ == '__main__':
    unittest.main()
