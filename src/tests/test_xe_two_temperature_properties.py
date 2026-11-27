from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from config import SETTINGS
from plasma.cli import main
from plasma.data import load_atomic_data
from plasma.model import prepare_state
from plasma.runner import run
from plasma.solver import solve_state
from plasma.xe_dataset import XeDataset
from plasma.xe_domain import ratio_grid, temperature_grid, temperature_interval
from plasma.xe_physics import (caloric_state, pair_temperatures, partition_moments,
    reaction_conductivity, reaction_conductivity_two_temperature, transport_state)
from plasma.xe_properties import compute_row

ROOT = Path(__file__).resolve().parents[1]


class XeTwoTemperaturePropertiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = replace(SETTINGS, elements=('Xe',), models=('ideal',),
                               sweep_mode='fixed_ratio', fixed_temperature_ratio=2.)
        cls.atom = load_atomic_data('Xe', cls.settings)
        cls.dataset = XeDataset(cls.settings.xe_lte_data_path)

    def state(self, th, ratio, pressure=None, initial=None):
        settings = replace(self.settings, pressure_pa=pressure or self.settings.pressure_pa,
                           relative_tolerance=1e-11)
        solution = solve_state(prepare_state(self.atom, th, settings, electron_temperature_k=ratio*th),
                               settings, 'ideal', initial)
        self.assertTrue(solution.converged, solution.message)
        return solution

    def test_caloric_energy_and_enthalpy_use_both_temperatures(self):
        th, te = 6000., 12000.
        density = np.array([3e23, 2e23, 1e23, 5e22, 1e20, 1e18])
        d = self.dataset; kb = d.constant('k_B'); ev = d.constant('e')
        means, _ = partition_moments(self.atom, te, kb, ev)
        formation = np.r_[0., np.cumsum(self.atom.ionization_ev[:4])]*ev
        pressure = kb*(density[0]*te+density[1:].sum()*th)
        energy = 1.5*pressure+density[1:]@(formation+means)
        rho = density@d.masses
        out = caloric_state(self.atom, th, density, d, electron_temperature_k=te)
        self.assertAlmostEqual(out['u_J_kg']/(energy/rho), 1., places=14)
        self.assertAlmostEqual(out['h_J_kg']/((energy+pressure)/rho), 1., places=14)
        self.assertEqual(out['pressure_excess_Pa'], 0.)

    def test_relative_pair_temperatures_and_two_temperature_debye(self):
        sol = self.state(7000., 1.5)
        ns = sol.state.density_star*self.settings.reference_density_m3
        d = self.dataset; kb = d.constant('k_B'); charge = d.constant('e')
        T = np.r_[10500., np.full(5, 7000.)]
        relative = pair_temperatures(T, d.masses)
        np.testing.assert_allclose(relative, relative.T, rtol=1e-15)
        np.testing.assert_allclose(np.diag(relative), T, rtol=1e-15)
        self.assertEqual(relative[2,3], 7000.)
        self.assertLess(abs(relative[0,1]/10500.-1), 1e-5)
        row = compute_row(sol, self.atom, d, self.settings)
        self.assertEqual(row['properties_status'], 'ok', row['message'])
        expected = np.sqrt(d.constant('epsilon_0')*kb/(charge**2*(ns[0]/10500.+ns[1:]@np.arange(5)**2/7000.)))
        self.assertAlmostEqual(row['debye_length_m']/expected, 1., places=14)

    def test_two_temperature_cp_and_cv_against_independent_fixed_density_path(self):
        for ratio, th in [(2., 6000.), (.8, 12000.)]:
            with self.subTest(ratio=ratio):
                sol = self.state(th, ratio)
                row = compute_row(sol, self.atom, self.dataset, self.settings)
                self.assertEqual(row['properties_status'], 'ok', row['message'])
                pressure = sol.pressure_pa
                def caloric(t, p):
                    at = self.state(t, ratio, p, sol.state.y)
                    return caloric_state(self.atom, t, at.state.density_star*self.settings.reference_density_m3,
                                         self.dataset, electron_temperature_k=ratio*t)
                step = .5
                cp = (caloric(th+step, pressure)['h_J_kg']-caloric(th-step, pressure)['h_J_kg'])/(2*step)
                def fixed_density(t):
                    p = brentq(lambda p: caloric(t,p)['rho_kg_m3']-row['rho_kg_m3'],
                               .8*pressure, 1.2*pressure, xtol=1e-7)
                    return caloric(t,p)['u_J_kg']
                cv = (fixed_density(th+step)-fixed_density(th-step))/(2*step)
                self.assertAlmostEqual(row['cp_J_kg_K']/cp, 1., delta=2e-6)
                self.assertAlmostEqual(row['cv_J_kg_K']/cv, 1., delta=2e-6)

    def test_reactive_momentum_closure_matches_unscaled_matrix_and_lte_limit(self):
        R = np.array([[-1,1,0,1], [0,-1,1,1]], float)
        x = np.array([.6,.2,.1,.1]); nD = np.full((4,4),1e20)
        H = np.array([1.,3.,6.,2.])*1e-19
        kb = self.dataset.constant('k_B'); T = 8000.
        for relative, drive in [(np.full((4,4), T), R@H/T),
                                (np.array([[8000,7000,6000,5000], [7000,6000,5000,4000],
                                           [6000,5000,4000,3000], [5000,4000,3000,2000]]),
                                 np.array([4.,7.])*kb)]:
            L = np.zeros((2,2))
            for i in range(4):
                for j in range(i+1,4):
                    v = R[:,i]/x[i]-R[:,j]/x[j]
                    L += kb*relative[i,j]*x[i]*x[j]/nD[i,j]*np.outer(v,v)
            expected = (R@H)@np.linalg.solve(L,drive)
            result = reaction_conductivity_two_temperature(x,nD,relative,H,R,drive,kb)
            self.assertAlmostEqual(result/expected,1.,places=12)
        lte = reaction_conductivity(x,nD,H,R,kb,T)
        value = reaction_conductivity_two_temperature(x,nD,np.full((4,4),T),H,R,R@H/T,kb)
        self.assertAlmostEqual(value/lte,1.,places=12)
        for tiny in (1e-40,1e-150,1e-250):
            trace = np.array([.8,.1,tiny,.1]); trace /= trace.sum()
            value = reaction_conductivity_two_temperature(trace,nD,np.full((4,4),T),H,R,R@H/T,kb)
            self.assertTrue(np.isfinite(value)); self.assertGreaterEqual(value,0.)

    def test_effective_lambda_weights_electron_and_internal_gradients(self):
        row = compute_row(self.state(6000.,2.),self.atom,self.dataset,self.settings)
        total = row['lambda_h_W_m_K']+row['lambda_reaction_W_m_K']+2*(row['lambda_e_W_m_K']+row['lambda_internal_W_m_K'])
        self.assertAlmostEqual(row['lambda_W_m_K']/total,1.,places=14)
        self.assertEqual(row['lambda_e_effective_h_W_m_K'],2*row['lambda_e_W_m_K'])
        self.assertEqual(row['lambda_temperature_gradient'],'T_h_at_fixed_gamma')
        self.assertEqual(row['heat_capacity_temperature'],'T_h')
        self.assertEqual(row['reference_status'],'not_applicable_two_temperature')
        self.assertTrue(all(v=='' for k,v in row.items() if k.startswith('reference_') and k not in ('reference_model','reference_status')))

    def test_new_scheme_continuously_recovers_gamma_one(self):
        first = compute_row(self.state(10000.,1.),self.atom,self.dataset,self.settings)
        second = compute_row(self.state(10000.,1.000001),self.atom,self.dataset,self.settings)
        self.assertEqual(second['properties_status'],'ok',second['message'])
        for key in ('u_J_kg','h_J_kg','cp_J_kg_K','cv_J_kg_K','sigma_S_m','eta_Pa_s',
                    'lambda_W_m_K','lambda_reaction_W_m_K'):
            self.assertAlmostEqual(second[key]/first[key],1.,delta=2e-5,msg=key)

    def test_upper_endpoint_uses_one_sided_caloric_derivative_and_no_extrapolation(self):
        calls=[]
        original=prepare_state
        def capture(atom,t,settings,**kwargs):
            calls.append((t,kwargs['electron_temperature_k']))
            return original(atom,t,settings,**kwargs)
        with patch('plasma.xe_properties.prepare_state',side_effect=capture):
            row = compute_row(self.state(15000.,2.),self.atom,self.dataset,self.settings)
        self.assertEqual(row['properties_status'],'ok',row['message'])
        self.assertTrue(calls)
        self.assertTrue(all(300<=t<=30000 and 300<=te<=30000 for t,te in calls))
        self.assertLessEqual(row['coulomb_Tstar_max'],10000.)

    def test_temperature_limits_keep_concentration_only_and_cs_grids_unchanged(self):
        for gamma, expected in [(1.,30000.),(1.25,24000.),(1.5,20000.),(2.,15000.)]:
            settings=replace(self.settings,fixed_temperature_ratio=gamma)
            grid=temperature_grid(settings,'Xe')
            self.assertEqual(grid[0],3000.);self.assertEqual(grid[-1],expected)
            np.testing.assert_array_equal(temperature_grid(settings,'Cs'),settings.temperatures())
            np.testing.assert_array_equal(temperature_grid(replace(settings,xe_properties_enabled=False),'Xe'),settings.temperatures())
        self.assertEqual(temperature_interval(.5),(600.,30000.))
        with self.assertRaisesRegex(ValueError,'нет допустимых температур'):
            temperature_grid(replace(self.settings,temperature_start_k=20000.),'Xe')
        ratios=replace(self.settings,sweep_mode='temperature_ratio',temperature_ratios=(1.,2.,8.))
        np.testing.assert_array_equal(ratio_grid(ratios,'Xe'),[1.,2.])
        np.testing.assert_array_equal(ratio_grid(ratios,'Cs'),[1.,2.,8.])

    def test_coulomb_domain_failure_preserves_caloric_values_and_reports_cli_failure(self):
        row=compute_row(self.state(4000.,5.),self.atom,self.dataset,self.settings)
        self.assertEqual(row['properties_status'],'outside_collision_domain')
        self.assertEqual(row['sigma_S_m'],'');self.assertEqual(row['lambda_W_m_K'],'')
        self.assertGreater(row['u_J_kg'],0);self.assertGreater(row['cv_J_kg_K'],0)
        self.assertIn('T*=',row['message'])
        with tempfile.TemporaryDirectory() as temp,patch('matplotlib.figure.Figure.savefig',autospec=True),redirect_stderr(io.StringIO()) as errors:
            code=main(replace(self.settings,output_dir=Path(temp)),
                      ['--fixed-gamma','5','--temperature-start-k','4000','--temperature-stop-k','4000','--quiet'])
            self.assertEqual(code,1);self.assertIn('Неполные свойства Xe',errors.getvalue())

    def test_fixed_gamma_run_clips_grid_and_writes_properties_metadata_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            settings=replace(self.settings,output_dir=Path(temp),temperature_start_k=10000.,
                             temperature_stop_k=30000.,temperature_step_k=5000.,
                             plot_groups=(1,),plot_all_species=False)
            with patch('matplotlib.figure.Figure.savefig',autospec=True) as saved:
                result=run(settings,quiet=True)
            self.assertEqual(result.failed_points,0);self.assertEqual(result.xe_property_failed_points,0)
            self.assertEqual([s.temperature_k for s in result.solutions],[10000.,15000.])
            self.assertEqual(saved.call_count,3)
            self.assertTrue(all('fixed_gamma_2' in str(p) for p in result.plot_paths))
            with result.xe_properties_path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual([float(r['T_e_K']) for r in rows],[20000.,30000.])
            metadata=json.loads(result.metadata_path.read_text())['xe_properties']
            self.assertEqual(metadata['temperature_domain']['actual_Th_K'],[10000.,15000.])
            self.assertEqual(metadata['temperature_domain']['requested_Th_K'],[10000.,30000.])
            self.assertEqual(metadata['status_counts'],{'ok':2})

    def test_ratio_sweep_property_graphs_use_gamma_axis(self):
        snapshots=[]
        def capture(fig,path,**kwargs):
            if str(path).endswith('_properties.png'):
                snapshots.append((str(path),fig.axes[0].get_xlabel(),fig.axes[0].lines[0].get_xdata()))
        with tempfile.TemporaryDirectory() as temp,patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=capture):
            settings=replace(self.settings,output_dir=Path(temp),sweep_mode='temperature_ratio',
                             heavy_temperature_k=4000.,temperature_ratios=(1.,1.5,2.),
                             plot_groups=(1,),plot_all_species=False)
            result=run(settings,quiet=True)
        self.assertEqual(result.xe_property_failed_points,0)
        self.assertEqual(len(snapshots),1)
        name,label,grid=snapshots[0]
        self.assertIn('two_temperature_properties',name);self.assertIn('γ',label)
        np.testing.assert_array_equal(grid,[1.,1.5,2.])


if __name__=='__main__':
    unittest.main()
