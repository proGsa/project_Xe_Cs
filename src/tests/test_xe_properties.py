from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.optimize import brentq

from config import SETTINGS
from plasma.cli import main
from plasma.data import DataError, load_atomic_data
from plasma.model import prepare_state, solve_gamma
from plasma.runner import run
from plasma.solver import solve_state
from plasma.xe_dataset import PropertyDomainError, XeDataset
from plasma.xe_physics import caloric_state, partition_moments, reaction_conductivity, transport_state
from plasma.xe_properties import compute_row

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'input/xe_lte_data.zip'


class XePropertiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings=replace(SETTINGS,input_dir=ROOT/'input',xe_lte_data_path=DATA,
                             elements=('Xe',),models=('ideal','nonideal'),max_charge=4)
        cls.atom=load_atomic_data('Xe',cls.settings)
        cls.dataset=XeDataset(DATA)

    def solution(self,T,mode='ideal',settings=None):
        settings=settings or self.settings
        result=solve_state(prepare_state(self.atom,T,settings),settings,mode)
        self.assertTrue(result.converged,result.message)
        return result

    def test_neutral_gas_caloric_and_transport_limits(self):
        d=self.dataset;T=300.;n=np.array([0.,1e23,0.,0.,0.,0.])
        out=caloric_state(self.atom,T,n,d)
        mass=d.masses[1];kb=d.constant('k_B')
        self.assertAlmostEqual(out['u_J_kg']/(1.5*kb*T/mass),1.,places=12)
        self.assertAlmostEqual(out['h_J_kg']/(2.5*kb*T/mass),1.,places=12)
        tr=transport_state(self.atom,T,n,d)
        Q=d.area(T,1,1,2,2,np.inf,'traditional','error')
        eta=5/16*np.sqrt(np.pi*mass*kb*T)/Q
        self.assertAlmostEqual(tr['eta_Pa_s']/eta,1.,places=12)
        self.assertAlmostEqual(tr['lambda_h_W_m_K']/(15/4*kb/mass*eta),1.,places=12)
        self.assertEqual(tr['sigma_S_m'],0.)
        self.assertEqual(tr['lambda_e_W_m_K'],0.)
        self.assertEqual(tr['lambda_reaction_W_m_K'],0.)

    def test_excitation_capacity_is_temperature_derivative(self):
        T=12000.;d=self.dataset
        means,cint=partition_moments(self.atom,T,d.constant('k_B'),d.constant('e'))
        plus,_=partition_moments(self.atom,T+.1,d.constant('k_B'),d.constant('e'))
        minus,_=partition_moments(self.atom,T-.1,d.constant('k_B'),d.constant('e'))
        np.testing.assert_allclose((plus-minus)/.2,cint,rtol=2e-8,atol=1e-35)

    def test_seminar_excess_free_energy_recovers_all_three_corrections(self):
        d=self.dataset;s=replace(self.settings,max_charge=2,gamma_tolerance=1e-14)
        from plasma.data import AtomicData
        atom=AtomicData('Xe',self.atom.levels[:3],self.atom.ionization_ev[:3])
        ns=np.array([2e23,5e23,1.6e23,2e22]);T=8000.

        def energy(t,density):
            prep=prepare_state(atom,t,s)
            gamma,_=solve_gamma(prep,density/s.reference_density_m3,s.gamma_tolerance)
            alpha=prep.physical.c_alpha*(gamma*t)**3*s.reference_density_m3
            return caloric_state(atom,t,density,d,alpha_density=alpha,gamma=gamma),gamma

        center,gamma=energy(T,ns);mu=[]
        for i in range(len(ns)):
            h=ns[i]*1e-4
            dn=np.eye(len(ns))[i]*h
            mu.append((energy(T,ns+dn)[0]['free_energy_excess_J_m3']-
                       energy(T,ns-dn)[0]['free_energy_excess_J_m3'])/(2*h))
        expected=-d.constant('k_B')*T*np.log1p(np.array([1,0,1,4])*gamma/2)
        np.testing.assert_allclose(mu,expected,rtol=2e-7,atol=1e-30)
        p_from_gradient=float(ns@mu)-center['free_energy_excess_J_m3']
        self.assertAlmostEqual(p_from_gradient/center['pressure_excess_Pa'],1.,delta=2e-7)
        step=T*1e-4
        dfdt=(energy(T+step,ns)[0]['free_energy_excess_J_m3']-
              energy(T-step,ns)[0]['free_energy_excess_J_m3'])/(2*step)
        internal=center['free_energy_excess_J_m3']-T*dfdt
        self.assertAlmostEqual(internal/center['internal_energy_excess_J_m3'],1.,delta=2e-7)

    def test_cp_and_cv_against_independent_paths(self):
        d=self.dataset;T=10000.;s=self.settings
        for mode in ('ideal','nonideal'):
            sol=self.solution(T,mode);row=compute_row(sol,self.atom,d,s)
            self.assertEqual(row['properties_status'],'ok',row['message'])
            np.testing.assert_allclose(row['h_J_kg']-row['u_J_kg'],
                                       row['eos_pressure_Pa']/row['rho_kg_m3'],rtol=1e-12)
            def at(t,p):
                local=replace(s,pressure_pa=p,relative_tolerance=1e-11,gamma_tolerance=1e-13)
                result=solve_state(prepare_state(self.atom,t,local),local,mode,sol.state.y)
                self.assertTrue(result.converged,result.message)
                st=result.state
                return caloric_state(self.atom,t,st.density_star*s.reference_density_m3,d,
                    alpha_density=st.alpha*s.reference_density_m3,gamma=st.gamma)
            h=2.
            cp=(at(T+h,s.pressure_pa)['h_J_kg']-at(T-h,s.pressure_pa)['h_J_kg'])/(2*h)
            self.assertAlmostEqual(row['cp_J_kg_K']/cp,1.,delta=2e-6)
            rho=row['rho_kg_m3']
            def fixed_rho(t):
                pressure=brentq(lambda p:at(t,p)['rho_kg_m3']-rho,.8*s.pressure_pa,1.2*s.pressure_pa,xtol=1e-6)
                return at(t,pressure)['u_J_kg']
            cv=(fixed_rho(T+h)-fixed_rho(T-h))/(2*h)
            self.assertAlmostEqual(row['cv_J_kg_K']/cv,1.,delta=2e-6)

    def test_langevin_model_has_expected_rate_and_diffusion_scaling(self):
        d=self.dataset;T=8000.;kb=d.constant('k_B')
        for z in (2,3,4):
            c4=d.c4[z]
            Q=d.langevin_area(T,c4,1)
            mu=d.masses[1]*d.masses[z+1]/(d.masses[1]+d.masses[z+1])
            nD=3/16*np.sqrt(2*np.pi*kb*T/mu)/Q
            rate=2*np.pi*np.sqrt(2*c4/mu)
            self.assertAlmostEqual(nD/(kb*T/(mu*rate)),1.,places=13)
            self.assertAlmostEqual(d.langevin_area(4*T,c4,1)/Q,.5,places=13)
        with self.assertRaisesRegex(PropertyDomainError,'Нет таблицы'):
            d.area(T,1,3,1,1,1e-7,'traditional','error')

    def test_reaction_scaling_handles_trace_species_and_matches_unscaled_form(self):
        R=np.array([[-1,1,0,1],[0,-1,1,1]],dtype=float)
        x=np.array([.6,.2,.1,.1]);nD=np.full((4,4),1e20);H=np.array([1.,3.,6.,2.])*1e-19
        A=np.zeros((2,2))
        for i in range(4):
            for j in range(i+1,4):
                v=R[:,i]/x[i]-R[:,j]/x[j]
                A+=x[i]*x[j]/nD[i,j]*np.outer(v,v)
        kb=self.dataset.constant('k_B');T=8000.
        expected=(R@H)@np.linalg.solve(A,R@H)/(kb*T*T)
        self.assertAlmostEqual(reaction_conductivity(x,nD,H,R,kb,T)/expected,1.,places=12)
        for tiny in (1e-40,1e-150,1e-250):
            trace=np.array([.8,.1,tiny,.1]);trace/=trace.sum()
            value=reaction_conductivity(trace,nD,H,R,kb,T)
            self.assertTrue(np.isfinite(value));self.assertGreaterEqual(value,0)

    def test_source_values_are_separate_and_flagged_lambda_is_not_interpolated(self):
        d=self.dataset
        reference,status=d.reference(8300,2e6,'traditional')
        self.assertEqual(status,'source_lambda_invalid')
        self.assertNotIn('lambda_total_W_m_K',reference)
        self.assertIn('sigma_S_m',reference)
        self.assertEqual(d.reference(8250,2e6,'traditional')[1],'source_lambda_invalid')
        self.assertEqual(d.reference(10000,100001,'traditional')[1],'pressure_not_tabulated')
        sol=self.solution(10000)
        first=compute_row(sol,self.atom,d,self.settings)
        with patch.object(d,'reference',return_value=({'sigma_S_m':1e30,'cp_J_kg_K':999.,'adiabatic_index_cp_cv':2.},'ok')):
            second=compute_row(sol,self.atom,d,self.settings)
        for k in ('u_J_kg','h_J_kg','cp_J_kg_K','cv_J_kg_K','sigma_S_m','eta_Pa_s','lambda_W_m_K'):
            self.assertEqual(first[k],second[k])
        self.assertEqual(second['reference_sigma_S_m'],1e30)

    def test_bad_archive_and_extrapolation_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'tampered.zip'
            with ZipFile(DATA) as src,ZipFile(target,'w',ZIP_DEFLATED) as dst:
                for name in src.namelist():
                    raw=src.read(name)
                    if name.endswith('collision_integrals/Xe0_e_shared.csv'):
                        raw=raw.replace(b'300.0',b'301.0',1)
                    dst.writestr(name,raw)
            with self.assertRaisesRegex(DataError,'контрольная сумма'):
                XeDataset(target)
        with self.assertRaises(PropertyDomainError):
            self.dataset.area(30001,0,1,1,1,1e-7,'traditional','error')

    def test_two_temperature_properties_are_computed_without_lte_reference(self):
        settings=replace(self.settings,sweep_mode='fixed_ratio',fixed_temperature_ratio=2.,models=('ideal',))
        sol=solve_state(prepare_state(self.atom,8000.,settings,electron_temperature_k=16000.),settings,'ideal')
        row=compute_row(sol,self.atom,self.dataset,settings)
        self.assertEqual(row['properties_status'],'ok',row['message'])
        self.assertGreater(row['u_J_kg'],0);self.assertGreater(row['sigma_S_m'],0)
        self.assertEqual(row['reference_status'],'not_applicable_two_temperature')
        self.assertEqual(row['reference_sigma_S_m'],'')
        self.assertGreater(row['n_e_m3'],0)

    def test_end_to_end_csv_metadata_and_property_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            settings=replace(self.settings,output_dir=Path(temp),temperature_start_k=9750.,
                temperature_stop_k=10250.,temperature_step_k=250.,plot_groups=(1,),plot_all_species=False)
            result=run(settings,quiet=True)
            self.assertEqual(result.failed_points,0)
            self.assertEqual(result.xe_property_failed_points,0)
            self.assertEqual(len(result.plot_paths),6)
            with result.xe_properties_path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertEqual(row['properties_status'],'ok')
                for k in ('u_J_kg','h_J_kg','cp_J_kg_K','cv_J_kg_K','sigma_S_m','eta_Pa_s','lambda_W_m_K'):
                    self.assertGreater(float(row[k]),0)
                total=sum(float(row[k]) for k in ('lambda_e_W_m_K','lambda_h_W_m_K','lambda_reaction_W_m_K','lambda_internal_W_m_K'))
                self.assertAlmostEqual(total/float(row['lambda_W_m_K']),1.,places=12)
            with result.csv_path.open(encoding='utf-8-sig') as f:concentrations=list(csv.DictReader(f))
            self.assertEqual(concentrations[0]['u_J_kg'],rows[0]['u_J_kg'])
            metadata=json.loads(result.metadata_path.read_text())['xe_properties']
            self.assertEqual(metadata['status_counts'],{'ok':6})
            self.assertEqual(metadata['neutral_high_ion_model'],'langevin_isotropic_capture')
            self.assertTrue(metadata['data']['verified_files'])
            self.assertTrue(all(p.stat().st_size>5000 for p in result.plot_paths))


if __name__=='__main__':
    unittest.main()
