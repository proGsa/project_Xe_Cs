"""Read Xe LTE data with the Python standard library. No silent extrapolation.

The full mixture transport calculation is not implemented here. Reference
interpolation, collision-integral input and ideal thermodynamic sums are provided.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
from functools import lru_cache
from pathlib import Path


class XeData:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
        self.constants = json.loads((self.directory / 'constants.json').read_text(encoding='utf-8'))

    @lru_cache(maxsize=32)
    def _rows(self, relative):
        with (self.directory / relative).open(encoding='utf-8', newline='') as file:
            return tuple({k: float(v) for k, v in row.items()} for row in csv.DictReader(file))

    @staticmethod
    def _interpolate(rows, x_column, value, *, logarithmic=False):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError('The interpolation coordinate must be finite.')
        grid = [r[x_column] for r in rows]
        if x < grid[0] or x > grid[-1]:
            raise ValueError(f'Value {x} outside [{grid[0]}, {grid[-1]}]; extrapolation disabled.')
        i = bisect.bisect_left(grid, x)
        if grid[i] == x:
            return dict(rows[i])
        left, right = rows[i-1], rows[i]
        transform = math.log if logarithmic else lambda t: t
        f = (transform(x)-transform(grid[i-1]))/(transform(grid[i])-transform(grid[i-1]))
        result = {}
        for key in left:
            a, b = left[key], right[key]
            if logarithmic:
                if a <= 0 or b <= 0:
                    raise ValueError('Log interpolation requires positive values.')
                result[key] = math.exp(math.log(a)+f*(math.log(b)-math.log(a)))
            else:
                result[key] = a+f*(b-a)
        result[x_column] = x
        return result

    def reference(self, temperature_K, pressure_Pa=100000.0, branch='traditional', *, allow_flagged=False):
        """Linear interpolation of published LTE results at an available pressure.

        These are source-model results, not a recalculation with the user's
        composition. Interpolation does not enforce exact conservation identities.
        """
        if branch not in ('traditional', 'phenomenological'):
            raise ValueError('branch must be traditional or phenomenological')
        pressure = float(pressure_Pa)
        if pressure not in (100000.,200000.,500000.,1000000.,2000000.,5000000.,10000000.):
            raise ValueError('Available pressures: 1, 2, 5, 10, 20, 50, 100 bar; no pressure interpolation.')
        rows = self._rows(f'reference/xe_lte_{int(pressure)}Pa_{branch}.csv')
        x = float(temperature_K)
        if math.isfinite(x) and rows[0]['T_h_K'] <= x <= rows[-1]['T_h_K']:
            grid=[r['T_h_K'] for r in rows]
            i=bisect.bisect_left(grid,x)
            used=[rows[i]] if grid[i]==x else [rows[i-1],rows[i]]
            if not allow_flagged and any(r['lambda_total_W_m_K'] <= 0 for r in used):
                raise ValueError('This source row/interpolation interval uses a nonphysical total thermal conductivity. Inspect validation.json; allow_flagged=True returns source values explicitly.')
        return self._interpolate(rows, 'T_h_K', temperature_K)

    def collision_integral(self, temperature_K, pair, l=1, s=1, branch='traditional'):
        """Return pi*Omega^(l,s) in m², log-log interpolated from source tables.

        Accepted pairs: Xe0_e, Xe0_Xe0, Xe0_Xe1. This quantity is not a
        collision frequency and is not an energy-resolved cross section.
        """
        if pair not in ('Xe0_e','Xe0_Xe0','Xe0_Xe1'):
            raise ValueError('This pair has no tabulated non-Coulomb integral in the bundle.')
        if branch not in ('traditional','phenomenological'):
            raise ValueError('Unknown branch.')
        label = 'shared' if pair == 'Xe0_e' else branch
        rows = self._rows(f'collision_integrals/{pair}_{label}.csv')
        key = f'pi_omega_l{l}_s{s}_m2'
        if key not in rows[0]:
            raise ValueError(f'Integral order ({l},{s}) is unavailable.')
        return self._interpolate(rows, 'temperature_K', temperature_K, logarithmic=True)[key]

    def debye_length(self, temperature_K, concentrations):
        """All charged species contribute at Te=Th. Concentrations are in m^-3."""
        T, ns = self._state(temperature_K, concentrations)
        weighted = ns['e']+sum(z*z*ns[f'Xe{z}'] for z in range(1,5))
        if weighted == 0:
            return math.inf
        return math.sqrt(self._c('epsilon_0')*self._c('k_B')*T/(self._c('e')**2*weighted))

    def coulomb_integral(self, temperature_K, charge_i, charge_j, debye_length_m, l=1, s=1):
        """Return pi*Omega^(l,s) in m²; require explicit local screening length.

        Reduced Tstar must lie in [0.1, 10000]. No asymptote is substituted.
        """
        T, lam = float(temperature_K), float(debye_length_m)
        if not (math.isfinite(T) and math.isfinite(lam) and T > 0 and lam > 0):
            raise ValueError('Temperature and Debye length must be positive and finite.')
        if any(isinstance(z, bool) or int(z) != z or z == 0 for z in (charge_i,charge_j)):
            raise ValueError('Both charge numbers must be nonzero integers.')
        product = charge_i*charge_j
        tstar = 4*math.pi*self._c('epsilon_0')*self._c('k_B')*T*lam/(abs(product)*self._c('e')**2)
        rows = self._rows('collision_integrals/coulomb_reduced.csv')
        sign = 'attractive' if product < 0 else 'repulsive'
        key = f'F_{sign}_l{l}_s{s}'
        if key not in rows[0]:
            raise ValueError('Unsupported integral order.')
        f = self._interpolate(rows,'reduced_temperature_Tstar',tstar,logarithmic=True)[key]
        return math.pi*lam*lam*f/(tstar*tstar)

    def partition_moments(self, charge, temperature_K):
        """Q, mean excitation energy J, variance J² for fixed bound-level set.

        No Debye lowering is applied by this helper. Excitation levels use a
        separate ground-state zero for each charge.
        """
        if isinstance(charge, bool) or charge not in range(5):
            raise ValueError('Charge must be an integer from 0 to 4.')
        T = float(temperature_K)
        if not math.isfinite(T) or T <= 0:
            raise ValueError('Temperature must be positive and finite.')
        rows = [r for r in self._rows('atomic/energy_levels.csv') if r['charge'] == charge]
        terms = [(r['statistical_weight']*math.exp(-r['excitation_energy_J']/(self._c('k_B')*T)),
                  r['excitation_energy_J']) for r in rows]
        Q = math.fsum(w for w,_ in terms)
        mean = math.fsum(w*E for w,E in terms)/Q
        variance = math.fsum(w*(E-mean)**2 for w,E in terms)/Q
        return {'Q':Q,'mean_excitation_J':mean,'excitation_variance_J2':variance}

    def ideal_thermodynamics(self, temperature_K, concentrations):
        """Ideal u [J/m³], e/h [J/kg], rho and pressure from a supplied composition.

        This does not compute Cp, Cv, eta, sigma or lambda. To obtain equilibrium
        Cp one must recompute composition at neighboring T at constant pressure.
        """
        T, ns = self._state(temperature_K, concentrations)
        with (self.directory/'atomic/species.csv').open(encoding='utf-8',newline='') as file:
            species = list(csv.DictReader(file))
        rho = math.fsum(ns[r['species']]*float(r['mass_kg']) for r in species)
        if rho <= 0:
            raise ValueError('The mixture must have positive mass density.')
        p = self._c('k_B')*T*math.fsum(ns.values())
        excitation = math.fsum(ns[f'Xe{z}']*self.partition_moments(z,T)['mean_excitation_J'] for z in range(5))
        formation = math.fsum(ns[r['species']]*float(r['formation_energy_J']) for r in species)
        u = 1.5*p+excitation+formation
        return {'rho_kg_m3':rho,'pressure_Pa':p,'u_J_m3':u,'e_J_kg':u/rho,'h_J_kg':(u+p)/rho}

    def _c(self, key):
        return self.constants[key]['value']

    @staticmethod
    def _state(temperature_K, concentrations):
        T = float(temperature_K)
        if not math.isfinite(T) or T <= 0:
            raise ValueError('Temperature must be positive and finite.')
        keys = {'e', 'Xe0', 'Xe1', 'Xe2', 'Xe3', 'Xe4'}
        unknown = set(concentrations)-keys
        if unknown:
            raise ValueError(f'Unknown species: {sorted(unknown)}')
        ns = {key:float(concentrations.get(key,0.0)) for key in keys}
        if any(not math.isfinite(n) or n < 0 for n in ns.values()):
            raise ValueError('Concentrations must be finite and nonnegative.')
        return T,ns


if __name__ == '__main__':
    data = XeData()
    row = data.reference(10000.)
    print(json.dumps({k:row[k] for k in ('T_h_K','T_e_K','gamma_Te_Th','pressure_Pa',
          'rho_kg_m3','h_J_kg','e_J_kg_derived','eta_Pa_s','sigma_S_m','lambda_total_W_m_K')},indent=2))
