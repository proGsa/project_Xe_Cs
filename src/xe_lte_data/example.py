"""Run with Python or uv: uv run --no-project python example.py"""
from xe_data import XeData

data=XeData()
T=10000.
row=data.reference(T,pressure_Pa=100000.,branch='traditional')
print(f'Te = Th = {T:g} K; p = {row["pressure_Pa"]:g} Pa')
for key in ('rho_kg_m3','h_J_kg','e_J_kg_derived','cp_J_kg_K','sigma_S_m','eta_Pa_s','lambda_total_W_m_K'):
    print(f'{key}: {row[key]:.8g}')
print('pi*Omega(1,1), electron-neutral, m2:',data.collision_integral(T,'Xe0_e'))
composition={'e':row['n_e_m3'],**{f'Xe{z}':row[f'n_Xe{z}_m3'] for z in range(5)}}
debye=data.debye_length(T,composition)
print('Debye length, m:',debye)
print('pi*Omega(1,1), electron-Xe+, m2:',data.coulomb_integral(T,-1,1,debye))
