"""Validate file integrity, schema, unit transformations and reader boundaries."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from xe_data import XeData


def validate():
    root=Path(__file__).resolve().parent
    checks=0
    manifest=json.loads((root/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        assert actual == digest, f'Checksum mismatch: {name}'
        checks+=1
    data=XeData(root)
    catalog=json.loads((root/'reference/catalog.json').read_text(encoding='utf-8'))
    source_validation=json.loads((root/'validation.json').read_text(encoding='utf-8'))
    flagged={(r['branch'],r['pressure_Pa'],r['temperature_K'])
             for r in source_validation['nonpositive_total_thermal_conductivity']}
    row_count=0
    for entry in catalog:
        rows=data._rows('reference/'+entry['file'])
        assert len(rows)==298
        assert [r['T_h_K'] for r in rows] == list(range(300,30001,100))
        for r in rows:
            assert all(math.isfinite(v) for v in r.values())
            assert r['gamma_Te_Th']==1 and r['T_e_K']==r['T_h_K']
            assert r['pressure_Pa']==entry['pressure_Pa']
            assert math.isclose(r['h_J_kg']-r['e_J_kg_derived'],r['pressure_Pa']/r['rho_kg_m3'],rel_tol=1e-12)
            assert math.isclose(r['lambda_total_W_m_K'],sum(r[k] for k in ('lambda_e_W_m_K','lambda_h_W_m_K','lambda_reaction_W_m_K','lambda_internal_W_m_K')),rel_tol=1e-12)
            assert all(r[k]>0 for k in ('rho_kg_m3','eta_Pa_s','sigma_S_m'))
            if r['lambda_total_W_m_K']<=0:
                assert (entry['branch'],entry['pressure_Pa'],r['T_h_K']) in flagged
        selected=data.reference(10000,entry['pressure_Pa'],entry['branch'])
        assert selected==rows[97]
        checks+=7
        row_count+=len(rows)
    ci=json.loads((root/'collision_integrals/catalog.json').read_text(encoding='utf-8'))
    for name,entry in ci['non_coulomb'].items():
        raw=list(csv.reader(io.StringIO((root/entry['source_file']).read_text(encoding='utf-8-sig'))))
        normalized=data._rows('collision_integrals/'+name)
        assert len(normalized)==len(raw)-1
        for rawrow,clean in zip(raw[1:],normalized):
            assert clean['temperature_K']==float(rawrow[0])
            for (l,s),value in zip(ci['orders'],rawrow[1:]):
                assert math.isclose(clean[f'pi_omega_l{l}_s{s}_m2'],float(value)*1e-20,rel_tol=1e-14)
                assert clean[f'pi_omega_l{l}_s{s}_m2']>0
        checks+=3
    for invalid in (299,30001,float('nan')):
        try:
            data.reference(invalid)
        except ValueError:
            checks+=1
        else:
            raise AssertionError('Out-of-range reference accepted')
    for branch,pressure,T in flagged:
        assert data.reference(T,pressure,branch,allow_flagged=True)['lambda_total_W_m_K'] <= 0
        try:
            data.reference(T,pressure,branch)
        except ValueError:
            checks+=1
        else:
            raise AssertionError('Nonphysical source row was not blocked')
    for charge in range(5):
        at300=data.partition_moments(charge,300)
        assert at300['Q']>0 and at300['mean_excitation_J']>=0
        assert at300['excitation_variance_J2']>=0
        checks+=1
    # Controlled neutral gas checks translational energy and h-e=p/rho independently.
    n=1e23
    result=data.ideal_thermodynamics(300,{'Xe0':n})
    assert math.isclose(result['u_J_m3'],1.5*n*data._c('k_B')*300,rel_tol=1e-12)
    assert math.isclose(result['h_J_kg']-result['e_J_kg'],result['pressure_Pa']/result['rho_kg_m3'],rel_tol=1e-12)
    row=data.reference(10000)
    ns={'e':row['n_e_m3'],**{f'Xe{z}':row[f'n_Xe{z}_m3'] for z in range(5)}}
    lam=data.debye_length(10000,ns)
    assert math.isclose(lam,row['debye_length_m_derived'],rel_tol=1e-14)
    assert data.coulomb_integral(10000,-1,1,lam)>0
    checks+=4
    # Use an exact reduced-temperature knot to verify prefactor and table mapping.
    T=10000.
    lam=data._c('e')**2/(4*math.pi*data._c('epsilon_0')*data._c('k_B')*T)
    coulomb=data._rows('collision_integrals/coulomb_reduced.csv')
    knot=next(r for r in coulomb if r['reduced_temperature_Tstar']==1.)
    for l,s in ((1,1),(2,2),(3,3),(3,4),(3,5),(4,4)):
        actual=data.coulomb_integral(T,-1,1,lam,l,s)
        expected=math.pi*lam**2*knot[f'F_attractive_l{l}_s{s}']
        assert math.isclose(actual,expected,rel_tol=1e-12)
        checks+=1
    return {'status':'passed_with_source_flags','checks':checks,'reference_tables':len(catalog),'reference_rows':row_count,
            'nonphysical_source_rows':len(flagged),'default_reader_blocks_flagged_rows':True,
            'scope':'Integrity, schemas, units, derived identities, interpolation and ideal neutral-gas limit; not experimental validation.'}


if __name__=='__main__':
    print(json.dumps(validate(),ensure_ascii=False,indent=2))
