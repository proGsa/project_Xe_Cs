"""Postprocess computed Xe compositions; reference interpolation stays separate."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .data import AtomicData
from .model import prepare_state
from .settings import RunSettings
from .solver import Solution, solve_state
from .xe_dataset import PropertyDomainError, XeDataset
from .xe_domain import CollisionDomainError, temperature_interval
from .xe_physics import caloric_state, transport_state


@dataclass(frozen=True)
class XePropertyRun:
    csv_path: Path
    plot_paths: tuple[Path, ...]
    rows: tuple[dict, ...]
    metadata: dict

    @property
    def failed_points(self) -> int:
        return sum(r["properties_status"] in ("failed", "outside_collision_domain") for r in self.rows)


def _lte_heat_capacities(solution: Solution, atom: AtomicData, dataset: XeDataset,
                    settings: RunSettings, center: dict) -> dict[str, float]:
    """Cp along fixed p; Cv from equilibrium density responses, not frozen Cp-R.

    Cp-Cv=T/rho² * [(d rho/dT)_p]² / (d rho/dp)_T.
    Each perturbed state independently solves the original composition model.
    Richardson refinement verifies that the derivative is resolved.
    """
    T,p=solution.temperature_k,solution.pressure_pa
    tight=replace(settings,relative_tolerance=min(settings.relative_tolerance,1e-11),
                  gamma_tolerance=min(settings.gamma_tolerance,1e-13))

    def state(t,pressure):
        local=replace(tight,pressure_pa=pressure)
        result=solve_state(prepare_state(atom,t,local),local,solution.mode,solution.state.y)
        if not result.converged or result.state is None:
            raise PropertyDomainError("Не сошлась соседняя точка для производной теплоёмкости: "+result.message)
        s=result.state
        density=s.density_star*settings.reference_density_m3
        return caloric_state(atom,t,density,dataset,
            alpha_density=s.alpha*settings.reference_density_m3,gamma=s.gamma)

    def derivative(axis):
        step=(T if axis=="T" else p)*settings.xe_heat_capacity_relative_step

        def evaluate(h):
            plus=state(T+h,p) if axis=="T" else state(T,p+h)
            minus=state(T-h,p) if axis=="T" else state(T,p-h)
            keys=("h_J_kg","rho_kg_m3") if axis=="T" else ("rho_kg_m3",)
            return np.array([(plus[k]-minus[k])/(2*h) for k in keys])

        coarse=evaluate(step)
        for _ in range(3):
            step/=2
            fine=evaluate(step)
            error=np.abs(fine-coarse)/3/np.maximum(np.maximum(np.abs(fine),np.abs(coarse)),1e-30)
            refined=fine+(fine-coarse)/3
            if np.max(error) <= settings.xe_heat_capacity_tolerance:
                return refined,float(np.max(error))
            coarse=fine
        raise PropertyDomainError("Производная теплоёмкости не сошлась при уменьшении шага.")

    (cp,rho_T),T_error=derivative("T")
    (rho_p,),p_error=derivative("p")
    if rho_p <= 0:
        raise PropertyDomainError("Неположительная изотермическая сжимаемость; Cv не определена выбранной устойчивой моделью.")
    cv=cp-T/center["rho_kg_m3"]**2*rho_T**2/rho_p
    if not np.all(np.isfinite([cp,cv])) or min(cp,cv) <= 0:
        raise PropertyDomainError("Неположительная/неконечная равновесная теплоёмкость.")
    return {"cp_J_kg_K":float(cp),"cv_J_kg_K":float(cv),
            "heat_capacity_relative_estimate":max(T_error,p_error)}


def two_temperature_response(solution: Solution, atom: AtomicData, dataset: XeDataset,
                             settings: RunSettings, center: dict) -> tuple[dict, np.ndarray]:
    """Response to Th at fixed gamma. Cv uses a chain rule valid away from LTE.

    Cv=(du/dTh)_(p,gamma) - (du/dp)_(Th,gamma)*rho_Th/rho_p.
    This is a derivative at fixed rho,gamma, not the LTE Cp-Cv identity.
    Neighboring compositions are solved anew; step halving and Richardson
    extrapolation check all caloric and logarithmic concentration derivatives.
    """
    if solution.mode != "ideal":
        raise PropertyDomainError("Двухтемпературные свойства пока определены только для ideal.")
    th, pressure, ratio = solution.temperature_k, solution.pressure_pa, solution.temperature_ratio
    tight = replace(settings, relative_tolerance=min(settings.relative_tolerance, 1e-11))
    base = np.r_[center["h_J_kg"], center["u_J_kg"], center["rho_kg_m3"], solution.state.y]
    cache = {(th, pressure): base}

    def state(t, p):
        key = (t, p)
        if key not in cache:
            local = replace(tight, pressure_pa=p)
            solved = solve_state(prepare_state(atom,t,local,electron_temperature_k=ratio*t),
                                 local,"ideal",solution.state.y)
            if not solved.converged or solved.state is None:
                raise PropertyDomainError("Не сошлась соседняя точка двухтемпературной производной: "+solved.message)
            result = caloric_state(atom,t,solved.state.density_star*settings.reference_density_m3,
                                   dataset,electron_temperature_k=ratio*t)
            cache[key] = np.r_[result["h_J_kg"],result["u_J_kg"],result["rho_kg_m3"],solved.state.y]
        return cache[key]

    def derivative(axis):
        argument = th if axis == "Th" else pressure
        step = argument*settings.xe_heat_capacity_relative_step
        lower, upper = temperature_interval(ratio) if axis == "Th" else (0., np.inf)
        if not lower <= argument <= upper:
            raise PropertyDomainError("Температуры вне диапазона двухтемпературных свойств Xe.")
        step = min(step, (upper-lower)/4)
        direction = 1 if argument-step < lower else -1 if argument+step > upper else 0
        def evaluate(h):
            def at(offset):
                return state(th+offset,pressure) if axis == "Th" else state(th,pressure+offset)
            if direction:
                return direction*(-3*base+4*at(direction*h)-at(2*direction*h))/(2*h)
            return (at(h)-at(-h))/(2*h)
        if step <= 0:
            raise PropertyDomainError("Нет интервала для вычисления производной теплоёмкости.")
        coarse = evaluate(step)
        for _ in range(3):
            step /= 2
            fine = evaluate(step)
            # A derivative close to zero is judged against a small physical scale.
            floor = 1e-8*np.maximum(np.abs(base),1e-12)/argument
            scale = np.maximum(np.maximum(np.abs(fine),np.abs(coarse)),floor)
            error = np.abs(fine-coarse)/(3*scale)
            # Pressure derivatives of log(trace n) are not used in the result.
            relevant = error if axis == "Th" else error[1:3]
            if np.max(relevant) <= settings.xe_heat_capacity_tolerance:
                return fine+(fine-coarse)/3, float(np.max(relevant))
            coarse = fine
        raise PropertyDomainError("Двухтемпературная производная не сошлась при уменьшении шага.")

    dth, t_error = derivative("Th")
    dp, p_error = derivative("p")
    if dp[2] <= 0:
        raise PropertyDomainError("Неположительная производная плотности по давлению.")
    cp = float(dth[0])
    cv = float(dth[1]-dp[1]*dth[2]/dp[2])
    if not np.all(np.isfinite([cp,cv])) or min(cp,cv) <= 0:
        raise PropertyDomainError("Неположительная/неконечная двухтемпературная теплоёмкость.")
    return {"cp_J_kg_K":cp,"cv_J_kg_K":cv,
            "heat_capacity_relative_estimate":max(t_error,p_error)},dth[3:]


def heat_capacities(solution: Solution, atom: AtomicData, dataset: XeDataset,
                    settings: RunSettings, center: dict) -> dict[str, float]:
    if solution.temperature_ratio == 1.:
        return _lte_heat_capacities(solution,atom,dataset,settings,center)
    return two_temperature_response(solution,atom,dataset,settings,center)[0]


_VALUES=("rho_kg_m3","u_J_kg","h_J_kg","cp_J_kg_K","cv_J_kg_K",
         "sigma_S_m","eta_Pa_s","lambda_W_m_K",
         "lambda_e_W_m_K","lambda_h_W_m_K","lambda_reaction_W_m_K","lambda_internal_W_m_K",
         "lambda_e_effective_h_W_m_K","lambda_internal_effective_h_W_m_K",
         "coulomb_Tstar_min","coulomb_Tstar_max",
         "debye_length_m","internal_energy_J_m3","eos_pressure_Pa","pressure_excess_Pa",
         "internal_energy_excess_J_m3","free_energy_excess_J_m3","heat_capacity_relative_estimate")
_REFERENCE={"rho_kg_m3":"rho_kg_m3","u_J_kg":"e_J_kg_derived","h_J_kg":"h_J_kg",
            "cp_J_kg_K":"cp_J_kg_K","sigma_S_m":"sigma_S_m","eta_Pa_s":"eta_Pa_s",
            "lambda_W_m_K":"lambda_total_W_m_K"}


def compute_row(solution: Solution, atom: AtomicData, dataset: XeDataset, settings: RunSettings) -> dict:
    row={"element":"Xe","model":solution.mode,"T_h_K":solution.temperature_k,
         "T_e_K":solution.temperature_k*solution.temperature_ratio,
         "gamma_Te_Th":solution.temperature_ratio,"p_Pa":solution.pressure_pa,
         "max_charge":atom.max_charge,"n_e_m3":"",
         **{f"n_{z}_m3":"" for z in range(atom.max_charge+1)},
         **{k:"" for k in _VALUES},
         **{"reference_"+k:"" for k in _REFERENCE},"reference_cv_J_kg_K":"",
         **{k+"_relative_difference":"" for k in _REFERENCE if k!="rho_kg_m3"},
         "thermodynamic_model":"ideal_fixed_levels" if solution.mode=="ideal" else "seminar_regularized_fixed_levels",
         "collision_branch":settings.xe_collision_branch,"high_ion_neutral_model":settings.xe_high_ion_neutral_model,
         "heat_capacity_temperature":"T_h","heat_capacity_constraint":"fixed_gamma; Cp:fixed_p; Cv:fixed_rho",
         "lambda_temperature_gradient":"T_h_at_fixed_gamma",
         "reactive_transport_model":"Butler_Brokaw_LTE" if solution.temperature_ratio==1 else "projected_momentum_2T",
         "reference_model":"MDPI_2026_LTE_DH",
         "thermodynamics_status":"pending","heat_capacity_status":"pending","transport_status":"pending",
         "reference_status":"pending","properties_status":"pending","message":""}
    if not solution.converged or solution.state is None:
        row.update(properties_status="failed",thermodynamics_status="composition_failed",
                   heat_capacity_status="composition_failed",transport_status="composition_failed",
                   reference_status="not_requested",message="Нет сошедшегося состава.")
        return row
    density=solution.state.density_star*settings.reference_density_m3
    row["n_e_m3"]=float(density[0])
    row.update({f"n_{z}_m3":float(n) for z,n in enumerate(density[1:])})
    if solution.temperature_ratio != 1.0 and solution.mode != "ideal":
        row.update(properties_status="unsupported_temperature",thermodynamics_status="unsupported_temperature",
                   heat_capacity_status="unsupported_temperature",transport_status="unsupported_temperature",
                   reference_status="not_requested",message="Двухтемпературные свойства поддерживаются только для ideal.")
        return row
    errors=[]
    dlog_density = None
    try:
        center=caloric_state(atom,solution.temperature_k,density,dataset,
            alpha_density=solution.state.alpha*settings.reference_density_m3,gamma=solution.state.gamma,
            electron_temperature_k=solution.temperature_k*solution.temperature_ratio)
        row.update(center,thermodynamics_status="ok")
    except (PropertyDomainError,FloatingPointError) as exc:
        row.update(thermodynamics_status="failed",heat_capacity_status="failed")
        errors.append(str(exc))
    else:
        try:
            if solution.temperature_ratio == 1.:
                values = _lte_heat_capacities(solution,atom,dataset,settings,center)
            else:
                values,dlog_density = two_temperature_response(solution,atom,dataset,settings,center)
            row.update(values,heat_capacity_status="ok")
        except (PropertyDomainError,FloatingPointError) as exc:
            row["heat_capacity_status"]="failed"
            errors.append(str(exc))
    try:
        row.update(transport_state(atom,solution.temperature_k,density,dataset,
            branch=settings.xe_collision_branch,high_ion_model=settings.xe_high_ion_neutral_model,
            electron_temperature_k=solution.temperature_k*solution.temperature_ratio,
            dlog_density_dth=dlog_density),transport_status="ok")
    except CollisionDomainError as exc:
        row["transport_status"]="outside_collision_domain"
        errors.append(str(exc))
    except (PropertyDomainError,FloatingPointError) as exc:
        row["transport_status"]="failed"
        errors.append(str(exc))
    reference,status=(dataset.reference(solution.temperature_k,solution.pressure_pa,settings.xe_collision_branch)
                      if solution.temperature_ratio == 1. else ({},"not_applicable_two_temperature"))
    row["reference_status"]=status
    for key,source in _REFERENCE.items():
        if source in reference:
            row["reference_"+key]=reference[source]
            if key!="rho_kg_m3" and row[key]!="" and reference[source]!=0:
                row[key+"_relative_difference"]=(row[key]-reference[source])/reference[source]
    if reference:
        row["reference_cv_J_kg_K"]=reference["cp_J_kg_K"]/reference["adiabatic_index_cp_cv"]
    row["properties_status"]=("outside_collision_domain" if row["transport_status"]=="outside_collision_domain"
                              and row["heat_capacity_status"]=="ok" else "failed" if errors else "ok")
    row["message"]="; ".join(errors)
    return row


def plot_properties(rows: list[dict], settings: RunSettings) -> list[Path]:
    import matplotlib
    if not settings.show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .settings import format_temperature_ratio
    folder=Path(settings.output_dir)/"plots"
    folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    ratio_sweep=settings.sweep_mode=="temperature_ratio"
    fixed=settings.sweep_mode=="fixed_ratio"
    xlabel="Отношение температур γ = Tₑ/Tₕ" if ratio_sweep else "Температура тяжёлых частиц Tₕ, К"
    capacity_label="Теплоёмкость по Tₕ при γ=const, Дж/(кг·К)"
    specs=[("u_J_kg","Внутренняя энергия, Дж/кг"),("h_J_kg","Энтальпия, Дж/кг"),
           ("cp_J_kg_K",capacity_label),("sigma_S_m","Электропроводность, См/м"),
           ("eta_Pa_s","Динамическая вязкость, Па·с"),("lambda_W_m_K","λ по градиенту Tₕ, Вт/(м·К)")]
    for mode in settings.models:
        selected=sorted((r for r in rows if r["model"]==mode),
                        key=lambda r:r["gamma_Te_Th"] if ratio_sweep else r["T_h_K"])
        if not selected:
            continue
        x=[r["gamma_Te_Th"] if ratio_sweep else r["T_h_K"] for r in selected]
        values=lambda key:np.array([r[key] if r[key]!="" else np.nan for r in selected])
        condition=(f"Tₕ={settings.heavy_temperature_k:g} К; Tₑ=γTₕ" if ratio_sweep else
                   f"γ={format_temperature_ratio(selected[0]['gamma_Te_Th'])}; Tₑ=γTₕ")
        prefix=f"Xe_{mode}"
        if ratio_sweep:
            prefix+="_two_temperature"
        elif fixed:
            prefix+="_fixed_gamma_"+format_temperature_ratio(settings.fixed_temperature_ratio)
        incomplete=sum(r["properties_status"]!="ok" for r in selected)
        footer=(f"Неполных точек: {incomplete}; пропуски показаны разрывами. " if incomplete else "")
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,(key,label) in zip(axes.flat,specs):
            curve="Cp: p=const, γ=const" if key=="cp_J_kg_K" else "Расчёт программы"
            ax.plot(x,values(key),label=curve,linewidth=1.8,marker="." if len(x)<5 else None)
            if np.any(np.isfinite(values("reference_"+key))):
                ax.plot(x,values("reference_"+key),"--",marker="." if ratio_sweep else None,
                        label="MDPI: только γ=1 (другая модель)",linewidth=1.3)
            if key=="cp_J_kg_K":
                ax.plot(x,values("cv_J_kg_K"),":",label="Cv: ρ=const, γ=const")
            if key in ("sigma_S_m","eta_Pa_s","lambda_W_m_K"):
                ax.set_yscale("log")
            ax.set_xlabel(xlabel);ax.set_ylabel(label)
            ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8)
        fig.suptitle(f"Xe: {mode}; {condition}; p={settings.pressure_pa:g} Па; {settings.xe_collision_branch}\n"
                     +footer+("Xe–Xe2+,3+,4+: изотропный захват Ланжевена" if settings.xe_high_ion_neutral_model!="error"
                              else "Без замены отсутствующих интегралов"),fontsize=11)
        path=folder/f"{prefix}_properties.png"
        fig.savefig(path,dpi=160);paths.append(path);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
        contributions=(("lambda_e_effective_h_W_m_K","γ λₑ: электроны"),
                       ("lambda_h_W_m_K","λₕ: тяжёлые частицы"),
                       ("lambda_reaction_W_m_K","λᵣ: реакции, по градиенту Tₕ"),
                       ("lambda_internal_effective_h_W_m_K","γ Λᵢₙₜ: возбуждение"),
                       ("lambda_W_m_K","Сумма по градиенту Tₕ"))
        has_negative=any(np.any(values(key)<0) for key,_ in contributions)
        for key,label in contributions:
            y=values(key)
            if not has_negative:
                y=np.where(y>0,y,np.nan)
            ax.plot(x,y,label=label,linestyle="--" if key=="lambda_W_m_K" else "-")
        if has_negative:
            ax.set_yscale("symlog",linthresh=1e-3)
        else:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel);ax.set_ylabel("Вклад в λ по градиенту Tₕ, Вт/(м·К)")
        ax.set_title(f"Xe: {mode}; {condition}; p={settings.pressure_pa:g} Па\n"+footer+"Составляющие эффективной теплопроводности")
        ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=9)
        path=folder/f"{prefix}_lambda_components.png"
        fig.savefig(path,dpi=160);paths.append(path);plt.close(fig)
    return paths


def run_xe_properties(solutions: list[Solution], atom: AtomicData, dataset: XeDataset,
                      settings: RunSettings, progress: Callable[[str],None] | None = None) -> XePropertyRun:
    rows=[]
    for solution in solutions:
        if solution.element!="Xe":
            continue
        row=compute_row(solution,atom,dataset,settings)
        rows.append(row)
        if progress:
            progress(f"Свойства Xe {solution.mode}: T={solution.temperature_k:g} K, "
                     f"gamma={solution.temperature_ratio:g}: {row['properties_status']}"+
                     (f"; {row['message']}" if row["message"] else ""))
    path=Path(settings.output_dir)/"xe_properties.csv"
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    plots=plot_properties(rows,settings)
    metadata={"data":dataset.provenance(),"status_counts":dict(Counter(r["properties_status"] for r in rows)),
        "computed_columns":list(_VALUES),"reference_columns":"reference_* are interpolated published results, never solver outputs",
        "gamma_scope":"Te/Th=1: ideal/nonideal; Te/Th!=1: ideal, specified approximate 2T transport closure",
        "heat_capacities":"Derivatives with respect to Th at fixed gamma: Cp at fixed p, Cv at fixed rho. LTE preserves the previous equilibrium identity; 2T uses Cv=u_Th|p-u_p|Th*rho_Th|p/rho_p|Th (chain rule). Adaptive Richardson differences.",
        "thermodynamics":"fixed atomic levels; u_ex=-3*kB*T*n_ref*alpha, p_ex=-kB*T*n_ref*alpha; derived from stationary seminar excess free energy",
        "excess_free_energy":"f_ex/(kBT)=-sum n_i*log(1+Zi^2*Gamma/2)+A*T^3*Gamma^3; A=1/(24*pi*b^3), b=e^2/(4*pi*epsilon0*kB)",
        "transport":"Devoto sigma2 and electronic heat3; CE viscosity1, Muckenfuss-Curtiss heavy heat; Butler-Brokaw reaction heat, Eucken internal heat",
        "neutral_high_ion_model":settings.xe_high_ion_neutral_model,
        "missing_pair_closure":"If enabled: V=-C4/r4; isotropic redistribution at capture; Q(l,s)=2*pi*sqrt(C4/kBT)*Gamma(s+1.5)/Gamma(s+2). Not an empirical MDPI integral; no glancing/explicit charge-transfer channels.",
        "screening":"lambda_D^-2=e^2/(epsilon0*kB)*(ne/Te+sum z^2*nz/Th); pair temperature=(mj*Ti+mi*Tj)/(mi+mj)",
        "two_temperature_transport":"Electron transport matrices at Te; heavy matrices at Th; integrals at relative pair temperatures. Auxiliary pair diffusion and projected first-moment reactive closure L*zeta=b, b=kB*R*(tau+Ti*dlog(ni)/dTh). Approximation, not a complete 2T Chapman-Enskog theory.",
        "effective_conductivity":"q=-lambda*grad(Th), constant spatial gamma; lambda=lambda_h+gamma*(lambda_e+Lambda_internal_e)+lambda_reaction_h. Raw electron/internal columns multiply grad(Te); effective_h columns are weighted contributions.",
        "temperature_domain":{"table_temperature_K":[300,30000],
            "allowed_Th":"max(300,300/gamma) <= Th <= min(30000,30000/gamma)",
            "requested_Th_K":[settings.temperature_start_k,settings.temperature_stop_k],
            "requested_ratios":settings.ratios().tolist() if settings.sweep_mode=="temperature_ratio" else None,
            "actual_Th_K":sorted(set(r["T_h_K"] for r in rows)),
            "actual_ratios":sorted(set(r["gamma_Te_Th"] for r in rows)),
            "coulomb_Tstar":[0.1,10000],
            "outside_collision_domain":[{"T_h_K":r["T_h_K"],"gamma":r["gamma_Te_Th"],"reason":r["message"]}
                for r in rows if r["transport_status"]=="outside_collision_domain"],
            "policy":"Limit Xe temperature/ratio grid to table temperatures; leave Cs unchanged. Check each required charged pair Tstar separately; outside points retain caloric/composition data, no extrapolated transport, CLI reports incomplete properties."},
        "comparison":"Different nonideality, level cutoff, transport orders and missing-pair closure can change the result; differences are not numerical error estimates.",
        "source_anomaly":"Nonpositive source lambda and interpolation intervals using it are excluded; other source properties are preserved."}
    return XePropertyRun(path,tuple(plots),tuple(rows),metadata)
