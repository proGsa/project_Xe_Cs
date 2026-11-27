"""Caloric properties and a specified one/two-temperature transport approximation.

All energies are J, physical number densities m^-3, area integrals m^2.
No tabulated transport coefficient is used in these calculations.
"""
from __future__ import annotations

import numpy as np

from .data import AtomicData
from .xe_dataset import PropertyDomainError, XeDataset
from .xe_domain import CollisionDomainError


def partition_moments(atom: AtomicData, T: float, kb: float, ev_j: float) -> tuple[np.ndarray, np.ndarray]:
    means, capacities = [], []
    for level in atom.levels:
        energies = level.energies_ev * ev_j
        log_weights = np.log(level.weights) - energies/(kb*T)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = float(weights @ energies)
        means.append(mean)
        capacities.append(float(weights @ (energies-mean)**2/(kb*T*T)))
    return np.array(means), np.array(capacities)


def caloric_state(atom: AtomicData, T: float, density: np.ndarray, dataset: XeDataset,
                  *, alpha_density: float = 0.0, gamma: float = 0.0,
                  electron_temperature_k: float | None = None) -> dict[str, float]:
    """Seminar regularized excess is derived, NOT a replacement by simple DH.

    b=e^2/(4*pi*epsilon0*kB), A=1/(24*pi*b^3), alpha_density=A*T^3*Gamma^3.
    f_ex/kT=-sum n_i log(1+Zi^2*Gamma/2)+A*T^3*Gamma^3.
    The Gamma equation is df_ex/dGamma=0. Envelope differentiation gives
    mu_i_ex=-kT log(1+Zi^2*Gamma/2), p_ex=-kT*alpha_density,
    U_ex=3*p_ex, matching both the Saha and pressure corrections in model.py.
    Levels remain fixed, exactly as in the composition model.
    """
    te = T if electron_temperature_k is None else electron_temperature_k
    if min(T, te) <= 0 or not np.all(np.isfinite([T, te])) or np.any(density < 0) or not np.all(np.isfinite(density)):
        raise PropertyDomainError("Некорректное состояние для термодинамики Xe.")
    if te != T and (alpha_density != 0 or gamma != 0):
        raise PropertyDomainError("Неидеальные калорические поправки заданы только при Te=Th.")
    kb, ev_j = dataset.constant("k_B"), dataset.constant("e")
    means, _ = partition_moments(atom, te, kb, ev_j)
    masses = dataset.masses[:len(density)]
    rho = float(density @ masses)
    if rho <= 0:
        raise PropertyDomainError("Массовая плотность должна быть положительна.")
    formation = np.r_[0., np.cumsum(atom.ionization_ev[:atom.max_charge])] * ev_j
    ideal_pressure = (float(density.sum()*kb*T) if te == T else
                      float(kb*(density[0]*te+density[1:].sum()*T)))
    energy_ideal = 1.5*ideal_pressure + float(density[1:] @ (formation+means))
    pressure_excess = -kb*T*alpha_density
    energy_excess = 3*pressure_excess
    pressure = ideal_pressure+pressure_excess
    energy = energy_ideal+energy_excess
    charges = np.r_[-1., np.arange(atom.max_charge+1)]
    free_energy_excess = kb*T*(alpha_density-float(density @ np.log1p(charges**2*gamma/2)))
    return {"rho_kg_m3":rho, "u_J_kg":energy/rho, "h_J_kg":(energy+pressure)/rho,
            "internal_energy_J_m3":energy, "eos_pressure_Pa":pressure,
            "pressure_excess_Pa":pressure_excess,
            "internal_energy_excess_J_m3":energy_excess,
            "free_energy_excess_J_m3":free_energy_excess}


def solve_scaled(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Row and column equilibration for transport systems with trace species."""
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise PropertyDomainError("Неконечные элементы транспортной системы.")
    scale = np.sqrt(np.abs(np.diag(matrix)))
    if np.any(scale <= 0):
        raise PropertyDomainError("Нулевой диагональный элемент транспортной системы.")
    scaled = matrix/scale[:,None]/scale[None,:]
    try:
        y = np.linalg.solve(scaled, rhs/scale)
    except np.linalg.LinAlgError as exc:
        raise PropertyDomainError("Вырожденная транспортная система.") from exc
    result = y/scale
    residual = np.linalg.norm(scaled@y-rhs/scale, ord=np.inf)
    norm = np.linalg.norm(scaled, ord=np.inf)*np.linalg.norm(y, ord=np.inf)+np.linalg.norm(rhs/scale, ord=np.inf)
    if not np.all(np.isfinite(result)) or residual > 1e-9*max(norm, 1e-300):
        raise PropertyDomainError("Транспортная система не решена с требуемой точностью.")
    return result


def reaction_conductivity(x: np.ndarray, nD: np.ndarray, enthalpies: np.ndarray,
                          reactions: np.ndarray, kb: float, T: float) -> float:
    """Butler-Brokaw Gram matrix with reaction scaling BEFORE squaring tiny x.

    A=sum v_ij v_ij^T/(n*Dij), v_ij=R_i sqrt(xj/xi)-R_j sqrt(xi/xj).
    Scaling each reaction by sqrt(min participating x) makes the factors
    bounded even for Xe4+ fractions around 1e-134 at low temperature.
    """
    if len(reactions) == 0:
        return 0.0
    logx = np.log(x)
    log_scale = .5*np.min(np.where(reactions != 0, logx[None,:], np.inf), axis=1)
    A = np.zeros((len(reactions), len(reactions)))
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            ei = np.where(reactions[:,i] != 0, log_scale+.5*(logx[j]-logx[i]), -np.inf)
            ej = np.where(reactions[:,j] != 0, log_scale+.5*(logx[i]-logx[j]), -np.inf)
            v = reactions[:,i]*np.exp(ei)-reactions[:,j]*np.exp(ej)
            A += np.outer(v,v)/nD[i,j]
    rhs = np.exp(log_scale)*(reactions@enthalpies)
    y = solve_scaled(A,rhs)
    return float(rhs@y/(kb*T*T))


def pair_temperatures(temperatures: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Relative Maxwellian temperature: (m_j T_i + m_i T_j)/(m_i+m_j)."""
    return ((temperatures[:, None]*masses[None, :]+masses[:, None]*temperatures[None, :])
            /(masses[:, None]+masses[None, :]))


def reaction_conductivity_two_temperature(
    x: np.ndarray, nD: np.ndarray, relative_temperatures: np.ndarray,
    enthalpies: np.ndarray, reactions: np.ndarray, drive: np.ndarray, kb: float,
) -> float:
    """Projected momentum closure L*zeta=b; lambda_r^(h,gamma)=DeltaH*zeta.

    b = kB R (dTi/dTh + Ti*dlog(ni)/dTh) at fixed p,gamma.
    Reaction scaling is applied before trace fractions are squared. This is
    the specified approximate 2T closure, not a full 2T kinetic CE solution.
    """
    if len(reactions) == 0:
        return 0.0
    logx = np.log(x)
    log_scale = .5*np.min(np.where(reactions != 0, logx[None, :], np.inf), axis=1)
    matrix = np.zeros((len(reactions), len(reactions)))
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            ei = np.where(reactions[:, i] != 0, log_scale+.5*(logx[j]-logx[i]), -np.inf)
            ej = np.where(reactions[:, j] != 0, log_scale+.5*(logx[i]-logx[j]), -np.inf)
            v = reactions[:, i]*np.exp(ei)-reactions[:, j]*np.exp(ej)
            matrix += kb*relative_temperatures[i, j]/nD[i, j]*np.outer(v, v)
    scale = np.exp(log_scale)
    response = solve_scaled(matrix, scale*drive)
    return float((scale*(reactions@enthalpies))@response)


def transport_state(atom: AtomicData, T: float, density: np.ndarray, dataset: XeDataset,
                    *, branch: str = "traditional", high_ion_model: str = "langevin_isotropic_capture",
                    electron_temperature_k: float | None = None,
                    dlog_density_dth: np.ndarray | None = None) -> dict[str, float]:
    te = T if electron_temperature_k is None else electron_temperature_k
    ratio = te/T
    if not np.all(np.isfinite([T, te])) or not (300 <= T <= 30000 and 300 <= te <= 30000):
        raise PropertyDomainError("Интегралы Xe поддерживают 300 <= Th,Te <= 30000 K.")
    if np.any(density < 0) or not np.all(np.isfinite(density)) or density[1:].sum() <= 0:
        raise PropertyDomainError("Некорректные концентрации для переноса Xe.")
    kb, charge, eps = (dataset.constant(k) for k in ("k_B","e","epsilon_0"))
    masses = dataset.masses[:len(density)]
    ns = len(density)
    temperatures = np.r_[te, np.full(ns-1, T)]
    relative = (np.full((ns, ns), T) if te == T else pair_temperatures(temperatures, masses))
    charges = np.r_[-1., np.arange(atom.max_charge+1)]
    weighted = float(density @ charges**2)
    screened = float(np.sum(density*charges**2/temperatures))
    debye = (float(np.sqrt(eps*kb*T/(charge**2*weighted))) if te == T and weighted > 0 else
             float(np.sqrt(eps*kb/(charge**2*screened))) if screened > 0 else np.inf)
    active = np.flatnonzero(density > 0)
    heavy = active[active > 0]
    charged = active[charges[active] != 0]
    tstars = []
    if len(charged):
        grid, _ = dataset.table("collision_integrals/coulomb_reduced.csv", "reduced_temperature_Tstar")
        for a, i in enumerate(charged):
            for j in charged[a:]:
                tstar = float(4*np.pi*eps*kb*relative[i,j]*debye/(abs(charges[i]*charges[j])*charge**2))
                if not grid[0] <= tstar <= grid[-1]:
                    name = lambda k: "e" if k == 0 else f"Xe{k-1}+"
                    raise CollisionDomainError(
                        f"Пара {name(i)}–{name(j)}: T*={tstar:.8g} вне "
                        f"[{grid[0]:g}, {grid[-1]:g}]; экстраполяция запрещена.")
                tstars.append(tstar)
    n_total = float(density.sum())
    cache = {}

    def Q(i,j,l,s):
        i,j = sorted((int(i),int(j)))
        key=(i,j,l,s)
        if key not in cache:
            cache[key]=dataset.area(relative[i, j],i,j,l,s,debye,branch,high_ion_model)
        return cache[key]

    means,cint=partition_moments(atom,te,kb,charge)
    formation=np.r_[0.,np.cumsum(atom.ionization_ev[:atom.max_charge])]*charge
    enthalpies=np.r_[2.5*kb*te, 2.5*kb*T+formation+means]
    sigma,le=0.,0.
    ne=float(density[0])
    if ne > 0:
        S={s:sum(density[j]*Q(0,j,1,s) for j in heavy) for s in range(1,6)}
        C={s:np.sqrt(2)*ne*Q(0,0,2,s) for s in (2,3,4)}
        m00=S[1]
        m01=2.5*S[1]-3*S[2]
        m11=25/4*S[1]-15*S[2]+12*S[3]+C[2]
        m12=175/16*S[1]-315/8*S[2]+57*S[3]-30*S[4]+7/4*C[2]-2*C[3]
        m22=1225/64*S[1]-735/8*S[2]+399/2*S[3]-210*S[4]+90*S[5]+77/16*C[2]-7*C[3]+5*C[4]
        if min(m00,m11,m22,m00-m01*m01/m11,m11-m12*m12/m22) <= 0:
            raise PropertyDomainError("Электронная транспортная матрица не положительно определена; проверь нормировку/интерполяцию интегралов.")
        a=solve_scaled(np.array([[m00,m01],[m01,m11]]),np.array([1.,0.]))
        b=solve_scaled(np.array([[m11,m12],[m12,m22]]),np.array([1.,0.]))
        sigma=float(3*ne*charge**2/8*np.sqrt(np.pi/(2*masses[0]*kb*te))*a[0])
        le=float(75/64*ne*kb*np.sqrt(2*np.pi*kb*te/masses[0])*b[0])

    w=density[heavy]/density[heavy].sum()
    V,K=np.zeros((len(heavy),len(heavy))),np.zeros((len(heavy),len(heavy)))
    for a,i in enumerate(heavy):
        for b,j in enumerate(heavy):
            mi,mj=masses[i],masses[j]
            mu=mi*mj/(mi+mj)
            G=mi*mj/(mi+mj)**2
            A=Q(i,j,2,2)/Q(i,j,1,1)
            B=(5*Q(i,j,1,2)-4*Q(i,j,1,3))/Q(i,j,1,1)
            v=5/16*np.sqrt(2*np.pi*mu*kb*T)/Q(i,j,2,2)
            ell=75/128*kb*np.sqrt(2*np.pi*kb*T/mu)/Q(i,j,2,2)
            if a == b:
                V[a,a]+=w[a]/v
                K[a,a]+=w[a]/ell
            else:
                V[a,a]+=2*w[b]*G/v*(5/(3*A)+mj/mi)
                V[a,b]=-2*w[b]*G/v*(5/(3*A)-1)
                K[a,a]+=w[b]/(2*ell)*(15/2*mi**2+25/4*mj**2-3*mj**2*B+4*mi*mj*A)/(A*(mi+mj)**2)
                K[a,b]=-w[b]*G/(2*ell)*(55/4-3*B-4*A)/A
    eta=float(w@solve_scaled(V,np.ones(len(heavy))))
    lh=float(w@solve_scaled(K,np.ones(len(heavy))))
    nD=np.zeros((len(active),len(active)))
    for a,i in enumerate(active):
        for b,j in enumerate(active):
            mu=masses[i]*masses[j]/(masses[i]+masses[j])
            nD[a,b]=3/16*np.sqrt(2*np.pi*kb*relative[i,j]/mu)/Q(i,j,1,1)
    x=density[active]/n_total
    capacities=np.r_[0.,cint][active]
    # n*sum x*c/(sum x/D)=sum x*c/(sum x/(nD)); no huge/small n factors.
    lint=float(np.sum(x*capacities/np.sum(x[:,None]/nD,axis=0)))
    reactions=[]
    present=set(active)
    for z in range(atom.max_charge):
        if {0,z+1,z+2} <= present:
            row=np.zeros(ns);row[0]=1;row[z+1]=-1;row[z+2]=1
            reactions.append(row[active])
    if te == T:
        # Preserve the original LTE/nonideal implementation exactly.
        lr=reaction_conductivity(x,nD,enthalpies[active],np.array(reactions),kb,T)
    elif reactions:
        derivative = np.asarray(dlog_density_dth, dtype=float)
        if derivative.shape != (ns,) or not np.all(np.isfinite(derivative)):
            raise PropertyDomainError("Для реакционного вклада нужны dln(n_i)/dTh при фиксированных p,gamma.")
        R = np.array(reactions)
        tau = np.r_[ratio, np.ones(ns-1)]
        drive = kb*(R@(tau[active]+temperatures[active]*derivative[active]))
        lr=reaction_conductivity_two_temperature(
            x,nD,relative[np.ix_(active,active)],enthalpies[active],R,drive,kb)
    else:
        lr=0.0
    # All contributions below multiply grad(Th); raw le/lint multiply grad(Te).
    total = le+lh+lr+lint if te == T else lh+ratio*(le+lint)+lr
    result={"sigma_S_m":sigma,"eta_Pa_s":eta,"lambda_e_W_m_K":le,
            "lambda_h_W_m_K":lh,"lambda_reaction_W_m_K":lr,
            "lambda_internal_W_m_K":lint,"lambda_W_m_K":total,
            "lambda_e_effective_h_W_m_K":ratio*le,
            "lambda_internal_effective_h_W_m_K":ratio*lint,
            "coulomb_Tstar_min":min(tstars) if tstars else "",
            "coulomb_Tstar_max":max(tstars) if tstars else "",
            "debye_length_m":debye if np.isfinite(debye) else ""}
    for key,value in result.items():
        if value == "":
            continue
        # The projected reactive contribution need not be positive away from LTE.
        if not np.isfinite(value) or (value < 0 and key != "lambda_reaction_W_m_K"):
            raise PropertyDomainError(f"Нефизический результат {key}={value}.")
    if eta <= 0 or result["lambda_W_m_K"] <= 0:
        raise PropertyDomainError("Неположительные вязкость/теплопроводность.")
    return result
