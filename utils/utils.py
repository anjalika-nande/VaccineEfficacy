import numpy as np
from scipy.integrate import odeint, solve_ivp
import pandas as pd
from tqdm import tqdm
import math

def seir(t, y, beta, sigma, gamma):
    """
    Basic SEIR model. Used with ``scipy.integrate.solve_ivp``
    """
    s, e, i, r = y
    dsdt = - beta*s*i # susceptible
    dedt = beta*s*i - sigma*e # exposed
    didt = sigma*e - gamma*i # infected
    drdt = gamma*i # recovered
    
    return [dsdt, dedt, didt, drdt]


def modified_seir(t, y, beta, sigma, gamma, epsL, w):
    """
    Modified SEIR model for instantaneous vaccination. Used with ``scipy.integrate.solve_ivp``
    """
    s, vs, vr, e, i, r, cv, cu = y
    
    dsdt = - beta*s*i # susceptible
    dvsdt = - beta*vs*i + w*vr # vaccinated and susceptible
    dvrdt = - beta*(1-epsL)*vr*i - w*vr # vaccinated and immune
    dedt = beta*(s+vs+(1-epsL)*vr)*i - sigma*e # exposed
    didt = sigma*e - gamma*i # infected
    drdt = gamma*i # recovered
    dcvdt = beta*(vs+(1-epsL)*vr)*i # cumulative infections from V
    dcudt = beta*s*i # cumulative infections from S

    return [dsdt, dvsdt, dvrdt, dedt, didt, drdt, dcvdt, dcudt]


def modified_seir_cont(t, y, beta, sigma, gamma, v, fv, epsL, epsA, w):
    """
    Modified SEIR model for continuous vaccination. Used with ``scipy.integrate.solve_ivp``
    """
    s, vs, vr, e, i, r, cv, cu = y
    
    dsdt = - beta*s*i - v*(1-(vs+vr)/fv)*s # susceptible
    dvsdt = (1 - epsA)*v*(1-(vs+vr)/fv)*s - beta*vs*i + w*vr # vaccinated and susceptible
    dvrdt = epsA*v*(1-(vs+vr)/fv)*s - beta*(1-epsL)*vr*i - w*vr # vaccinated and immune
    dedt = beta*(s+vs+(1-epsL)*vr)*i - sigma*e # exposed
    didt = sigma*e - gamma*i # infected
    drdt = gamma*i # recovered
    dcvdt = beta*(vs+(1-epsL)*vr)*i # cumulative infections from V
    dcudt = beta*s*i # cumulative infections from S

    return [dsdt, dvsdt, dvrdt, dedt, didt, drdt, dcvdt, dcudt]


def run_modified_seir(
    y0: list,
    t: int,
    tv: float,
    beta: float,
    sigma: float,
    gamma: float,
    w: float,
    fv: float,
    eps: float,
    mode: str = "leaky",
):
    """
    Run modified SEIR model with given parameters for instantaneous vaccination. Uses ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    y0: list
        Initial conditions of model. In the form [S(0), V_S(0), V_R(0), E(0), I(0), R(0), C_V(0), C_U(0)]. All the 
        values should sum up to 1. 
    t: int
        Number of days to run simulation (days). The model assumes the simulation begins at Day 0.
    tv: float
        Time of vaccination (days).
    beta: float
        Infectiousness of infected individuals.
    sigma: float
        1/sigma is the duration of the latent period (days). 
    gamma: float
        1/gamma is the duration of the infectious period (days).
    w: float
        Rate at which vaccine efficacy wanes over time (1/days). 
    fv: float
        Fraction of population that is vaccinated.
    eps: float
        Vaccine efficacy. A vaccine that is 50% effective would have an epsilon value of 0.5 For a 
        leaky vaccine, (epsL, epsA) = (eps, 1), and for an all-or-nothing vaccine, (epsL, epsA) = (1, eps).
    mode: str
        


    """
    s0, e0, i0, r0 = y0
    if mode == "leaky":
        epsL = eps
        epsA = 1
    elif mode == "aon":
        epsL = 1
        epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    if tv == -1:
        vs0 = fv * (1 - epsA)
        vr0 = fv * epsA
        s0_vax = s0 - fv
        if s0_vax < 0:
            s0_vax = 0
        y0_vax = [s0_vax, vs0, vr0, e0, i0, r0, 0, 0]
        sol_vax = solve_ivp(
            modified_seir,
            [0, t],
            y0_vax,
            args=(beta, sigma, gamma, epsL, w),
            dense_output=True,
            t_eval=np.linspace(0, t, t+1)
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu

    else:
        if math.floor(tv) == tv:
            t_eval1 = np.linspace(0, tv, tv+1)
            t_eval2 = np.linspace(tv, t, t-tv+1)
        else:
            t_eval1 = np.append(np.linspace(0, math.floor(tv), math.floor(tv)+1), [tv])
            t_eval2 = np.linspace(math.floor(tv)+1, t, t-math.floor(tv)+1)
        sol = solve_ivp(
            seir,
            [0, tv],
            y0,
            args=(beta, sigma, gamma),
            dense_output=True,
            t_eval = t_eval1
        )
        s = sol.y[0]
        e = sol.y[1]
        i = sol.y[2]
        r = sol.y[3]

        vs0 = (1 - epsA) * fv
        vr0 = epsA * fv
        s0_vax = s[-1] - fv
        if s0_vax < 0:
            s0_vax = 0
        y0_vax = [s0_vax, vs0, vr0, e[-1], i[-1], r[-1], 0, 0]
        sol_vax = solve_ivp(
            modified_seir,
            [tv, t],
            y0_vax,
            args=(beta, sigma, gamma, epsL, w),
            dense_output=True,
            t_eval = t_eval2
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
        cv = np.concatenate((np.zeros(np.shape(s[:-1])), cv), axis=None)
        cu = np.concatenate((np.zeros(np.shape(s[:-1])), cu), axis=None)

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu


def run_modified_seir_cont(
    y0: list,
    t: float,
    tv: float,
    beta: float,
    sigma: float,
    gamma: float,
    w: float,
    fv: float,
    eps: float,
    v: float,
    mode: str = "leaky",
):
    s0, e0, i0, r0 = y0
    if mode == "leaky":
        epsL = eps
        epsA = 1
    elif mode == "aon":
        epsL = 1
        epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    if tv == -1:
        vs0 = 0
        vr0 = 0
        y0_vax = [s0, vs0, vr0, e0, i0, r0, 0, 0]
        sol_vax = solve_ivp(
            modified_seir_cont,
            [0, t],
            y0_vax,
            args=(beta, sigma, gamma, v, fv, epsL, epsA, w),
            dense_output=True,
            t_eval=np.linspace(0, t, t+1)
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu

    else:
        if math.floor(tv) == tv:
            t_eval1 = np.linspace(0, tv, tv+1)
            t_eval2 = np.linspace(tv, t, t-tv+1)
        else:
            t_eval1 = np.append(np.linspace(0, math.floor(tv), math.floor(tv)+1), [tv])
            t_eval2 = np.linspace(math.floor(tv)+1, t, t-math.floor(t)+1)
        sol = solve_ivp(
            seir,
            [0, tv],
            y0,
            args=(beta, sigma, gamma),
            dense_output=True,
            t_eval = t_eval1
        )
        s = sol.y[0]
        e = sol.y[1]
        i = sol.y[2]
        r = sol.y[3]

        vs0 = 0
        vr0 = 0
        y0_vax = [s[-1], vs0, vr0, e[-1], i[-1], r[-1], 0, 0]
        sol_vax = solve_ivp(
            modified_seir_cont,
            [tv, t],
            y0_vax,
            args=(beta, sigma, gamma, v, fv, epsL, epsA, w),
            dense_output=True,
            t_eval = t_eval2
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
        cv = np.concatenate((np.zeros(np.shape(s[:-1])), cv), axis=None)
        cu = np.concatenate((np.zeros(np.shape(s[:-1])), cu), axis=None)

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu

def run_modified_seir_days(y0: list, t: int, tv: int, beta: float, sigma: float, gamma: float, w: float, fv: float, \
    eps: float, mode: str = 'leaky'):
    s0, e0, i0, r0 = y0
    if mode == 'leaky':
        epsL = eps; epsA = 1
    elif mode == 'aon':
        epsL = 1; epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")
    
    if tv == -1:
        vs0 = fv*(1-epsA)*s0; vr0 = fv*epsA*s0; s0_vax = s0*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e0, i0, r0, 0, 0]
        sol_vax = solve_ivp(
            modified_seir,
            [0, t],
            y0_vax,
            args=(beta, sigma, gamma, epsL, w),
            dense_output=True,
            t_eval=np.linspace(0, t, t+1)
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu
    
    else:
        t_eval1 = np.linspace(0, tv, tv+1)
        t_eval2 = np.linspace(tv, t, t-tv+1)
        sol = solve_ivp(
            seir,
            [0, tv],
            y0,
            args=(beta, sigma, gamma),
            dense_output=True,
            t_eval = t_eval1
        )
        s = sol.y[0]
        e = sol.y[1]
        i = sol.y[2]
        r = sol.y[3]

        #sim = odeint(seir, y0, np.linspace(0, tv, tv+1), args=(beta, sigma, gamma))
        #s, e, i, r = sim.T

        vs0 = (1-epsA)*fv*s[-1]; vr0 = epsA*fv*s[-1]; s0_vax = s[-1]*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e[-1], i[-1], r[-1], 0, 0]
        sol_vax = solve_ivp(
            modified_seir_cont,
            [tv, t],
            y0_vax,
            args=(beta, sigma, gamma, v, fv, epsL, epsA, w),
            dense_output=True,
            t_eval = t_eval2
        )
        s_vax = sol_vax.y[0]
        vs = sol_vax.y[1]
        vr = sol_vax.y[2]
        e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]
        r_vax = sol_vax.y[5]
        v = vs + vr
        cv = sol_vax.y[6]
        cu = sol_vax.y[7]

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
        cv = np.concatenate((np.zeros(np.shape(s[:-1])), cv), axis=None)
        cu = np.concatenate((np.zeros(np.shape(s[:-1])), cu), axis=None)
        """
        sim_vax = odeint(modified_seir, y0_vax, np.linspace(0, t-tv, t-tv+1), args=(beta, sigma, gamma, epsL, w))
        s_vax, vs, vr, e_vax, i_vax, r_vax = sim_vax.T
        v = vs + vr

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
        """
    
        return s_vax, vs, vr, v, e_vax, i_vax, r_vax, cv, cu


"""
def run_scenarios_size(y0: list, t: int, size: float, R0s: np.ndarray, sigma: float, gamma: float, \
    epss: np.ndarray, measured: int):  
    s0, e0, i0, r0 = y0
    df_R0s = []; df_epss = []; df_fvs = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []
    df_rs = []; df_rleakys = []; df_raons = []

    for R0 in R0s:
        beta = R0 * gamma
        sol = solve_ivp(seir_ivp, [0, t], y0, args=(beta, sigma, gamma), dense_output=True)
        #r = sol.y[3]; r10 = r[-1]*0.1; r25 = r[-1]*0.25

        def _reach_size10(t, y, beta, sigma, gamma): return y[3] - 0.1
        def _reach_size25(t, y, beta, sigma, gamma): return y[3] - 0.25

        _reach_size10.terminate=True
        _reach_size25.terminate=True
                
        for eps in epss:
            if size == 0:
                fc = 1/eps * (1 - 1/R0)
                if fc < 0:
                    fc = 99999 # make blank on plot
                elif fc > 0.98:
                    fc = 0.98
                tv = -1
                t_new = measured
            else:
                if size == 0.1:
                    sol = solve_ivp(seir_ivp, [0, t], y0, args=(beta, sigma, gamma), \
                        events=_reach_size10, dense_output=True)
                elif size == 0.25:
                    sol = solve_ivp(seir_ivp, [0, t], y0, args=(beta, sigma, gamma), \
                        events=_reach_size25, dense_output=True)

                if np.array(sol.t_events).size == 0:
                    fc = 99999; fv = 99999; r_perc_leaky = 99999; r_perc_aon = 99999; r_perc_diff = 99999
                else:
                    s_temp = np.ravel(np.array(sol.y_events[0]))[0]
                    tv = np.ravel(np.array(sol.t_events))[0]
                    fc = 1/eps * (1 - 1/(R0*s_temp))
                    if fc < 0:
                        fc = 99999
                    elif fc > 0.98:
                        fc = 0.98
                    t_new = tv + measured
            
                        
            for cov in covs:
                if fc != 99999:
                    if cov == 'Below fc':
                        fv = fc * 0.8
                    elif cov == 'Slightly Above fc':
                        fv = 0.98 - ((0.98 - fc) * 0.8)
                    else:
                        fv = 0.98 - ((0.98 - fc) * 0.5)
                        
                    sol_vax = solve_ivp(seir_ivp, [0, t_new], y0, args=(beta, sigma, gamma), dense_output=True)
                    r_vax = sol_vax.y[3]
                        
                    _, _, _, _, _, _, r_leaky = run_modified_seir_ivp(y0, t_new, tv, beta, sigma, gamma, fv, eps, mode='leaky')
                    _, _, _, _, _, _, r_aon = run_modified_seir_ivp(y0, t_new, tv, beta, sigma, gamma, fv, eps, mode='aon')

                    r_perc_leaky = (r_vax[-1] - r_leaky[-1]) / r_vax[-1] * 100
                    r_perc_aon = (r_vax[-1] - r_aon[-1]) / r_vax[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                df_R0s.append(R0)
                df_epss.append(eps)
                df_fvs.append(fv)
                df_covs.append(cov)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe                        
    data = {'R0': df_R0s, 'VE': df_epss, 'Vax Coverage': df_covs, 'fv': df_fvs, \
        'Leaky': df_r_perc_leakys, 'AON': df_r_perc_aons, 'Diff': df_r_perc_diffs}
    vax_df = pd.DataFrame(data=data)

    return vax_df
"""


def run_scenarios(
    y0: list,
    t: int,
    size: float,
    covs: np.ndarray,
    R0s: np.ndarray,
    sigma: float,
    gamma: float,
    w: float,
    epss: np.ndarray,
    measured: int,
):
    df_R0s = []
    df_epss = []
    df_covs = []
    df_fvs = []
    df_fcs = []
    df_hit = []
    df_r_perc_leakys = []
    df_r_perc_aons = []
    df_r_perc_diffs = []

    s0, _, _, _ = y0

    for i in tqdm(range(0, len(R0s))):
        R0 = R0s[i]
        beta = R0 * gamma
        sol = solve_ivp(
            seir, [0, t], y0, args=(beta, sigma, gamma), dense_output=True
        )

        def _reach_size10(t, y, beta, sigma, gamma):
            return y[3] - 0.1

        def _reach_size25(t, y, beta, sigma, gamma):
            return y[3] - 0.25

        _reach_size10.terminate = True
        _reach_size25.terminate = True

        for eps in epss:
            if size == 0:
                fc = 1 / eps * (1 - 1 / R0)
                fs = s0 # remaining S
                tv = -1
                t_new = measured
            else:
                if size == 0.1:
                    sol = solve_ivp(
                        seir,
                        [0, t],
                        y0,
                        args=(beta, sigma, gamma),
                        events=_reach_size10,
                        dense_output=True,
                    )
                elif size == 0.25:
                    sol = solve_ivp(
                        seir,
                        [0, t],
                        y0,
                        args=(beta, sigma, gamma),
                        events=_reach_size25,
                        dense_output=True,
                    )

                if np.array(sol.t_events).size == 0:  # never reaches target size
                    fc = 99999
                else:
                    fs = np.ravel(np.array(sol.y_events[0]))[0]  # remaining S
                    tv = np.ravel(np.array(sol.t_events))[0]
                    fc = 1 / eps * (1 - 1 / (R0 * fs))
                    t_new = math.floor(tv + measured)
            if fc < 0:
                fc = 99999
            elif (fc > 0.98) and (fc != 99999):
                fc = 0.98

            for cov in covs:
                if fc == 99999:
                    fv = 99999
                    r_perc_leaky = 99999
                    r_perc_aon = 99999
                    r_perc_diff = 99999
                else:
                    fv = cov * fs
                    sol_vax = solve_ivp(
                        seir,
                        [0, t_new],
                        y0,
                        args=(beta, sigma, gamma),
                        dense_output=True,
                    )
                    r_vax = sol_vax.y[3]

                    _, _, _, _, _, _, r_leaky, _, _ = run_modified_seir(
                        y0, t_new, tv, beta, sigma, gamma, w, fv, eps, mode="leaky"
                    )
                    _, _, _, _, _, _, r_aon, _, _ = run_modified_seir(
                        y0, t_new, tv, beta, sigma, gamma, w, fv, eps, mode="aon"
                    )

                    r_perc_leaky = (r_vax[-1] - r_leaky[-1]) / r_vax[-1] * 100
                    r_perc_aon = (r_vax[-1] - r_aon[-1]) / r_vax[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                df_R0s.append(R0)
                df_epss.append(eps)
                df_covs.append(cov)
                df_fvs.append(fv)
                df_fcs.append(fc)
                df_hit.append(fv > fc) # only for 'valid' values (0 <= fc <= S(0))
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe
    data = {
        "R0": df_R0s,
        "VE": df_epss,
        "Vax Coverage": df_covs,
        "fv*": df_fcs,
        "fv": df_fvs,
        "Above HIT": df_hit,
        "Leaky": df_r_perc_leakys,
        "AON": df_r_perc_aons,
        "Diff": df_r_perc_diffs,
    }
    vax_df = pd.DataFrame(data=data)

    return vax_df


def run_scenarios_days(y0: list, t: int, tv: int, R0s: np.ndarray, sigma: float, gamma: float, w:float, epss: np.ndarray):  
    df_R0s = []; df_epss = []; df_fvs = []; df_fcs = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []

    for i in tqdm(range(0, len(R0s))):
        R0 = R0s[i]
        beta = R0 * gamma
        sol = solve_ivp(
            seir, [0, t], y0, args=(beta, sigma, gamma), dense_output=True
        )
        r = sol.y[3]
                
        for eps in epss:
            if tv == -1:
                fc = 1/eps * (1 - 1/R0)
            else:
                sol_temp = solve_ivp(
                    seir, [0, tv], y0, args=(beta, sigma, gamma), dense_output=True
                )
                s_temp = sol_temp.y[0]
                fc = 1/eps * (1 - 1/(R0*s_temp[-1]))
            
            if fc < 0:
                fc = 99999
            elif (fc > 0.98) and (fc != 99999):
                fc = 0.98

            for cov in covs:
                if fc == 99999:
                    fv = 99999
                    r_perc_leaky = 99999
                    r_perc_aon = 99999
                    r_perc_diff = 99999
                else:
                    if cov == 'Below fc':
                        fv = fc * 0.8
                    elif cov == 'Slightly Above fc':
                        fv = 1 - ((1 - fc) * 0.8)
                    else:
                        fv = 1 - ((1 - fc) * 0.5)

                    _, _, _, _, _, _, r_leaky, _, _ = run_modified_seir(y0, t, tv, beta, sigma, gamma, w, fv, eps, mode='leaky')
                    _, _, _, _, _, _, r_aon, _, _ = run_modified_seir(y0, t, tv, beta, sigma, gamma, w, fv, eps, mode='aon')

                    r_perc_leaky = (r[-1] - r_leaky[-1]) / r[-1] * 100
                    r_perc_aon = (r[-1] - r_aon[-1]) / r[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                df_R0s.append(R0)
                df_epss.append(eps)
                df_covs.append(cov)
                df_fvs.append(fv)
                df_fcs.append(fc)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe                        
    data = {
        "R0": df_R0s,
        "VE": df_epss,
        "Vax Coverage": df_covs,
        "fv*": df_fcs,
        "fv": df_fvs,
        "Leaky": df_r_perc_leakys,
        "AON": df_r_perc_aons,
        "Diff": df_r_perc_diffs,
    }
    vax_df = pd.DataFrame(data=data)

    return vax_df

"""
def run_scenarios_below_remS(
    y0: list,
    t: int,
    size: float,
    covs: list,
    R0s: np.ndarray,
    sigma: float,
    gamma: float,
    epss: np.ndarray,
    measured: int,
):
    df_R0s = []
    df_epss = []
    df_covs = []
    df_fvs = []
    df_fcs = []
    df_hit = []
    df_r_perc_leakys = []
    df_r_perc_aons = []
    df_r_perc_diffs = []

    s0, _, _, _ = y0

    for i in tqdm(range(0, len(R0s))):
        R0 = R0s[i]
        beta = R0 * gamma
        sol = solve_ivp(
            seir_ivp, [0, t], y0, args=(beta, sigma, gamma), dense_output=True
        )

        def _reach_size10(t, y, beta, sigma, gamma):
            return y[3] - 0.1

        def _reach_size25(t, y, beta, sigma, gamma):
            return y[3] - 0.25

        _reach_size10.terminate = True
        _reach_size25.terminate = True

        for eps in epss:
            if size == 0:
                fc = 1 / eps * (1 - 1 / R0)
                fs = s0 # remaining S
                tv = -1
                t_new = measured
            else:
                if size == 0.1:
                    sol = solve_ivp(
                        seir_ivp,
                        [0, t],
                        y0,
                        args=(beta, sigma, gamma),
                        events=_reach_size10,
                        dense_output=True,
                    )
                elif size == 0.25:
                    sol = solve_ivp(
                        seir_ivp,
                        [0, t],
                        y0,
                        args=(beta, sigma, gamma),
                        events=_reach_size25,
                        dense_output=True,
                    )

                if np.array(sol.t_events).size == 0:  # never reaches target size
                    fc = 99999
                else:
                    fs = np.ravel(np.array(sol.y_events[0]))[0]  # remaining S
                    tv = np.ravel(np.array(sol.t_events))[0]
                    fc = 1 / eps * (1 - 1 / (R0 * fs))
                    t_new = tv + measured
            if fc < 0:
                fc = 99999
            elif (fc > 0.98) and (fc != 99999):
                fc = 0.98

            for cov in covs:
                if fc == 99999:
                    fv = 99999
                    r_perc_leaky = 99999
                    r_perc_aon = 99999
                    r_perc_diff = 99999
                else:
                    if cov == 'Below fc':
                        fv = 0.8 * fc
                    else: 
                        fv = fs

                    sol_vax = solve_ivp(
                        seir_ivp,
                        [0, t_new],
                        y0,
                        args=(beta, sigma, gamma),
                        dense_output=True,
                    )
                    r_vax = sol_vax.y[3]

                    _, _, _, _, _, _, r_leaky = run_modified_seir_ivp(
                        y0, t_new, tv, beta, sigma, gamma, fv, eps, mode="leaky"
                    )
                    _, _, _, _, _, _, r_aon = run_modified_seir_ivp(
                        y0, t_new, tv, beta, sigma, gamma, fv, eps, mode="aon"
                    )

                    r_perc_leaky = (r_vax[-1] - r_leaky[-1]) / r_vax[-1] * 100
                    r_perc_aon = (r_vax[-1] - r_aon[-1]) / r_vax[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                df_R0s.append(R0)
                df_epss.append(eps)
                df_covs.append(cov)
                df_fvs.append(fv)
                df_fcs.append(fc)
                df_hit.append(fv > fc) # only for 'valid' values (0 <= fc <= S(0))
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe
    data = {
        "R0": df_R0s,
        "VE": df_epss,
        "Vax Coverage": df_covs,
        "fv*": df_fcs,
        "fv": df_fvs,
        "Above HIT": df_hit,
        "Leaky": df_r_perc_leakys,
        "AON": df_r_perc_aons,
        "Diff": df_r_perc_diffs,
    }
    vax_df = pd.DataFrame(data=data)

    return vax_df
"""