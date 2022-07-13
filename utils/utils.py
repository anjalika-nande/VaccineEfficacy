import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

def seir (y, t, beta, sigma, gamma):
    """
    Basic SEIR model
    """
    s, e, i, r = y
    dsdt = - beta*s*i
    dedt = beta*s*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i
    
    return dsdt, dedt, didt, drdt


def seir_ivp(t, y, beta, sigma, gamma):
    """
    Basic SEIR model
    """
    s, e, i, r = y
    dsdt = - beta*s*i
    dedt = beta*s*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i
    
    return [dsdt, dedt, didt, drdt]


def modified_seir(y, t, beta, sigma, gamma, epsL):
    """
    Modified SEIR model for instantaneous vaccination. 
    """
    s, vs, vr, e, i, r = y
    
    dsdt = - beta*s*i
    dvsdt = - beta*vs*i
    dvrdt = - beta*(1-epsL)*vr*i
    dedt = beta*(s+vs+(1-epsL)*vr)*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return dsdt, dvsdt, dvrdt, dedt, didt, drdt


def modified_seir_ivp(t, y, beta, sigma, gamma, epsL):
    """
    Modified SEIR model for instantaneous vaccination. 
    """
    s, vs, vr, e, i, r = y
    
    dsdt = - beta*s*i
    dvsdt = - beta*vs*i
    dvrdt = - beta*(1-epsL)*vr*i
    dedt = beta*(s+vs+(1-epsL)*vr)*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return [dsdt, dvsdt, dvrdt, dedt, didt, drdt]


def modified_seir_waning(y, t, beta, sigma, gamma, epsL, w):
    """
    Modified SEIR model for instantaneous vaccination. 
    """
    s, vs, vr, e, i, r = y
    
    dsdt = - beta*s*i
    dvsdt = - beta*vs*i + w*vr
    dvrdt = - beta*(1-epsL)*vr*i - w*vr
    dedt = beta*(s+vs+(1-epsL)*vr)*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return dsdt, dvsdt, dvrdt, dedt, didt, drdt


def run_modified_seir(y0: list, t: int, tv: int, beta: float, sigma: float, gamma: float, fv: float, \
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
        y0_vax = [s0_vax, vs0, vr0, e0, i0, r0]
        sim_vax = odeint(modified_seir, y0_vax, np.linspace(0, t, t+1), args=(beta, sigma, gamma, epsL))
        s_vax, vs, vr, e_vax, i_vax, r_vax = sim_vax.T
        v = vs + vr

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax
    
    else:
        sim = odeint(seir, y0, np.linspace(0, tv, tv+1), args=(beta, sigma, gamma))
        s, e, i, r = sim.T

        vs0 = (1-epsA)*fv*s[-1]; vr0 = epsA*fv*s[-1]; s0_vax = s[-1]*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e[-1], i[-1], r[-1]]
        sim_vax = odeint(modified_seir, y0_vax, np.linspace(0, t-tv, t-tv+1), args=(beta, sigma, gamma, epsL))
        s_vax, vs, vr, e_vax, i_vax, r_vax = sim_vax.T
        v = vs + vr

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
    
        return s_vax, vs, vr, v, e_vax, i_vax, r_vax


def run_modified_seir_ivp(y0: list, t: float, tv: float, beta: float, sigma: float, gamma: float, fv: float, \
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
        y0_vax = [s0_vax, vs0, vr0, e0, i0, r0]
        sol_vax = solve_ivp(modified_seir_ivp, [0, t], y0_vax, args=(beta, sigma, gamma, epsL), \
            dense_output=True)
        s_vax = sol_vax.y[0]; vs = sol_vax.y[1]; vr = sol_vax.y[2]; e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]; r_vax = sol_vax.y[5]
        v = vs + vr

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax
    
    else:
        sol = solve_ivp(seir_ivp, [0, tv], y0, args=(beta, sigma, gamma), dense_output=True)
        s = sol.y[0]; e = sol.y[1]; i = sol.y[2]; r = sol.y[3]

        vs0 = (1-epsA)*fv*s[-1]; vr0 = epsA*fv*s[-1]; s0_vax = s[-1]*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e[-1], i[-1], r[-1]]
        sol_vax = solve_ivp(modified_seir_ivp, [tv, t], y0_vax, args=(beta, sigma, gamma, epsL), \
            dense_output=True)
        s_vax = sol_vax.y[0]; vs = sol_vax.y[1]; vr = sol_vax.y[2]; e_vax = sol_vax.y[3]
        i_vax = sol_vax.y[4]; r_vax = sol_vax.y[5]
        v = vs + vr

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
    
        return s_vax, vs, vr, v, e_vax, i_vax, r_vax


def run_modified_seir_waning(y0: list, t: int, tv: int, beta: float, sigma: float, gamma: float, fv: float, \
    eps: float, w: float, mode: str = 'leaky'):
    s0, e0, i0, r0 = y0
    if mode == 'leaky':
        epsL = eps; epsA = 1
    elif mode == 'aon':
        epsL = 1; epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")
    
    if tv == -1:
        vs0 = fv*(1-epsA)*s0; vr0 = fv*epsA*s0; s0_vax = s0*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e0, i0, r0]
        sim_vax = odeint(modified_seir_waning, y0_vax, np.linspace(0, t, t+1), args=(beta, sigma, gamma, epsL, w))
        s_vax, vs, vr, e_vax, i_vax, r_vax = sim_vax.T
        v = vs + vr

        return s_vax, vs, vr, v, e_vax, i_vax, r_vax
    
    else:
        sim = odeint(seir, y0, np.linspace(0, tv, tv+1), args=(beta, sigma, gamma))
        s, e, i, r = sim.T

        vs0 = (1-epsA)*fv*s[-1]; vr0 = epsA*fv*s[-1]; s0_vax = s[-1]*(1-fv)
        y0_vax = [s0_vax, vs0, vr0, e[-1], i[-1], r[-1]]
        sim_vax = odeint(modified_seir_waning, y0_vax, np.linspace(0, t-tv, t-tv+1), args=(beta, sigma, gamma, epsL, w))
        s_vax, vs, vr, e_vax, i_vax, r_vax = sim_vax.T
        v = vs + vr

        s_vax = np.concatenate((s[:-1], s_vax), axis=None)
        vs = np.concatenate((np.zeros(np.shape(s[:-1])), vs), axis=None)
        vr = np.concatenate((np.zeros(np.shape(s[:-1])), vr), axis=None)
        v = np.concatenate((np.zeros(np.shape(s[:-1])), v), axis=None)
        e_vax = np.concatenate((e[:-1], e_vax), axis=None)
        i_vax = np.concatenate((i[:-1], i_vax), axis=None)
        r_vax = np.concatenate((r[:-1], r_vax), axis=None)
    
        return s_vax, vs, vr, v, e_vax, i_vax, r_vax


def plot_timeseries(sim_novax, sim_leaky, sim_aon, figsize=(22, 10), savefig=False, filename: str = None):
    s, e, i, r = sim_novax
    s_leaky, vs_leaky, vr_leaky, v_leaky, e_leaky, i_leaky, r_leaky = sim_leaky
    s_aon, vs_aon, vr_aon, v_aon, e_aon, i_aon, r_aon = sim_aon
    t = np.linspace(0, len(s)-1, len(s))

    fig = plt.figure(facecolor='w', figsize=figsize)

    ax1 = fig.add_subplot(231, axisbelow=True)
    ax1.plot(t, s_leaky, 'b', alpha=0.5, lw=2, label='$S$')
    ax1.plot(t, i_leaky, 'r', alpha=0.5, lw=2, label='$I$')
    ax1.plot(t, r_leaky, 'g', alpha=0.5, lw=2, label='$R$')
    ax1.plot(t, v_leaky, 'y', alpha=0.5, lw=2, label='$V_{S} + V_{R}$')
    ax1.plot(t, vs_leaky, 'y--', alpha=0.5, lw=2, label='$V_{S}$')
    ax1.plot(t, vr_leaky, 'y:', alpha=0.5, lw=2, label='$V_{R}$')
    ax1.set_title("With Leaky Vaccine")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of Population")
    ax1.set_xlabel('Time (Days)')
    ax1.grid(linewidth=0.5)
    legend = ax1.legend(loc='upper right'); legend.get_frame().set_alpha(0.5)

    ax2 = fig.add_subplot(232, axisbelow=True)
    ax2.plot(t, s_aon, 'b', alpha=0.5, lw=2)
    ax2.plot(t, i_aon, 'r', alpha=0.5, lw=2)
    ax2.plot(t, r_aon, 'g', alpha=0.5, lw=2)
    ax2.plot(t, v_aon, 'y', alpha=0.5, lw=2)
    ax2.plot(t, vs_aon, 'y--', alpha=0.5, lw=2)
    ax2.plot(t, vr_aon, 'y:', alpha=0.5, lw=2)
    ax2.set_title("With All-or-None Vaccine")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of Population")
    ax2.set_xlabel('Time (Days)')
    ax2.grid(linewidth=0.5)
    
    ax3 = fig.add_subplot(233, axisbelow=True)
    ax3.plot(t, s, 'b', alpha=0.5, lw=2)
    ax3.plot(t, i, 'r', alpha=0.5, lw=2)
    ax3.plot(t, r, 'g', alpha=0.5, lw=2)
    ax3.set_title('Without Vaccine')
    ax3.set_ylabel('Fraction of Population')
    ax3.set_xlabel('Time (Days)')
    ax3.set_ylim(0, 1)
    ax3.grid(linewidth=0.5)
    
    ax4 = fig.add_subplot(234, axisbelow=True)
    ax4.plot(t, i_leaky, 'r--', alpha=0.5, lw=2, label='$I$ - Leaky')
    ax4.plot(t, i_aon, 'r:', alpha=0.5, lw=2, label= '$I$ - AON')
    ax4.plot(t, i, 'r', alpha=0.5, lw=2, label='$I$ - No Vax')
    ax4.grid(linewidth=0.5)
    ax4.set_ylabel('Fraction of Population')
    ax4.set_xlabel('Time (Days)')
    ax4.set_title('Comparing Infected Population')
    legend = ax4.legend(); legend.get_frame().set_alpha(0.5)

    ax5 = fig.add_subplot(235, axisbelow=True)
    ax5.plot(t, r_leaky, 'g--', alpha=0.5, lw=2, label='$R$ - Leaky')
    ax5.plot(t, r_aon, 'g:', alpha=0.5, lw=2, label='$R$ - AON')
    ax5.plot(t, r, 'g', alpha=0.5, lw=2, label='$R$ - No Vax')
    ax5.grid(linewidth=0.5)
    ax5.set_ylabel('Fraction of Population')
    ax5.set_xlabel('Time (Days)')
    ax5.set_title('Comparing Recovered Population')
    legend = ax5.legend(); legend.get_frame().set_alpha(0.5)

    if savefig:
        plt.savefig(filename, bbox_inches='tight')


def run_scenarios(y0: list, t: int, tv: int, R0s: np.ndarray, sigma: float, gamma: float, epss: np.ndarray):  
    s0, e0, i0, r0 = y0
    df_R0s = []; df_epss = []; df_fvs = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []
    df_rs = []; df_rleakys = []; df_raons = []

    for R0 in R0s:
        beta = R0 * gamma
        sim = odeint(seir, y0, np.linspace(0, t, t+1), args=(beta, sigma, gamma))
        _, _, _, r = sim.T
        
        r_tot = r[-1]
                
        for eps in epss:
            if tv == -1:
                fc = 1/eps * (1 - 1/R0)
            else:
                sim_temp = odeint(seir, y0, np.linspace(0, tv, tv+1), args=(beta, sigma, gamma))
                s_temp, _, _, _ = sim_temp.T
                fc = 1/eps * (1 - 1/(R0*s_temp[-1]))
                
            for cov in covs:
                if cov == 'Below fc':
                    fv = fc * 0.8
                elif cov == 'Slightly Above fc':
                    fv = 1 - ((1 - fc) * 0.8)
                else:
                    fv = 1 - ((1 - fc) * 0.5)

                if fv < 0:
                    fv = 0
                elif fv > 0.98:
                    fv = 0.98
                else:
                    fv = fv
                
                _, _, _, _, _, _, r_leaky = run_modified_seir(y0, t, tv, beta, sigma, gamma, fv, eps, mode='leaky')
                _, _, _, _, _, _, r_aon = run_modified_seir(y0, t, tv, beta, sigma, gamma, fv, eps, mode='aon')

                r_perc_leaky = (r[-1] - r_leaky[-1]) / r[-1] * 100
                r_perc_aon = (r[-1] - r_aon[-1]) / r[-1] * 100
                r_perc_diff = r_perc_aon - r_perc_leaky

                df_R0s.append(R0)
                df_epss.append(eps)
                df_fvs.append(fv)
                df_covs.append(cov)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)
                df_rs.append(r[-1])
                df_rleakys.append(r_leaky[-1])
                df_raons.append(r_aon[-1])

    # build dataframe                        
    data = {'R0': df_R0s, 'VE': df_epss, 'Vax Coverage': df_covs, 'fv': df_fvs, \
        'Leaky': df_r_perc_leakys, 'AON': df_r_perc_aons, 'Diff': df_r_perc_diffs, \
        'r': df_rs, 'r_leaky': df_rleakys, 'r_aon': df_raons}
    vax_df = pd.DataFrame(data=data)

    return vax_df


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
                    t_new = tv + measured
                        
            for cov in covs:
                if fc != 99999:
                    if cov == 'Below fc':
                        fv = fc * 0.8
                    elif cov == 'Slightly Above fc':
                        fv = 1 - ((1 - fc) * 0.8)
                    else:
                        fv = 1 - ((1 - fc) * 0.5)

                    if fv < 0:
                        fv = 0
                    elif fv > 0.98:
                        fv = 0.98
                    else:
                        fv = fv
                        
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
    

def run_scenarios_waning(y0: list, t: int, tv: int, R0s: np.ndarray, sigma: float, gamma: float, epss: np.ndarray, w: float):  
    s0, e0, i0, r0 = y0
    df_R0s = []; df_epss = []; df_fvs = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []
    df_rs = []; df_rleakys = []; df_raons = []

    for R0 in R0s:
        beta = R0 * gamma
        sim = odeint(seir, y0, np.linspace(0, t, t+1), args=(beta, sigma, gamma))
        _, _, _, r = sim.T
        
        r_tot = r[-1]
                
        for eps in epss:
            if tv == -1:
                fc = 1/eps * (1 - 1/R0)
            else:
                sim_temp = odeint(seir, y0, np.linspace(0, tv, tv+1), args=(beta, sigma, gamma))
                s_temp, _, _, _ = sim_temp.T
                fc = 1/eps * (1 - 1/(R0*s_temp[-1]))
                
            for cov in covs:
                if cov == 'Below fc':
                    fv = fc * 0.8
                elif cov == 'Slightly Above fc':
                    fv = 1 - ((1 - fc) * 0.8)
                else:
                    fv = 1 - ((1 - fc) * 0.5)

                if fv < 0:
                    fv = 0
                elif fv > 0.98:
                    fv = 0.98
                else:
                    fv = fv
                
                _, _, _, _, _, _, r_leaky = run_modified_seir_waning(y0, t, tv, beta, sigma, gamma, fv, eps, w, mode='leaky')
                _, _, _, _, _, _, r_aon = run_modified_seir_waning(y0, t, tv, beta, sigma, gamma, fv, eps, w, mode='aon')

                r_perc_leaky = (r[-1] - r_leaky[-1]) / r[-1] * 100
                r_perc_aon = (r[-1] - r_aon[-1]) / r[-1] * 100
                r_perc_diff = abs(r_perc_aon - r_perc_leaky)
                
                df_R0s.append(R0)
                df_epss.append(eps)
                df_fvs.append(fv)
                df_covs.append(cov)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)
                df_rs.append(r[-1])
                df_rleakys.append(r_leaky[-1])
                df_raons.append(r_aon[-1])

    # build dataframe                        
    data = {'R0': df_R0s, 'VE': df_epss, 'Vax Coverage': df_covs, 'fv': df_fvs, \
        'Leaky': df_r_perc_leakys, 'AON': df_r_perc_aons, 'Diff': df_r_perc_diffs, \
        'r': df_rs, 'r_leaky': df_rleakys, 'r_aon': df_raons}
    vax_df = pd.DataFrame(data=data)

    return vax_df


def plot_scenarios(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, dim: int = 2):
    r0s = np.arange(1.0, 3.0, 0.01); epss = np.arange(0.01, 1.0, 0.01)
    plot_r0, plot_eps = np.nan_to_num(np.meshgrid(r0s, epss, indexing='ij'))

    # pre
    below_df1 = df1[df1['Vax Coverage'] == 'Below fc']
    slabove_df1 = df1[df1['Vax Coverage'] == 'Slightly Above fc']
    above_df1 = df1[df1['Vax Coverage'] == 'Above fc']

    pre_below = np.nan_to_num(np.reshape(below_df1['Diff'].to_numpy(), np.shape(plot_r0)))    
    pre_slabove = np.nan_to_num(np.reshape(slabove_df1['Diff'].to_numpy(), np.shape(plot_r0)))
    pre_above = np.nan_to_num(np.reshape(above_df1['Diff'].to_numpy(), np.shape(plot_r0)))

    # post10
    below_df2 = df2[df2['Vax Coverage'] == 'Below fc']
    slabove_df2 = df2[df2['Vax Coverage'] == 'Slightly Above fc']
    above_df2 = df2[df2['Vax Coverage'] == 'Above fc']

    post10_below = np.nan_to_num(np.reshape(below_df2['Diff'].to_numpy(), np.shape(plot_r0)))    
    post10_slabove = np.nan_to_num(np.reshape(slabove_df2['Diff'].to_numpy(), np.shape(plot_r0)))
    post10_above = np.nan_to_num(np.reshape(above_df2['Diff'].to_numpy(), np.shape(plot_r0)))

    # post30
    below_df3 = df3[df3['Vax Coverage'] == 'Below fc']
    slabove_df3 = df3[df3['Vax Coverage'] == 'Slightly Above fc']
    above_df3 = df3[df3['Vax Coverage'] == 'Above fc']

    post30_below = np.nan_to_num(np.reshape(below_df3['Diff'].to_numpy(), np.shape(plot_r0)))    
    post30_slabove = np.nan_to_num(np.reshape(slabove_df3['Diff'].to_numpy(), np.shape(plot_r0)))
    post30_above = np.nan_to_num(np.reshape(above_df3['Diff'].to_numpy(), np.shape(plot_r0)))

    if dim == 3:
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(20,20), gridspec_kw=dict(width_ratios=[1,1,1]), \
            subplot_kw={'projection': '3d'})
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission
        surf1 = axes[0,0].plot_surface(plot_r0, plot_eps, np.log(pre_below+1), rstride=1, norm=norm, cstride=1, cmap='viridis')
        axes[0,0].set_title('Pre | Below $f^*_V$')

        axes[0,1].plot_surface(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,1].set_title('Pre | Slightly Above $f^*_V$')

        axes[0,2].plot_surface(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,2].set_title('Pre | Above $f^*_V$')
        axes[0,2].set_zlabel('$log(P_A - P_L + 1)$')


        # 10 days post-tranmission
        axes[1,0].plot_surface(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,0].set_title('10 Days Post | Below $f^*_V$')

        axes[1,1].plot_surface(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,1].set_title('10 Days Post | Slightly Above $f^*_V$')

        axes[1,2].plot_surface(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,2].set_title('10 Days Post | Above $f^*_V$')
        axes[1,2].set_zlabel('$log(P_A - P_L + 1)$')

        # 30 days post-transmission
        axes[2,0].plot_surface(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,0].set_title('30 Days Post | Below $f^*_V$')

        axes[2,1].plot_surface(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,1].set_title('30 Days Post | Slightly Above $f^*_V$')

        axes[2,2].plot_surface(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,2].set_title('30 Days Post | Above $f^*_V$')
        axes[2,2].set_zlabel('$log(P_A - P_L + 1)$')

        axs = np.array(axes)
        for ax in axs.reshape(-1):
            ax.set_xlabel('$R_{0,V}$')
            ax.set_ylabel('Vaccine Efficacy')
            ax.view_init(elev=30, azim=120)

        cb = fig.colorbar(mappable=surf1, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), surf1.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
    
        return fig
    
    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(15,15), sharex=True, sharey=True, \
            gridspec_kw=dict(width_ratios=[1,1,1]))
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission 
        proj = axes[0,0].contourf(plot_r0, plot_eps, np.log(pre_below+1), norm=norm, cmap='viridis')
        axes[0,0].set_title('Pre | Below $f^*_V$')
        axes[0,0].set_ylabel('Vaccine Efficacy')

        axes[0,1].contourf(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, cmap='viridis')
        axes[0,1].set_title('Pre | Slightly Above $f^*_V$')

        #ax3 = fig.add_subplot(133)
        axes[0,2].contourf(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, cmap='viridis')
        axes[0,2].set_title('Pre | Above $f^*_V$')

        # 10 days post-transmission
        axes[1,0].contourf(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, cmap='viridis')
        axes[1,0].set_title('10 Days Post | Below $f^*_V$')
        axes[1,0].set_ylabel('Vaccine Efficacy')

        axes[1,1].contourf(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, cmap='viridis')
        axes[1,1].set_title('10 Days Post | Slightly Above $f^*_V$')

        axes[1,2].contourf(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, cmap='viridis')
        axes[1,2].set_title('10 Days Post | Above $f^*_V$')

        # 30 days post-transmission
        axes[2,0].contourf(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, cmap='viridis')
        axes[2,0].set_title('30 Days Post | Below $f^*_V$')
        axes[2,0].set_ylabel('Vaccine Efficacy')
        axes[2,0].set_xlabel('$R_{0,V}$')

        axes[2,1].contourf(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, cmap='viridis')
        axes[2,1].set_title('30 Days Post | Slightly Above $f^*_V$')
        axes[2,1].set_xlabel('$R_{0,V}$')

        axes[2,2].contourf(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, cmap='viridis')
        axes[2,2].set_title('30 Days Post | Above $f^*_V$')
        axes[2,2].set_xlabel('$R_{0,V}$')

        fig.tight_layout(pad=0.1)
        cb = fig.colorbar(mappable=proj, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), proj.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        
        return fig


def plot_scenarios_size(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, dim: int = 2):
    r0s = np.arange(1.0, 3.0, 0.01); epss = np.arange(0.01, 1.0, 0.01)
    plot_r0, plot_eps = np.nan_to_num(np.meshgrid(r0s, epss, indexing='ij'))

    # pre
    below_df1 = df1[df1['Vax Coverage'] == 'Below fc']
    slabove_df1 = df1[df1['Vax Coverage'] == 'Slightly Above fc']
    above_df1 = df1[df1['Vax Coverage'] == 'Above fc']

    #pre_below = np.nan_to_num(np.reshape(below_df1['Diff'].to_numpy(), np.shape(plot_r0)))  
    pre_below = np.nan_to_num(np.reshape(below_df1['Diff'].to_numpy(), np.shape(plot_r0))) 
    pre_slabove = np.nan_to_num(np.reshape(slabove_df1['Diff'].to_numpy(), np.shape(plot_r0)))
    pre_above = np.nan_to_num(np.reshape(above_df1['Diff'].to_numpy(), np.shape(plot_r0)))

    pre_below = np.ma.masked_where(pre_below == 99999, pre_below)
    pre_slabove = np.ma.masked_where(pre_slabove == 99999, pre_slabove)
    pre_above = np.ma.masked_where(pre_above == 99999, pre_above)

    # post10
    below_df2 = df2[df2['Vax Coverage'] == 'Below fc']
    slabove_df2 = df2[df2['Vax Coverage'] == 'Slightly Above fc']
    above_df2 = df2[df2['Vax Coverage'] == 'Above fc']

    post10_below = np.nan_to_num(np.reshape(below_df2['Diff'].to_numpy(), np.shape(plot_r0)))    
    post10_slabove = np.nan_to_num(np.reshape(slabove_df2['Diff'].to_numpy(), np.shape(plot_r0)))
    post10_above = np.nan_to_num(np.reshape(above_df2['Diff'].to_numpy(), np.shape(plot_r0)))

    post10_below = np.ma.masked_where(post10_below == 99999, post10_below)
    post10_slabove = np.ma.masked_where(post10_slabove == 99999, post10_slabove)
    post10_above = np.ma.masked_where(post10_above == 99999, post10_above)

    # post30
    below_df3 = df3[df3['Vax Coverage'] == 'Below fc']
    slabove_df3 = df3[df3['Vax Coverage'] == 'Slightly Above fc']
    above_df3 = df3[df3['Vax Coverage'] == 'Above fc']

    post30_below = np.nan_to_num(np.reshape(below_df3['Diff'].to_numpy(), np.shape(plot_r0)))    
    post30_slabove = np.nan_to_num(np.reshape(slabove_df3['Diff'].to_numpy(), np.shape(plot_r0)))
    post30_above = np.nan_to_num(np.reshape(above_df3['Diff'].to_numpy(), np.shape(plot_r0)))

    post30_below = np.ma.masked_where(post30_below == 99999, post30_below)
    post30_slabove = np.ma.masked_where(post30_slabove == 99999, post30_slabove)
    post30_above = np.ma.masked_where(post30_above == 99999, post30_above)

    if dim == 3:
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(20,20), gridspec_kw=dict(width_ratios=[1,1,1]), \
            subplot_kw={'projection': '3d'})
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission
        surf1 = axes[0,0].plot_surface(plot_r0, plot_eps, np.log(pre_below+1), rstride=1, norm=norm, cstride=1, cmap='viridis')
        axes[0,0].set_title('0% | Below $f^*_V$')

        axes[0,1].plot_surface(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,1].set_title('0% | Slightly Above $f^*_V$')

        axes[0,2].plot_surface(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,2].set_title('0% | Above $f^*_V$')
        axes[0,2].set_zlabel('$log(P_A - P_L + 1)$')


        # 10 days post-tranmission
        axes[1,0].plot_surface(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,0].set_title('10% | Below $f^*_V$')

        axes[1,1].plot_surface(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,1].set_title('10% | Slightly Above $f^*_V$')

        axes[1,2].plot_surface(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,2].set_title('10% | Above $f^*_V$')
        axes[1,2].set_zlabel('$log(P_A - P_L + 1)$')

        # 30 days post-transmission
        axes[2,0].plot_surface(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,0].set_title('25% | Below $f^*_V$')

        axes[2,1].plot_surface(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,1].set_title('25% | Slightly Above $f^*_V$')

        axes[2,2].plot_surface(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,2].set_title('25% | Above $f^*_V$')
        axes[2,2].set_zlabel('$log(P_A - P_L + 1)$')

        axs = np.array(axes)
        for ax in axs.reshape(-1):
            ax.set_xlabel('$R_{0,V}$')
            ax.set_ylabel('Vaccine Efficacy')
            ax.view_init(elev=30, azim=120)

        cb = fig.colorbar(mappable=surf1, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), surf1.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
    
        return fig
    
    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(15,15), sharex=True, sharey=True, \
            gridspec_kw=dict(width_ratios=[1,1,1]))
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission 
        proj = axes[0,0].contourf(plot_r0, plot_eps, np.log(pre_below+1), norm=norm, cmap='viridis')
        axes[0,0].set_title('0% | Below $f^*_V$')
        axes[0,0].set_ylabel('Vaccine Efficacy')

        axes[0,1].contourf(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, cmap='viridis')
        axes[0,1].set_title('0% | Slightly Above $f^*_V$')

        #ax3 = fig.add_subplot(133)
        axes[0,2].contourf(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, cmap='viridis')
        axes[0,2].set_title('0% | Above $f^*_V$')

        # 10 days post-transmission
        axes[1,0].contourf(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, cmap='viridis')
        axes[1,0].set_title('10% | Below $f^*_V$')
        axes[1,0].set_ylabel('Vaccine Efficacy')

        axes[1,1].contourf(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, cmap='viridis')
        axes[1,1].set_title('10%| Slightly Above $f^*_V$')

        axes[1,2].contourf(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, cmap='viridis')
        axes[1,2].set_title('10% | Above $f^*_V$')

        # 30 days post-transmission
        axes[2,0].contourf(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, cmap='viridis')
        axes[2,0].set_title('30%  | Below $f^*_V$')
        axes[2,0].set_ylabel('Vaccine Efficacy')
        axes[2,0].set_xlabel('$R_{0,V}$')

        axes[2,1].contourf(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, cmap='viridis')
        axes[2,1].set_title('30% | Slightly Above $f^*_V$')
        axes[2,1].set_xlabel('$R_{0,V}$')

        axes[2,2].contourf(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, cmap='viridis')
        axes[2,2].set_title('30% | Above $f^*_V$')
        axes[2,2].set_xlabel('$R_{0,V}$')

        fig.tight_layout(pad=0.1)
        cb = fig.colorbar(mappable=proj, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), proj.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        
        return fig


def plot_comparison(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, df4: pd.DataFrame, df5: pd.DataFrame, \
    df6: pd.DataFrame, eps: float):
    plot_r0 = np.arange(1.0, 3.0, 0.01); 
    belows = np.zeros((6, len(plot_r0)))
    slaboves = np.zeros((6, len(plot_r0)))
    aboves = np.zeros((6, len(plot_r0)))

    for i, df in enumerate([df1, df2, df3, df4, df5, df6]):
        df_eps = df[df['VE'] == eps]
        below_df_eps = df_eps[df_eps['Vax Coverage'] == 'Below fc']
        slabove_df_eps = df_eps[df_eps['Vax Coverage'] == 'Slightly Above fc']
        above_df_eps = df_eps[df_eps['Vax Coverage'] == 'Above fc']

        below = np.nan_to_num(np.reshape(below_df_eps['Diff'].to_numpy(), np.shape(plot_r0)))    
        slabove = np.nan_to_num(np.reshape(slabove_df_eps['Diff'].to_numpy(), np.shape(plot_r0)))
        above = np.nan_to_num(np.reshape(above_df_eps['Diff'].to_numpy(), np.shape(plot_r0)))

        belows[i] = below
        slaboves[i] = slabove
        aboves[i] = above

    fig, axes = plt.subplots(3,3, facecolor='w', figsize=(10,10), sharex=True, sharey=True, \
        gridspec_kw=dict(width_ratios=[1,1,1]))
        
    # Pre-transmission 
    axes[0,0].plot(plot_r0, belows[0], 'b--', label='Without Waning')
    axes[0,0].plot(plot_r0, belows[3], 'r:', label='With Waning')
    axes[0,0].set_title('Pre | Below $f^*_V$')
    axes[0,0].set_ylabel('Difference in % Reduction of $R$')
    legend = axes[0,0].legend(); legend.get_frame().set_alpha(0.5)

    axes[0,1].plot(plot_r0, slaboves[0], 'b--')
    axes[0,1].plot(plot_r0, slaboves[3], 'r:')
    axes[0,1].set_title('Pre | Slightly Above $f^*_V$')

    #ax3 = fig.add_subplot(133)
    axes[0,2].plot(plot_r0, aboves[0], 'b--')
    axes[0,2].plot(plot_r0, aboves[3], 'r:')
    axes[0,2].set_title('Pre | Above $f^*_V$')

    # 10 days post-transmission
    axes[1,0].plot(plot_r0, belows[1], 'b--')
    axes[1,0].plot(plot_r0, belows[4], 'r:')
    #axes[1,0].set_ylim([-0.2, 17.7])
    axes[1,0].set_title('10 Days Post | Below $f^*_V$')
    axes[1,0].set_ylabel('Difference in % Reduction of $R$')

    axes[1,1].plot(plot_r0, slaboves[1], 'b--')
    axes[1,1].plot(plot_r0, slaboves[4], 'r:')
    axes[1,1].set_title('10 Days Post | Slightly Above $f^*_V$')

    axes[1,2].plot(plot_r0, aboves[1], 'b--')
    axes[1,2].plot(plot_r0, aboves[4], 'r:')
    axes[1,2].set_title('10 Days Post | Above $f^*_V$')

    # 30 days post-transmission
    axes[2,0].plot(plot_r0, belows[2], 'b--')
    axes[2,0].plot(plot_r0, belows[5], 'r:')
    axes[2,0].set_title('30 Days Post | Below $f^*_V$')
    axes[2,0].set_ylabel('Difference in % Reduction of $R$')
    axes[2,0].set_xlabel('$R_{0,V}$')

    axes[2,1].plot(plot_r0, slaboves[2], 'b--')
    axes[2,1].plot(plot_r0, slaboves[5], 'r:')
    axes[2,1].set_title('30 Days Post | Slightly Above $f^*_V$')
    axes[2,1].set_xlabel('$R_{0,V}$')

    axes[2,2].plot(plot_r0, aboves[2], 'b--')
    axes[2,2].plot(plot_r0, aboves[5], 'r:')
    axes[2,2].set_title('30 Days Post | Above $f^*_V$')
    axes[2,2].set_xlabel('$R_{0,V}$')

    fig.tight_layout(pad=0.1)

    return fig