import numpy as np
from scipy.integrate import odeint
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


def _delta(t, tv):
    if (t > tv - 0.8) and (t < tv + 0.8):
        return 1
    else:
        return 0


def modified_seir(y, t, tv, beta, sigma, gamma, fv, epsL, epsA):
    """
    Modified SEIR model for instantaneous vaccination. 
    """
    s, v_s, v_r, e, i, r = y
    v = fv * _delta(t, tv)
    
    dsdt = - beta*s*i - v*s
    dvsdt = (1-epsA)*v*s - beta*v_s*i
    dvrdt = epsA*v*s - beta*(1-epsL)*v_r*i
    dedt = beta*(s+v_s+(1-epsL)*v_r)*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return dsdt, dvsdt, dvrdt, dedt, didt, drdt


def modified_seir_waning(y, t, tv, beta, sigma, gamma, fv, epsL, epsA, w):
    """
    Modified SEIR model for instantaneous vaccination. 
    """
    s, v_s, v_r, e, i, r = y
    v = fv * _delta(t, tv)
    
    dsdt = - beta*s*i - v*s
    dvsdt = (1-epsA)*v*s - beta*v_s*i + w*v_r
    dvrdt = epsA*v*s - beta*(1-epsL)*v_r*i - w*v_r
    dedt = beta*(s+v_s+(1-epsL)*v_r)*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return dsdt, dvsdt, dvrdt, dedt, didt, drdt


def run_modified_seir(y0: list, t: np.ndarray, tv: int, beta: float, sigma: float, gamma: float, fv: float, \
    eps: float, mode: str = 'leaky'):
    if mode == 'leaky':
        epsL = eps; epsA = 1
    elif mode == 'aon':
        epsL = 1; epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    sim = odeint(modified_seir, y0, t, args=(tv, beta, sigma, gamma, fv, epsL, epsA))
    s, v_s, v_r, e, i, r = sim.T
    v = v_s + v_r

    return s, v_s, v_r, v, e, i, r


def run_modified_seir_waning(y0: list, t: np.ndarray, tv: int, beta: float, sigma: float, gamma: float, fv: float, \
    eps: float, w: float, mode: str = 'leaky'):
    if mode == 'leaky':
        epsL = eps; epsA = 1
    elif mode == 'aon':
        epsL = 1; epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    sim = odeint(modified_seir_waning, y0, t, args=(tv, beta, sigma, gamma, fv, epsL, epsA, w))
    s, v_s, v_r, e, i, r = sim.T
    v = v_s + v_r

    return s, v_s, v_r, v, e, i, r


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


def run_scenarios(y0: list, t: np.ndarray, r0s: np.ndarray, sigma: float, gamma: float, \
    epss: np.ndarray, scenario: str = 'pre'):  
    s0, e0, i0, r0 = y0
    df_R0s = []; df_epss = []; df_fvs = []; df_sig = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []

    for r0 in r0s:
        beta = r0 * gamma
        sim = odeint(seir, y0, t, args=(beta, sigma, gamma))
        _, _, _, r = sim.T
        
        r_tot = r[-1]
                
        for eps in epss:
            fc = 1/eps * (1 - 1/r0)
            for cov in covs:
                if cov == 'Below fc':
                    fv = fc * 0.8
                elif cov == 'Slightly Above fc':
                    fv = 1 - ((1 - fc) * 0.8)
                else:
                    fv = 1 - ((1 - fc) * 0.5)

                s0_vax = 0.98-fv 

                if scenario == 'pre':
                    tv = -1
                
                    # leaky
                    vs0_leaky = 0; vr0_leaky = fv; y0_leaky = [s0_vax, vs0_leaky, vr0_leaky, e0, i0, r0]
                    sim_leaky = run_modified_seir(y0_leaky, t, tv, beta, sigma, gamma, fv, eps, mode = 'leaky')

                    # aon
                    vs0_aon = fv*(1-eps); vr0_aon = fv*eps; y0_aon = [s0_vax, vs0_aon, vr0_aon, e0, i0, r0]
                    sim_aon = run_modified_seir(y0_aon, t, tv, beta, sigma, gamma, fv, eps, mode = 'aon')   

                elif scenario == 'post10':
                    tv = 10; vs0 = 0; vr0 = 0
                    y0_vax = [s0, vs0, vr0, e0, i0, r0]

                    # leaky
                    sim_leaky = run_modified_seir(y0_vax, t, tv, beta, sigma, gamma, fv, eps, mode = 'leaky')

                    # aon
                    sim_aon = run_modified_seir(y0_vax, t, tv, beta, sigma, gamma, fv, eps, mode = 'aon')
                
                elif scenario == 'post30':
                    tv = 30; vs0 = 0; vr0 = 0
                    y0_vax = [s0, vs0, vr0, e0, i0, r0]

                    # leaky
                    sim_leaky = run_modified_seir(y0_vax, t, tv, beta, sigma, gamma, fv, eps, mode = 'leaky')

                    # aon
                    sim_aon = run_modified_seir(y0_vax, t, tv, beta, sigma, gamma, fv, eps, mode = 'aon')
                
                _, _, _, _, _, _, r_leaky = sim_leaky
                r_tot_leaky = r_leaky[-1]
                r_perc_leaky = r_tot - r_tot_leaky / r_tot * 100

                _, _, _, _, _, _, r_aon = sim_aon
                r_tot_aon = r_aon[-1]
                r_perc_aon = r_tot - r_tot_aon / r_tot * 100

                r_perc_diff = r_perc_aon - r_perc_leaky

                if r_perc_diff >= 2:
                    df_sig.append('Yes')
                else:
                    df_sig.append('No')

                df_R0s.append(r0)
                df_epss.append(eps)
                df_fvs.append(fv)
                df_covs.append(cov)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe                        
    data = {'R0': df_R0s, 'VE': df_epss, 'Vax Coverage': df_covs, 'fv': df_fvs, \
        'Leaky': df_r_perc_leakys, 'AON': df_r_perc_aons, 'Diff': df_r_perc_diffs, 'Significant': df_sig}
    vax_df = pd.DataFrame(data=data)

    return vax_df


def run_scenarios_waning(y0: list, t: np.ndarray, r0s: np.ndarray, sigma: float, gamma: float, \
    epss: np.ndarray, w: float, scenario: str = 'pre'):  
    s0, e0, i0, r0 = y0
    df_R0s = []; df_epss = []; df_fvs = []; df_sig = []
    covs = ['Below fc', 'Slightly Above fc', 'Above fc']; df_covs = []
    df_r_perc_leakys = []; df_r_perc_aons = []; df_r_perc_diffs = []

    for r0 in r0s:
        beta = r0 * gamma
        sim = odeint(seir, y0, t, args=(beta, sigma, gamma))
        _, _, _, r = sim.T
        
        r_tot = r[-1]
                
        for eps in epss:
            fc = 1/eps * (1 - 1/r0)
            for cov in covs:
                if cov == 'Below fc':
                    fv = fc * 0.8
                elif cov == 'Slightly Above fc':
                    fv = 1 - ((1 - fc) * 0.8)
                else:
                    fv = 1 - ((1 - fc) * 0.5)

                s0_vax = 0.98-fv 

                if scenario == 'pre':
                    tv = -1
                
                    # leaky
                    vs0_leaky = 0; vr0_leaky = fv; y0_leaky = [s0_vax, vs0_leaky, vr0_leaky, e0, i0, r0]
                    sim_leaky = run_modified_seir_waning(y0_leaky, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'leaky')

                    # aon
                    vs0_aon = fv*(1-eps); vr0_aon = fv*eps; y0_aon = [s0_vax, vs0_aon, vr0_aon, e0, i0, r0]
                    sim_aon = run_modified_seir_waning(y0_aon, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'aon')   

                elif scenario == 'post10':
                    tv = 10; vs0 = 0; vr0 = 0
                    y0_vax = [s0, vs0, vr0, e0, i0, r0]

                    # leaky
                    sim_leaky = run_modified_seir_waning(y0_vax, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'leaky')

                    # aon
                    sim_aon = run_modified_seir_waning(y0_vax, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'aon')
                
                elif scenario == 'post30':
                    tv = 30; vs0 = 0; vr0 = 0
                    y0_vax = [s0, vs0, vr0, e0, i0, r0]

                    # leaky
                    sim_leaky = run_modified_seir_waning(y0_vax, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'leaky')

                    # aon
                    sim_aon = run_modified_seir_waning(y0_vax, t, tv, beta, sigma, gamma, fv, eps, w, mode = 'aon')
                
                _, _, _, _, _, _, r_leaky = sim_leaky
                r_tot_leaky = r_leaky[-1]
                r_perc_leaky = r_tot - r_tot_leaky / r_tot * 100

                _, _, _, _, _, _, r_aon = sim_aon
                r_tot_aon = r_aon[-1]
                r_perc_aon = r_tot - r_tot_aon / r_tot * 100

                r_perc_diff = r_perc_aon - r_perc_leaky

                if r_perc_diff >= 2:
                    df_sig.append('Yes')
                else:
                    df_sig.append('No')

                df_R0s.append(r0)
                df_epss.append(eps)
                df_fvs.append(fv)
                df_covs.append(cov)
                df_r_perc_leakys.append(r_perc_leaky)
                df_r_perc_aons.append(r_perc_aon)
                df_r_perc_diffs.append(r_perc_diff)

    # build dataframe                        
    data = {'R0': df_R0s, 'VE': df_epss, 'Vax Coverage': df_covs, 'fv': df_fvs, \
        'Leaky': df_r_perc_leakys, 'AON': df_r_perc_aons, 'Diff': df_r_perc_diffs, 'Significant': df_sig}
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

