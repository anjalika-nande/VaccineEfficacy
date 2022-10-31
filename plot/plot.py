import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_timeseries(sim_novax, sim_leaky, sim_aon, figsize=(22, 10), savefig=False, filename: str = None):
    s, e, i, r = sim_novax
    s_leaky, vs_leaky, vr_leaky, v_leaky, e_leaky, i_leaky, r_leaky = sim_leaky
    s_aon, vs_aon, vr_aon, v_aon, e_aon, i_aon, r_aon = sim_aon
    t = np.linspace(0, len(s)-1, len(s))

    fig = plt.figure(facecolor='w', figsize=figsize)

    ax1 = fig.add_subplot(231, axisbelow=True)
    ax1.plot(t, s_leaky, 'y', alpha=0.5, lw=2, label='$S$')
    ax1.plot(t, v_leaky, 'tab:orange', alpha=0.5, lw=2, label='$V_{S} + V_{R}$')
    ax1.plot(t, vs_leaky, 'tab:orange', ls='--', alpha=0.5, lw=2, label='$V_{S}$')
    ax1.plot(t, vr_leaky, 'tab:orange', ls=':', alpha=0.5, lw=2, label='$V_{R}$')
    ax1.plot(t, i_leaky, 'b', alpha=0.5, lw=2, label='$I$')
    ax1.plot(t, r_leaky, 'r', alpha=0.5, lw=2, label='$R$')
    ax1.set_title("With Leaky Vaccine")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of Population")
    ax1.set_xlabel('Time (Days)')
    ax1.grid(linewidth=0.5)
    legend = ax1.legend(loc='upper right'); legend.get_frame().set_alpha(0.5)

    ax2 = fig.add_subplot(232, axisbelow=True)
    ax2.plot(t, s_aon, 'y', alpha=0.5, lw=2)
    ax2.plot(t, v_aon, 'tab:orange', alpha=0.5, lw=2)
    ax2.plot(t, vs_aon, 'tab:orange', ls='--', alpha=0.5, lw=2)
    ax2.plot(t, vr_aon, 'tab:orange', ls=':', alpha=0.5, lw=2)
    ax2.plot(t, i_aon, 'b', alpha=0.5, lw=2)
    ax2.plot(t, r_aon, 'r', alpha=0.5, lw=2)
    ax2.set_title("With All-or-None Vaccine")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of Population")
    ax2.set_xlabel('Time (Days)')
    ax2.grid(linewidth=0.5)
    
    ax3 = fig.add_subplot(233, axisbelow=True)
    ax3.plot(t, s, 'y', alpha=0.5, lw=2)
    ax3.plot(t, i, 'b', alpha=0.5, lw=2)
    ax3.plot(t, r, 'r', alpha=0.5, lw=2)
    ax3.set_title('Without Vaccine')
    ax3.set_ylabel('Fraction of Population')
    ax3.set_xlabel('Time (Days)')
    ax3.set_ylim(0, 1)
    ax3.grid(linewidth=0.5)
    
    ax4 = fig.add_subplot(234, axisbelow=True)
    ax4.plot(t, i_leaky, 'b--', alpha=0.5, lw=2, label='$I$ - Leaky')
    ax4.plot(t, i_aon, 'b:', alpha=0.5, lw=2, label= '$I$ - AON')
    ax4.plot(t, i, 'b', alpha=0.5, lw=2, label='$I$ - No Vax')
    ax4.grid(linewidth=0.5)
    ax4.set_ylabel('Fraction of Population')
    ax4.set_xlabel('Time (Days)')
    ax4.set_title('Comparing Infected Population')
    legend = ax4.legend(); legend.get_frame().set_alpha(0.5)

    ax5 = fig.add_subplot(235, axisbelow=True)
    ax5.plot(t, r_leaky, 'r--', alpha=0.5, lw=2, label='$R$ - Leaky')
    ax5.plot(t, r_aon, 'r:', alpha=0.5, lw=2, label='$R$ - AON')
    ax5.plot(t, r, 'r', alpha=0.5, lw=2, label='$R$ - No Vax')
    ax5.grid(linewidth=0.5)
    ax5.set_ylabel('Fraction of Population')
    ax5.set_xlabel('Time (Days)')
    ax5.set_title('Comparing Recovered Population')
    legend = ax5.legend(); legend.get_frame().set_alpha(0.5)

    if savefig:
        plt.savefig(filename, bbox_inches='tight')


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
        axes[0,0].set_title('Pre | $f_{V, Below}$')

        axes[0,1].plot_surface(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,1].set_title('Pre | $f_{V, Slightly Above}$')

        axes[0,2].plot_surface(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,2].set_title('Pre | $f_{V, Above}$')


        # 10 days post-tranmission
        axes[1,0].plot_surface(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,0].set_title('10 Days Post | $f_{V, Below}$')

        axes[1,1].plot_surface(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,1].set_title('10 Days Post | $f_{V, Slightly Above}$')

        axes[1,2].plot_surface(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,2].set_title('10 Days Post | $f_{V, Above}$')

        # 30 days post-transmission
        axes[2,0].plot_surface(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,0].set_title('30 Days Post | $f_{V, Below}$')

        axes[2,1].plot_surface(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,1].set_title('30 Days Post | $f_{V, Slightly Above}$')

        axes[2,2].plot_surface(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,2].set_title('30 Days Post | $f_{V, Above}$')

        axs = np.array(axes)
        for ax in axs.reshape(-1):
            ax.set_xlabel('$R_0$')
            ax.set_ylabel('Vaccine Efficacy')
            ax.view_init(elev=30, azim=240)
            ax.set_zlabel('$log(abs(P_A - P_L) + 1)$', rotation=180)

        cb = fig.colorbar(mappable=surf1, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), surf1.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        cb.set_label('Difference in Total Effectiveness (%)')
    
        return fig
    
    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(15,15), sharex=True, sharey=True, \
            gridspec_kw=dict(width_ratios=[1,1,1]))
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission 
        proj = axes[0,0].contourf(plot_r0, plot_eps, np.log(pre_below+1), norm=norm, cmap='viridis')
        axes[0,0].set_title('Pre | $f_{V, Below}$')
        axes[0,0].set_ylabel('Vaccine Efficacy')

        axes[0,1].contourf(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, cmap='viridis')
        axes[0,1].set_title('Pre | $f_{V, Slightly Above}$')

        #ax3 = fig.add_subplot(133)
        axes[0,2].contourf(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, cmap='viridis')
        axes[0,2].set_title('Pre | $f_{V, Above}$')

        # 10 days post-transmission
        axes[1,0].contourf(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, cmap='viridis')
        axes[1,0].set_title('10 Days Post | $f_{V, Below}$')
        axes[1,0].set_ylabel('Vaccine Efficacy')

        axes[1,1].contourf(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, cmap='viridis')
        axes[1,1].set_title('10 Days Post | $f_{V, Slightly Above}$')

        axes[1,2].contourf(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, cmap='viridis')
        axes[1,2].set_title('10 Days Post | $f_{V, Above}$')

        # 30 days post-transmission
        axes[2,0].contourf(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, cmap='viridis')
        axes[2,0].set_title('30 Days Post | $f_{V, Below}$')
        axes[2,0].set_ylabel('Vaccine Efficacy')
        axes[2,0].set_xlabel('$R_0$')

        axes[2,1].contourf(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, cmap='viridis')
        axes[2,1].set_title('30 Days Post | $f_{V, Slightly Above}$')
        axes[2,1].set_xlabel('$R_{0,V}$')

        axes[2,2].contourf(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, cmap='viridis')
        axes[2,2].set_title('30 Days Post | $f_{V, Above}$')
        axes[2,2].set_xlabel('$R_0$')

        fig.tight_layout(pad=0.1)
        cb = fig.colorbar(mappable=proj, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), proj.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        cb.set_label('Difference in Total Effectiveness (%)')
        
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
        axes[0,0].set_title('0% | $f_{V, Below}$')

        axes[0,1].plot_surface(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,1].set_title('0% | $f_{V, Slightly Above}$')

        axes[0,2].plot_surface(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[0,2].set_title('0% | $f_{V, Above}$')


        # 10 days post-tranmission
        axes[1,0].plot_surface(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,0].set_title('10% | $f_{V, Below}$')

        axes[1,1].plot_surface(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,1].set_title('10% | $f_{V, Slightly Above}$')

        axes[1,2].plot_surface(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[1,2].set_title('10% | $f_{V, Above}$')

        # 30 days post-transmission
        axes[2,0].plot_surface(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,0].set_title('25% | $f_{V, Below}$')

        axes[2,1].plot_surface(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,1].set_title('25% | $f_{V, Slightly Above}$')

        axes[2,2].plot_surface(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, rstride=1, cstride=1, cmap='viridis')
        axes[2,2].set_title('25% | $f_{V, Above}$')

        axs = np.array(axes)
        for ax in axs.reshape(-1):
            ax.set_xlabel('$R_0$')
            ax.set_ylabel('Vaccine Efficacy')
            ax.view_init(elev=30, azim=240)
            ax.set_zlabel('$log(P_A - P_L + 1)$', rotation=180)

        cb = fig.colorbar(mappable=surf1, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), surf1.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        cb.set_label('Difference in Total Effectiveness (%)')
    
        return fig
    
    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(3,3, facecolor='w', figsize=(15,15), sharex=True, sharey=True, \
            gridspec_kw=dict(width_ratios=[1,1,1]))
        norm = plt.Normalize(np.min(np.log(pre_below+1)), np.max(np.log(pre_below+1)))

        # Pre-transmission 
        proj = axes[0,0].contourf(plot_r0, plot_eps, np.log(pre_below+1), norm=norm, cmap='viridis')
        axes[0,0].set_title('0% | $f_{V, Below}$')
        axes[0,0].set_ylabel('Vaccine Efficacy')

        axes[0,1].contourf(plot_r0, plot_eps, np.log(pre_slabove+1), norm=norm, cmap='viridis')
        axes[0,1].set_title('0% | $f_{V, Slightly Above}$')

        #ax3 = fig.add_subplot(133)
        axes[0,2].contourf(plot_r0, plot_eps, np.log(pre_above+1), norm=norm, cmap='viridis')
        axes[0,2].set_title('0% | $f_{V, Above}$')

        # 10 days post-transmission
        axes[1,0].contourf(plot_r0, plot_eps, np.log(post10_below+1), norm=norm, cmap='viridis')
        axes[1,0].set_title('10% | $f_{V, Below}$')
        axes[1,0].set_ylabel('Vaccine Efficacy')

        axes[1,1].contourf(plot_r0, plot_eps, np.log(post10_slabove+1), norm=norm, cmap='viridis')
        axes[1,1].set_title('10% | $f_{V, Slightly Above}$')

        axes[1,2].contourf(plot_r0, plot_eps, np.log(post10_above+1), norm=norm, cmap='viridis')
        axes[1,2].set_title('10% | $f_{V, Above}$')

        # 30 days post-transmission
        axes[2,0].contourf(plot_r0, plot_eps, np.log(post30_below+1), norm=norm, cmap='viridis')
        axes[2,0].set_title('25%  | $f_{V, Below}$')
        axes[2,0].set_ylabel('Vaccine Efficacy')
        axes[2,0].set_xlabel('$R_0$')

        axes[2,1].contourf(plot_r0, plot_eps, np.log(post30_slabove+1), norm=norm, cmap='viridis')
        axes[2,1].set_title('25% | $f_{V, Slightly Above}$')
        axes[2,1].set_xlabel('$R_0$')

        axes[2,2].contourf(plot_r0, plot_eps, np.log(post30_above+1), norm=norm, cmap='viridis')
        axes[2,2].set_title('25% | $f_{V, Above}$')
        axes[2,2].set_xlabel('$R_0$')

        fig.tight_layout(pad=0.1)
        cb = fig.colorbar(mappable=proj, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), proj.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels)-1).astype(int))
        cb.set_label('Difference in Total Effectiveness (%)')
        
        return fig


def plot_scenarios_Spop(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    R0s: np.ndarray,
    epss: np.ndarray,
    dim: int = 2,
    mask_hit: bool = False,
):

    covs = np.unique(df1["Vax Coverage"].to_numpy())
    plot_R0, plot_eps = np.nan_to_num(np.meshgrid(R0s, epss, indexing="ij"))

    # construct arrays of difference in percentage reduction for each coverage level
    # construct arrays of largest vaccine efficacy for each R0 for which fv <= fv*
    n = len(epss)
    hit_eps = np.zeros((9, len(R0s)))
    hit_diff = np.zeros((9, len(R0s)))
    plot_dfs = []
    i = 0  # counter
    for df in [df1, df2, df3]:
        for cov in covs:
            temp_cov_df = df[df["Vax Coverage"] == cov]
            plot_df = np.reshape(temp_cov_df["Diff"].to_numpy(), np.shape(plot_R0))
            plot_df = np.ma.masked_where(
                plot_df == 99999, plot_df
            )  # mask where fv* < 0 or R never reaches target size
            if mask_hit:
                hit_df = np.reshape(temp_cov_df["Above HIT"].to_numpy(), np.shape(plot_R0))
                plot_df = np.ma.masked_where(hit_df, plot_df)  # mask where fv > fv*
            plot_dfs.append(plot_df)

            for j in range(len(R0s)):
                temp_df = temp_cov_df.iloc[n * j : n * (j + 1), :]
                temp_df = temp_df[temp_df.fv != 99999]
                if not temp_df.empty:
                    true_df = temp_df[temp_df["Above HIT"] == True]
                    false_df = temp_df[temp_df["Above HIT"] == False]
                    if not false_df.empty:
                        if (
                            not true_df.empty
                        ):  # above vs. below fv* divided within eps [0,1]
                            hit_eps_val = false_df.iloc[-1, 1]
                        else:  # fv always below fv*
                            hit_eps_val = 1
                        hit_diff_val = false_df.iloc[-1, -1]

                        hit_eps[i, j] = hit_eps_val
                        hit_diff[i, j] = np.log(hit_diff_val + 1)
            i += 1

    cov1_df1 = plot_dfs[0]
    min_diff = 9999999
    max_diff = -9999999
    for df in plot_dfs:
        temp_min = np.min(df)
        if temp_min < min_diff:
            min_diff = temp_min

        temp_max = np.max(df)
        if temp_max > max_diff:
            max_diff = temp_max

    covs_str = ["50%", "75%", "100%"] * 3
    tvs_str = ["0"] * 3 + ["0.10"] * 3 + ["0.25"] * 3

    if dim == 3:
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(16,16),
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
            subplot_kw={"projection": "3d"},
        )
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot_surface(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),
                norm=norm,
                rstride=1,
                cstride=1,
                cmap="viridis",
            )
            ax.plot(R0s, hit_eps[i, :], hit_diff[i, :], "r", linewidth=2)
            ax.set_title(
                "$t_V$: R = {tv} | $f_V$: {cov} of (1- $f_R$)".format(
                    tv=tvs_str[i], cov=covs_str[i]
                )
            )
            ax.set_xlabel("$R_0$")
            ax.set_ylabel("Vaccine Efficacy")
            ax.view_init(elev=30, azim=240)
            ax.set_zlabel("$log(P_A - P_L + 1)$", rotation=180)

        im = cm.ScalarMappable(norm=norm)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig

    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(15,15),
            sharex=True,
            sharey=True,
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
        )
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            im = ax.contourf(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),
                norm=norm,
                cmap="viridis",
            )
            ax.plot(R0s, hit_eps[i, :], "r", linewidth=2)
            ax.set_title(
                "$t_V$: R = {tv} | $f_V$: {cov} of (1- $f_R$)".format(
                    tv=tvs_str[i], cov=covs_str[i]
                )
            )
            ax.set_ylim([0, 1])
            ax.set_xlim([1, 3])

            if i % 3 == 0:
                ax.set_ylabel("Vaccine Efficacy")
            if i >= 6:
                ax.set_xlabel("$R_0$")

        im = cm.ScalarMappable(norm=norm)
        fig.tight_layout(pad=1.0)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig


def plot_scenarios_below_remS(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    R0s: np.ndarray,
    epss: np.ndarray,
    dim: int = 2,
    mask_hit: bool = False,
):

    covs = np.unique(df1["Vax Coverage"].to_numpy())
    plot_R0, plot_eps = np.nan_to_num(np.meshgrid(R0s, epss, indexing="ij"))

    # construct arrays of difference in percentage reduction for each coverage level
    # construct arrays of largest vaccine efficacy for each R0 for which fv <= fv*
    n = len(epss)
    hit_eps = np.zeros((6, len(R0s)))
    hit_diff = np.zeros((6, len(R0s)))
    plot_dfs = []
    i = 0  # counter
    for df in [df1, df2, df3]:
        for cov in covs:
            temp_cov_df = df[df["Vax Coverage"] == cov]
            plot_df = np.reshape(temp_cov_df["Diff"].to_numpy(), np.shape(plot_R0))
            plot_df = np.ma.masked_where(
                plot_df == 99999, plot_df
            )  # mask where fv* < 0 or R never reaches target size
            if mask_hit:
                hit_df = np.reshape(
                    temp_cov_df["Above HIT"].to_numpy(), np.shape(plot_R0)
                )
                plot_df = np.ma.masked_where(hit_df, plot_df)  # mask where fv > fv*
            plot_dfs.append(plot_df)

            if cov == "Remaining S":
                for j in range(len(R0s)):
                    temp_df = temp_cov_df.iloc[n * j : n * (j + 1), :]
                    temp_df = temp_df[temp_df.fv != 99999]
                    if not temp_df.empty:
                        true_df = temp_df[temp_df["Above HIT"] == True]
                        false_df = temp_df[temp_df["Above HIT"] == False]
                        if not false_df.empty:
                            if (
                                not true_df.empty
                            ):  # above vs. below fv* divided within eps [0,1]
                                hit_eps_val = false_df.iloc[-1, 1]
                            else:  # fv always below fv*
                                hit_eps_val = 1
                            hit_diff_val = false_df.iloc[-1, -1]

                            hit_eps[i, j] = hit_eps_val
                            hit_diff[i, j] = np.log(hit_diff_val + 1)
            i += 1

    min_diff = 9999999
    max_diff = -9999999
    for df in plot_dfs:
        temp_min = np.min(df)
        if temp_min < min_diff:
            min_diff = temp_min

        temp_max = np.max(df)
        if temp_max > max_diff:
            max_diff = temp_max

    covs_str = ["Below $f_V^*$", "$S(t_V)$"] * 3
    tvs_str = ["0"] * 2 + ["0.10"] * 2 + ["0.25"] * 2

    if dim == 3:
        fig, axes = plt.subplots(
            3,
            2,
            facecolor="w",
            figsize=(12, 16),
            gridspec_kw=dict(width_ratios=[1, 1]),
            subplot_kw={"projection": "3d"},
        )
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot_surface(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),
                norm=norm,
                rstride=1,
                cstride=1,
                cmap="viridis",
            )
            if i % 2 == 1:
                ax.plot(R0s, hit_eps[i, :], hit_diff[i, :], "r", linewidth=2)
            ax.set_title(
                "$t_V$: R = {tv} | $f_V$: {cov}".format(tv=tvs_str[i], cov=covs_str[i])
            )
            ax.set_xlabel("$R_0$")
            ax.set_ylabel("Vaccine Efficacy")
            ax.view_init(elev=30, azim=240)
            ax.set_zlabel("$log(P_A - P_L + 1)$", rotation=180)

        im = cm.ScalarMappable(norm=norm)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig

    elif dim == 2:
        # set color-scale
        fig, axes = plt.subplots(
            3,
            2,
            facecolor="w",
            figsize=(10, 15),
            sharex=True,
            sharey=True,
            gridspec_kw=dict(width_ratios=[1, 1]),
        )
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            im = ax.contourf(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),
                norm=norm,
                cmap="viridis",
            )
            if i % 2 == 1:
                ax.plot(R0s, hit_eps[i, :], "r", linewidth=2)
            ax.set_title(
                "$t_V$: R = {tv} | $f_V$: {cov}".format(tv=tvs_str[i], cov=covs_str[i])
            )

            ax.set_ylim([0, 1])
            ax.set_xlim([1, 3])
            ax.set_box_aspect(1)

            if i % 2 == 0:
                ax.set_ylabel("Vaccine Efficacy")
            if i >= 4:
                ax.set_xlabel("$R_0$")

        im = cm.ScalarMappable(norm=norm)
        fig.tight_layout(pad=3.0)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

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
