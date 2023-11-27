import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_timeseries(sim_novax, sim_leaky, sim_aon, figsize=(22, 10)):
    """
    Plot time-series plots for S, V_S, V_R, V, I, R, comparing models with a 
    leaky vaccine, an all-or-nothing vaccine, and no vaccination. The fourth
    subplot compares only the infected population, and the fifth subplot
    compares only the recoverd population.

    Parameters
    ----------
    sim_novax: np.ndarrays
        Output from using ``scipy.integrate.solve_ivp`` to solve ``utils.seir``.
    sim_leaky: np.ndarrays
        Output from ``utils.run_modified_seir`` with ``mode='leaky'``.
    sim_aon: np.ndarrays
        Output from ``utils.run_modified_seir`` with ``mode='aon'``.
    figsize: tuple (Default: (22, 10))
        Size of figure.
    
    Returns
    -------
    fig: matplotlib.pyplot.figure
    """
    # parse inputs
    s, _, i, r = sim_novax
    s_leaky, vs_leaky, vr_leaky, v_leaky, _, i_leaky, r_leaky, _, _ = sim_leaky
    s_aon, vs_aon, vr_aon, v_aon, _, i_aon, r_aon, _, _ = sim_aon

    # get t
    t = np.linspace(0, len(s) - 1, len(s))

    # initalize figure
    fig = plt.figure(facecolor="w", figsize=figsize)

    # with vaccination - leaky
    ax1 = fig.add_subplot(231, axisbelow=True)
    ax1.plot(t, s_leaky, "y", alpha=0.5, lw=2, label="$S$")
    ax1.plot(t, v_leaky, "tab:orange", alpha=0.5, lw=2, label="$V_{S} + V_{R}$")
    ax1.plot(t, vs_leaky, "tab:orange", ls="--", alpha=0.5, lw=2, label="$V_{S}$")
    ax1.plot(t, vr_leaky, "tab:orange", ls=":", alpha=0.5, lw=2, label="$V_{R}$")
    ax1.plot(t, i_leaky, "b", alpha=0.5, lw=2, label="$I$")
    ax1.plot(t, r_leaky, "r", alpha=0.5, lw=2, label="$R$")
    ax1.set_title("With Leaky Vaccine")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of Population")
    ax1.set_xlabel("Time (Days)")
    ax1.grid(linewidth=0.5)
    legend = ax1.legend(loc="upper right")
    legend.get_frame().set_alpha(0.5)

    # with vaccination - all-or-nothing
    ax2 = fig.add_subplot(232, axisbelow=True)
    ax2.plot(t, s_aon, "y", alpha=0.5, lw=2)
    ax2.plot(t, v_aon, "tab:orange", alpha=0.5, lw=2)
    ax2.plot(t, vs_aon, "tab:orange", ls="--", alpha=0.5, lw=2)
    ax2.plot(t, vr_aon, "tab:orange", ls=":", alpha=0.5, lw=2)
    ax2.plot(t, i_aon, "b", alpha=0.5, lw=2)
    ax2.plot(t, r_aon, "r", alpha=0.5, lw=2)
    ax2.set_title("With All-or-None Vaccine")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction of Population")
    ax2.set_xlabel("Time (Days)")
    ax2.grid(linewidth=0.5)

    # without vaccination
    ax3 = fig.add_subplot(233, axisbelow=True)
    ax3.plot(t, s, "y", alpha=0.5, lw=2)
    ax3.plot(t, i, "b", alpha=0.5, lw=2)
    ax3.plot(t, r, "r", alpha=0.5, lw=2)
    ax3.set_title("Without Vaccine")
    ax3.set_ylabel("Fraction of Population")
    ax3.set_xlabel("Time (Days)")
    ax3.set_ylim(0, 1)
    ax3.grid(linewidth=0.5)

    # comparing infected population
    ax4 = fig.add_subplot(234, axisbelow=True)
    ax4.plot(t, i_leaky, "b--", alpha=0.5, lw=2, label="$I$ - Leaky")
    ax4.plot(t, i_aon, "b:", alpha=0.5, lw=2, label="$I$ - AON")
    ax4.plot(t, i, "b", alpha=0.5, lw=2, label="$I$ - No Vax")
    ax4.grid(linewidth=0.5)
    ax4.set_ylabel("Fraction of Population")
    ax4.set_xlabel("Time (Days)")
    ax4.set_title("Comparing Infected Population")
    legend = ax4.legend()
    legend.get_frame().set_alpha(0.5)

    # comparing recovered population
    ax5 = fig.add_subplot(235, axisbelow=True)
    ax5.plot(t, r_leaky, "r--", alpha=0.5, lw=2, label="$R$ - Leaky")
    ax5.plot(t, r_aon, "r:", alpha=0.5, lw=2, label="$R$ - AON")
    ax5.plot(t, r, "r", alpha=0.5, lw=2, label="$R$ - No Vax")
    ax5.grid(linewidth=0.5)
    ax5.set_ylabel("Fraction of Population")
    ax5.set_xlabel("Time (Days)")
    ax5.set_title("Comparing Recovered Population")
    ax5.set_ylim(0, 1)
    legend = ax5.legend()
    legend.get_frame().set_alpha(0.5)

    return fig


def plot_timeseries_cat(sim_novax, sim_leaky, sim_aon, figsize=(15, 6)):
    """
    Plot time-series plots for I and R, comparing models with a leaky vaccine,
    an all-or-nothing vaccine, and no vaccination. The first subplot shows 
    the infected population, and the second subplot shows the recovered
    population as well as cumulative infections from V and S.

    Parameters
    ----------
    sim_novax: np.ndarrays
        Output from using ``scipy.integrate.solve_ivp`` to solve ``utils.seir``.
    sim_leaky: np.ndarrays
        Output from ``utils.run_modified_seir`` with ``mode='leaky'``.
    sim_aon: np.ndarrays
        Output from ``utils.run_modified_seir`` with ``mode='aon'``.
    figsize: tuple (Default: (15, 6))
        Size of figure.
    
    Returns
    -------
    fig: matplotlib.pyplot.figure
    """
    # parse inputs
    _, _, i, r = sim_novax
    _, _, _, _, _, i_leaky, r_leaky, cv_leaky, cu_leaky = sim_leaky
    _, _, _, _, _, i_aon, r_aon, cv_aon, cu_aon = sim_aon

    # get t
    t = np.linspace(0, len(i) - 1, len(i))

    # intialize figure
    fig = plt.figure(facecolor="w", figsize=figsize)

    # comparing infected population
    ax1 = fig.add_subplot(121, axisbelow=True)
    ax1.plot(t, i, "r", alpha=0.5, lw=2, label="$I$ - No Vax")
    ax1.plot(t, i_leaky, "b", alpha=0.5, lw=2, label="$I$ - Leaky")
    ax1.plot(t, i_aon, "g", alpha=0.5, lw=2, label="$I$ - AON")
    ax1.set_title("Comparing Infected Population")
    ax1.set_ylabel("Fraction of Population")
    ax1.set_xlabel("Time (Days)")
    ax1.grid(linewidth=0.5)
    legend = ax1.legend()
    legend.get_frame().set_alpha(0.5)

    # comparing recovered population and cumulative infections from V, S
    ax2 = fig.add_subplot(122, axisbelow=True)
    ax2.plot(t, r, "r", alpha=0.5, lw=2, label="$R$ - No Vax")
    ax2.plot(t, r_leaky, "b", alpha=0.5, lw=2, label="$R$ - Leaky")
    ax2.plot(
        t, cv_leaky, "b--", alpha=0.5, lw=2, label="Cumulative $I$ from $V$ - Leaky"
    )
    ax2.plot(
        t, cu_leaky, "b:", alpha=0.5, lw=2, label="Cumulative $I$ from $S$ - Leaky"
    )
    ax2.plot(t, r_aon, "g", alpha=0.5, lw=2, label="$R$ - AON")
    ax2.plot(t, cv_aon, "g--", alpha=0.5, lw=2, label="Cumulative $I$ from $V$ - AON")
    ax2.plot(t, cu_aon, "g:", alpha=0.5, lw=2, label="Cumulative $I$ from $S$ - AON")
    ax2.set_title("Comparing Recovered Population and Cumulative Infected Population")
    ax2.set_ylabel("Fraction of Population")
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylim(0, 1)
    ax2.grid(linewidth=0.5)
    legend = ax2.legend()
    legend.get_frame().set_alpha(0.5)

    return fig


def plot_scenarios(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    R0s: np.ndarray,
    epss: np.ndarray,
    dim: int = 2,
    mask_hit: bool = False,
):
    """
    Plot difference in percentage reduction of total recovered population between
    leaky and all-or-nothing vaccines compared to that of without vaccination for
    all scenarios (defined in Scenarios/1_Scenarios.ipynb). The plots are 
    log-scaled while the colorbars show the actual values before transformation.

    Parameters
    ----------
    df1: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0``.
    df2: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.1``.
    df3: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.25``.
    R0s: np.ndarray
        Array of varying R0 values (ex. np.linspace(1.0, 3.0, 601))
    epss: np.ndarray
        Array of varying vaccine efficacy values. (ex. np.linspace(0.01, 1, 201)) 
        A vaccine that is 50% effective would have an epsilon value of 0.5 For a 
        leaky vaccine, (epsL, epsA) = (eps, 1), and for an all-or-nothing vaccine,
        (epsL, epsA) = (1, eps).
    dim: int (Default: 2)
        ``dim=2`` plots a 2D contour plot using ``matplotlib.axes.Axes.contourf``
        and ``dim=3`` plots a 3D surface plot using Matplotlib's ``plot_surface``
        method.
    mask_hit: boolean (Default: False)
        If ``mask_hit=True``, only scenarios where fv < fv* are shown. If 
        ``mask_hit=False``, all scenarios are shown. 
    
    Returns
    -------
    fig: matplotlib.pyplot.figure
    """
    # parse inputs
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
                hit_df = np.reshape(
                    temp_cov_df["Above HIT"].to_numpy(), np.shape(plot_R0)
                )
                plot_df = np.ma.masked_where(hit_df, plot_df)  # mask where fv > fv*
            plot_dfs.append(plot_df)

            # determine where fv >= fv* and fv < fv*
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

    # find min and max values to normalize colormap
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

    # plot 3D surface plot
    if dim == 3:
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(16, 16),
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
            subplot_kw={"projection": "3d"},
        )
        # set colormap
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot_surface(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),  # log-transform
                norm=norm,
                rstride=1,
                cstride=1,
                cmap="viridis",
            )
            # plot red line for max fv while fv < fv*
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

        # add colorbar for actual values
        im = cm.ScalarMappable(norm=norm)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig

    # plot 2d contour plot
    elif dim == 2:
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(15, 15),
            sharex=True,
            sharey=True,
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
        )
        # set colormap
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            im = ax.contourf(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),  # log-transform
                norm=norm,
                cmap="viridis",
            )
            # plot red line for max fv while fv < fv*
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

        # add colorbar for actual values
        im = cm.ScalarMappable(norm=norm)
        fig.tight_layout(pad=1.0)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig


def plot_scenarios_days(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    R0s: np.ndarray,
    epss: np.ndarray,
    dim: int = 2,
):
    """
    Plot difference in percentage reduction of total recovered population between
    leaky and all-or-nothing vaccines compared to that of without vaccination for
    all scenarios (defined in Appendix/1_Scenarios_days.ipynb). The plots are 
    log-scaled while the colorbars show the actual values before transformation.

    Parameters
    ----------
    df1: pd.DataFrame
        Output from ``utils.run_scenarios_days`` with ``tv=-1``.
    df2: pd.DataFrame
        Output from ``utils.run_scenarios_days`` with ``tv=10``.
    df3: pd.DataFrame
        Output from ``utils.run_scenarios_days`` with ``tv=30``.
    R0s: np.ndarray
        Array of varying R0 values (ex. np.linspace(1.0, 3.0, 601))
    epss: np.ndarray
        Array of varying vaccine efficacy values. (ex. np.linspace(0.01, 1, 201)) 
        A vaccine that is 50% effective would have an epsilon value of 0.5 For a 
        leaky vaccine, (epsL, epsA) = (eps, 1), and for an all-or-nothing vaccine,
        (epsL, epsA) = (1, eps).
    dim: int (Default: 2)
        ``dim=2`` plots a 2D contour plot using ``matplotlib.axes.Axes.contourf``
        and ``dim=3`` plots a 3D surface plot using Matplotlib's ``plot_surface``
        method.
    
    Returns
    -------
    fig: matplotlib.pyplot.figure
    """
    # parse inputs
    covs = np.unique(df1["Vax Coverage"].to_numpy())
    plot_R0, plot_eps = np.nan_to_num(np.meshgrid(R0s, epss, indexing="ij"))

    # construct arrays of difference in percentage reduction for each coverage level
    plot_dfs = []
    for df in [df1, df2, df3]:
        for cov in covs:
            temp_cov_df = df[df["Vax Coverage"] == cov]
            plot_df = np.reshape(temp_cov_df["Diff"].to_numpy(), np.shape(plot_R0))
            plot_df = np.ma.masked_where(plot_df == 99999, plot_df)
            plot_dfs.append(plot_df)

    # define min, max values to normalize colormap
    min_diff = 9999999
    max_diff = -9999999
    for df in plot_dfs:
        temp_min = np.min(df)
        if temp_min < min_diff:
            min_diff = temp_min

        temp_max = np.max(df)
        if temp_max > max_diff:
            max_diff = temp_max

    covs_str = ["Below", "Slightly Above", "Above"] * 3
    tvs_str = ["Pre"] * 3 + ["10 Days Post"] * 3 + ["30 Days Post"] * 3

    # plot 3d surface plot
    if dim == 3:
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(16, 16),
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
            subplot_kw={"projection": "3d"},
        )
        # set colormap
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot_surface(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),  # log-transform
                norm=norm,
                rstride=1,
                cstride=1,
                cmap="viridis",
            )
            ax.set_title(
                "$t_V$: {tv} | $f_V$: {cov} $f_V^*$".format(
                    tv=tvs_str[i], cov=covs_str[i]
                )
            )
            ax.set_xlabel("$R_0$")
            ax.set_ylabel("Vaccine Efficacy")
            ax.view_init(elev=30, azim=240)
            ax.set_zlabel("$log(P_A - P_L + 1)$", rotation=180)

        # set colorbar for actual values
        im = cm.ScalarMappable(norm=norm)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig

    # plot 2d contour plot
    elif dim == 2:
        fig, axes = plt.subplots(
            3,
            3,
            facecolor="w",
            figsize=(15, 15),
            sharex=True,
            sharey=True,
            gridspec_kw=dict(width_ratios=[1, 1, 1]),
        )
        # set colorbar
        norm = Normalize(np.log(min_diff + 1), np.log(max_diff + 1))

        axs = np.array(axes)
        for i, ax in enumerate(axs.reshape(-1)):
            im = ax.contourf(
                plot_R0,
                plot_eps,
                np.log(plot_dfs[i] + 1),  # log-transform
                norm=norm,
                cmap="viridis",
            )
            ax.set_title(
                "$t_V$: {tv} | $f_V$: {cov} $f_V^*$".format(
                    tv=tvs_str[i], cov=covs_str[i]
                )
            )
            ax.set_ylim([0, 1])
            ax.set_xlim([1, 3])

            if i % 3 == 0:
                ax.set_ylabel("Vaccine Efficacy")
            if i >= 6:
                ax.set_xlabel("$R_0$")

        # set colorbar for actual values
        im = cm.ScalarMappable(norm=norm)
        fig.tight_layout(pad=1.0)
        cb = fig.colorbar(mappable=im, ax=axes, fraction=0.02, shrink=0.5)
        cblabels = np.interp(cb.ax.get_yticks(), cb.ax.get_ylim(), im.get_clim())
        cb.ax.set_yticklabels(np.round(np.exp(cblabels) - 1).astype(int))
        cb.set_label("Difference in Percentage Reduction (%)")

        return fig


def plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    df4: pd.DataFrame,
    df5: pd.DataFrame,
    df6: pd.DataFrame,
    R0s: np.ndarray,
    eps: float,
    tol: float,
):
    """
    Plot difference in percentage reduction of total recovered population between
    leaky and all-or-nothing vaccines compared to that of without vaccination for
    when vaccine efficacy wanes vs. doesn't wane. All scenarios are shown for a 
    given vaccine efficacy value. 

    Parameters
    ----------
    df1: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0``, ``w=0``.
    df2: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.1``, ``w=0``.
    df3: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.25``, ``w=0``.
    df4: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0`` and nonzero ``w``.
    df5: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.1`` and nonzero ``w``.
    df6: pd.DataFrame
        Output from ``utils.run_scenarios`` with ``size=0.25`` and nonzero ``w``.
    R0s: np.ndarray
        Array of varying R0 values (ex. np.linspace(1.0, 3.0, 601))
    eps: float
        Vaccine efficacy. A vaccine that is 50% effective would have an epsilon 
        value of 0.5. For a leaky vaccine, (epsL, epsA) = (eps, 1), and for an 
        all-or-nothing vaccine, (epsL, epsA) = (1, eps).
    tol: float
        Tolerance value for selecting values with given ``eps`` from dataframes,
        as they might not be evaluated at the exact ``eps`` given. Depends on 
        step size of ``epss`` used for ``utils.run_scenarios``.
    
    Returns
    -------
    fig: matplotlib.pyplot.figure
    """
    # initialize lists for different fv values
    fv50s = []
    fv75s = []
    fv100s = []

    # separate dataframes based on fv used
    for i, df in enumerate([df1, df2, df3, df4, df5, df6]):
        df_eps = df[df["VE"] <= eps + tol]
        df_eps = df_eps[df_eps["VE"] > eps - tol]
        fv50_df_eps = df_eps[df_eps["Vax Coverage"] == 0.5]
        fv75_df_eps = df_eps[df_eps["Vax Coverage"] == 0.75]
        fv100_df_eps = df_eps[df_eps["Vax Coverage"] == 1.0]

        fv50 = np.reshape(fv50_df_eps["Diff"].to_numpy(), np.shape(R0s))
        fv50 = np.ma.masked_where(fv50 == 99999, fv50)
        fv75 = np.reshape(fv75_df_eps["Diff"].to_numpy(), np.shape(R0s))
        fv75 = np.ma.masked_where(fv75 == 99999, fv75)
        fv100 = np.reshape(fv100_df_eps["Diff"].to_numpy(), np.shape(R0s))
        fv100 = np.ma.masked_where(fv100 == 99999, fv100)
        fv50s.append(fv50)
        fv75s.append(fv75)
        fv100s.append(fv100)

    fvs = [fv50s, fv75s, fv100s]

    covs_str = ["50%", "75%", "100%"] * 3
    tvs_str = ["0"] * 3 + ["0.10"] * 3 + ["0.25"] * 3

    # plot comparison plots
    fig, axes = plt.subplots(
        3,
        3,
        facecolor="w",
        figsize=(10, 10),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(width_ratios=[1, 1, 1]),
    )

    for i in range(3):
        for j in range(3):
            plot_fv = fvs[j]
            axes[i, j].plot(R0s, plot_fv[i], "b--", label="Without Waning")
            axes[i, j].plot(R0s, plot_fv[i + 3], "r:", label="With Waning")
            axes[i, j].set_title(
                "$t_V$: R = {tv} | $f_V$: {cov} of (1- $f_R$)".format(
                    tv=tvs_str[i], cov=covs_str[i]
                )
            )
            axes[i, j].set_xlim([1, 3])

            if j % 3 == 0:
                axes[i, j].set_ylabel("Difference in Percentage Reduction (%) of $R$")
            if i >= 2:
                axes[i, j].set_xlabel("$R_0$")
    legend = axes[0, 0].legend()
    legend.get_frame().set_alpha(0.5)

    fig.tight_layout(pad=0.1)

    return fig
