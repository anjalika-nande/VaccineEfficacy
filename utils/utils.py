import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from tqdm import tqdm
import math


def seir(t, y, beta, sigma, gamma):
    """
    Basic SEIR model. Used with ``scipy.integrate.solve_ivp``
    """
    s, e, i, r = y
    dsdt = -beta * s * i  # susceptible
    dedt = beta * s * i - sigma * e  # exposed
    didt = sigma * e - gamma * i  # infected
    drdt = gamma * i  # recovered

    return [dsdt, dedt, didt, drdt]


def modified_seir(t, y, beta, sigma, gamma, epsL, w):
    """
    Modified SEIR model for instantaneous vaccination. Used with 
    ``scipy.integrate.solve_ivp``
    """
    s, vs, vr, e, i, r, cv, cu = y

    dsdt = -beta * s * i  # susceptible
    dvsdt = -beta * vs * i + w * vr  # vaccinated and susceptible
    dvrdt = -beta * (1 - epsL) * vr * i - w * vr  # vaccinated and immune
    dedt = beta * (s + vs + (1 - epsL) * vr) * i - sigma * e  # exposed
    didt = sigma * e - gamma * i  # infected
    drdt = gamma * i  # recovered
    dcvdt = beta * (vs + (1 - epsL) * vr) * i  # cumulative infections from V
    dcudt = beta * s * i  # cumulative infections from S

    return [dsdt, dvsdt, dvrdt, dedt, didt, drdt, dcvdt, dcudt]


def modified_seir_cont(t, y, beta, sigma, gamma, v, fv, epsL, epsA, w):
    """
    Modified SEIR model for continuous vaccination. Used with 
    ``scipy.integrate.solve_ivp``
    """
    s, vs, vr, e, i, r, cv, cu = y

    dsdt = -beta * s * i - v * (1 - (vs + vr) / fv) * s  # susceptible
    dvsdt = (
        (1 - epsA) * v * (1 - (vs + vr) / fv) * s - beta * vs * i + w * vr
    )  # vaccinated and susceptible
    dvrdt = (
        epsA * v * (1 - (vs + vr) / fv) * s - beta * (1 - epsL) * vr * i - w * vr
    )  # vaccinated and immune
    dedt = beta * (s + vs + (1 - epsL) * vr) * i - sigma * e  # exposed
    didt = sigma * e - gamma * i  # infected
    drdt = gamma * i  # recovered
    dcvdt = beta * (vs + (1 - epsL) * vr) * i  # cumulative infections from V
    dcudt = beta * s * i  # cumulative infections from S

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
    Run modified SEIR model with given parameters for instantaneous vaccination. 
    Used with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    y0: list
        Initial conditions of model. In the form [S(0), V_S(0), V_R(0), E(0), 
        I(0), R(0), C_V(0), C_U(0)]. All the values should sum up to 1. 
    t: int
        Number of days to run simulation (days). The model assumes the simulation 
        begins at Day 0.
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
        Vaccine efficacy. A vaccine that is 50% effective would have an epsilon 
        value of 0.5 For a leaky vaccine, (epsL, epsA) = (eps, 1), and for an 
        all-or-nothing vaccine, (epsL, epsA) = (1, eps).
    mode: str
        Vaccine failure mode. Accepts 'leaky' for leaky vaccine, and 'aon' for 
        an all-or-nothing vaccine.
    
    Returns
    -------
    s_vax, vs, vr, v, e_vax, i_vax, r_vax : np.ndarray
        S, V_S, V_R, V, E, I, R for [0, t]. V = V_S + V_R.
    cv, cu: np.ndarray
        Cumulative infections for [0, t] from V, S. 
    """
    # parse initial condition
    s0, e0, i0, r0 = y0

    # determine vaccine failure mode
    if mode == "leaky":
        epsL = eps
        epsA = 1
    elif mode == "aon":
        epsL = 1
        epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    # solve modified_seir for given vaccination timing tv
    if tv == -1:  # before epidemic
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
            t_eval=np.linspace(0, t, t + 1),
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

    else:  # during epidemic
        # set timepoints at which solutions are evaluated (for time-series plots)
        if math.floor(tv) == tv:
            t_eval1 = np.linspace(0, tv, tv + 1)
            t_eval2 = np.linspace(tv, t, t - tv + 1)
        else:
            t_eval1 = np.append(
                np.linspace(0, math.floor(tv), math.floor(tv) + 1), [tv]
            )
            t_eval2 = np.linspace(math.floor(tv) + 1, t, t - math.floor(tv) + 1)

        # solve seir for [0, tv]
        sol = solve_ivp(
            seir,
            [0, tv],
            y0,
            args=(beta, sigma, gamma),
            dense_output=True,
            t_eval=t_eval1,
        )
        s = sol.y[0]
        e = sol.y[1]
        i = sol.y[2]
        r = sol.y[3]

        # solve modified_seir for [tv, t]
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
            t_eval=t_eval2,
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

        # concatenate results from [0, tv] and [tv, t]
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
    """
    Run modified SEIR model with given parameters for continuous vaccination. 
    Used with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    y0: list
        Initial conditions of model. In the form [S(0), V_S(0), V_R(0), E(0), 
        I(0), R(0), C_V(0), C_U(0)]. All the values should sum up to 1. 
    t: int
        Number of days to run simulation (days). The model assumes the simulation
        begins at Day 0.
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
        Vaccine efficacy. A vaccine that is 50% effective would have an epsilon 
        value of 0.5 For a leaky vaccine, (epsL, epsA) = (eps, 1), and for an 
        all-or-nothing vaccine, (epsL, epsA) = (1, eps).
    v: float
        Constant vaccination rate (1/days).
    mode: str
        Vaccine failure mode. Accepts 'leaky' for leaky vaccine, and 'aon' for 
        an all-or-nothing vaccine.
    
    Returns
    -------
    s_vax, vs, vr, v, e_vax, i_vax, r_vax : np.ndarray
        S, V_S, V_R, V, E, I, R for [0, t]. V = V_S + V_R.
    cv, cu: np.ndarray
        Cumulative infections for [0, t] from V, S. 
    """
    # parse initial condition
    s0, e0, i0, r0 = y0

    # determine vaccine failure mode
    if mode == "leaky":
        epsL = eps
        epsA = 1
    elif mode == "aon":
        epsL = 1
        epsA = eps
    else:
        print("Mode must be 'leaky' or 'aon'.")

    # solve modified_seir_cont for given vaccination timing tv
    if tv == -1:  # before epidemic
        vs0 = 0
        vr0 = 0
        y0_vax = [s0, vs0, vr0, e0, i0, r0, 0, 0]
        sol_vax = solve_ivp(
            modified_seir_cont,
            [0, t],
            y0_vax,
            args=(beta, sigma, gamma, v, fv, epsL, epsA, w),
            dense_output=True,
            t_eval=np.linspace(0, t, t + 1),
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

    else:  # during epidemic
        # set timepoints at which solutions are evaluated (for time-series plots)
        if math.floor(tv) == tv:
            t_eval1 = np.linspace(0, tv, tv + 1)
            t_eval2 = np.linspace(tv, t, t - tv + 1)
        else:
            t_eval1 = np.append(
                np.linspace(0, math.floor(tv), math.floor(tv) + 1), [tv]
            )
            t_eval2 = np.linspace(math.floor(tv) + 1, t, t - math.floor(t) + 1)

        # solve seir for [0, tv]
        sol = solve_ivp(
            seir,
            [0, tv],
            y0,
            args=(beta, sigma, gamma),
            dense_output=True,
            t_eval=t_eval1,
        )
        s = sol.y[0]
        e = sol.y[1]
        i = sol.y[2]
        r = sol.y[3]

        # solve modified_seir_cont for [tv, t]
        vs0 = 0
        vr0 = 0
        y0_vax = [s[-1], vs0, vr0, e[-1], i[-1], r[-1], 0, 0]
        sol_vax = solve_ivp(
            modified_seir_cont,
            [tv, t],
            y0_vax,
            args=(beta, sigma, gamma, v, fv, epsL, epsA, w),
            dense_output=True,
            t_eval=t_eval2,
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

        # concatenate results from [0, tv] and [tv, t]
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
    """
    Run scenarios (defined in Scenarios/1_Scenarios.ipynb) with varying R0 and 
    vaccine efficacy values for both leaky and all-or-nothing vaccine failure modes.
    Used with ``scipy.integrate.solve_ivp``. Note that the vaccination campaign 
    is assumed to be instantaneous.

    Parameters
    ----------
    y0: list
        Initial conditions of model. In the form [S(0), V_S(0), V_R(0), E(0), 
        I(0), R(0), C_V(0), C_U(0)]. All the values should sum up to 1. 
    t: int
        Number of days to run simulation (days). The model assumes the simulation 
        begins at Day 0.
    size: float
        Size of epidemic at which the vaccine is administered. Measured by total 
        recovered population. Should be either 0, 0.1, or 0.25. 
    covs: np.ndarray
        Array of floats representing the fraction of remaining susceptible 
        population that is vaccinated. 
    R0s: np.ndarray
        Array of varying R0 values (ex. np.linspace(1.0, 3.0, 601))
    sigma: float
        1/sigma is the duration of the latent period (days). 
    gamma: float
        1/gamma is the duration of the infectious period (days).
    w: float
        Rate at which vaccine efficacy wanes over time (1/days). 
    epss: np.ndarray
        Array of varying vaccine efficacy values. (ex. np.linspace(0.01, 1, 201)) 
        A vaccine that is 50% effective would have an epsilon value of 0.5 For a 
        leaky vaccine, (epsL, epsA) = (eps, 1), and for an all-or-nothing vaccine, 
        (epsL, epsA) = (1, eps).
    
    Returns
    -------
    vax_df: pd.DataFrame
        Dataframe with the following columns:
            - 'R0': R0 value (float)
            - 'VE': Vaccine efficacy for leaky and all-or-nothing vaccine (float)
            - 'Vax Coverage': Fraction of remaining susceptible population 
            vaccinated (float)
            - 'fv*': Calculated critical vaccination level (float)
            - 'fv': Fraction of total population vaccinated (float)
            - 'Above HIT': Whether fv is greater than fv* (boolean)
            - 'Leaky': Percentage reduction of total recovered population for 
            leaky vaccine
            - 'AON': Percentage reduction of total recovered population for 
            all-or-nothing vaccine
            - 'Diff': Difference in percentage reduction of total recovered 
            population between leaky and all-or-nothing vaccines
    """
    # initialize lists for each column in vax_df
    df_R0s = []
    df_epss = []
    df_covs = []
    df_fvs = []
    df_fcs = []
    df_hit = []
    df_r_perc_leakys = []
    df_r_perc_aons = []
    df_r_perc_diffs = []

    # parse initial condition
    s0, _, _, _ = y0

    # run model for each scenario
    for i in tqdm(range(0, len(R0s))):  # iterate through R0 values
        R0 = R0s[i]
        beta = R0 * gamma

        # find vaccination timing tv
        sol = solve_ivp(seir, [0, t], y0, args=(beta, sigma, gamma), dense_output=True)

        def _reach_size10(t, y, beta, sigma, gamma):
            return y[3] - 0.1

        def _reach_size25(t, y, beta, sigma, gamma):
            return y[3] - 0.25

        _reach_size10.terminate = True
        _reach_size25.terminate = True

        for eps in epss:  # iterate through vaccine efficacy values
            # calculate critical vaccination threshold fc
            if size == 0:
                fc = 1 / eps * (1 - 1 / R0)
                fs = s0  # remaining S
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

            # fc is bound by 0 and s0
            if fc < 0:
                fc = 99999
            elif (fc > s0) and (fc != 99999):
                fc = s0

            for cov in covs:  # iterate through vaccine coverage levels
                if fc == 99999:
                    fv = 99999
                    r_perc_leaky = 99999
                    r_perc_aon = 99999
                    r_perc_diff = 99999
                else:
                    # calculate fraction of population vaccinated (fv)
                    fv = cov * fs

                    # without vaccination
                    sol_vax = solve_ivp(
                        seir,
                        [0, t_new],
                        y0,
                        args=(beta, sigma, gamma),
                        dense_output=True,
                    )
                    r_vax = sol_vax.y[3]

                    # with vaccination - leaky
                    _, _, _, _, _, _, r_leaky, _, _ = run_modified_seir(
                        y0, t_new, tv, beta, sigma, gamma, w, fv, eps, mode="leaky"
                    )

                    # with vaccination - all-or-nothing
                    _, _, _, _, _, _, r_aon, _, _ = run_modified_seir(
                        y0, t_new, tv, beta, sigma, gamma, w, fv, eps, mode="aon"
                    )

                    # calculate percentage reduction of total recovered population
                    r_perc_leaky = (r_vax[-1] - r_leaky[-1]) / r_vax[-1] * 100
                    r_perc_aon = (r_vax[-1] - r_aon[-1]) / r_vax[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                # store values
                df_R0s.append(R0)
                df_epss.append(eps)
                df_covs.append(cov)
                df_fvs.append(fv)
                df_fcs.append(fc)
                df_hit.append(fv > fc)  # only for 'valid' values (0 <= fc <= S(0))
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


def run_scenarios_days(
    y0: list,
    t: int,
    tv: int,
    R0s: np.ndarray,
    sigma: float,
    gamma: float,
    w: float,
    epss: np.ndarray,
):
    """
    Run scenarios (defined in Appendix/1_Scenarios_days.ipynb) with varying R0 
    and vaccine efficacy values for both leaky and all-or-nothing vaccine failure 
    modes. Used with ``scipy.integrate.solve_ivp``. Note that the vaccination 
    campaign is assumed to be instantaneous.

    Parameters
    ----------
    y0: list
        Initial conditions of model. In the form [S(0), V_S(0), V_R(0), E(0), 
        I(0), R(0), C_V(0), C_U(0)]. All the values should sum up to 1. 
    t: int
        Number of days to run simulation (days). The model assumes the simulation 
        begins at Day 0.
    size: float
        Size of epidemic at which the vaccine is administered. Measured by total 
        recovered population.
    covs: np.ndarray
        Array of floats representing the fraction of remaining susceptible 
        population that is vaccinated. 
    R0s: np.ndarray
        Array of varying R0 values (ex. np.linspace(1.0, 3.0, 601))
    sigma: float
        1/sigma is the duration of the latent period (days). 
    gamma: float
        1/gamma is the duration of the infectious period (days).
    w: float
        Rate at which vaccine efficacy wanes over time (1/days). 
    epss: np.ndarray
        Array of varying vaccine efficacy values. (ex. np.linspace(0.01, 1, 201)) 
        A vaccine that is 50% effective would have an epsilon value of 0.5 For a 
        leaky vaccine, (epsL, epsA) = (eps, 1), and for an all-or-nothing vaccine,
        (epsL, epsA) = (1, eps).
    
    Returns
    -------
    vax_df: pd.DataFrame
        Dataframe with the following columns:
            - 'R0': R0 value (float)
            - 'VE': Vaccine efficacy for leaky and all-or-nothing vaccine (float)
            - 'Vax Coverage': Fraction of remaining susceptible population 
            vaccinated (float)
            - 'fv*': Calculated critical vaccination level (float)
            - 'fv': Fraction of total population vaccinated (float)
            - 'Above HIT': Whether fv is greater than fv* (boolean)
            - 'Leaky': Percentage reduction of total recovered population for 
            leaky vaccine
            - 'AON': Percentage reduction of total recovered population for 
            all-or-nothing vaccine
            - 'Diff': Difference in percentage reduction of total recovered 
            population between leaky and all-or-nothing vaccines
    """
    # initialize lists for each column in vax_df
    df_R0s = []
    df_epss = []
    df_fvs = []
    df_fcs = []
    covs = ["Below fc", "Slightly Above fc", "Above fc"]
    df_covs = []
    df_r_perc_leakys = []
    df_r_perc_aons = []
    df_r_perc_diffs = []

    # parse initial condition
    s0, _, _, _ = y0

    # run model for each scenario
    for i in tqdm(range(0, len(R0s))):  # iterate through R0 values
        R0 = R0s[i]
        beta = R0 * gamma

        # without vaccination
        sol = solve_ivp(seir, [0, t], y0, args=(beta, sigma, gamma), dense_output=True)
        r = sol.y[3]

        # iterate through vaccine eficacy values
        for eps in epss:
            # calculate critical vaccination threshold fc
            if tv == -1:  # before epidemic
                fc = 1 / eps * (1 - 1 / R0)
            else:
                sol_temp = solve_ivp(
                    seir, [0, tv], y0, args=(beta, sigma, gamma), dense_output=True
                )
                s_temp = sol_temp.y[0]
                fc = 1 / eps * (1 - 1 / (R0 * s_temp[-1]))

            # fc is bound from 0 to s0
            if fc < 0:
                fc = 99999
            elif (fc > s0) and (fc != 99999):
                fc = s0

            # iterate through vaccine coverage levels
            for cov in covs:
                # nonsense values if fc < 0
                if fc == 99999:
                    fv = 99999
                    r_perc_leaky = 99999
                    r_perc_aon = 99999
                    r_perc_diff = 99999
                else:
                    # calculate fraction of population vaccinated (fv)
                    if cov == "Below fc":
                        fv = fc * 0.8
                    elif cov == "Slightly Above fc":
                        fv = 1 - ((1 - fc) * 0.8)
                    else:
                        fv = 1 - ((1 - fc) * 0.5)

                    # with vaccination - leaky
                    _, _, _, _, _, _, r_leaky, _, _ = run_modified_seir(
                        y0, t, tv, beta, sigma, gamma, w, fv, eps, mode="leaky"
                    )

                    # with vaccination - all-or-nothing
                    _, _, _, _, _, _, r_aon, _, _ = run_modified_seir(
                        y0, t, tv, beta, sigma, gamma, w, fv, eps, mode="aon"
                    )

                    # calculate percentage reduction of total recovered population
                    r_perc_leaky = (r[-1] - r_leaky[-1]) / r[-1] * 100
                    r_perc_aon = (r[-1] - r_aon[-1]) / r[-1] * 100
                    r_perc_diff = r_perc_aon - r_perc_leaky

                # store values
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
