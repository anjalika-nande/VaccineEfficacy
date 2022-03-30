import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
    

