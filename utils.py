import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

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


def modified_seir(y, t, beta, sigma, gamma, f, epsL, epsA, v):
    """
    Modified SEIR model. Accomodates two different modes of vaccine failure - 'Leaky'
    and 'All-or-Nothing'. 
    """
    s, v_es, v_rs, e, i, r = y
    dsdt = - beta*s*i - f*(1-(v_es+v_rs)/v)*s
    dvesdt = f*(1-epsA)*(1-(v_es+v_rs)/v)*s - beta*v_es*i
    dvrsdt = f*epsA*(1-(v_es+v_rs)/v)*s - (1-epsL)*beta*v_rs*i
    dedt = beta*s*i + beta*v_es*i + (1-epsL)*beta*v_rs*i - sigma*e
    didt = sigma*e - gamma*i
    drdt = gamma*i

    return dsdt, dvesdt, dvrsdt, dedt, didt, drdt


