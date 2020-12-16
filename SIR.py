#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Här längst upp är modellen som använder sig
av SEIR metoden, alltså den vi fick av GS-duden
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# describe the model
def deriv(y, t, N, beta, k, delta):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - k * I
    dRdt = k * I
    return dSdt, dEdt, dIdt, dRdt


# describe the parameters
N =  1000           # population
delta = 1.0 / 5.0       # incubation period of five days
beta = 2.0           
k = 1 / 5                   
S0, E0, I0, R0 = N-1, 1, 0, 0  # initial conditions: one infected, rest susceptible


t = np.linspace(0, 99, 100) # Grid of time points (in days)
y0 = S0, E0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, k, delta))
S, E, I, R = ret.T


def plotsir(t, S, E, I, R):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')  
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        plt.savefig("Plot.png")
        plt.show();

plotsir(t, S, E, I, R) 
#%%

"""
Här är egentligen samma modell fast jag har tagit bort
Exposed variabeln, alltså har vi en vanlig SIR-modell
"""

# describe the model
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * I * S / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# describe the parameters
N =  10000000             # population
beta = 1.0            
gamma = 1/1.5                  
S0, I0, R0 = N-1, 1, 0  # initial conditions: one infected, rest susceptible


t = np.linspace(0, 99, 100) # Grid of time points (in days)
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


def plotsir(t, S, I, R):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible') 
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        plt.savefig("Plot.png")
        plt.show();

plotsir(t, S, I, R)