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
def deriv(y, t, N, beta, gamma, delta):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# describe the parameters
N =  10000000           # population
delta = 1.0 / 5.0       # incubation period of five days
beta = 0.25           
gamma = 1 / 7                
S0, E0, I0, R0 = N-1, 1, 0, 0  # initial conditions: one infected, rest susceptible


t = np.linspace(0, 365, 367) # Grid of time points (in days)
y0 = S0, E0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
S, E, I, R = ret.T


def plotsir(t, S, E, I, R):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')  
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')

    ax.grid(b=True, which='major', c='black', lw=0.35, ls='-')
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
beta = 0.5            
gamma = 1 / 7                  
S0, I0, R0 = N-1, 1, 0  # initial conditions: one infected, rest susceptible


t = np.linspace(0, 100, 101) # Grid of time points (in days)
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
    ax.set_ylabel('Population')

    ax.grid(b=True, which='major', c='black', lw=0.35, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        plt.savefig("Plot.png")
        plt.show();

plotsir(t, S, I, R)
#%%
"""
Här har vi gjort en version som har med både en andrakurva och vaccin.
Lösningen med vaccin har lånats från grupp 15.
"""


from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import math

# describe the model
def deriv(y, t, N, gamma, alpha, delta, phi, omega, vacc_doses):
    S, E, I, R, B, H, D, C, V = y
    
    vacc = 150 # introduction day for vaccine
    
    vacc_pop = S + E + I + R # the people remaining to be vaccinated
    
    k = 1 if vacc < t < vacc + (N-D)/vacc_doses else 0 
    
    dSdt = -B * S * I / N + alpha * R - S/vacc_pop * vacc_doses * k + alpha * V   # susceptible-compartment 
    dEdt = B * S * I / N - delta * E - E/vacc_pop * vacc_doses * k   # exposed-compartment
    dIdt = delta * E - (1 - phi) * gamma * I - phi * omega * I - I/vacc_pop * vacc_doses * k   # infected-compartment
    dRdt = (1 - phi) * gamma * I - alpha * R - R/vacc_pop * vacc_doses * k   # recovered-compartment 
    dDdt = phi * omega * I   # dead-compartment 
    dCdt = 0.05 * dIdt * 1 / 10  # intensive care, 5 % of the infected require intensive care, 10 days from infected to critically ill
    def dBdt(t):   # varying transmission rate 
        return 1 / 50 * math.cos(t / 20) if 230 < t < 330 else B * (-1 / 0.7 * math.sin(2 * math.pi / 700))   # adding a second wave
        #return 0 if t < 200 else B * (-1 / 0.7 * math.sin(2 * math.pi / 70))   # lockdown after 200 days  
        #if b   
    dHdt = 0   # healthcare's COVID-19 capacity
    dVdt = vacc_doses * k - alpha * V
    return dSdt, dEdt, dIdt, dRdt, dBdt(t), dHdt, dDdt, dCdt, dVdt

# describe the parameters
N = 10336399          # population of Sweden (January 2020 source: www.scb.se)
delta = 1.0 / 5.0     # incubation period of five days 
D = 7.0               # number of days that an infected person has and can spread the disease
gamma = 1.0 / D       # removal rate 
alpha = 1/180         # immunity lost after six months
phi = 0.02            # 2 % fatality rate (number of deaths from disease / number of confirmed cases of disease * 100)
omega = 1/14          # 14 days from infection until death 
vacc_doses = 12000           # vaccination doses per day

S0, E0, I0, R0, B0, H0, D0, C0, V0 = N-1, 1, 0, 0, 1, 680, 0, 0, 0  # initial conditions: one infected, rest susceptible, initial transmission rate B=3, 680 intensive care spots, zero dead, zero in critical condition

t = np.linspace(0, 365, 366) # Grid of time points (in days)
y0 = S0, E0, I0, R0, B0, H0, D0, C0, V0 # Initial conditions vector

# Integrate the SEIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, gamma, alpha, delta, phi, omega, vacc_doses))
S, E, I, R, B, H, D, C, V = ret.T

def plotseir(t, S, E, I, R, D, V):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')  
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'c', alpha=0.7, linewidth=2, label='Dead')
    ax.plot(t, V, 'purple', alpha=0.7, linewidth=2, label='Vacinated')

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
#plot the graph

plotseir(t, S, E, I, R, D, V)