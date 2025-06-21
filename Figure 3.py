
import os
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from scipy import integrate, interpolate, optimize, misc
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.optimize import minimize, minimize_scalar

import cmocean
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, ConnectionPatch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter, FixedFormatter, FixedLocator, SymmetricalLogLocator

# key functions and parameters
from IRF_functions import *
from IRF_parameters import *

dir_path = 'D:/科研相关/我的/tCDR/tCDR'
os.chdir(dir_path)

plt.style.use('default')


#==============================================================================
# IRF functions
#==============================================================================

# CO2, F(t)
AGWPCO2_partial = partial(AGWPCO2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2)
AGTPCO2_partial = partial(AGTPCO2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2)

# NonCO2, F(t,AANonCO2, tauNonCO2)
AGWPNonCO2_Final_partial = partial(AGWPNonCO2_Final, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPNonCO2_Final_partial = partial(AGTPNonCO2_Final, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)

# CH4 different sources,  F(t)
AGWPCH4NonFossil_Final_partial = partial(AGWPCH4NonFossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPCH4NonFossil_Final_partial = partial(AGTPCH4NonFossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGWPCH4Fossil_Final_partial = partial(AGWPCH4Fossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPCH4Fossil_Final_partial = partial(AGTPCH4Fossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)

# tCDR,  F(t, decay)
AGWPPRF_Exp_partial = partial(AGWPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Exp_partial = partial(AGTPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)


# tCDR vs NonCO2, F(t, aloha, decay, AANonCO2, tauNonCO2)
def NetAGTP_Exp(t, alpha, decay, AANonCO2,tauNonCO2):
    return AGTPNonCO2_Final_partial(t, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)

# tCDR vs CO2, F(t, aloha, decay)
def NetAGTP_Exp_CO2(t, alpha, decay):
    return AGTPCO2_partial(t) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)


#==============================================================================
# other functions
#==============================================================================

# format with zeros, rounded to three significant figures for clarity
def format_with_zeros(x):
    try:
        x = float(x)
    except ValueError:

        return x
    except TypeError:

        if pd.isna(x):
            return x

    if pd.isna(x):  
        return x
    formatted = format(x, '.3g')  
    if 'e' in formatted:  
        significand, exponent = formatted.split('e')
        exponent = int(exponent)
        if exponent < 0:
            zeros = '0' * (-exponent - 1)
            return '0.' + zeros + significand.replace('.', '')
        else:
            if '.' in significand:
                significand = significand.replace('.', '')
                return significand.ljust(exponent + 1, '0')
            else:
                return significand + '0' * exponent
    else:
        return formatted

# covert rgb to 0-1    
def convert_rgb_to_01(rgb):
    return [x/255.0 for x in rgb]


#==============================================================================
# Figure 3
#==============================================================================


plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})


dt = 0.1
tmax = 200
t = np.arange(0, tmax + dt, dt)

decays = [5, 10, 15, 50, 100]
tCDR_colors = plt.cm.summer(np.linspace(0, 0.9, len(decays)))

NonCO2_colors = plt.cm.summer(np.linspace(0, 0.9, len(decays)))

gas_params = {
    # 'BC': {'display_name': 'BC', 'tauNonCO2': tauBC, 'AANonCO2': AABC, 'scaling_factor': 1},
    'HFC32': {'display_name': 'HFC-32', 'tauNonCO2': tauHFC32, 'AANonCO2': AAHFC32, 'scaling_factor': 1},
    'CH4': {'display_name': r'CH$_4$', 'tauNonCO2': tauCH4, 'AANonCO2': AACH4, 'scaling_factor': 1},
    'HFC134a': {'display_name': 'HFC-134a', 'tauNonCO2': tauHFC134a, 'AANonCO2': AAHFC134a, 'scaling_factor': 1},
    'CFC11': {'display_name': 'CFC-11', 'tauNonCO2': tauCFC11, 'AANonCO2': AACFC11, 'scaling_factor': 1},
    'N2O': {'display_name': r'N$_2$O', 'tauNonCO2': tauN2O, 'AANonCO2': AAN2O, 'scaling_factor': 1},
    # 'PFC14': {'display_name': 'PFC-14', 'tauNonCO2': tauPFC14, 'AANonCO2': AAPFC14, 'scaling_factor': 1},
}


if False:
    # find TpeakCO2
    def find_TpeakCO2(decay):
        """
        Find the Tpeak time for CO2 given a decay parameter
        Args:
            decay: decay parameter (tau)
        Returns:
            Time of Tpeak
        """
        def objective(t):
            return AGTPPRF_Exp_partial(t=t, decay=decay)
        
        result = minimize_scalar(objective)
        
        if not result.success:
            warnings.warn(f"Optimization failed for decay={decay}")
            return np.nan
            
        return result.x

    # Calculate TpeakCO2( values for each decay 
    def calculate_TpeakCO2(taus):
        """
        Calculate the Tpeak time for CO2
        Args:
            taus: array of decay parameters
        Returns:
            Array of Tpeak times
        """
        TpeakCO2_values = []
        for tau in taus:
            peak_time = find_TpeakCO2(tau)
            TpeakCO2_values.append(peak_time)
        return np.array(TpeakCO2_values)

    # find tauNonCO2
    def find_tauNonCO2(tpeak, AA=1):
        """
        Find tau value for NonCO2 that gives maximum at tpeak using minimization
        """
        def objective(tau):
            # Calculate derivative at tpeak
            dt = 1e-6
            val1 = AGTPNonCO2_Final_partial(t=tpeak + dt, AANonCO2=AA, tauNonCO2=tau)
            val2 = AGTPNonCO2_Final_partial(t=tpeak - dt, AANonCO2=AA, tauNonCO2=tau)
            derivative = (val1 - val2)/(2*dt)
            return abs(derivative)  # Return absolute value for minimization
        
        # Use minimize_scalar instead of root_scalar
        result = minimize_scalar(
            objective,
            bounds=(1, 100),
            method='bounded'
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed for tpeak={tpeak}")
            return np.nan
            
        return result.x

    # Calculate tauNonCO2 values for each TpeakCO2
    def calculate_tauNonCO2(TpeakCO2_values):
        tauNonCO2_values = []
        for tpeak in TpeakCO2_values:
            tau_nonco2 = find_tauNonCO2(tpeak)
            tauNonCO2_values.append(tau_nonco2)
        return np.array(tauNonCO2_values)


    # get TpeakCO2 and tauNonCO2 values
    taus = np.logspace(0, 5, num=1000)  
    taus = np.unique(taus.astype(int))  

    TpeakCO2_values = calculate_TpeakCO2(taus)
    tauNonCO2_values = calculate_tauNonCO2(TpeakCO2_values)

    # save 
    np.savetxt('data/taus.csv', taus, delimiter=',')
    np.savetxt('data/TpeakCO2_values.csv', TpeakCO2_values, delimiter=',')
    np.savetxt('data/tauNonCO2_values.csv', tauNonCO2_values, delimiter=',')

# read TpeakCO2 and tauNonCO2 values from CSV files
taus = np.loadtxt('data/taus.csv', delimiter=',')
TpeakCO2_values = np.loadtxt('data/TpeakCO2_values.csv', delimiter=',')
tauNonCO2_values = np.loadtxt('data/tauNonCO2_values.csv', delimiter=',')


#==============================================================================
# Figure 3
#==============================================================================

fig = plt.figure(figsize=(18, 10)) 
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3)

ax0 = fig.add_subplot(gs[:, 0])  
ax1 = fig.add_subplot(gs[0, 1])  
ax2 = fig.add_subplot(gs[1, 1]) 

## panel a
ax = ax0

# NonCO2
for (gas_name, params), color in zip(
    [(k, v) for k, v in gas_params.items()], 
    NonCO2_colors
):
    label = f'{params["display_name"]} ({params["tauNonCO2"]}, scaled)*'
    ax.plot(t, 1E15*AGTPNonCO2_Final_partial(t,
            AANonCO2=AACO2,
            tauNonCO2=params['tauNonCO2']),
            '-', color=color, label=label)

# CO2
ax.plot(t, 1E15*AGTPCO2_partial(t), '-', color='k', label=r'CO$_2$ (multiple and near infinite)')

# temporary CDR
for decay, color in zip(decays, tCDR_colors):
    ax.plot(t, 1E15*AGTPPRF_Exp_partial(t=t, decay=decay), '--',
            color=color, label=f'$\\tau={decay}$')
ax.plot(t, -1*1E15*AGTPCO2_partial(t), '--', color='k', label=r'$\tau \; \to \; \infty$')

ax.legend(frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1.03, 0.0001), labelspacing=0.1)

# TpeakCO2
ax.axvline(x=TpeakCO2_values[-1], color='gray', linestyle=':')

ax.axhline(y=0, color='k', linestyle='-')

ax.set_xticks(np.arange(0, 111, 10))

ax.set_xlim([-10, 100])
ax.set_ylim([-1, 1])

ax.set_xlabel('Year')
ax.set_ylabel(r'Global temperature response (K kg$^{-1}$ × 10$^{-15}$)')

ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False)  
ax.text(-0.15, 1, "a", transform=ax.transAxes, size=18, weight='bold')  


ax.annotate('', xy=(20, 0.82), xytext=(9, 0.82),
           arrowprops=dict(arrowstyle="->", linewidth=1))


ax.text(22, 0.83,r'Timing of temperature peak for CO$_2$ emission', fontsize=14)
ax.text(22, 0.77,r'and pCDR ($\tau \; \to \; \infty$)', fontsize=14)

ax.text(21, -0.58,r'Pulse emissions', fontsize=14)
ax.text(87, -0.58,r'CDR', fontsize=14)

## panel b
ax = ax1

ax.plot(taus, TpeakCO2_values, linestyle='-', color='b')

f = interp1d(taus, TpeakCO2_values)
x_ref = 100
y_ref = float(f(x_ref))

ax.plot([x_ref, x_ref], [0, y_ref], 
        color='gray', linestyle=':')

ax.plot([0, x_ref], [y_ref, y_ref], 
        color='gray', linestyle=':')


ax.set_xscale('log')
ax.set_xlabel(r'CDR Storage Timescale ($\tau$, years)', fontsize=14)
ax.set_ylabel(r'Timing of temperature peak for CDR (years)', fontsize=14)

ax.axhline(y=9.01, color='gray', linestyle=':')

ax.text(0.6, 9.2, "9.0 years for near infinite storage", size=14)  
ax.text(0.6, 6.6, "6.4 years for 100-year storage", size=14)  
ax.text(-0.15, 1, "b", transform=ax.transAxes, size=18, weight='bold')  

ax.set_xlim([-10, 1E5])
ax.set_ylim([0, 10])

ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False)  


## panel c

ax = ax2

ax.plot(taus, tauNonCO2_values, linestyle='-', color='b')

f = interp1d(taus, tauNonCO2_values)
x_ref = 100
y_ref = float(f(x_ref))

ax.plot([x_ref, x_ref], [ax.get_ylim()[0], y_ref], 
        color='gray', linestyle=':')

ax.plot([ax.get_xlim()[0], x_ref], [y_ref, y_ref], 
        color='gray', linestyle=':')


ax.set_xscale('log')
ax.set_xlabel(r'CDR Storage Timescale ($\tau$, years)', fontsize=14)
ax.set_ylabel('Lifetime threshold (years)', fontsize=14)

ax.axhline(y=30.9, color='gray', linestyle=':')

ax.text(0.6, 31.9, "30.9 years for near infinite storage", size=14)  
ax.text(0.6, 13.7, "12.7 years for 100-year storage", size=14)  
ax.text(-0.15, 1, "c", transform=ax.transAxes, size=18, weight='bold')  

ax.set_xlim([-10, 1E5])
ax.set_ylim([0, 40])

ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False)  

plt.savefig('figure\Figure 3.png', bbox_inches='tight', pad_inches=0.1)

