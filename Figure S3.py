
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

# Impulse + Exp,  F(t, decay)
AGWPPRF_Exp_partial = partial(AGWPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Exp_partial = partial(AGTPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)
AGWPPRF_Constant_partial = partial(AGWPPRF_Constant, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Constant_partial = partial(AGTPPRF_Constant, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)
AGWPPRF_Impulse_partial = partial(AGWPPRF_Impulse, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Impulse_partial = partial(AGTPPRF_Impulse, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)
AGWPPRF_Trian_Exp_partial = partial(AGWPPRF_Trian_Linear, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2)
AGTPPRF_Trian_Exp_partial = partial(AGTPPRF_Trian_Linear, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2)


AGWPPRF_Linear_partial = partial(AGWPPRF_Linear, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Linear_partial = partial(AGTPPRF_Linear, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)


# CH4 different sources,  F(t)
AGWPCH4NonFossil_Final_partial = partial(AGWPCH4NonFossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPCH4NonFossil_Final_partial = partial(AGTPCH4NonFossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGWPCH4Fossil_Final_partial = partial(AGWPCH4Fossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPCH4Fossil_Final_partial = partial(AGTPCH4Fossil_Final, tauNonCO2=tauCH4, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AANonCO2=AACH4, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)


# tCDR vs NonCO2, F(t, aloha, decay, AANonCO2, tauNonCO2)
def NetAGTP_Exp(t, alpha, decay, AANonCO2,tauNonCO2):
    return AGTPNonCO2_Final_partial(t, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)

# tCDR vs CO2, F(t, aloha, decay)
def NetAGTP_Exp_CO2(t, alpha, decay):
    return AGTPCO2_partial(t) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)


#==============================================================================
# other functions
#==============================================================================

# arrowed_spines
def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw*0.8, head_length=hl*0.8, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw*0.8, head_length=yhl*0.8, overhang = ohg, 
             length_includes_head= True, clip_on = False)

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
# Figure S3
#==============================================================================

# Anthropogenic emissions F(t,AANonCO2, tauNonCO2)
AGWPNonCO2_Final_partial = partial(AGWPNonCO2_Final, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)
AGTPNonCO2_Final_partial = partial(AGTPNonCO2_Final, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, gamma=gamma, aS1=aS1, aS2=aS2, aS3=aS3, tauS1=tauS1, tauS2=tauS2, tauS3=tauS3)

# Impulse + Exp  F(t, decay)
AGWPPRF_Exp_partial = partial(AGWPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2, p=p, rotation=rotation)
AGTPPRF_Exp_partial = partial(AGTPPRF_Exp, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2, p=p, rotation=rotation)

# CO2 emissions F(t)
AGWPCO2_partial = partial(AGWPCO2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, AACO2=AACO2)
AGTPCO2_partial = partial(AGTPCO2, aC1=aC1, aC2=aC2, aC3=aC3, aC4=aC4, tauC1=tauC1, tauC2=tauC2, tauC3=tauC3, kPulseT=kPulseT, aT1=aT1, tauT1=tauT1, aT2=aT2, tauT2=tauT2, AACO2=AACO2)

tmax = 1000
dt = 0.1
t = np.arange(0, tmax + dt, dt)

# %matplotlib inline 

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14

dt = 0.1
tmax = 200
t = np.arange(0, tmax + dt, dt)

decays = [1, 5, 10, 50, 100, 300]
colors = plt.cm.summer(np.linspace(0, 0.9, len(decays)))

tab20_colors = plt.cm.tab20.colors
NonCO2_colors = [tab20_colors[i] for i in [0, 2, 4, 6, 10, 8, 12, 14, 16]]


scaling_factors = [1E-4, 1E-4, 1E-2, 1E-4, 1E-4, 1E-3, 1E-4]
taus = [tauBC, tauHFC32, tauCH4, tauHFC134a, tauCFC11, tauN2O, tauPFC14]
AAs = [AABC, AAHFC32, AACH4, AAHFC134a, AACFC11, AAN2O, AAPFC14]
labels = [r'$\mathrm{BC}$ (0.02)', 
        r'$\mathrm{HFC32}$ (5.4)', 
        r'$\mathrm{CH}_4$ (11.8)', 
        r'$\mathrm{HFC134a}$  (14.0)', 
        r'$\mathrm{CFC11}$ (52.0)', 
        r'$\mathrm{N}_2$O (109.0)', 
        r'$\mathrm{PFC14}$ (50000.0)']

labels_scaling_factors = [
    r'$\mathrm{-1 \times 10^{-4}\; BC}$ (0.02)',
    r'$\mathrm{-1 \times 10^{-4}\; HFC32}$ (5.4)',
    r'$\mathrm{-1 \times 10^{-2}\; CH}_4$ (11.8)',
    r'$\mathrm{-1 \times 10^{-4}\; HFC134a}$ (14.0)',
    r'$\mathrm{-1 \times 10^{-4}\; CFC11}$ (52.0)',
    r'$\mathrm{-1 \times 10^{-3}\; N}_2$O (109.0)',
    r'$\mathrm{-1 \times 10^{-4}\; PFC14}$ (50000.0)']

# Impulse + Exp
if True:

    fig, axs = plt.subplots(2, 2, figsize=(16, 14))

    plt.subplot(2, 2, 1) 
    current_ax = plt.gca() 
    plt.text(-0.2, 1.0, "a", transform=current_ax.transAxes, size=18, weight='bold')  

    for decay, color in zip(decays, colors):
        plt.plot(t, PRF_Exp(t, decay, aC1, aC2, aC3, aC4, tauC1, tauC2, tauC3), '-', color=color, label=f'$\\tau={decay}$')
    plt.plot(t, -1 * IRFCO2(t, aC1, aC2, aC3, aC4, tauC1, tauC2, tauC3), '-', color='k', label=r'$\tau \; \to \; \infty$')

    for i in range(len(taus[:])):
        plt.plot(t, -1 * IRFNonCO2(t, taus[i]), '--', color=NonCO2_colors[i], label=labels[i])

    plt.plot([0, 300], [0, 0], ':k')
    plt.legend(frameon=False, loc='best')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim([0,200])
    plt.ylim([-1.2, 0.2])
    plt.xlabel('Year')
    plt.ylabel('Fraction in atmosphere')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)


    plt.subplot(2, 2, 2)  
    current_ax = plt.gca() 
    plt.text(-0.2, 1.0, "b", transform=current_ax.transAxes, size=18, weight='bold')  

    for decay, color in zip(decays, colors):
        plt.plot(t, AGTPPRF_Exp_partial(t=t, decay=decay), '-', color=color, label=f'$\\tau={decay}$')
    plt.plot(t, (-1 * AGTPCO2_partial(t))*1E16 , '-', color='k', label=r'$\tau \; \to \; \infty$')

    for i in range(len(taus[:])):
        plt.plot(t, (-1 * scaling_factors[i] * AGTPNonCO2_Final_partial(t, AANonCO2=AAs[i], tauNonCO2=taus[i]))*1E16, '--', color=NonCO2_colors[i], label=labels_scaling_factors[i])


    lines = plt.gca().get_lines()

    selected_lines = lines[-7:]  

    plt.legend(handles=selected_lines, frameon=False, loc='best')


    plt.plot([0, 300], [0, 0], ':k')
    plt.xlim([0,200])
    plt.ylim([-7.5, 1])
    plt.xlabel('Year')
    plt.ylabel(r'$\mathrm{AGTP\ (}\mathrm{K\ kg^{-1}\ \times 10^{-16})}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)


    plt.subplot(2, 2, 3)  
    current_ax = plt.gca() 
    plt.text(-0.2, 1.0, "c", transform=current_ax.transAxes, size=18, weight='bold')  

    for decay, color in zip(decays, colors):
        plt.plot(t, -1 * AGWPPRF_Exp_partial(t=t, decay=decay), '-', color=color, label=f'$\\tau={decay}$')
    plt.plot(t, AGWPCO2_partial(t), '-', color='k', label=r'$\tau \; \to \; \infty$')

    for i in range(len(taus[:])):
        plt.plot(t, scaling_factors[i] * AGWPNonCO2_Final_partial(t, AANonCO2=AAs[i], tauNonCO2=taus[i]), '--', color=NonCO2_colors[i], label=labels_scaling_factors[i])


    lines = plt.gca().get_lines()

    selected_lines = lines[-7:]  

    plt.legend(handles=selected_lines, frameon=False, loc='best')

    plt.yscale('log')  
    plt.yticks([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12], [r'$-10^{-18}$',r'$-10^{-17}$',r'$-10^{-16}$', r'$-10^{-15}$', r'$-10^{-14}$', r'$-10^{-13}$', r'$-10^{-12}$'])
    plt.xlabel('Year')
    plt.ylabel(r'$\mathrm{AGWP\ (W\ m^{-2}\ kg^{-1})}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)
    plt.xlim([0,200])
    plt.ylim([1e-18,4*1e-13])
    plt.gca().invert_yaxis() 



    plt.subplot(2, 2, 4)  
    current_ax = plt.gca() 
    plt.text(-0.2, 1.0, "d", transform=current_ax.transAxes, size=18, weight='bold')  

    for decay, color in zip(decays, colors):
        tmp = -1 * dt * np.cumsum(AGTPPRF_Exp_partial(t=t, decay=decay))
        plt.plot(t, tmp, '-', color=color, label=f'$\\tau={decay}$')

    plt.plot(t, dt * np.cumsum(AGTPCO2_partial(t)), '-', color='k', label=r'$\tau \; \to \; \infty$')

    for i in range(len(taus[:])):
        plt.plot(t, scaling_factors[i] * dt * np.cumsum(AGTPNonCO2_Final_partial(t, AANonCO2=AAs[i], tauNonCO2=taus[i])), '--', color=NonCO2_colors[i], label=labels_scaling_factors[i])


    lines = plt.gca().get_lines()


    selected_lines = lines[-7:] 

    plt.legend(handles=selected_lines, frameon=False, loc='best')

    # plt.legend(frameon=False, loc='best')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.yscale('log') 
    plt.yticks([1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12], [r'$-10^{-18}$',r'$-10^{-17}$',r'$-10^{-16}$', r'$-10^{-15}$', r'$-10^{-14}$', r'$-10^{-13}$', r'$-10^{-12}$'])
    plt.xlabel('Year')
    plt.ylabel(r'$\mathrm{iAGTP\ (}\mathrm{K\ year\ kg^{-1})}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)
    plt.xlim([0,200])
    plt.ylim([1e-18,4*1e-13])
    plt.gca().invert_yaxis() 



    plt.tight_layout()


    plt.subplots_adjust(wspace=0.3) 
    plt.savefig('figure\Figure S3.png')


