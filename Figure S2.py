
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
# Figure S2
#==============================================================================
color1 = convert_rgb_to_01((204,0,0))
color2 = convert_rgb_to_01((230,145,56))
color3 = convert_rgb_to_01((61,133,198))


fig, axs = plt.subplots(2, 3, figsize=(21, 12))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.3)

# Exponential
ax = axs[0] 
if True:
    dt = 0.1
    xlag = 5
    decayValue = 100
    tmax = decayValue + 100 
    tlimit = tmax + xlag + 10
    tt = np.arange(0, tmax + dt, dt)
    Fontsize=14
    scalefactor = 1E13

    tCDR = 1E2 * scalefactor * AGTPPRF_Exp_partial(t=tt, decay=100)
    pCDR = -1E2 *scalefactor * AGTPCO2_partial(t=tt)

    ax.plot(tt + xlag, tCDR,linewidth=2, color=color3)
    ax.plot(tt + xlag, pCDR, linewidth=2, color='k')

    ax.set_xlim([0,tlimit])
    ax.set_ylim([-0.7,0.7])
    ax.plot([decayValue + xlag, decayValue + xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)
    ax.plot([xlag, xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)

    arrowed_spines(fig, ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(-0.1, 1.02, 'a', transform=ax.transAxes,size=16, weight='bold')  
    ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes,size=Fontsize)    
    ax.text(0.95, 0.42, 'Year', transform=ax.transAxes,size=Fontsize)  
    ax.text(0.3, 0.35, r'tCDR ($\tau$ = 100 years)', transform=ax.transAxes,size=Fontsize, color=color3)  
    ax.text(0.5, 0.15, r'pCDR ($\tau \; \to \; \infty$)', transform=ax.transAxes,size=Fontsize, color='k')  
    ax.text(0.05, 0.53, 't = 0', transform=ax.transAxes,  size=Fontsize, color='k')  
    ax.text(-0.08, 0.65, 'Warming', 
            transform=ax.transAxes, size=Fontsize, rotation=90)
    ax.text(-0.08, 0.15, 'Cooling', 
            transform=ax.transAxes, size=Fontsize, rotation=90)


    arrow = FancyArrowPatch((xlag, 0.45), (decayValue+xlag, 0.45),
                            arrowstyle='<|-|>', 
                            mutation_scale=20, 
                            linestyle='--',
                            color='black')
    ax.add_patch(arrow)

    ax.text(0.25, 0.88, r'Storage timescale', 
            transform=ax.transAxes, size=Fontsize, 
            horizontalalignment='center', verticalalignment='center')

# Linear
ax = axs[1] 
if True:
    dt = 0.1
    xlag = 5
    decayValue = 100
    tmax = decayValue + 100 
    tlimit = tmax + xlag + 10
    tt = np.arange(0, tmax + dt, dt)
    Fontsize=14
    scalefactor = 1E13

    tCDR = 1E2 * scalefactor * AGTPPRF_Linear_partial(t=tt, decay=100)
    pCDR = -1E2 *scalefactor * AGTPCO2_partial(t=tt)

    ax.plot(tt + xlag, tCDR,linewidth=2, color=color3)
    ax.plot(tt + xlag, pCDR, linewidth=2, color='k')

    ax.set_xlim([0,tlimit])
    ax.set_ylim([-0.7,0.7])
    ax.plot([decayValue + xlag, decayValue + xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)
    ax.plot([xlag, xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)

    arrowed_spines(fig, ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(-0.1, 1.02, 'b', transform=ax.transAxes,size=16, weight='bold')  
    ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes,size=Fontsize)    
    ax.text(0.95, 0.42, 'Year', transform=ax.transAxes,size=Fontsize)  
    ax.text(0.3, 0.35, r'tCDR ($\tau$ = 100 years)', transform=ax.transAxes,size=Fontsize, color=color3)  
    ax.text(0.5, 0.15, r'pCDR ($\tau \; \to \; \infty$)', transform=ax.transAxes,size=Fontsize, color='k')  
    ax.text(0.05, 0.53, 't = 0', transform=ax.transAxes,  size=Fontsize, color='k')  
    ax.text(-0.08, 0.65, 'Warming', 
            transform=ax.transAxes, size=Fontsize, rotation=90)
    ax.text(-0.08, 0.15, 'Cooling', 
            transform=ax.transAxes, size=Fontsize, rotation=90)


    arrow = FancyArrowPatch((xlag, 0.45), (decayValue+xlag, 0.45),
                            arrowstyle='<|-|>', 
                            mutation_scale=20, 
                            linestyle='--',
                            color='black')
    ax.add_patch(arrow)

    ax.text(0.25, 0.88, r'Storage timescale', 
            transform=ax.transAxes, size=Fontsize, 
            horizontalalignment='center', verticalalignment='center')

# Constant
ax = axs[3] 
if True:
    dt = 0.1
    xlag = 5
    decayValue = 100
    tmax = decayValue + 100 
    tlimit = tmax + xlag + 10
    tt = np.arange(0, tmax + dt, dt)
    Fontsize=14
    scalefactor = 1E13

    tCDR = 1E2 * scalefactor * AGTPPRF_Constant_partial(t=tt, decay=100)
    pCDR = -1E2 *scalefactor * AGTPCO2_partial(t=tt)

    ax.plot(tt + xlag, tCDR,linewidth=2, color=color3)
    ax.plot(tt + xlag, pCDR, linewidth=2, color='k')

    ax.set_xlim([0,tlimit])
    ax.set_ylim([-0.7,0.7])
    ax.plot([decayValue + xlag, decayValue + xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)
    ax.plot([xlag, xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)

    arrowed_spines(fig, ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(-0.1, 1.02, 'c', transform=ax.transAxes,size=16, weight='bold')  
    ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes,size=Fontsize)    
    ax.text(0.95, 0.42, 'Year', transform=ax.transAxes,size=Fontsize)  
    ax.text(0.3, 0.35, r'tCDR ($\tau$ = 100 years)', transform=ax.transAxes,size=Fontsize, color=color3)  
    ax.text(0.5, 0.15, r'pCDR ($\tau \; \to \; \infty$)', transform=ax.transAxes,size=Fontsize, color='k')  
    ax.text(0.05, 0.53, 't = 0', transform=ax.transAxes,  size=Fontsize, color='k')  
    ax.text(-0.08, 0.65, 'Warming', 
            transform=ax.transAxes, size=Fontsize, rotation=90)
    ax.text(-0.08, 0.15, 'Cooling', 
            transform=ax.transAxes, size=Fontsize, rotation=90)


    arrow = FancyArrowPatch((xlag, 0.45), (decayValue+xlag, 0.45),
                            arrowstyle='<|-|>', 
                            mutation_scale=20, 
                            linestyle='--',
                            color='black')
    ax.add_patch(arrow)

    ax.text(0.25, 0.88, r'Storage timescale', 
            transform=ax.transAxes, size=Fontsize, 
            horizontalalignment='center', verticalalignment='center')

# Impluse
ax = axs[4] 
if True:
    dt = 0.1
    xlag = 5
    decayValue = 100
    tmax = decayValue + 100 
    tlimit = tmax + xlag + 10
    tt = np.arange(0, tmax + dt, dt)
    Fontsize=14
    scalefactor = 1E13

    tCDR = 1E2 * scalefactor * AGTPPRF_Impulse_partial(t=tt, decay=100)
    pCDR = -1E2 *scalefactor * AGTPCO2_partial(t=tt)

    ax.plot(tt + xlag, tCDR,linewidth=2, color=color3)
    ax.plot(tt + xlag, pCDR, linewidth=2, color='k')

    ax.set_xlim([0,tlimit])
    ax.set_ylim([-0.7,0.7])
    ax.plot([decayValue + xlag, decayValue + xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)
    ax.plot([xlag, xlag],[-0.55,0.55], linestyle='--', color='gray', linewidth=2)

    arrowed_spines(fig, ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.text(-0.1, 1.02, 'd', transform=ax.transAxes,size=16, weight='bold')  
    ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes,size=Fontsize)    
    ax.text(0.95, 0.42, 'Year', transform=ax.transAxes,size=Fontsize)  
    ax.text(0.3, 0.35, r'tCDR ($\tau$ = 100 years)', transform=ax.transAxes,size=Fontsize, color=color3)  
    ax.text(0.5, 0.15, r'pCDR ($\tau \; \to \; \infty$)', transform=ax.transAxes,size=Fontsize, color='k')  
    ax.text(0.05, 0.53, 't = 0', transform=ax.transAxes,  size=Fontsize, color='k')  
    ax.text(-0.08, 0.65, 'Warming', 
            transform=ax.transAxes, size=Fontsize, rotation=90)
    ax.text(-0.08, 0.15, 'Cooling', 
            transform=ax.transAxes, size=Fontsize, rotation=90)


    arrow = FancyArrowPatch((xlag, 0.45), (decayValue+xlag, 0.45),
                            arrowstyle='<|-|>', 
                            mutation_scale=20, 
                            linestyle='--',
                            color='black')
    ax.add_patch(arrow)

    ax.text(0.25, 0.88, r'Storage timescale', 
            transform=ax.transAxes, size=Fontsize, 
            horizontalalignment='center', verticalalignment='center')

# delete axs 
for i in [2,5]:
        fig.delaxes(axs[i])  

plt.savefig('figure\Figure S2.png', bbox_inches='tight', pad_inches=0.1)
