
import os
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

import cmocean
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ConnectionPatch, PathPatch
from matplotlib.path import Path

# key functions and parameters
from IRF_functions import *
from IRF_parameters import *

dir_path = os.path.dirname(os.path.abspath(__file__)) 
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


# tCDR vs NonCO2, F(t, alpha, decay, AANonCO2, tauNonCO2)
def NetAGTP_Exp(t, alpha, decay, AANonCO2,tauNonCO2):
    return AGTPNonCO2_Final_partial(t, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)

# tCDR vs CO2, F(t, alpha, decay)
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
# Figure 1 - Conceptual framework of temporary CDR
#==============================================================================

# Color definitions
color1 = convert_rgb_to_01((204, 0, 0))    # Red for CH4
color2 = convert_rgb_to_01((230, 145, 56)) # Orange for N2O
color3 = convert_rgb_to_01((61, 133, 198)) # Blue for tCDR

# Parameters
dt = 0.1
xlag = 5
tmax = 100
decayValue = 100
tlimit = tmax + xlag + 10
tt = np.arange(0, tmax + dt, dt)
Fontsize = 14
scalefactor = 1E13

# Create figure with 2 rows × 4 columns
fig, axs = plt.subplots(2, 4, figsize=(25, 12))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.3, hspace=0.25)  # 调整垂直间距

# Common x-axis tick settings
time_ticks = np.arange(0, tmax + 1, 20)
x_positions = time_ticks + xlag


#==============================================================================
# Panel a: GHG pulse emission
#==============================================================================
ax = axs[0]
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.plot([xlag, xlag], [0, 1], color='k', linewidth=2)
ax.plot([xlag, xlag], [1, 1], 'ko', linewidth=2)

ax.set_xlim([0, tlimit])
ax.set_ylim([-1.3, 1.3])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

ax.set_xticks(x_positions)
ax.set_yticks([-1, 0, 1])
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)
ax.set_yticklabels([-1, 0, 1], fontsize=Fontsize)


ax.text(-0.1, 1.02, 'a', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, 'Emissions', transform=ax.transAxes, size=Fontsize)
ax.text(xlag + 5, 1, r'CH$_4$ or N$_2$O pulse', color='k', size=Fontsize)
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.6, 'Positive', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.2, 'Negative', transform=ax.transAxes, size=Fontsize, rotation=90)


#==============================================================================
# Panel b: Atmoshperic concentration 
#==============================================================================
ax = axs[1]
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.plot(tt + xlag, IRFNonCO2(t=tt, tauNonCO2=tauCH4), 
        linewidth=2, color=color1)
ax.plot(tt + xlag, IRFNonCO2(t=tt, tauNonCO2=tauN2O), 
        linewidth=2, color=color2)

ax.set_xlim([0, tlimit])
ax.set_ylim([-1.3, 1.3])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

ax.set_xticks(x_positions)
ax.set_yticks([-1, 0, 1])
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)
ax.set_yticklabels([-1, 0, 1], fontsize=Fontsize)

ax.text(-0.1, 1.02, 'b', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, r'Atmospheric fraction', transform=ax.transAxes, size=Fontsize)
ax.text(0.2, 0.6, r'CH$_4$ (11.8 years)', transform=ax.transAxes, size=Fontsize, color=color1)
ax.text(0.4, 0.8, r'N$_2$O (109 years)', transform=ax.transAxes, size=Fontsize, color=color2)
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.6, 'Positive', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.2, 'Negative', transform=ax.transAxes, size=Fontsize, rotation=90)


#==============================================================================
# Panel c: Temperature response (AGTP)
#==============================================================================
ax = axs[2]
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

ax.plot(tt + xlag, scalefactor * AGTPNonCO2_Final_partial(t=tt, AANonCO2=AACH4, tauNonCO2=tauCH4), 
        linewidth=2, color=color1, linestyle='-')
ax.plot(tt + xlag, 0.1 * scalefactor * AGTPNonCO2_Final_partial(t=tt, AANonCO2=AAN2O, tauNonCO2=tauN2O), 
        linewidth=2, color=color2, linestyle='-')

ax.set_xlim([0, tlimit])
ax.set_ylim([-0.7, 0.7])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(False)

ax.set_xticks(x_positions)
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)


ax.text(-0.1, 1.02, 'c', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes, size=Fontsize)
ax.text(0.2, 0.8, r'CH$_4$ (11.8 years)', transform=ax.transAxes, size=Fontsize, color=color1)
ax.text(0.4, 0.62, r'0.1 $\times$ N$_2$O (109 years)', transform=ax.transAxes, size=Fontsize, color=color2)
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.6, 'Warming', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.2, 'Cooling', transform=ax.transAxes, size=Fontsize, rotation=90)


#==============================================================================
# Delete panel (top right corner)
#==============================================================================
fig.delaxes(axs[3])


#==============================================================================
# Panel d: CDR
#==============================================================================
ax = axs[4]
ax.spines['bottom'].set_position(('data', 0.18))
ax.spines['left'].set_position(('data', 0))

ax.plot([xlag, xlag], [0, -1], color='k', linewidth=2)
ax.plot([xlag, xlag], [-1, -1], 'ko', linewidth=2)
ax.plot(np.arange(dt, decayValue + dt, dt) + xlag, 
        10 * F_Exp(np.arange(dt, decayValue + dt, dt), decay=decayValue), 
        color=color3, linewidth=2)
ax.plot([decayValue + xlag, decayValue + xlag], [-1, 1], 
        linestyle='--', color='k', linewidth=2)

ax.set_xlim([0, tlimit])
ax.set_ylim([-1.3, 1.3])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

ax.set_xticks(x_positions)
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([-1,0,1], fontsize=Fontsize)  


ax.text(-0.1, 1.02, 'd', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, 'Emissions', transform=ax.transAxes, size=Fontsize)
ax.text(0.3, 0.65, 'tCDR only (release)', color=color3, 
        transform=ax.transAxes, size=Fontsize, ha='center', va='center')
ax.text(0.33, 0.06, 'tCDR or pCDR (capture)', 
        transform=ax.transAxes, size=Fontsize, ha='center', va='center')
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.6, 'Positive', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.2, 'Negative', transform=ax.transAxes, size=Fontsize, rotation=90)

arrow = FancyArrowPatch((xlag, -0.8), (tmax + xlag, -0.8),
                        arrowstyle='<|-|>', mutation_scale=20, 
                        linestyle='--', color='black')
ax.add_patch(arrow)
ax.text(0.5, 0.24, r'Storage timescale ($\tau$)', 
        transform=ax.transAxes, size=Fontsize, ha='center', va='center')


#==============================================================================
# Panel e: Atmospheric fraction
#==============================================================================
ax = axs[5]
ax.spines['bottom'].set_position(('data', 0.18))
ax.spines['left'].set_position(('data', 0))

ax.plot(tt + xlag, PRF_Exp(tt, 100, aC1, aC2, aC3, aC4, tauC1, tauC2, tauC3), 
        linewidth=2, color=color3)
ax.plot(tt + xlag, -1 * IRFCO2(tt, aC1, aC2, aC3, aC4, tauC1, tauC2, tauC3),
        linewidth=2, color='k')

ax.set_xlim([0, tlimit])
ax.set_ylim([-1.3, 1.3])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

ax.set_xticks(x_positions)
ax.set_yticks([-1, 0, 1])
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)
ax.set_yticklabels([-1, 0, 1], fontsize=Fontsize)

ax.text(-0.1, 1.02, 'e', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, r'Atmospheric fraction', transform=ax.transAxes, size=Fontsize)
ax.text(0.35, 0.38, r'tCDR ($\tau$ = 100 years)', 
        transform=ax.transAxes, size=Fontsize, color=color3)
ax.text(0.5, 0.25, r'pCDR ($\tau \to \infty$)', 
        transform=ax.transAxes, size=Fontsize, color='k')
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.6, 'Positive', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.2, 'Negative', transform=ax.transAxes, size=Fontsize, rotation=90)


#==============================================================================
# Panel f: Temperature response (AGTP)
#==============================================================================
ax = axs[6]
ax.spines['bottom'].set_position(('data', 0.1))
ax.spines['left'].set_position(('data', 0))

ax.plot(tt + xlag, 1E2 * scalefactor * AGTPPRF_Exp_partial(t=tt, decay=50), 
        linewidth=2, color=color3, linestyle='-')
ax.plot(tt + xlag, -1E2 * scalefactor * AGTPCO2_partial(t=tt), 
        linewidth=2, color='k')

ax.set_xlim([0, tlimit])
ax.set_ylim([-0.7, 0.7])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(False)

ax.set_xticks(x_positions)
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)


ax.text(-0.1, 1.02, 'f', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, r'Temperature response (AGTP)', transform=ax.transAxes, size=Fontsize)
ax.text(0.3, 0.38, r'tCDR ($\tau$ = 100 years)', 
        transform=ax.transAxes, size=Fontsize, color=color3)
ax.text(0.4, 0.1, r'pCDR ($\tau \to \infty$)', 
        transform=ax.transAxes, size=Fontsize, color='k')
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.65, 'Warming', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.15, 'Cooling', transform=ax.transAxes, size=Fontsize, rotation=90)


#==============================================================================
# Panel g: Temperature response (AGTP)
#==============================================================================
pos_ax = axs[7].get_position()
axs[7].set_position([pos_ax.x0, pos_ax.y0 + pos_ax.height * 0.55, pos_ax.width, pos_ax.height])
ax = axs[7]

ax.spines['bottom'].set_position(('data', -0.01))
ax.spines['left'].set_position(('data', 0))

alphas = 120
decays = 82.6

xx = tt + xlag
y1 = scalefactor * AGTPNonCO2_Final_partial(t=tt, AANonCO2=AACH4, tauNonCO2=tauCH4)
y2 = scalefactor * alphas * AGTPPRF_Exp_partial(t=tt, decay=decays)
y3 = scalefactor * NetAGTP_Exp(tt, alpha=alphas, decay=decays, AANonCO2=AACH4, tauNonCO2=tauCH4)

path1 = Path(np.array([xx, y1]).transpose())
patch1 = PathPatch(path1, facecolor='none')
ax.add_patch(patch1)

path2 = Path(np.array([xx, y2]).transpose())
patch2 = PathPatch(path2, facecolor='none')
ax.add_patch(patch2)

ax.plot(xx, y1, color=color1, linewidth=2)
ax.plot(xx, y2, color=color3, linewidth=2)
ax.plot(xx, y3, color='gray', linestyle='-', linewidth=2)

ax.fill_between(xx, y1, color='none', edgecolor=color1, alpha=0.4, hatch='//')
ax.fill_between(xx, y2, color='none', edgecolor=color3, alpha=0.4, hatch='//')

ax.set_xlim([0, tlimit])
ax.set_ylim([-0.7, 0.7])
arrowed_spines(fig, ax)
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(False)

ax.set_xticks(x_positions)
ax.set_xticklabels(time_ticks.astype(int), fontsize=Fontsize)


ax.text(-0.1, 1.02, 'g', transform=ax.transAxes, size=16, weight='bold')
ax.text(0.05, 1.02, 'Temperature response (AGTP)', transform=ax.transAxes, size=Fontsize)
ax.text(0.98, 0.42, 'Year', transform=ax.transAxes, size=Fontsize)
ax.text(-0.08, 0.65, 'Warming', transform=ax.transAxes, size=Fontsize, rotation=90)
ax.text(-0.08, 0.15, 'Cooling', transform=ax.transAxes, size=Fontsize, rotation=90)

text1 = r'Cumulative warming due to 1 kg CH$_4$'
text2 = r'(iAGTP$_{\mathrm{CH}_4}$)'
text3 = r'Cumulative cooling due to $\alpha$ kg CO$_2$ of tCDR'
text4 = r'($\alpha \times$ iAGTP$_{\mathrm{tCDR}}$)'

ax.text(0.25, 0.7, text1, transform=ax.transAxes, size=Fontsize, 
        multialignment='center', color=color1)
ax.text(0.55, 0.62, text2, transform=ax.transAxes, size=Fontsize, 
        multialignment='center', color=color1)
ax.text(0.25, 0.25, text3, transform=ax.transAxes, size=Fontsize, 
        multialignment='center', color=color3)
ax.text(0.55, 0.17, text4, transform=ax.transAxes, size=Fontsize, 
        multialignment='center', color=color3)


#==============================================================================
# Connection arrows between panels
#==============================================================================
con_upper = ConnectionPatch(
    xyA=(2.1, 0.9), xyB=(0.5, 1.2),
    coordsA="axes fraction", coordsB="axes fraction",
    axesA=axs[1], axesB=axs[7],
    arrowstyle='->',
    color="black",
    connectionstyle="angle,angleA=180,angleB=-90,rad=0",
    mutation_scale=20,
    linewidth=1.5
)

con_lower = ConnectionPatch(
    xyA=(2.1, 0.1), xyB=(0.5, -0.1),
    coordsA="axes fraction", coordsB="axes fraction",
    axesA=axs[5], axesB=axs[7],
    arrowstyle='->',
    color="black",
    connectionstyle="angle,angleA=180,angleB=-90,rad=0",
    mutation_scale=20,
    linewidth=1.5
)

fig.add_artist(con_upper)
fig.add_artist(con_lower)


plt.savefig('figure/Figure 1.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.savefig('figure/Figure 1.pdf', bbox_inches='tight', pad_inches=0.1)