
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


# tCDR vs NonCO2, F(t, alpha, decay, AANonCO2, tauNonCO2)
def NetAGTP_Exp(t, alpha, decay, AANonCO2,tauNonCO2):
    return AGTPNonCO2_Final_partial(t, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2) + alpha * AGTPPRF_Exp_partial(t=t, decay=decay)

# tCDR vs CO2, F(t, alpha, decay)
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

# iAGTP
def plot_alpha_ratio_iAGTP(gas_name, AGTPPRF_Exp_partial, AGTPNonCO2_Final_partial, params, IfCO2=None, ax=None, fmt = '%1.0f'):

    # example: plot_alpha_ratio_iAGTP('CH4', AGTPPRF_Exp_partial, AGTPNonCO2_Final_partial, gas_params, IfCO2=None, ax=None)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  

    # Retrieve parameters from the dictionary
    Amountmax = params[gas_name]['Amountmax']
    dAmount = params[gas_name]['dAmount']
    levels_lines = params[gas_name]['levels_lines']
    levels_filled = np.linspace(0, Amountmax, 50)

    # X：time horizon
    tmax = 1000  
    dt = 1
    t_values = np.arange(1,tmax+dt,dt)

    # Y：storage timescale
    taumax = 1000       
    dtau = 1
    tau_values = np.arange(1,taumax+dtau,dtau)

    # Prepare a 2D grid 
    X, Y = np.meshgrid(t_values,tau_values)
    Z = np.zeros(Y.shape)

    # GHG Emission
    if IfCO2 is None:
        tauNonCO2 = params[gas_name]['tauNonCO2']
        AANonCO2 = params[gas_name]['AANonCO2']
        Emission = AGTPNonCO2_Final_partial(t=t_values, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2)  

    else:
        Emission = AGTPNonCO2_Final_partial(t=t_values) 


    for i, tau in enumerate(tau_values):
        Capture = AGTPPRF_Exp_partial(t=t_values, decay=tau)
        Z[i,:] = -1 * np.cumsum(Emission) /  np.cumsum(Capture)

    font_size = 12
    contour_line_color = 'k' 
    original_cmap = plt.cm.YlGnBu_r

    original_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=original_cmap.name, a=0.15, b=0.95),
        original_cmap(np.linspace(0.15, 0.95, 256))
    )

    # main results
    contour = ax.contour(X, Y, Z, levels=levels_lines, colors=contour_line_color, linewidths=0.5)
    contour_filled = ax.contourf(X, Y, Z, levels=levels_filled, cmap=original_cmap, extend='max')

    # colorbar and label
    cbar = fig.colorbar(contour_filled, ax=ax, ticks=np.arange(0, Amountmax+1, dAmount), extend='max')  
    cbar.ax.tick_params(labelsize=10)

    # labels and titles
    ax.set_ylabel(r'Storage Timescale ($\tau$, years)', fontsize=font_size)
    ax.set_xlabel(r'Time Horizon (TH, years)', fontsize=font_size)


    tau_value = params[gas_name]["tauNonCO2"]

    if isinstance(tau_value, (list, tuple)):
        tau_str = ", ".join(map(str, tau_value)) 
    else:
        tau_str = str(tau_value)  

    title_line1 = r'kg CO$_2$ of tCDR to offset 1 kg {}'.format(params[gas_name]["display_name"])
    title_line2 = r'(lifetime: {} years)'.format(tau_str)

    ax.set_title(
        f"{title_line1}\n{title_line2}", 
        fontsize=font_size
    )


    # linear + log scale
    ax.set_xscale('symlog', linthresh=100, linscale=1)
    ax.set_yscale('symlog', linthresh=100, linscale=1)

    # Linear interval
    ax.plot([0, 100], [100, 100], linestyle='--', color='gray')
    ax.plot([100,100], [0, 100], linestyle='--', color='gray')

    linear_ticks = [0, 50, 100]  
    log_ticks = [200, 500, 1000] 
    major_ticks = linear_ticks + log_ticks
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    major_tick_labels = ['0', '50', '100', '200', '500', '1000']
    ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
    ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))

    # subticks
    subs = np.arange(2, 10) 
    log_locator = SymmetricalLogLocator(linthresh=100, base=10, subs=subs)
    ax.xaxis.set_minor_locator(log_locator)
    ax.yaxis.set_minor_locator(log_locator)

    ax.xaxis.set_minor_formatter(FixedFormatter([]))
    ax.yaxis.set_minor_formatter(FixedFormatter([]))

    ax.set_xlim([0,1000])
    ax.set_ylim([0,1000])

    # position of contour line labels
    linear_thresh = 40
    xmin, xmax, ymin, ymax = ax.axis()
    mid = (xmin + xmax) / 2, (ymin + ymax) / 2
    label_positions = {}
    labeled_levels = set()

    mid_log = (np.log10(xmax) + np.log10(max(xmin, linear_thresh))) / 2, \
            (np.log10(ymax) + np.log10(max(ymin, linear_thresh))) / 2

    for collection, level in zip(contour.collections, contour.levels):
        paths = collection.get_paths() 
        if not paths or level in labeled_levels:
            continue
        
        longest_path = max(paths, key=len)
        verts = longest_path.vertices
        is_log_scale = verts[:, 0] > linear_thresh

        if np.any(is_log_scale):
            verts_log = np.log10(verts[is_log_scale])
            dist = np.linalg.norm(verts_log - mid_log, axis=1)
            min_ind = np.argmin(dist)
            label_positions[level] = 10 ** verts_log[min_ind]
        else:
            dist = np.linalg.norm(verts - mid, axis=1)
            min_ind = np.argmin(dist)
            label_positions[level] = verts[min_ind]

        labeled_levels.add(level)  

    for level, position in label_positions.items():
        try:
            labels=ax.clabel(contour, levels=[level], inline=True, fontsize=9, fmt=fmt, manual=[position])
        except ValueError:
            pass

    return Z

# AGWP
def plot_alpha_ratio_AGWP(gas_name, AGWPPRF_Exp_partial, AGWPNonCO2_Final_partial, params, IfCO2=None, ax=None, fmt = '%1.0f'):

    # example: plot_alpha_ratio_AGWP('CH4', AGWPPRF_Exp_partial, AGWPNonCO2_Final_partial, gas_params, IfCO2=None, ax=None)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  

    # Retrieve parameters from the dictionary
    Amountmax = params[gas_name]['Amountmax']
    dAmount = params[gas_name]['dAmount']
    levels_lines = params[gas_name]['levels_lines']
    levels_filled = np.linspace(0, Amountmax, 50)

    # X：time horizon
    tmax = 1000  
    dt = 1
    t_values = np.arange(1,tmax+dt,dt)

    # Y：storage timescale
    taumax = 1000       
    dtau = 1
    tau_values = np.arange(1,taumax+dtau,dtau)

    # Prepare a 2D grid 
    X, Y = np.meshgrid(t_values,tau_values)
    Z = np.zeros(Y.shape)


    # GHG Emission
    if IfCO2 is None:
        tauNonCO2 = params[gas_name]['tauNonCO2']
        AANonCO2 = params[gas_name]['AANonCO2']
        Emission = AGWPNonCO2_Final_partial(t=t_values, AANonCO2=AANonCO2, tauNonCO2=tauNonCO2)  

    else:
        Emission = AGWPNonCO2_Final_partial(t=t_values) 

        
    for i, tau in enumerate(tau_values):
        Capture = AGWPPRF_Exp_partial(t=t_values, decay=tau)
        Z[i,:] = -1 * Emission /  Capture

    font_size = 12
    contour_line_color = 'k' 
    original_cmap = plt.cm.YlGnBu_r

    # Create a new colormap from the original one, using only the range from 0.15 to 0.95
    original_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=original_cmap.name, a=0.15, b=0.95),
        original_cmap(np.linspace(0.15, 0.95, 256))
    )

    # main results
    contour = ax.contour(X, Y, Z, levels=levels_lines, colors=contour_line_color, linewidths=0.5)
    contour_filled = ax.contourf(X, Y, Z, levels=levels_filled, cmap=original_cmap, extend='max')

    # colorbar and label
    cbar = fig.colorbar(contour_filled, ax=ax, ticks=np.arange(0, Amountmax+1, dAmount), extend='max')  
    cbar.ax.tick_params(labelsize=10)

    # labels and titles
    ax.set_ylabel(r'Storage Timescale ($\tau$, years)', fontsize=font_size)
    ax.set_xlabel(r'Time Horizon (TH, years)', fontsize=font_size)


    tau_value = params[gas_name]["tauNonCO2"]

    if isinstance(tau_value, (list, tuple)):
        tau_str = ", ".join(map(str, tau_value))  
    else:
        tau_str = str(tau_value)  
    title_line1 = r'kg CO$_2$ of tCDR to offset 1 kg {}'.format(params[gas_name]["display_name"])
    title_line2 = r'(lifetime: {} years)'.format(tau_str)

    ax.set_title(
        f"{title_line1}\n{title_line2}", 
        fontsize=font_size
    )

    # linear + log scale
    ax.set_xscale('symlog', linthresh=100, linscale=1)
    ax.set_yscale('symlog', linthresh=100, linscale=1)

    # Linear interval
    ax.plot([0, 100], [100, 100], linestyle='--', color='gray')
    ax.plot([100,100], [0, 100], linestyle='--', color='gray')

    linear_ticks = [0, 50, 100]  
    log_ticks = [200, 500, 1000] 
    major_ticks = linear_ticks + log_ticks
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    major_tick_labels = ['0', '50', '100', '200', '500', '1000']
    ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
    ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))

    # subticks
    subs = np.arange(2, 10) 
    log_locator = SymmetricalLogLocator(linthresh=100, base=10, subs=subs)
    ax.xaxis.set_minor_locator(log_locator)
    ax.yaxis.set_minor_locator(log_locator)

    ax.xaxis.set_minor_formatter(FixedFormatter([]))
    ax.yaxis.set_minor_formatter(FixedFormatter([]))

    ax.set_xlim([0,1000])
    ax.set_ylim([0,1000])

    # position of contour line labels
    linear_thresh = 40
    xmin, xmax, ymin, ymax = ax.axis()
    mid = (xmin + xmax) / 2, (ymin + ymax) / 2
    label_positions = {}
    labeled_levels = set()

    mid_log = (np.log10(xmax) + np.log10(max(xmin, linear_thresh))) / 2, \
            (np.log10(ymax) + np.log10(max(ymin, linear_thresh))) / 2

    for collection, level in zip(contour.collections, contour.levels):
        paths = collection.get_paths() 
        if not paths or level in labeled_levels:
            continue
        
        longest_path = max(paths, key=len)
        verts = longest_path.vertices
        is_log_scale = verts[:, 0] > linear_thresh

        if np.any(is_log_scale):
            verts_log = np.log10(verts[is_log_scale])
            dist = np.linalg.norm(verts_log - mid_log, axis=1)
            min_ind = np.argmin(dist)
            label_positions[level] = 10 ** verts_log[min_ind]
        else:
            dist = np.linalg.norm(verts - mid, axis=1)
            min_ind = np.argmin(dist)
            label_positions[level] = verts[min_ind]

        labeled_levels.add(level)  

    for level, position in label_positions.items():
        try:
            labels=ax.clabel(contour, levels=[level], inline=True, fontsize=9, fmt=fmt, manual=[position])
        except ValueError:
            pass

    return Z


#==============================================================================
# Figure 2 contour plot (iAGTP)
#==============================================================================

# parameters
gas_params = {
    'BC': {'display_name': 'BC', 'tauNonCO2': tauBC, 'AANonCO2': AABC, 
           'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'HFC32': {'display_name': 'HFC-32', 'tauNonCO2': tauHFC32, 'AANonCO2': AAHFC32, 
              'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'CH4': {'display_name': r'CH$_4$', 'tauNonCO2': tauCH4, 'AANonCO2': AACH4, 
            'Amountmax': 300, 'dAmount': 50, 'levels_lines': [10,20,30,40,60,80,100,150, 200, 300, 500, 2000]},
    'HFC134a': {'display_name': 'HFC-134a', 'tauNonCO2': tauHFC134a, 'AANonCO2': AAHFC134a, 
                'Amountmax': 10000, 'dAmount': 2000, 'levels_lines': [100,500,1000,1500,2000,3000,4000,5000,8000,14000,20000,50000]},
    'CFC11': {'display_name': 'CFC-11', 'tauNonCO2': tauCFC11, 'AANonCO2': AACFC11, 
              'Amountmax': 30000, 'dAmount': 5000, 'levels_lines': [100,500,1000,3000,5000,8000,10000,15000,20000,30000,50000,80000,150000,300000]},
    'N2O': {'display_name': r'N$_2$O', 'tauNonCO2': tauN2O, 'AANonCO2': AAN2O, 
            'Amountmax': 1500, 'dAmount': 300,  'levels_lines': [100,200,300,400,500,600,800,1000,1500,2000,3000,5000,10000,20000]},
    'PFC14': {'display_name': 'PFC-14', 'tauNonCO2': tauPFC14, 'AANonCO2': AAPFC14, 
              'Amountmax': 100000, 'dAmount': 20000, 'levels_lines': [1000,2000,4000,6000, 8000,10000,15000,20000,30000, 60000,150000,300000,600000,1000000]},
    'CO2': {'display_name': r'CO$_2$', 'tauNonCO2': 'multiple', 'AANonCO2': 0, 
            'Amountmax': 20, 'dAmount': 5, 'levels_lines': [1, 1.2, 1.5, 2, 3,5,10,20,30,40,60]},
}


fig, axs = plt.subplots(2, 4, figsize=(26, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.4)

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  

for i, (gas_name, param_dict) in enumerate(gas_params.items()):
    ax = axs[i]

    if i < 7:
        plot_alpha_ratio_iAGTP(gas_name, AGTPPRF_Exp_partial, AGTPNonCO2_Final_partial, params=gas_params, IfCO2=None, ax=ax, fmt = '%1.0f')
    elif i == 7:
        plot_alpha_ratio_iAGTP('CO2', AGTPPRF_Exp_partial, AGTPCO2_partial, params=gas_params, IfCO2=1, ax=ax, fmt = '%1.1f')
    
    # at the top left corner of each subplot, add a label
    ax.text(-0.23, 1.08, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')

fig.savefig('figure\Figure 2.png', bbox_inches='tight', pad_inches=0)


#==============================================================================
# Figure S4 contour plot (AGWP)
#==============================================================================

# parameters
gas_params = {
    'BC': {'display_name': 'BC', 'tauNonCO2': tauBC, 'AANonCO2': AABC, 
           'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'HFC32': {'display_name': 'HFC-32', 'tauNonCO2': tauHFC32, 'AANonCO2': AAHFC32, 
              'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'CH4': {'display_name': r'CH$_4$', 'tauNonCO2': tauCH4, 'AANonCO2': AACH4, 
            'Amountmax': 300, 'dAmount': 50, 'levels_lines': [10,20,30,40,60,80,100,150, 200, 300, 500, 2000]},
    'HFC134a': {'display_name': 'HFC-134a', 'tauNonCO2': tauHFC134a, 'AANonCO2': AAHFC134a, 
                'Amountmax': 10000, 'dAmount': 2000, 'levels_lines': [100,500,1000,1500,2000,3000,4000,5000,8000,14000,20000,50000]},
    'CFC11': {'display_name': 'CFC-11', 'tauNonCO2': tauCFC11, 'AANonCO2': AACFC11, 
              'Amountmax': 30000, 'dAmount': 5000, 'levels_lines': [100,500,1000,3000,5000,8000,10000,15000,20000,30000,50000,80000,150000,300000]},
    'N2O': {'display_name': r'N$_2$O', 'tauNonCO2': tauN2O, 'AANonCO2': AAN2O, 
            'Amountmax': 1500, 'dAmount': 300,  'levels_lines': [100,200,300,400,500,600,800,1000,1500,2000,3000,5000,10000,20000]},
    'PFC14': {'display_name': 'PFC-14', 'tauNonCO2': tauPFC14, 'AANonCO2': AAPFC14, 
              'Amountmax': 100000, 'dAmount': 20000, 'levels_lines': [1000,2000,4000,6000, 8000,10000,15000,20000,30000, 60000,150000,300000,600000,1000000]},
    'CO2': {'display_name': r'CO$_2$', 'tauNonCO2': 'multiple', 'AANonCO2': 0, 
            'Amountmax': 20, 'dAmount': 5, 'levels_lines': [1, 1.2, 1.5, 2, 3,5,10,20,30,40,60]},
}


fig, axs = plt.subplots(2, 4, figsize=(26, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.4)

panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] 

for i, (gas_name, param_dict) in enumerate(gas_params.items()):
    ax = axs[i]

    if i < 7:
        plot_alpha_ratio_AGWP(gas_name, AGWPPRF_Exp_partial, AGWPNonCO2_Final_partial, params=gas_params, IfCO2=None, ax=ax, fmt = '%1.0f')
    elif i == 7:
        plot_alpha_ratio_AGWP('CO2', AGWPPRF_Exp_partial, AGWPCO2_partial, params=gas_params, IfCO2=1, ax=ax, fmt = '%1.1f')
    
    # at the top left corner of each subplot, add a label
    ax.text(-0.23, 1.08, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')

fig.savefig('figure\Figure S4.png', bbox_inches='tight', pad_inches=0)



#==============================================================================
# Figure S6 contour plot (iAGTP, CH4 different sources)
#==============================================================================

# parameters
gas_params = {
    'BC': {'display_name': 'BC', 'tauNonCO2': tauBC, 'AANonCO2': AABC, 
           'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'HFC32': {'display_name': 'HFC-32', 'tauNonCO2': tauHFC32, 'AANonCO2': AAHFC32, 
              'Amountmax': 5000, 'dAmount': 1000, 'levels_lines': [10,20,30,40,50,100,200,500, 800, 1000,1500,2000,3000,4000,6000,10000,20000]},
    'CH4': {'display_name': r'CH$_4$', 'tauNonCO2': tauCH4, 'AANonCO2': AACH4, 
            'Amountmax': 300, 'dAmount': 50, 'levels_lines': [10,20,30,40,60,80,100,150, 200, 300, 500, 2000]},
    'HFC134a': {'display_name': 'HFC-134a', 'tauNonCO2': tauHFC134a, 'AANonCO2': AAHFC134a, 
                'Amountmax': 10000, 'dAmount': 2000, 'levels_lines': [100,500,1000,1500,2000,3000,4000,5000,8000,14000,20000,50000]},
    'CFC11': {'display_name': 'CFC-11', 'tauNonCO2': tauCFC11, 'AANonCO2': AACFC11, 
              'Amountmax': 30000, 'dAmount': 5000, 'levels_lines': [100,500,1000,3000,5000,8000,10000,15000,20000,30000,50000,80000,150000,300000]},
    'N2O': {'display_name': r'N$_2$O', 'tauNonCO2': tauN2O, 'AANonCO2': AAN2O, 
            'Amountmax': 1500, 'dAmount': 300,  'levels_lines': [100,200,300,400,500,600,800,1000,1500,2000,3000,5000,10000,20000]},
    'PFC14': {'display_name': 'PFC-14', 'tauNonCO2': tauPFC14, 'AANonCO2': AAPFC14, 
              'Amountmax': 100000, 'dAmount': 20000, 'levels_lines': [1000,2000,4000,6000, 8000,10000,15000,20000,30000, 60000,150000,300000,600000,1000000]},
    'CO2': {'display_name': r'CO$_2$', 'tauNonCO2': 0, 'AANonCO2': 0, 
            'Amountmax': 20, 'dAmount': 5, 'levels_lines': [1, 1.2, 1.5, 2, 3,5,10,20,30,40,60]},
}


panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',] 

fig, axs = plt.subplots(2, 4, figsize=(26, 10))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.4)

for i in np.arange(0,8):

    
    if i == 0:
        ax = axs[i]
        plot_alpha_ratio_iAGTP('CH4', AGTPPRF_Exp_partial, AGTPCH4NonFossil_Final_partial, params=gas_params, IfCO2=None, ax=ax, fmt = '%1.0f')

        title_line1 = r'kg CO$_2$ of tCDR to offset 1 kg CH$_4$ (biogenic)'
        title_line2 = r'(lifetime: 11.8 years)'

        ax.set_title(
            f"{title_line1}\n{title_line2}",  
            fontsize=12
        )
        ax.text(-0.23, 1.08, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')
    
    if i == 1:
        ax = axs[i]
        plot_alpha_ratio_iAGTP('CH4', AGTPPRF_Exp_partial, AGTPCH4Fossil_Final_partial, params=gas_params, IfCO2=None, ax=ax, fmt = '%1.0f')
  
        title_line1 = r'kg CO$_2$ of tCDR to offset 1 kg CH$_4$ (fossil)'
        title_line2 = r'(lifetime: 11.8 years)'

        ax.set_title(
            f"{title_line1}\n{title_line2}",  
            fontsize=12
        )
        ax.text(-0.23, 1.08, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom', ha='left')

    elif i > 1:
        axs[i].remove()

fig.savefig('figure\Figure S6.png', bbox_inches='tight', pad_inches=0)


