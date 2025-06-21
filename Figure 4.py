
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
# Figure 3 data source: FullCode Optimization CTC100=0.nb

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

# Figure 3c
def compute_optimal_data_quad_netzero(gas_name, params, NetAGTPFunction, IfCO2=None, TH=100):

    ScaleFactor2 = 1E12
    tauNonCO2 = params[gas_name]['tauNonCO2']
    AANonCO2 = params[gas_name]['AANonCO2']
    alphaNmax = params[gas_name]['alphaNmax']
    decayNmax = params[gas_name]['decayNmax']


    # GHG Emission
    if IfCO2 is None:
        
        # zero net warming
        def objective2(params):
            alpha, decay = params
            objective_value = np.abs(quad(lambda t: ScaleFactor2 *  NetAGTPFunction(t, alpha, decay, AANonCO2, tauNonCO2), 0, TH)[0])
            return objective_value
        
        initial_guess = [1, 1]
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0]},  # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[1]}   # decay > 0
        )
        result2 = minimize(objective2, initial_guess, method='SLSQP', constraints=constraints, options={'ftol': 1e-20})
        
        alpha_range = np.linspace(1, alphaNmax, 1000)
        decay_range = np.linspace(1, decayNmax, 1000)
        Z2 = quad_vec(lambda t: NetAGTPFunction(t, alpha_range, decay_range[:, np.newaxis], AANonCO2, tauNonCO2), 0, TH)[0]
    
    else:    
        # zero net warming
        def objective2(params):
            alpha, decay = params
            objective_value = np.abs(quad(lambda t: ScaleFactor2 *  NetAGTPFunction(t, alpha, decay), 0, TH)[0])
            return objective_value
        
        initial_guess = [1, 1]
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0]},  # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[1]}   # decay > 0
        )
        result2 = minimize(objective2, initial_guess, method='SLSQP', constraints=constraints, options={'ftol': 1e-20})
        
        alpha_range = np.linspace(1, alphaNmax, 1000)
        decay_range = np.linspace(1, decayNmax, 1000)
        Z2 = quad_vec(lambda t: NetAGTPFunction(t, alpha_range, decay_range[:, np.newaxis]), 0, TH)[0]
    
    # Save Z values to CSV
    csv_file_path1 = 'Z_values_TH' + str(TH) + '_' + NetAGTPFunction.__name__ + '_' + gas_name + '_netzero.csv'
    np.savetxt(dir_path + '\\data\\' + csv_file_path1, Z2, delimiter=',')
    print(csv_file_path1)

    return Z2, result2.x

# Figure 3d
def compute_optimal_data_quad_stability(gas_name, params, NetAGTPFunction, IfCO2=None, TH=100):

    ScaleFactor1 = 1E30
    tauNonCO2 = params[gas_name]['tauNonCO2']
    AANonCO2 = params[gas_name]['AANonCO2']
    alphaNmax = params[gas_name]['alphaNmax']
    decayNmax = params[gas_name]['decayNmax']

    # GHG Emission
    if IfCO2 is None:
        
        # climate stability
        def objective1(params):
            alpha, decay = params
            objective_value = quad(lambda t: ScaleFactor1 *  NetAGTPFunction(t, alpha, decay, AANonCO2, tauNonCO2)**2, 0, TH)[0]
            return objective_value

        initial_guess = [1, 1]
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0]},  # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[1]}   # decay > 0
        )
        result1 = minimize(objective1, initial_guess, method='SLSQP', constraints=constraints, options={'ftol': 1e-20})

        alpha_range = np.linspace(1, alphaNmax, 1000)
        decay_range = np.linspace(1, decayNmax, 1000)
        Z1 = quad_vec(lambda t: NetAGTPFunction(t, alpha_range, decay_range[:, np.newaxis], AANonCO2, tauNonCO2)**2, 0, TH)[0]

    else:
        # climate stability
        def objective1(params):
            alpha, decay = params
            objective_value = quad(lambda t: ScaleFactor1 *  NetAGTPFunction(t, alpha, decay)**2, 0, TH)[0]
            return objective_value
        
        initial_guess = [1, 1]
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[0]},  # alpha > 0
            {'type': 'ineq', 'fun': lambda x: x[1]}   # decay > 0
        )
        result1 = minimize(objective1, initial_guess, method='SLSQP', constraints=constraints, options={'ftol': 1e-20})
 
        alpha_range = np.linspace(1, alphaNmax, 1000)
        decay_range = np.linspace(1, decayNmax, 1000)
        Z1 = quad_vec(lambda t: NetAGTPFunction(t, alpha_range, decay_range[:, np.newaxis])**2, 0, TH)[0]

    # Save Z values to CSV
    csv_file_path1 = 'Z_values_TH' + str(TH) + '_' + NetAGTPFunction.__name__ + '_' + gas_name + '_stability.csv'
    np.savetxt(dir_path + '\\data\\' + csv_file_path1, Z1, delimiter=',')
    print(csv_file_path1)

    return Z1, result1.x

# Data multiplied by 1E30 and log10 applied to enhance colormap gradient
# 0-200 linear
# data source: compute_optimal_data_quad_stability
def plot_optimal_data_stability_200(gas_name, params, NetAGTPFunction, ax=None, TH=100):

    alphaNmax = params[gas_name]['alphaNmax']
    decayNmax = params[gas_name]['decayNmax']

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  

    alpha_range = np.linspace(0, alphaNmax, 1000)
    decay_range = np.linspace(0, decayNmax, 1000)
    X, Y = np.meshgrid(alpha_range, decay_range)

    
    # data source: def compute_optimal_data_quad_stability
    Z = np.genfromtxt(dir_path + '\\data\\' + 'Z_values_TH' + str(TH) + '_' + NetAGTPFunction + '_' + gas_name + '_stability.csv', delimiter=',')


    Zscaled=np.log10(Z*1E30)
    Zscaled=np.clip(Zscaled, 3.5, 7.5) 

    font_size = 12

    contour = ax.contour(X,Y,Zscaled, levels=[3,4,5,6,7,8], colors='k', linewidths=0.5)
    contour_filled = ax.contourf(X, Y, Zscaled, levels=np.linspace(3.5,7.5,100), cmap=cmocean.cm.ice, extend='max')

    cbar = fig.colorbar(contour_filled, ax=ax, ticks=np.arange(3.5,8,0.5), extend='max')  
    cbar_label = r'Relative levels'

    cbar.set_label(cbar_label, size=font_size)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r'The amount of tCDR ($\alpha$, kg CO$_2$ kg$^{-1}$ CH$_4$)', fontsize=font_size)
    ax.set_ylabel(r'Storage Timescale ($\tau$, years)', fontsize=font_size)


    ax.plot([200,200],[0,200],linestyle='--',color='gray')
    ax.plot([0,200],[200,200],linestyle='--',color='gray')

    ax.set_xscale('symlog', linthresh=200, linscale=1)
    ax.set_yscale('symlog', linthresh=200, linscale=1)

    if decayNmax == 1E6:
        linear_ticks = [100, 200]  
        log_ticks = [500, 1000, 1E4,1E5,1E6] 
        major_ticks = linear_ticks + log_ticks
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        major_tick_labels = ['100', '200', '500', '1000', '1E4','1E5','1E6']
        ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
        ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))

    else:
        linear_ticks = [0, 50, 100, 150, 200]  
        log_ticks = [500, 1000]  
        major_ticks = linear_ticks + log_ticks
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        major_tick_labels = ['0', '50', '100', '150', '200', '500', '1000']
        ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
        ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))


    subs = np.arange(2, 10) 
    log_locator = SymmetricalLogLocator(linthresh=200, base=10, subs=subs)

    ax.xaxis.set_minor_locator(log_locator)
    ax.yaxis.set_minor_locator(log_locator)

    ax.xaxis.set_minor_formatter(FixedFormatter([]))
    ax.yaxis.set_minor_formatter(FixedFormatter([]))

    ax.tick_params(axis='x', labelsize=10)  
    ax.tick_params(axis='y', labelsize=10)

    ax.set_title(r'Cumulative net temperature deviation', fontsize=font_size)

    ax.set_xlim([0,alphaNmax])
    ax.set_ylim([0,decayNmax])

    linear_thresh = 400
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
            labels=ax.clabel(contour, levels=[level], colors='k', inline=True, fontsize=9, fmt='%1.0f', manual=[position])
        except ValueError:
            pass

# Data multiplied by 1E12
# 0-200 linear
# data source: def compute_optimal_data_quad_netzero
def plot_optimal_data_netzero_200(gas_name, params, NetAGTPFunction, ax=None, TH=100):

    alphaNmax = params[gas_name]['alphaNmax']
    decayNmax = params[gas_name]['decayNmax']
    levels_lines = params[gas_name]['levels_lines']
    ticks = params[gas_name]['ticks']
    levels_filled = np.linspace(ticks[0],ticks[-1],1000)


    def adjust_amount(amount):
        return np.ceil(amount / 2) * 2 if amount % 2 == 1 else amount

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  


    alpha_range = np.linspace(1, alphaNmax, 1000)
    decay_range = np.linspace(1, decayNmax, 1000)
    X, Y = np.meshgrid(alpha_range, decay_range)
    
    # data source: def compute_optimal_data_quad_netzero
    Z = np.genfromtxt(dir_path + '\\data\\' + 'Z_values_TH' + str(TH) + '_' + NetAGTPFunction + '_' + gas_name + '_netzero.csv', delimiter=',')


    Zscaled = Z * 1E12
    font_size = 12

    # Round to the nearest integer, then adjust to the closest even number
    Amountmax = np.ceil(Zscaled.max())
    Amountmin = np.floor(Zscaled.min())
    Amountmax = adjust_amount(Amountmax)
    Amountmin = adjust_amount(Amountmin)

    class MidpointNormalize(mcolors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            mcolors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    if Amountmin > 0:
        midpoint = Amountmin
        contour = ax.contour(X,Y,Zscaled,levels=levels_lines, colors='k', linewidths=0.5)
        contour_filled = ax.contourf(X, Y, Zscaled, levels=levels_filled, norm=MidpointNormalize(midpoint=midpoint), cmap='RdBu_r', extend='max')
   
    else:
        midpoint = 0. 
        contour = ax.contour(X,Y,Zscaled,levels=levels_lines, colors='k', linewidths=0.5)
        contour_filled = ax.contourf(X, Y, Zscaled, levels=levels_filled, norm=MidpointNormalize(midpoint=midpoint), cmap='RdBu_r', extend='both')

    cbar = fig.colorbar(contour_filled, ax=ax, ticks=ticks, extend='max')  
    cbar_label = r'($10^{-12}$ K)'

    cbar.set_label(cbar_label, size=font_size)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r'The amount of tCDR ($\alpha$, kg CO$_2$ kg$^{-1}$ CH$_4$)', fontsize=font_size)
    ax.set_ylabel(r'Storage Timescale ($\tau$, years)', fontsize=font_size)

    ax.plot([200,200],[0,200],linestyle='--',color='gray')
    ax.plot([0,200],[200,200],linestyle='--',color='gray')

    ax.set_xscale('symlog', linthresh=200, linscale=1)
    ax.set_yscale('symlog', linthresh=200, linscale=1)


    if decayNmax == 1E6:
       
        linear_ticks = [100, 200]  
        log_ticks = [500, 1000, 1E4,1E5,1E6]  
        major_ticks = linear_ticks + log_ticks
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        major_tick_labels = ['100', '200', '500', '1000', '1E4','1E5','1E6']
        ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
        ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))

    else:
        linear_ticks = [0, 50, 100, 150, 200]  
        log_ticks = [500, 1000]  
        major_ticks = linear_ticks + log_ticks
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        major_tick_labels = ['0', '50', '100', '150', '200', '500', '1000']
        ax.xaxis.set_major_formatter(FixedFormatter(major_tick_labels))
        ax.yaxis.set_major_formatter(FixedFormatter(major_tick_labels))


    subs = np.arange(2, 10) 
    log_locator = SymmetricalLogLocator(linthresh=200, base=10, subs=subs)

    ax.xaxis.set_minor_locator(log_locator)
    ax.yaxis.set_minor_locator(log_locator)

    ax.xaxis.set_minor_formatter(FixedFormatter([]))
    ax.yaxis.set_minor_formatter(FixedFormatter([]))

    ax.tick_params(axis='x', labelsize=10)  
    ax.tick_params(axis='y', labelsize=10)

    ax.set_title(r'Cumulative net temperature change', fontsize=font_size)

    ax.set_xlim([0,alphaNmax])
    ax.set_ylim([0,decayNmax])


    linear_thresh = 400
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
            labels=ax.clabel(contour, levels=[level], colors='k', inline=True, fontsize=9, fmt='%1.0f', manual=[position])
        except ValueError:
            pass


#==============================================================================
# Figure 4
#==============================================================================

# parameters
gas_params_optimal = {
    'BC': {'display_name': 'BC', 'tauNonCO2': tauBC, 'AANonCO2': AABC, 
           'alphaNmax': 5000, 'decayNmax': 5000, 
           'levels_lines': [-140, -100, -60, -40, -20, -10, 0, 10, 20, 30, 35, 40], 
           'ticks': np.arange(-180, 60, 20)},
    
    'HFC32': {'display_name': 'HFC-32', 'tauNonCO2': tauHFC32, 'AANonCO2': AAHFC32, 
              'alphaNmax': 5000, 'decayNmax': 5000, 
              'levels_lines': [-30, 0, 30, 40, 45], 
              'ticks': np.arange(-120, 80, 20)},
    
    'CH4': {'display_name': r'CH$_4$', 'tauNonCO2': tauCH4, 'AANonCO2': AACH4, 
            'alphaNmax': 1000, 'decayNmax': 1000, 
            'levels_lines': [-30, -20, -10, -5, -1, 1, 2, 5, 10, 20, 30], 
            'ticks': np.append(np.arange(-30, 5, 5), 2)},
    
    'HFC134a': {'display_name': 'HFC-134a', 'tauNonCO2': tauHFC134a, 'AANonCO2': AAHFC134a, 
                'alphaNmax': 5000, 'decayNmax': 5000, 
                'levels_lines': [-30, 0, 30, 60, 70, 80, 90, 95, 100], 
                'ticks': np.arange(-100, 120, 20)},
    
    'CFC11': {'display_name': 'CFC-11', 'tauNonCO2': tauCFC11, 'AANonCO2': AACFC11, 
              'alphaNmax': 20000, 'decayNmax': 20000, 
              'levels_lines': [-200, -100, 0, 100, 200, 300, 320, 350], 
              'ticks': np.arange(-400, 500, 100)},
    
    'N2O': {'display_name': r'N$_2$O', 'tauNonCO2': tauN2O, 'AANonCO2': AAN2O, 
            'alphaNmax': 1000, 'decayNmax': 1000, 
            'levels_lines': [-20, -15, -10, -5, 0, 5, 10, 15, 20], 
            'ticks': np.arange(-30, 20, 5)},
    
    'PFC14': {'display_name': 'PFC-14', 'tauNonCO2': tauPFC14, 'AANonCO2': AAPFC14, 
              'alphaNmax': 20000, 'decayNmax': 20000, 
              'levels_lines': [-200, -100, 0, 100, 200, 300, 320, 350], 
              'ticks': np.arange(-400, 500, 100)},
    
    'CO2': {'display_name': r'CO$_2$', 'tauNonCO2': 0, 'AANonCO2': 0,
            'alphaNmax': 1000, 'decayNmax': 1000, 
            'levels_lines': [-30, -20, -10, -5, -1, 0, 1, 2, 5, 10, 20, 30], 
            'ticks': np.arange(-20, 8, 4)}
}

if False:
    Z1,X1=compute_optimal_data_quad_stability('CH4', gas_params_optimal, NetAGTP_Exp, IfCO2=None, TH=100)
    # output: Z_values_TH100_NetAGTP_Exp_CH4_stability.csv

    Z2,X2=compute_optimal_data_quad_netzero('CH4', gas_params_optimal, NetAGTP_Exp, IfCO2=None, TH=100)
    # output: Z_values_TH100_NetAGTP_Exp_CH4_netzero.csv


# Find P1 P2 positions based on Z_values_TH100_NetAGTP_Exp_CH4_netzero.csv
alpha_range = np.linspace(0, 1000, 1000)
decay_range = np.linspace(0, 1000, 1000)
X, Y = np.meshgrid(alpha_range, decay_range)
Z = 1E12*np.genfromtxt(dir_path + '\\data\\' + 'Z_values_TH100_NetAGTP_Exp_CH4_netzero.csv', delimiter=',')
contour = plt.contour(X, Y, Z, levels=[0])

zero_contour_paths = contour.collections[0].get_paths()
xy_values = []
for collection in contour.collections:
    for path in collection.get_paths():
        vertices = path.vertices
        xy_values.extend(vertices)


P1x = 34.3 # xy_values[-1]
P1y = 1000
P2x = 1000
P2y = 9 # xy_values[0]
P3x = 120  # Table 2, TH=100
P3y = 82.6 # Table 2, TH=100


color1 = convert_rgb_to_01((69,153,67))
color2 = convert_rgb_to_01((0,116,179))
color3 = convert_rgb_to_01((207,24,42))


fig, axs = plt.subplots(2, 2, figsize=(14, 11))
axs = axs.flatten()
plt.subplots_adjust(wspace=0.25)
plt.subplots_adjust(hspace=0.3)

## panel a
ax = axs[0]
ax.text(-0.15, 1.03, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')

pos_ax = ax.get_position()
ax.set_position([pos_ax.x0, pos_ax.y0, pos_ax.width * 0.9, pos_ax.height])

TH = 100
dt = 0.1
tt = np.arange(0,TH+dt,dt)

gas_name = 'CH4'
tauNonCO2 = gas_params_optimal[gas_name]['tauNonCO2']
AANonCO2 = gas_params_optimal[gas_name]['AANonCO2']

alphas = [P1x, P2x, P3x]
decays = [P1y, P2y, P3y]
colors = [color1,color2,color3]

ax.plot(tt,1E13*AGTPNonCO2_Final_partial(t=tt, AANonCO2=AANonCO2,tauNonCO2=tauNonCO2), color='k') 

for i in [0,1]:
    yy = alphas[i] * AGTPPRF_Exp_partial(t=tt, decay=decays[i])
    zz = NetAGTP_Exp(tt, alpha=alphas[i], decay=decays[i], AANonCO2=AANonCO2,tauNonCO2=tauNonCO2)

    ax.plot(tt, 1E13*yy, color=colors[i], label=rf'$\mathrm{{P{i+1}}}\ (\alpha = {alphas[i]}, \tau = {decays[i]})$')
    ax.plot(tt, 1E13*zz, color=colors[i], linestyle='--', linewidth=2)

    ax.axhline(0, color='k', linewidth=0.5)

ax.legend(frameon=False,loc='lower right')
ax.set_xlim([0, 100])
ax.set_ylim([-2.5, 1])

ax.set_xlabel(r'Years', fontsize=12)
ax.set_ylabel(r'Global temperature response (K $\times 10^{-13}$)', fontsize=12)


text1 = r'1 kg CH$_4$'
text2 = r'$\alpha$ kg CO$_2$ of tCDR stored for $\tau$ years'
text3 = r'net temperature difference'


ax.text(0.52, 0.17, 'TH = 100', transform=ax.transAxes, color='k')   

ax.text(0.17, 0.9, text1, transform=ax.transAxes,size=12, multialignment='center', color='k')   

ax.text(0.18, 0.48, text2, transform=ax.transAxes,size=12, multialignment='center', color=color2)   

ax.text(0.45, 0.8, text3, transform=ax.transAxes,size=12, multialignment='center', color=color2)   


ax.annotate("", xy=(9 , 0.53), xytext=(16.5, 0.7),
            arrowprops=dict(arrowstyle="->"))

ax.annotate("", xy=(9.5 , -0.5), xytext=(17, -0.75), 
            arrowprops=dict(arrowstyle="->", edgecolor=color2))

ax.annotate("", xy=(36, 0.2), xytext=(44, 0.3), 
            arrowprops=dict(arrowstyle="->", edgecolor=color2))



## panel b
ax = axs[1]
ax.text(-0.2, 1.03, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')

plot_optimal_data_netzero_200('CH4', gas_params_optimal, 'NetAGTP_Exp', ax=ax, TH=100)

# Plot the 0 level contour lines from Z1 onto Panel b
for path in zero_contour_paths:
    ax.plot(path.vertices[:, 0], path.vertices[:, 1], color='r')

ax.plot(P1x, P1y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P1x+5, P1y-150, 'P1', fontsize=10, color='r')

ax.plot(P2x, P2y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P2x-150, P2y+5, 'P2', fontsize=10, color='r')

ax.plot(P3x, P3y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P3x+5, P3y+5, 'P3', fontsize=10, color='r')



## panel c
ax = axs[2]
ax.text(-0.15, 1.03, 'c', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')

pos_ax = ax.get_position()
ax.set_position([pos_ax.x0, pos_ax.y0, pos_ax.width * 0.9, pos_ax.height])

TH = 100
dt =0.1
tt = np.arange(0,TH+dt,dt)

gas_name = 'CH4'
tauNonCO2 = gas_params_optimal[gas_name]['tauNonCO2']
AANonCO2 = gas_params_optimal[gas_name]['AANonCO2']

ax.plot(tt, 1E14*AGTPNonCO2_Final_partial(t=tt, AANonCO2=AANonCO2,tauNonCO2=tauNonCO2), color='k') 
i = 2
yy = alphas[i] * AGTPPRF_Exp_partial(t=tt, decay=decays[i])
zz = NetAGTP_Exp(tt, alpha=alphas[i], decay=decays[i], AANonCO2=AANonCO2,tauNonCO2=tauNonCO2)

ax.plot(tt, 1E14*yy, color=colors[i], label=rf'$\mathrm{{P{i+1}}}\ (\alpha = {alphas[i]}, \tau = {decays[i]})$')
ax.plot(tt, 1E14*zz, color=colors[i], linestyle='--', linewidth=2)

ax.axhline(0, color='k', linewidth=0.5)

ax.legend(frameon=False,loc='lower right')
ax.set_xlim([0, 100])
ax.set_ylim([-6,6])

ax.set_ylabel(r'Global temperature response (K $\times 10^{-14}$)', fontsize=12)
ax.set_xlabel(r'Years', fontsize=12)

ax.text(0.54, 0.1, 'TH = 100', transform=ax.transAxes, color='k')   



## panel d
ax = axs[3]
ax.text(-0.2, 1.03, 'd', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')

plot_optimal_data_stability_200('CH4', gas_params_optimal, 'NetAGTP_Exp', ax=ax, TH=100)

# Plot the 0 level contour lines from Z1 onto Panel b
for path in zero_contour_paths:
    ax.plot(path.vertices[:, 0], path.vertices[:, 1], color='r')

ax.plot(P1x, P1y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P1x+5, P1y-150, 'P1', fontsize=10, color='r')

ax.plot(P2x, P2y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P2x-150, P2y+5, 'P2', fontsize=10, color='r')

ax.plot(P3x, P3y, 'r*', marker='*', markersize=10, markeredgecolor='none', markerfacecolor='r', clip_on=False, zorder=10)
ax.text(P3x+5, P3y+5, 'P3', fontsize=10, color='r')

plt.savefig('figure\Figure 4.png', bbox_inches='tight', pad_inches=0.1)


