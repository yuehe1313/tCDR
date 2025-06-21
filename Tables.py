
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



#==============================================================================
# Table 1 iAGTP formatted
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table iAGTP.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table 1 iAGTP formatted.csv',
    index=True
)



#==============================================================================
# Table 2 
#==============================================================================

# data source: Table 2 Optimization TH100=0.nb 
# data source: Table 2 Optimization THInf=0.nb 

gas_name = ['BC', 'HFC32', 'CH4', 'HFC134a']
combined_data = []

for gas in gas_name:
    file_path2 = f'Table {gas}_Optimization_TH100.csv'
    file_path3 = f'Table {gas}_Optimization_THInfinity.csv'

    data2 = pd.read_csv(dir_path + '\\data\\' + file_path2, header=None)
    data3 = pd.read_csv(dir_path + '\\data\\' + file_path3, header=None)

    formatted_data = pd.DataFrame({
        'Optimal alpha 100': data2.iloc[0, -1],
        'Optimal tau 100': data2.iloc[1, -1],
        'Optimal alpha Inf': data3.iloc[0, -1],
        'Optimal tau Inf': data3.iloc[1, -1],
    }, index=[gas])
    combined_data.append(formatted_data)

# combined_data = pd.concat(combined_data).applymap(lambda x: format(x, '.3g'))
combined_data = pd.concat(combined_data).applymap(format_with_zeros)

combined_data.to_csv(dir_path + '\\data\\' + 'Table 2.csv') 


#==============================================================================
# Table S1
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table GWP GTP iGTP.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table S1 GWP GTP iGTP formatted.csv',
    index=True
)

#==============================================================================
# Table S4 AGWP formatted
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table AGWP.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table S4 AGWP formatted.csv',
    index=True
)


#==============================================================================
# Table S6 iAGTP Linear formatted
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table iAGTP Linear.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table S6 iAGTP Linear formatted.csv',
    index=True
)


#==============================================================================
# Table S7 iAGTP Constant formatted
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table iAGTP Constant.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table S7 iAGTP Constant formatted.csv',
    index=True
)



#==============================================================================
# Table S8 iAGTP Impulse formatted
#==============================================================================

xls_path = dir_path + '\\data\\' + 'Table iAGTP Impulse.xlsx'

data = pd.read_excel(xls_path, index_col=0)

numeric_data = data.apply(pd.to_numeric, errors='coerce')

formatted_data = numeric_data.applymap(format_with_zeros)

formatted_data.index = data.index
formatted_data.columns = data.columns

formatted_data.to_csv(
    dir_path + '\\data\\' + 'Table S8 iAGTP Impulse formatted.csv',
    index=True
)


