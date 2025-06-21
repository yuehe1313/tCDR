import os
import numpy as np
import pandas as pd
import xarray as xr

dir_path = 'D:/科研相关/我的/tCDR/tCDR'
os.chdir(dir_path)


#==============================================================================
# GHG lifetimes and radiative efficiencies 
#==============================================================================
# a=0.5640766947115513
# v1=0.34386279086127003
# v2=0.9596722614296296
rotation=60
p=rotation/2

#----------- IRFCO2 -----------#
# Joos 2013 
aC1=0.2173
aC2=0.2763
aC3=0.2824
aC4=0.2240
tauC1=4.304
tauC2=36.54
tauC3=394.4

#----------- IRFclim -----------#
# this is what Bill used (335_chapter7_generate_metrics.ipynb
kPulseT=0.7578
aT1=0.5856
aT2=0.4144
tauT1=3.424
tauT2=285.0

# Geoffery 2013
# kPulseT=0.885
# aT1=0.587
# aT2=0.413
# tauT1=4.1
# tauT2=249

#----------- IRFfdbk -----------#
# Gasser 2017
gamma=3.015
aS1=0.6368
aS2=0.3322
aS3=0.0310
tauS1=2.376
tauS2=30.14
tauS3=490.1

#----------- other parameters -----------#
# perturbation lifetimes (reference: IPCC AR6 Table 7.15)
tauCH4=11.8
tauN2O=109.
tauHFC32=5.4
tauHFC134a=14.
tauCFC11=52.
tauPFC14=50000.
tauBC=0.02 # Fuglestvedt et al. (2010)

# radiative efficiencies W/m2/ppb (original data used for IPCC AR6 Table 7.15) reference: 335_chapter7_generate_metrics.ipynb
reCO2 = 1.3330689487029318E-5 
reN2O = 2.7788125677697985E-3 # include chemical adjustments, re_n2o+f_n2o_ch4*re_ch4
reCH4 = 5.686440286086949E-4 # include chemical adjustments, meinshausen(np.array([co2, ch4+1, n2o]), np.array([co2, ch4, n2o]), scale_F2x=False)[1] * (1+ch4_ra) + ch4_o3+ch4_h2o
reHFC32 = 0.11144 # hodnebrog20.csv
reHFC134a = 0.16714 # hodnebrog20.csv
reCFC11 = 0.25941* (1+0.12) # CFC11 and CFC12 need a 12% rapid adjustment # hodnebrog20.csv
rePFC14 = 0.09859 # hodnebrog20.csv
reN2O_SM = 3.19550741640458E-3 # same as Table 7.SM.7, do not include chemical adjustments, meinshausen(np.array([co2, ch4, n2o+1]), np.array([co2, ch4, n2o]), scale_F2x=False)[2] * 1.07
reCH4_SM = 3.8864402860869495E-4 # same as Table 7.SM.7, do not include chemical adjustments, meinshausen(np.array([co2, ch4+1, n2o]), np.array([co2, ch4, n2o]), scale_F2x=False)[1] * (1+ch4_ra)

# Molar mass (reference: Hodnebrog_et_al_2020_revgeo\hodnebrog20.csv and ar6.metrics.constants)
M_ATMOS = 5.1352E18
M_AIR = 28.97
M_C = 12.0
M_CO2 = 44.01
M_CH4 = 16.043
M_N2O = 44.0
M_HFC32 = 52.03
M_HFC134a = 102.04
M_CFC11 = 137.36
M_PFC14 = 88.01

# the contribution to methane-emission metrics from CO2 from fossil methane oxidation (reference: AR6 7.SM.5.8)
Y=0.75 
tauOH=9.7 

# ppb2kg
AACO2 =reCO2/(( 1E-9*M_CO2/M_AIR)*M_ATMOS)
AAN2O = reN2O/((1E-9*M_N2O/M_AIR)*M_ATMOS)
AACH4 = reCH4/((1E-9*M_CH4/M_AIR)*M_ATMOS)
AAHFC32 = reHFC32/((1E-9*M_HFC32/M_AIR)*M_ATMOS)
AAHFC134a = reHFC134a/((1E-9*M_HFC134a/M_AIR)*M_ATMOS)
AACFC11 = reCFC11/((1E-9*M_CFC11/M_AIR)*M_ATMOS)
AAPFC14 = rePFC14/((1E-9*M_PFC14/M_AIR)*M_ATMOS)
AAN2O_SM = reN2O_SM/((1E-9*M_N2O/M_AIR)*M_ATMOS)
AACH4_SM = reCH4_SM/((1E-9*M_CH4/M_AIR)*M_ATMOS)
AABC = 2.54E-9 # IPCC AR6 7.SM.1.3.1
# BC in IPCC AR6 7SM: 50.8 mW yr m−2 MtC−1 = radiation efficiency * turnover time(0.02)
# BC radiation efficiency = （50.8 / 0.02）* 10^-3 * 10^-9 W m−2 kg-1 = 2.54 * 10^-9 W m−2 kg-1
# old BC radiation efficiency = 1.96×10−9 W m−2 kg−1 gasser 2017
