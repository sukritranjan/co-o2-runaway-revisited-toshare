#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:44:51 2023

@author: sukrit

Purpose of this script is to get some appendix parameters

"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy
import scipy.integrate
from scipy import interpolate as interp


########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################

#Unit conversions
km2m=1.e3 #1 km in m
m2km=1.0E-3 #1 m in km
km2cm=1.e5 #1 km in cm
cm2km=1.e-5 #1 cm in km
amu2g=1.66054e-24 #1 amu in g
bar2atm=0.9869 #1 bar in atm
Pa2bar=1.e-5 #1 Pascal in bar
bar2Pa=1.e5 #1 bar in Pascal
deg2rad=np.pi/180.
bar2barye=1.e+6 #1 Bar in Barye (the cgs unit of pressure)
barye2bar=1.e-6 #1 Barye in Bar
micron2m=1.e-6 #1 micron in m
micron2cm=1.e-4 #1 micron in cm
metricton2kg=1000. #1 metric ton in kg


#Fundamental constants
c=2.997924e10 #speed of light, cm/s
h=6.6260755e-27 #planck constant, erg/s
k=1.380658e-16 #boltzmann constant, erg/K
sigma=5.67051e-5 #Stefan-Boltzmann constant, erg/(cm^2 K^4 s)
R_earth=6371.*km2m#radius of earth in m
R_sun=69.63e9 #radius of sun in cm
AU=1.496e13#1AU in cm
hc=1.98645e-9 #value of h*c in erg*nm

########################
###Define key functions
########################

#Calculate the mean cross-section of CO2 in the UV
def calculate_mean_CO2_XC(MEAC_uv_file):
    """
    UV SED in MEAC format
    """
    
    ###UV
    ##Import UV
    MEAC_wav, MEAC_toa=np.genfromtxt(MEAC_uv_file, skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, W m**-2 nm**-1
    
    ##Convert to number flux
    MEAC_toa_cgs=MEAC_toa*1.0E+3 #convert W m^-2 nm^-1 to erg cm^-2 s^-1 nm^-1
    MEAC_toa_numflux=MEAC_toa_cgs*MEAC_wav/hc
    
    ###Import CO2 XC
    CO2_wav, CO2_XC=np.genfromtxt('./hu-code-sr-co-runaway/CO2', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, cm**2

    ##Functionalize
    stellaruv_func=interp.interp1d(MEAC_wav, MEAC_toa_numflux, kind='linear')  
    
    uv_weighted_co2_xc=CO2_XC*np.interp(CO2_wav, MEAC_wav, MEAC_toa_numflux, left=0, right=0)
    uv_weighted_co2_xc_func=interp.interp1d(CO2_wav, uv_weighted_co2_xc, kind='linear', fill_value=0.0)
    
    ###Calculate weighted mean
    leftedge=np.min(MEAC_wav)
    rightedge=np.max(CO2_wav)
    mean_co2_xc=scipy.integrate.quad(uv_weighted_co2_xc_func, leftedge, rightedge, epsabs=0., epsrel=1.e-3, limit=100000)[0]/scipy.integrate.quad(stellaruv_func, leftedge, rightedge, epsabs=0., epsrel=1.e-3, limit=100000)[0]
    return mean_co2_xc

#Plot the components of the calculation
def plot_mean_CO2_XC_calc():
    """
    UV SED in MEAC format
    """
    
    ###UV
    ##Import UV
    Sun_wav, Sun_toa=np.genfromtxt('./hu-code-sr-co-runaway/Data/solar00.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, W m**-2 nm**-1
    
    T1_wav, T1_toa=np.genfromtxt('./hu-code-sr-co-runaway/Data/trappist-1_00.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, W m**-2 nm**-1
    
    ##Convert to number flux
    Sun_toa_cgs=Sun_toa*1.0E+3 #convert W m^-2 nm^-1 to erg cm^-2 s^-1 nm^-1
    Sun_toa_numflux=Sun_toa_cgs*Sun_wav/hc
 
    T1_toa_cgs=T1_toa*1.0E+3 #convert W m^-2 nm^-1 to erg cm^-2 s^-1 nm^-1
    T1_toa_numflux=T1_toa_cgs*T1_wav/hc   
 
    ###Import CO2 XC
    CO2_wav, CO2_XC=np.genfromtxt('./hu-code-sr-co-runaway/CO2', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, cm**2
    
    ###Weight
    sun_uv_weighted_co2_xc=CO2_XC*np.interp(CO2_wav, Sun_wav, Sun_toa_numflux, left=0, right=0)
    T1_uv_weighted_co2_xc=CO2_XC*np.interp(CO2_wav, T1_wav, T1_toa_numflux, left=0, right=0)

    
    ###Plot
    fig1, ax=plt.subplots(2,1, figsize=(8., 8.), sharex=True)
    
    ax[0].plot(Sun_wav, Sun_toa_numflux, color='gold', linestyle='-', label='Sun')
    ax[0].plot(T1_wav, T1_toa_numflux, color='red', linestyle='-', label='TRAPPIST-1')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$F_{*}$ (cm$^{-2}$ s$^{-1}$ nm$^{-1}$)', fontsize=14)
    ax[0].legend(fontsize=14, loc='upper right', ncol=1, borderaxespad=0.)
    ax0alt=ax[0].twinx()
    ax0alt.plot(CO2_wav, CO2_XC, color='black', linestyle='-', label=r'CO$_2$')
    ax0alt.set_xscale('linear')
    ax0alt.set_yscale('log')    
    ax0alt.set_ylabel(r'$\sigma_{CO_{2}}$ (cm$^{2}$)', fontsize=14)  
    ax0alt.legend(fontsize=14, loc='lower right', ncol=1, borderaxespad=0.)
   
    ax[1].plot(CO2_wav, sun_uv_weighted_co2_xc, color='gold', linestyle='-', label='Sun')
    ax[1].plot(CO2_wav, T1_uv_weighted_co2_xc, color='red', linestyle='-', label='TRAPPIST-1')
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$F_{*}\times\sigma_{CO_{2}}$ (s$^{-1}$ nm$^{-1}$)', fontsize=14)
    ax[1].legend(fontsize=12, loc='upper right', ncol=1, borderaxespad=0.)
    ax[1].set_ylim([1E-12, 1E-4])
    
    ax[1].set_xscale('linear')
    ax[1].set_xlim([110.0, 202.0])
    ax[1].set_xlabel('Wavelength (nm)', fontsize=14)
    plt.savefig('./Plots/plot_co2meanuvxc.pdf', orientation='portrait', format='pdf')

def get_peak_photolysis(file):
    co2_mean_xc=calculate_mean_CO2_XC(file)
    
    mu_0=np.cos(57.3*deg2rad) #solar zenith angle
    mmm=42.4*amu2g ###mean molecular mass of 0.9 bar CO2 in g, 0.1 bar N2 atmospheres
    r_co2=0.9 #co2 mixing ratio
    g=981. #gravitational acceleration of Earth, cm/s^2
    
    p_peak_bar=(mu_0*mmm*g/(r_co2*co2_mean_xc))*barye2bar
    
    print('Mean CO2 XC (cm$^2$): {0:1.0e}. Peak photolysis pressure (bar):{1:1.0e}'.format(co2_mean_xc,p_peak_bar))



[A_Ar_air, s_Ar_air]=[6.73E16, 0.749]#from Banks & Kockarts 1973
[A_Ar_CO2, s_Ar_CO2]=[7.6E16, 0.646] #Converted from p 66 of Marrero & Mason 1972 by dividing by kT and evaluating exponential at 175K. 
[A_H_CO2, s_H_CO2]=[3.87E17, 0.75]#via Ranjan+2020 (many underlying references). NOTE: Implicit evaluation of exponential correction factor at 175K
[A_H2_CO2, s_H2_CO2]=[2.15E17, 0.75]# via Ranjan+2020 (many underlying references)

def D_Ar_air(T, n): #from Banks & Kockarts 1973
    return A_Ar_air*T**s_Ar_air/n 

#All below are from CRC Handbook of Chemistry and Physics, 14-19 US Standard Atmosphere. All in SI
TP_earth_z=np.array([60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 110000])*m2km #altitude in m converted to km
TP_earth_T=np.array([247.02, 233.29, 219.59, 208/40, 198.64, 188.89, 186.87, 188.42, 195.08, 240.0]) #T in K
TP_earth_P=np.array([2.196E1, 1.093E1, 5.221, 2.388, 1.052, 4.457E-1, 1.836E-1, 7.597E-2, 3.201E-2, 7.104E-3])*Pa2bar #P in Pa, converted to bar
TP_earth_n=np.array([6.439E21, 3.393E21, 1.722E21, 8.300E20, 3.3838E20, 1.709E20, 7.116E19, 2.920E19, 1.189E19, 2.144E18])*1E-6 #number density in m^-3, converted to cm^-3

def get_homopause_pressure(K_z, T, A12, s12):
    homopause_pressure=k*(A12*T**(1+s12))/K_z #units are cgs, therefore barye
    homopause_pressure_bar=homopause_pressure*barye2bar
    
    return homopause_pressure_bar

def validate_homopause_pressure_calculation():
    """
    Can we approximately recover the Earth homopause?
    Answer: yes, it looks sensible. Not a hard guarantee, but reasonable for an Appendix.
    """
    T_homo=195.08+(6.0/10.0)*(240.0-195.08) #Interpolate to T_homo on the vs z graph
    earth_homopause_pressure_ar=get_homopause_pressure(1.0E+6, T_homo, A_Ar_air, s_Ar_air) 

    fig, ax=plt.subplots(2, figsize=(8,8), sharex=True)
    ax[0].plot(D_Ar_air(TP_earth_T,TP_earth_n), TP_earth_z, color='red', label='D(Ar, Air)')
    ax[0].axvline(1.0E+6, color='black', label='Kzz(homopause, Earth)')
    ax[0].set_yscale('linear')
    ax[0].set_ylim([80, 110])
   
    ax[1].plot(D_Ar_air(TP_earth_T,TP_earth_n), TP_earth_P, color='red', label='D(Ar, Air)')
    ax[1].axvline(1.0E+6, color='black', label='Kzz(homopause, Earth)')
    ax[1].axhline(earth_homopause_pressure_ar, color='blue', label='Calculated P(homopause, Earth)')
    ax[1].set_yscale('log')
    ax[1].set_ylim([1.0E-5, 1.0E-8])
       
    ax[1].set_xscale('log')
    ax[1].set_xlim([1.0E+2, 1.0E+7])
    
### x=get_homopause_pressure(1.0E+6, 200, 6.73E16, 0.749)
### validate_homopause_pressure_calculation()

plot_mean_CO2_XC_calc()
get_peak_photolysis('./hu-code-sr-co-runaway/Data/solar00.txt')
get_peak_photolysis('./hu-code-sr-co-runaway/Data/trappist-1_120nm.txt')
get_peak_photolysis('./hu-code-sr-co-runaway/Data/trappist-1_00.txt')
print(get_homopause_pressure(7.0E+5, 175.0, A_Ar_CO2, s_Ar_CO2)*1E6)
print(get_homopause_pressure(7.0E+5, 175.0, A_H2_CO2, s_H2_CO2)*1E6)
print(get_homopause_pressure(7.0E+5, 175.0, A_H_CO2, s_H_CO2)*1E6)
