"""
Purpose of this code is to read and plot the output of the Hu+2012 photochemical code, and track our progress adding PH3. 
"""
########################
###Import useful libraries
########################
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb

########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################

#Unit conversions
km2m=1.e3 #1 km in m
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


########################
###Establish key
########################

#Corrected for Hu 1-indexing vs Python 0-indexing
ind_o=1-1 #O
ind_h=3-1 #H
ind_oh=4-1 #OH

ind_so2=43-1
ind_so=42-1
ind_ch4=21-1
ind_h2s=45-1
ind_h2=53-1
ind_h2o=7-1

ind_no=12-1
ind_n2o=11-1

ind_co2=52-1
ind_n2=55-1
ind_cho=61-1


ind_s8=79-1
ind_s8a=111-1
ind_ch4o=24-1
ind_c2h2=27-1

ind_ocs=49-1

ind_o1d=56-1
ind_co=20-1
ind_o2=54-1
ind_c2h6=31-1
ind_h2o2=6-1
ind_h2so4=73-1
ind_h2so4a=78-1
ind_ch2o=22-1
ind_o3=2-1
ind_ho2=5-1
ind_n2o5=15-1
ind_hno4=70-1

def plot_comparison_rad(base_file, base_numrad, base_numz, new_file, new_numrad, new_numz, name):
    """
    #Base file
    #New file
    #Title of plot and name of file. 
    """


    ########################
    ###Build 2D grid of radiation at each wavelength and altitude
    ########################
    
    ###Base file
    base_rad=np.zeros((base_numz, base_numrad))
    base_wavs=np.zeros(base_numrad)
    for ind1 in range(0, base_numz):
        linestart=1+ind1*(base_numrad+1)
        if ind1==0:
            base_wavs, base_rad[ind1,:]=np.genfromtxt(base_file, skip_header=linestart, max_rows=base_numrad, usecols=(0,1), unpack=True) 
        else:
            base_rad[ind1,:]=np.genfromtxt(base_file, skip_header=linestart, max_rows=base_numrad, usecols=(1), unpack=True)   
    
    ###New file
    new_rad=np.zeros((new_numz, new_numrad))
    new_wavs=np.zeros(new_numrad)
    for ind1 in range(0, new_numz):
        linestart=1+ind1*(new_numrad+1)
        if ind1==0:
            new_wavs, new_rad[ind1,:]=np.genfromtxt(new_file, skip_header=linestart, max_rows=new_numrad, usecols=(0,1), unpack=True) 
        else:
            new_rad[ind1,:]=np.genfromtxt(new_file, skip_header=linestart, max_rows=new_numrad, usecols=(1), unpack=True)  
            
            
    ########################
    ###Plot
    ########################
    linestyles=np.array(['-',':'])
    
    fig, ax=plt.subplots(1, figsize=(6,8))

    
    ###Top plot: Outgassed species
    ax.plot(base_wavs, base_rad[-1,:], linewidth=2, linestyle=linestyles[0], color='blue', label='Base Rad, TOA')
    ax.plot(new_wavs, new_rad[-1,:], linewidth=2, linestyle=linestyles[0], color='red', label='New Rad, TOA')
    ax.plot(base_wavs, base_rad[0,:], linewidth=2, linestyle=linestyles[1], color='blue', label='Base Rad, BOA')
    ax.plot(new_wavs, new_rad[0,:], linewidth=2, linestyle=linestyles[1], color='red', label='New Rad, BOA')

    ax.legend(loc='best', ncol=1, borderaxespad=0., fontsize=12)    

    
    ax.set_yscale('log')
    ax.set_ylabel('Actinic Flux')
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength')  
    ax.set_xlim([100, 400])
    ax.set_ylim([1E-10, 1E-1])
   
    # plt.savefig('./Plots/plot'+name+'.png', orientation='portrait', format='png')
    plt.show()


###############################
###Run
###############################
plot_comparison_rad('./scenario_library/TRAPPIST-1/CO2-noS-pCO2=0.01/Radiation.dat', 1000, 54,'./scenario_library/TRAPPIST-1/CO2-noS-pCO2=0.1/Radiation.dat', 1000, 54, 'rad_CO2-noS-pCO2=0.01_0.1')
plot_comparison_rad('./scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2=0.01/Radiation.dat', 1000, 100,'./scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2=0.1/Radiation.dat', 1000, 100, 'rad_bigzgrid-CO2-noS-pCO2=0.01_0.1')




