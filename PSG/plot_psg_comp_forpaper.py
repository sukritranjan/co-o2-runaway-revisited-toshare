#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:03:26 2023

@author: sukrit

Based on main PSG script, to plot two different transmission spectra on top of each other. 
"""

#Colors from Wong+2011 colorblind-friendly
bluishgreen=(0, 158/256., 115/256.)
reddishpurple=(204/256., 121/256., 167/256.)
vermillion=(213/256., 94/256., 0)
blue=(0, 114/256., 178/256.)
black=(0,0,0)
orange=(230/256., 159/256., 0)

import numpy as np
import matplotlib.pyplot as plt
import pdb


def psgplot_compare(file_list, label_list, color_list):
    ###Read inputs
    #Initial defaults
    # xunit = 'Wavelength [um]'; yunit = 'Contrast [ppm]'; cols = 'Total'
    
    #Read in files in loop
    wavels={} #in um
    tspecs={} #in ppm
    zs={} #in km 
    
    R_planet=0.5 * 12742.0  #km; taken from meac_sr file (input)
    R_star=0.1192 * 6.963e5 #km; taken from meac_sr file (input)
    
    for file in file_list:
        # fw = open("%s_rad.txt" % filebase); lines = fw.readlines(); fw.close()
        # for line in lines:
        #     if line[0]!='#': break
        #     if line[0:16]=='# Spectral unit:': xunit = line[17:-1]
        #     if line[0:16]=='# Radiance unit:': yunit = line[17:-1]
        #     if line[0:11]=='# Wave/freq': cols = line[12:-1]
        # #Endfor
        # data = np.genfromtxt("%s_rad.txt" % filebase)
        # cols = cols.split(); wnoise=0
        
        wavels[file], tspecs[file]=np.genfromtxt(file, skip_header=12, skip_footer=0, unpack=True, usecols=(0, 5)) #Import mapping between numerical ID in code and species name.
        zs[file]=1E-6*tspecs[file]*R_star**2/(2*R_planet)
        # pdb.set_trace()
            
    ###Make plot
    fig, ax1=plt.subplots(figsize=[8,4])
    for ind in range(0, len(file_list)):
        file=file_list[ind]
        linestyle='-'
        if ind>0:
            linestyle='--'
        ax1.plot(wavels[file],tspecs[file],label=label_list[ind], linestyle=linestyle, color=color_list[ind])
    #Endfors
    ax1.set_xscale('linear')
    ax1.set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    ax1.set_ylabel(r'd (ppm)', fontsize=14)
    ax1.set_ylim([0, 100])
    ax1.set_xlim([0.2, 11.0])
    
    
    #Secondary axis
    def tspec2z(tspec):
        return 1E-6*tspec*R_star**2/(2*R_planet)
    def z2tspec(z):
        return 1E6*2*R_planet*z/R_star**2
    secax=ax1.secondary_yaxis('right', functions=(tspec2z, z2tspec))
    secax.set_ylabel('z (km)', fontsize=14)
    
    #Annotate spectral features

  
    plt.annotate(r'O$_2$', (0.78,38), fontsize=10, color=bluishgreen)
    plt.annotate(r'O$_2$-X CIA', (5.8,35), fontsize=10, color=blue)
    plt.annotate(r'CO', (2.2,55), fontsize=10, color=vermillion)
    plt.annotate(r'CO', (4.55,70), fontsize=10, color=vermillion)
    plt.annotate(r'O$_3$', (0.25,80), fontsize=10, color=reddishpurple)
    plt.annotate(r'O$_3$', (0.55,50), fontsize=10, color=reddishpurple)
    plt.annotate(r'O$_3$', (9.5,60), fontsize=10, color=reddishpurple)

    plt.legend(ncol=1, fontsize=10)
    # plt.title('PSG Comparison of CO/O2 Runaway/Non-Runaway (from MEAC)')
    plt.tight_layout()
    plt.savefig('simulated_spectra_runwayvsnorunway.pdf')
    plt.show()


# psgplot('atmos_psg')
# psgplot('meac_psg')

psgplot_compare(['T1_CO2_noS_rad.txt', 'T1_CO2_noS_bigzgrid_rad.txt'], [r'Run 4 ($z_{max}=54$ km)','Run 5 ($z_{max}=100$ km)'], (black, orange))
