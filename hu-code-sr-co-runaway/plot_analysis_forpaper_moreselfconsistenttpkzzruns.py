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
import seaborn as sns

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

def return_reaction_rates(concSTD, ChemReac, reaction_label_list):
    """
    Takes filenames for a concentrationSTD.dat and ChemicalRate.dat file from the Hu code, and a list of Hu code reaction IDs.
    From concentrationSTD.dat file extracts z scale. From ChemicalRate file takes reaction rate (cm**-3 s**-1).
    Returns: z_centers (km), delta_zs (cm), vertically resolved rates for each reaction ID (cm**-3 s**-1), column integrated reaction rate for each reaction ID (cm**-2 s**-1)
    """
    ###Get z scale
    tp=np.genfromtxt(concSTD, skip_header=2, skip_footer=0, unpack=False) #Import simulation results
    z_centers=tp[:,0] # Center of altitude bins, km 
    deltazs=(tp[:,2]-tp[:,1]) #width of altitude bins, km    
    p_centers=tp[:,4]*Pa2bar*bar2barye #pressures at center of altitude bins, converted to barye.
    
    ###Get Chemical Reaction Rates
    chemlabels=np.genfromtxt(ChemReac, skip_header=1, skip_footer=0, unpack=True, usecols=(0), dtype=None, encoding='UTF-8')
    chemrates=np.genfromtxt(ChemReac, skip_header=1, skip_footer=0, unpack=False, dtype=float)[:,1:]
    
    ###Extract relevant chemical reaction rates.
    
    vert_reac_rates={} #Dict which will hold the vertical reaction rates (cm*-3 s**-1)
    colint_reac_rates={} #Dict which will hold the column-integrated reaction rates (cm**-2 s**-1)
    
    for target_chemlabel in reaction_label_list: #Step through the chemical reactions we are interested in...
        for ind in range(0, len(chemlabels)): #Step through each of the chemical reactions in the FILE
            chemlabel=chemlabels[ind]#.decode('UTF-8') #decode the indth chemical reaction in the FILE
            if chemlabel==target_chemlabel: #If they match, this is what we want
                vert_reac_rates[target_chemlabel]=chemrates[ind,:] #cm**-3 s**-1
                colint_reac_rates[target_chemlabel]=np.sum(chemrates[ind,:]*deltazs*km2cm) #cm**-2 s**-1

    return p_centers, z_centers, deltazs, vert_reac_rates, colint_reac_rates

def return_atm_concentrations(concSTD):
    ########################
    ###Read in base data
    ########################
    base_data=np.genfromtxt(concSTD, skip_header=2, skip_footer=0, unpack=False) #Import mapping between numerical ID in code and species name.
    
    z_center_base=base_data[:,0] # Center of altitude bins, km 
    T_z_base=base_data[:,3] # Temperature(z), in K
    P_z_base=base_data[:,4]*Pa2bar*bar2barye # Pressure(z), in Pa converted to Barye
    n_z_s_base=base_data[:,5:] #Number concentrations of the 111 chemical species, in cm**-3, as a function of (altitude, species)

    ###Get molar concentrations
    ###NOTE: May (probably) need to exclude condensed-phase species for molar concentration calculation...probably doesn't matter most of the time, but formally required and mioght matter in some weird edge cases.
    n_z_base=np.sum(n_z_s_base,1) #sum number densities across species. This is a profile for the whole atmosphere.
    n_z_bulkatm_base=P_z_base/(k*T_z_base)
    mc_z_s_base=np.zeros(np.shape(n_z_s_base))
    mr_z_s_base=np.zeros(np.shape(n_z_s_base))

    num_s=np.shape(n_z_s_base)[1]

    for ind2 in range(0, num_s):
        mc_z_s_base[:,ind2]=n_z_s_base[:,ind2]/n_z_base#molar concentration of each species.
        mr_z_s_base[:,ind2]=n_z_s_base[:,ind2]/n_z_bulkatm_base#mixing ratio of each species.
        
    return z_center_base, T_z_base, P_z_base, mc_z_s_base, mr_z_s_base, n_z_bulkatm_base


def plot_failure_to_resolve_photolysis_rates():
    """
    This script show the failure of the truncated calculation to resolve the reaction rates. 
    To get it right, it needs to do it for the truncated and untruncated, for pCO2=0.01, 0.1, and 0.9. So, it needs to be done programmatically. 
    """
    ###Specify what to load
    #Reactions
    phot_reaction_label_list=np.array(['P36','P37','P6']) #$CO_2+h\nu\rightarrow CO+O$',r'$CO_2+h\nu\rightarrow CO+O(^1D)$', r'$H_2O+h\nu\rightarrow H+OH$'
    
    #Addresses
    truncated_addresses=['./scenario_library/TRAPPIST-1/CO2-noS-pCO2=0.01/', './scenario_library/TRAPPIST-1/CO2-noS-pCO2=0.03/', './scenario_library/TRAPPIST-1/CO2-noS-pCO2=0.1/']
    extended_addresses=['./scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2=0.01/', './scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2=0.03/', './scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2=0.1/']
    
    #pCO2
    pCO2_list=['0.01', '0.03', '0.1']
    
    ###Initialize plot
    fig2, ax=plt.subplots(3,2, figsize=(8., 9.), sharey=True)
    markersizeval=5.
    linestyles=np.array(['-', '--', ':'])
    linewidthval=2.5
    colors=sns.color_palette('colorblind', 8)
    
    ###Go through loop
    
    for pCO2_ind in range(0, len(pCO2_list)):
        
        ##Build address
        truncated_concSTD=truncated_addresses[pCO2_ind]+'ConcentrationSTD.dat'
        truncated_ChemReac=truncated_addresses[pCO2_ind]+'ChemicalRate.dat'
        
        extended_concSTD=extended_addresses[pCO2_ind]+'ConcentrationSTD.dat'
        extended_ChemReac=extended_addresses[pCO2_ind]+'ChemicalRate.dat'

        ##Import reaction data
        truncated_p_centers, truncated_z_centers, truncated_deltazs, truncated_vert_reac_rates, truncated_colint_reac_rates=return_reaction_rates(truncated_concSTD, truncated_ChemReac, phot_reaction_label_list)
        extended_p_centers, extended_z_centers, extended_deltazs, extended_vert_reac_rates, extended_colint_reac_rates=return_reaction_rates(extended_concSTD, extended_ChemReac, phot_reaction_label_list)
        
        ##Import concentration data
        thunk, truncated_T_z_base, truncated_P_z_base, truncated_mc_z_s_base, truncated_mr_z_s_base, truncated_n_z_bulkatm_base=return_atm_concentrations(truncated_concSTD)
        thunk, extended_T_z_base, extended_P_z_base, extended_mc_z_s_base, extended_mr_z_s_base, extended_n_z_bulkatm_base=return_atm_concentrations(extended_concSTD)
       
        ###Plot 
        ax[pCO2_ind,0].set_title('pCO$_2$={0:s} bar'.format(pCO2_list[pCO2_ind]), x=1.1, y=1.0)
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_co2], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[0], label='CO$_2$')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_co2], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[0])
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_h2o], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[1], label='H$_2$O')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_h2o], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[1])          
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_co], truncated_p_centers*barye2bar, linewidth=3, linestyle=linestyles[0], color=colors[2], label='CO')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_co], extended_p_centers*barye2bar, linewidth=3, linestyle=linestyles[1], color=colors[2])
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_o2], truncated_p_centers*barye2bar, linewidth=3, linestyle=linestyles[0], color=colors[3], label='O$_2$')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_o2], extended_p_centers*barye2bar, linewidth=3, linestyle=linestyles[1], color=colors[3])  
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_o3], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[4], label='O$_3$')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_o3], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[4])
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_o], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[5], label='O')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_o], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[5])   
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_oh], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[6], label='OH')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_oh], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[6])   
        ax[pCO2_ind,0].plot(truncated_mr_z_s_base[:,ind_h], truncated_p_centers*barye2bar, linewidth=2, linestyle=linestyles[0], color=colors[7], label='H')
        ax[pCO2_ind,0].plot(extended_mr_z_s_base[:,ind_h], extended_p_centers*barye2bar, linewidth=2, linestyle=linestyles[1], color=colors[7])   
        ax[pCO2_ind,0].set_yscale('log')
        ax[pCO2_ind,0].set_ylabel('Dry Pressure (bar)', fontsize=14)
        ax[pCO2_ind,0].set_xscale('log')
        ax[pCO2_ind,0].set_xlim([2E-12, 2E0])
        ax[pCO2_ind,0].tick_params(axis='both', which='major', labelsize=12)        
        
        
        
        ax[pCO2_ind,1].plot(truncated_vert_reac_rates['P36']+truncated_vert_reac_rates['P37'], truncated_p_centers*barye2bar, linewidth=linewidthval, linestyle=linestyles[0], color='red', label='CO$_2$ Photolysis')
        ax[pCO2_ind,1].plot(extended_vert_reac_rates['P36']+extended_vert_reac_rates['P37'], extended_p_centers*barye2bar, linewidth=linewidthval, linestyle=linestyles[1], color='red')
        ax[pCO2_ind,1].plot(truncated_vert_reac_rates['P6'], truncated_p_centers*barye2bar, linewidth=linewidthval, linestyle=linestyles[0], color='blue', label='H$_2$O Photolysis')
        ax[pCO2_ind,1].plot(extended_vert_reac_rates['P6'], extended_p_centers*barye2bar, linewidth=linewidthval, linestyle=linestyles[1], color='blue')
        
        ax[pCO2_ind,1].set_yscale('log')
        ax[pCO2_ind,1].set_xscale('log')
        # ax[pCO2_ind,1].set_xlabel(r'Chemical Rate (cm$^{-3}$ s$^{-1}$)', fontsize=14)  
        ax[pCO2_ind,1].set_xlim([4.e-2, 4.0e7])
        ax[pCO2_ind,1].tick_params(axis='both', which='major', labelsize=12)
        
        if pCO2_ind<len(pCO2_list)-1:
            ax[pCO2_ind,0].xaxis.set_tick_params(labelbottom=False)
            ax[pCO2_ind,1].xaxis.set_tick_params(labelbottom=False)
    ax[0,0].set_ylim([1.0, 1.0E-8])
    ax[0,0].plot(0,0,linestyle=linestyles[0], color='black', label=r'$z_{max}|_{P=0.34~\mu bar}$')
    ax[0,0].plot(0,0,linestyle=linestyles[1], color='black', label='$z_{max}=100$ km')
    ax[0,1].plot(0,0,linestyle=linestyles[0], color='black', label=r'$z_{max}|_{P=0.34~\mu bar}$')
    ax[0,1].plot(0,0,linestyle=linestyles[1], color='black', label='$z_{max}=100$ km')
    ax[0,0].legend(loc='upper left', ncol=4, borderaxespad=0., fontsize=10, bbox_to_anchor=(-0.33,1.54))    
    ax[0,1].legend(loc='upper left', ncol=2, borderaxespad=0., fontsize=10,bbox_to_anchor=(-0.05,1.54))
    ax[-1,0].set_xlabel(r'Mixing Ratio (CO$_2$+N$_2$, v/v)', fontsize=14)  
    ax[-1,1].set_xlabel(r'Chemical Rate (cm$^{-3}$ s$^{-1}$)', fontsize=14)  

    
    plt.savefig('./Plots/plot_photolysis_unresolved_changedtpkzz.pdf', orientation='portrait', format='pdf')


def plot_runaway_mechanics_pCO2():
    """
    This script shows what's going on when runaway occurs by plotting various parameters as a function of pCO2
    """
    ###Specify what to load
    #Addresses
    truncated_address='./scenario_library/TRAPPIST-1/CO2-noS-pCO2='
    extended_address='./scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2='
    
    #pCO2
    pCO2_list=np.array([0.01, 0.03, 0.06, 0.1])
    
    #Species
    species_list=np.array(['O2', 'CO', 'O', 'OH', 'H'])
    species_indices=np.array([ind_o2, ind_co, ind_o, ind_oh, ind_h])
    species_colors=colors=sns.color_palette('colorblind', 5) #np.array(['green', 'grey', 'red','hotpink','orange'])
    
    #Reactions
    reaction_label_list=np.array(['R441','R192', 'R180', 'P7', 'P6','R445','M34', 'M16', 'M13', 'M18', 'R526'])
    
    oh_prod_reaction_label_list=np.array(['R441','R192', 'R180', 'P7', 'P6'])
    oh_prod_reaction_name_list=np.array([r'$O+HO_2\rightarrow OH+O_2$', r'$H+O_3\rightarrow OH+O_2$', r'$H+HO_2\rightarrow OH+OH$', r'$H_2O_2+h\nu\rightarrow OH+OH$',r'H$_2$O+h$\nu\rightarrow$ H + OH'])
    oh_prod_reaction_colors_list=colors=sns.color_palette('colorblind', 5) #np.array(['red','orange','blue', 'black', 'green']) #, 'gold' 
    
    radical_reaction_label_list=np.array(['R445','M34', 'M16', 'M13', 'M18', 'R526'])
    radical_reaction_name_list=np.array([ r'$O+OH\rightarrow O_2+H$', r'$O+O+M\rightarrow O_2+M$', r'$H+O+M\rightarrow$OH+M', r'$H+H+M\rightarrow H_2+M$', r'$H+OH+M\rightarrow H_2O+M$', r'$OH+OH\rightarrow H_2O+O$'])
    radical_reaction_colors_list=colors=sns.color_palette('colorblind',6) #np.array(['red','orange','blue', 'black', 'green', 'gold'])
    
    ###Programmatically initialize all
    conc_species_trunc={}
    conc_species_ext={}
    for specie in species_list:
        conc_species_trunc[specie]=np.zeros(np.shape(pCO2_list))
        conc_species_ext[specie]=np.zeros(np.shape(pCO2_list))
        
    colrate_rxn_trunc={}
    colrate_rxn_ext={}
    for rxn_label in reaction_label_list:
        colrate_rxn_trunc[rxn_label]=np.zeros(np.shape(pCO2_list))
        colrate_rxn_ext[rxn_label]=np.zeros(np.shape(pCO2_list))
    
    ###loop through files.
    for pCO2_ind in range(0, len(pCO2_list)):
        pCO2=pCO2_list[pCO2_ind]
        
        ##Build address
        truncated_concSTD=truncated_address+str(pCO2)+'/ConcentrationSTD.dat'
        truncated_ChemReac=truncated_address+str(pCO2)+'/ChemicalRate.dat'
        
        extended_concSTD=extended_address+str(pCO2)+'/ConcentrationSTD.dat'
        extended_ChemReac=extended_address+str(pCO2)+'/ChemicalRate.dat'

        ##Import reaction data
        truncated_p_centers, truncated_z_centers, truncated_deltazs, truncated_vert_reac_rates, truncated_colint_reac_rates=return_reaction_rates(truncated_concSTD, truncated_ChemReac, reaction_label_list)
        truncated_p_centers, extended_z_centers, extended_deltazs, extended_vert_reac_rates, extended_colint_reac_rates=return_reaction_rates(extended_concSTD, extended_ChemReac, reaction_label_list)
        
        ##Import concentration data
        thunk, truncated_T_z, truncated_P_z, truncated_mc_z_s, truncated_mr_z_s, truncated_n_z_bulkatm=return_atm_concentrations(truncated_concSTD)
        thunk, extended_T_z, extended_P_z, extended_mc_z_s, extended_mr_z_s, extended_n_z_bulkatm=return_atm_concentrations(extended_concSTD)
        
        #Extract concentration data. 
        for species_ind in range(0, len(species_list)):
            specie=species_list[species_ind]
            
            conc_species_trunc[specie][pCO2_ind]=np.sum(truncated_mr_z_s[:,species_indices[species_ind]]*truncated_n_z_bulkatm)/np.sum(truncated_n_z_bulkatm)
            conc_species_ext[specie][pCO2_ind]=np.sum(extended_mr_z_s[:,species_indices[species_ind]]*extended_n_z_bulkatm)/np.sum(extended_n_z_bulkatm)
        
        #Extract reaction data.
        
        for rxn_ind in range(0, len(reaction_label_list)):
            rxn_label=reaction_label_list[rxn_ind]
            
            colrate_rxn_trunc[rxn_label][pCO2_ind]=truncated_colint_reac_rates[rxn_label]
            colrate_rxn_ext[rxn_label][pCO2_ind]=extended_colint_reac_rates[rxn_label]
            
    
    ###Initialize plot
    fig1, (ax1, ax2, ax3)=plt.subplots(3,1, figsize=(8., 9.), sharex=True)
    markersizeval=5.
    linestyles=np.array(['-', '--'])
    linewidthval=2.5
    
    
    ax4=ax1.twinx()
    ax4.plot(0,0,linestyle=linestyles[0], color='black', label='$z_{max}|_{P=0.34~\mu bar}$')
    ax4.plot(0,0,linestyle=linestyles[1], color='black', label='$z_{max}=100$ km')
    ax4.legend(loc='upper left', ncol=2, borderaxespad=0., fontsize=10, bbox_to_anchor=(0,1.2), frameon=False)
    ax4.set_yticks([])
    
    ##Plot concentration data
    ax1.set_title('Concentrations of Key Species', loc='right')
    for species_ind in range(0, len(species_list)):
        specie=species_list[species_ind]
        ax1.plot(pCO2_list, conc_species_trunc[specie]/conc_species_trunc[specie][0], color=species_colors[species_ind], linestyle=linestyles[0], label='['+specie+']', marker='o')
        ax1.plot(pCO2_list, conc_species_ext[specie]/conc_species_ext[specie][0], color=species_colors[species_ind], linestyle=linestyles[1], marker='o')
    
    ax1.set_yscale('log')
    ax1.set_ylim([1E-2, 1E5])
    ax1.set_xscale('linear')
    ax1.legend(loc='upper left', ncol=2, borderaxespad=0., fontsize=10, frameon=False) #,bbox_to_anchor=(-0.1,1.5)
    # ax1.set_xlabel(r'pCO$_2$ (bar)', fontsize=14)  
    ax1.set_ylabel(r'Col.-Avg. Mixing Ratio'+'\n'+r'(Rel. to pCO$_2=0.01$ bar)', fontsize=12)

    ##Plot rate data: radical-radical rxns
    ax2.set_title('Main Radical-Radical Reactions', loc='right')
    for rxn_ind in range(0, len(radical_reaction_label_list)):
        rxn_label=radical_reaction_label_list[rxn_ind]
        ax2.plot(pCO2_list, colrate_rxn_trunc[rxn_label]/colrate_rxn_trunc[rxn_label][0], color=radical_reaction_colors_list[rxn_ind], linestyle=linestyles[0], label=radical_reaction_name_list[rxn_ind], marker='o') #reaction_colors[rxn_ind]
        ax2.plot(pCO2_list, colrate_rxn_ext[rxn_label]/colrate_rxn_ext[rxn_label][0], color=radical_reaction_colors_list[rxn_ind], linestyle=linestyles[1], marker='o')
        
    ax2.set_yscale('log')
    ax2.set_xscale('linear')
    ax2.legend(loc='lower left', ncol=2, borderaxespad=0., fontsize=10, frameon=False)    
    # ax2.set_xlabel(r'pCO$_2$ (bar)', fontsize=14)  
    ax2.set_ylabel(r'Col.-Int. Reaction Rate'+'\n'+r'(Rel. to pCO$_2=0.01$ bar)', fontsize=12)    
    
    ##Plot rate data: OH-production rxns
    ax3.set_title('Main OH-producing Reactions', loc='right')
    for rxn_ind in range(0, len(oh_prod_reaction_label_list)):
        rxn_label=oh_prod_reaction_label_list[rxn_ind]
        ax3.plot(pCO2_list, colrate_rxn_trunc[rxn_label]/colrate_rxn_trunc[rxn_label][0], color=oh_prod_reaction_colors_list[rxn_ind], linestyle=linestyles[0], label=oh_prod_reaction_name_list[rxn_ind], marker='o') #reaction_colors[rxn_ind]
        ax3.plot(pCO2_list, colrate_rxn_ext[rxn_label]/colrate_rxn_ext[rxn_label][0], color=oh_prod_reaction_colors_list[rxn_ind], linestyle=linestyles[1], marker='o')
        
    ax3.set_yscale('log')
    ax3.set_xscale('linear')
    ax3.set_ylim([0.1, 10.0])
    ax3.legend(loc='upper left', ncol=3, borderaxespad=0., fontsize=10, frameon=False)    
    ax3.set_xlabel(r'pCO$_2$ (bar)', fontsize=14)  
    ax3.set_ylabel(r'Col.-Int. Reaction Rate'+'\n'+r'(Rel. to pCO$_2=0.01$ bar)', fontsize=12)
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.savefig('./Plots/plot_runaway_pCO2_changedtpkzz.pdf', orientation='portrait', format='pdf')
    
    
def plot_vertrxnrates_pCO2():
    """
    Plot, for pCO2=0.01 bar and pCO2=0.1 bar, the OH-producing and rad0cal-radical reactions, for the extended and non-extended cases.
    """
    ###Specify what to load
    #Addresses
    truncated_address='./scenario_library/TRAPPIST-1/CO2-noS-pCO2='
    extended_address='./scenario_library/TRAPPIST-1/CO2-noS-bigzgrid-pCO2='
    
    #pCO2
    pCO2_list=np.array([0.01, 0.1])
    #Reactions
    reaction_label_list=np.array(['R441','R192', 'R180', 'P7','P6', 'R445','M34', 'M16', 'M13', 'M18', 'R526'])
    
    oh_prod_reaction_label_list=np.array(['R441','R192', 'R180', 'P7','P6'])
    oh_prod_reaction_name_list=np.array([ r'$O+HO_2\rightarrow OH+O_2$', r'$H+O_3\rightarrow OH+O_2$', r'$H+HO_2\rightarrow OH+OH$', r'$H_2O_2+h\nu\rightarrow OH+OH$', r'$H_2O+h\nu\rightarrow H+OH$'])
    oh_prod_reaction_colors_list=colors=sns.color_palette('colorblind', 5) #np.array(['hotpink','darkolivegreen','cyan', 'darkorchid']) #, 'gold' 
    
    radical_reaction_label_list=np.array(['R445','M34', 'M16', 'M13', 'M18', 'R526'])
    radical_reaction_name_list=np.array([ r'$O+OH\rightarrow O_2+H$', r'$O+O+M\rightarrow O_2+M$', r'$H+O+M\rightarrow$OH+M', r'$H+H+M\rightarrow H_2+M$', r'$H+OH+M\rightarrow H_2O+M$', r'$OH+OH\rightarrow H_2O+O$'])
    radical_reaction_colors_list=colors=sns.color_palette('colorblind', 6) #np.array(['red','orange','blue', 'black', 'green', 'gold'])
    
    ###Programmatically initialize all
    vertrate_rxn_trunc={}
    vertrate_rxn_ext={}
    zcenters_rxn_trunc={}
    zcenters_rxn_ext={}
    pcenters_rxn_trunc={}
    pcenters_rxn_ext={}
    
    ###Initialize Plot
    fig1, ax=plt.subplots(len(pCO2_list),2, figsize=(8., 8.), sharex=False, sharey=True)
    markersizeval=5.
    linestyles=np.array(['-', '--', ':'])
    linewidthval=2
    
    ###loop through pCO2.
    for pCO2_ind in range(0, len(pCO2_list)):
        pCO2=pCO2_list[pCO2_ind]
        
        ##Build address
        truncated_concSTD=truncated_address+str(pCO2)+'/ConcentrationSTD.dat'
        truncated_ChemReac=truncated_address+str(pCO2)+'/ChemicalRate.dat'
        
        extended_concSTD=extended_address+str(pCO2)+'/ConcentrationSTD.dat'
        extended_ChemReac=extended_address+str(pCO2)+'/ChemicalRate.dat'

        ##Import reaction data
        truncated_p_centers, truncated_z_centers, truncated_deltazs, truncated_vert_reac_rates, truncated_colint_reac_rates=return_reaction_rates(truncated_concSTD, truncated_ChemReac, reaction_label_list)
        extended_p_centers, extended_z_centers, extended_deltazs, extended_vert_reac_rates, extended_colint_reac_rates=return_reaction_rates(extended_concSTD, extended_ChemReac, reaction_label_list)
        
        zcenters_rxn_trunc[str(pCO2)]=truncated_z_centers
        zcenters_rxn_ext[str(pCO2)]=extended_z_centers
        pcenters_rxn_trunc[str(pCO2)]=truncated_p_centers
        pcenters_rxn_ext[str(pCO2)]=extended_p_centers
        
        #Extract reaction data.
        for rxn_ind in range(0, len(reaction_label_list)):
            rxn_label=reaction_label_list[rxn_ind]
            rxn_pCO2_label=rxn_label+str(pCO2)
            
            vertrate_rxn_trunc[rxn_pCO2_label]=truncated_vert_reac_rates[rxn_label]
            vertrate_rxn_ext[rxn_pCO2_label]=extended_vert_reac_rates[rxn_label]
        
        #Plot OH-prod reaction data
        for rxn_ind in range(0, len(oh_prod_reaction_label_list)):
            rxn_label=oh_prod_reaction_label_list[rxn_ind]
            rxn_pCO2_label=rxn_label+str(pCO2)
            
            ax[pCO2_ind, 0].plot(vertrate_rxn_trunc[rxn_pCO2_label], pcenters_rxn_trunc[str(pCO2)]*barye2bar, linewidth=linewidthval, linestyle=linestyles[0], color=oh_prod_reaction_colors_list[rxn_ind], label=oh_prod_reaction_name_list[rxn_ind])
            ax[pCO2_ind, 0].plot(vertrate_rxn_ext[rxn_pCO2_label], pcenters_rxn_ext[str(pCO2)]*barye2bar, linewidth=linewidthval, linestyle=linestyles[1], color=oh_prod_reaction_colors_list[rxn_ind])
        ax[pCO2_ind,0].set_xlim([1.e-6, 1.0e8])
        ax[pCO2_ind,0].set_ylabel('Pressure (bar)', fontsize=14)
        ax[pCO2_ind,0].set_xscale('log')
        ax[pCO2_ind,0].tick_params(axis='both', which='major', labelsize=12)
        ax[pCO2_ind,0].set_title('pCO$_2$='+str(pCO2)+' bar', x=1.1, y=1.0)

        #Plot radical-radical reaction data
        for rxn_ind in range(0, len(radical_reaction_label_list)):
            rxn_label=radical_reaction_label_list[rxn_ind]
            rxn_pCO2_label=rxn_label+str(pCO2)
            
            if rxn_label=='M34':
                linewidthval=3
            else:
                linewidthval=2
            
            ax[pCO2_ind, 1].plot(vertrate_rxn_trunc[rxn_pCO2_label], pcenters_rxn_trunc[str(pCO2)]*barye2bar, linewidth=linewidthval, linestyle=linestyles[0], color=radical_reaction_colors_list[rxn_ind], label=radical_reaction_name_list[rxn_ind])
            ax[pCO2_ind,1].plot(vertrate_rxn_ext[rxn_pCO2_label], pcenters_rxn_ext[str(pCO2)]*barye2bar, linewidth=linewidthval, linestyle=linestyles[1], color=radical_reaction_colors_list[rxn_ind])
        ax[pCO2_ind,1].set_xlim([1.e-9, 1.0e6])
        ax[pCO2_ind,1].set_xscale('log')
        ax[pCO2_ind,1].tick_params(axis='both', which='major', labelsize=12)
        # ax[pCO2_ind,1].set_title('pCO$_2$='+str(pCO2)+ ' bar')
                
        if pCO2_ind<len(pCO2_list)-1:
            ax[pCO2_ind,0].xaxis.set_tick_params(labelbottom=False)
            ax[pCO2_ind,1].xaxis.set_tick_params(labelbottom=False)


    #Configure plot
    ax[0,0].set_ylim([1.0, 1.0E-8])
    ax[0,0].set_yscale('log')
    ax[-1,0].set_xlabel(r'Chemical Rate (cm$^{-3}$ s$^{-1}$)', fontsize=14) 
    ax[-1,1].set_xlabel(r'Chemical Rate (cm$^{-3}$ s$^{-1}$)', fontsize=14) 
    ax[0,0].legend(loc='upper left', ncol=2, borderaxespad=0, fontsize=10, bbox_to_anchor=(-0.34, 1.35))
    ax[0,1].legend(loc='upper left', ncol=2, borderaxespad=0, fontsize=10, bbox_to_anchor=(-0.13, 1.35))  
    plt.subplots_adjust(hspace = 0.1)
    plt.savefig('./Plots/plot_vertrxn_pCO2_changedtpkzz.pdf', orientation='portrait', format='pdf')

plot_failure_to_resolve_photolysis_rates()
# plot_runaway_mechanics_pCO2()
# plot_vertrxnrates_pCO2()