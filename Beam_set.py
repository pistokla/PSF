#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:06:37 2024

@author: perivar
"""

#import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt

#%% import necessary modules:
#import necessary modules:
import Bjelker as Bj #importerer globale parametre og funksjoner


#Importing FORM analysis modules
#from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
#from FORM_V_beam_module import FORM_V_beam

#Partialfactors methods is imported from Partial_safety_factors.py
#import Partial_safety_factors as PF


import B_chi_plot as BC

import Alphas as AL


#%% Printing global parameters for control:
#Printing global parameters for control:
alpha_cc=Bj.alpha_cc
B_t=Bj.B_t
alpha_E=Bj.alpha_E
alpha_R=Bj.alpha_R
alpha_E2=Bj.alpha_E2
alpha_R2=Bj.alpha_R2
delta=Bj.delta
print('alpha_cc={},B_t={},[alpha_E,alpha_R, alpha_E2 ,alpha_R2]={},delta={}'.format(alpha_cc,B_t,np.round([alpha_E,alpha_R, alpha_E2 ,alpha_R2],2),delta))


theta_R_bending=Bj.theta_R_bending
theta_R_shear=Bj.theta_R_shear
theta_E=Bj.theta_E


#%% Specifiying limits and intermediate values for examples.
#Specifiying limits and intermediate values for examples.

hs=[200,500,1000]
bs=[150,300,500]
c=50

mu_fcs=[15,30,60]

mu_fys=[200,400,600]

muGps=[3,10]

L=8000

TYPEs=['indoor','roof']

Vfcs=[0.1,0.2]
Vfss=[0.01,0.05,0.1]

#%% Creating a set of beams including all different combinations.
#Creating a set of beams including all different combinations. 

#I could be discussed if all combinations is realistic...
#Unrealistic beams has been tried discarded.

Beams=[]
for h in hs:
    for b in bs:
        for mu_fy in mu_fys:
            for mu_fc in mu_fcs:
                for muGp in muGps:
                    for TYPE in TYPEs:
                        for Vfc in Vfcs:
                            for Vfs in Vfss:
                                fs = Bj.Create_dist_variable(mu_fy, Vfs, 'Lognormal')  # Creating example steel
                                Bj.add_fk(fs)
                                
                                fc = Bj.Create_dist_variable(mu_fc, Vfc, 'Lognormal')  # Creating example steel
                                Bj.add_fk(fc)
                                
                                
                                Beam = Bj.Create_beam_dict(h, b, b, (h-c), Bj.Create_dist_variable(0, 0, 'Normal'), Bj.Create_dist_variable(0, 0, 'Normal'), fc, fs, TYPE,L)
                                Beam['Gp']=muGp
                                Beams+=[Beam]

#rhos are calculated later

#Interesting to look more at cases with different CoV.

#%% Function that create chi and loading arrays



def create_chi_loading2(Beam,muGp,chiLim=[0.05,0.7]):
    chis=np.arange(chiLim[0],chiLim[1],0.01) 
    Loadings=[]
    
    Gs=Bj.calc_Gs(Beam,25,0.05)
    Gs['LoadType']='SelfWeight'
    Gp=Bj.Create_dist_variable(muGp, 0.1, 'Normal') #!!! 
    Gp['LoadType']='Permanent'

    Bj.add_qk(Gs, 0.5)
    Bj.add_qk(Gp, 0.5)

    for i, chi in enumerate(chis):
        Qk=chi/(1-chi)*(Gs['qk']+Gp['qk']) #Qk is calculated from the definition of chi.
        #Depending on the beam type, either imposed load or snowload is added to the beam in addition to self-weight and permanent load.
        if Beam['type']=='indoor':
            mQ = 0.53*1*Qk/1.35 #5 year mean value is calculated from characteristic load, JRC table values
            vQ = np.sqrt(0.49**2+0.10**2) #!!! CoV_Q from JRC table
        
            Q = Bj.Create_dist_variable(mQ, vQ, 'Gumbel')
            Q['LoadType']='Imposed'
            Q['qk']=Qk
        
        if Beam['type']=='roof':    
            mS = 0.39*0.81*Qk/0.82 #1 year mean value is calculated from characteristic load, JRC table values
            vS = np.sqrt(0.51**2+0.26**2)  #CoV_Q from JRC table

            Q = Bj.Create_dist_variable(mS, vS, 'Gumbel')
            Q['LoadType']='Climatic'
            Q['qk']=Qk  
        Loadings+=[[Gs,Gp,Q]]
    Loadings=np.array(Loadings)
    
    return chis, Loadings


#%%  Function that calculate necessary rho, with simple approach. And resulting beta and saves this as a dictionary, for a list of beams. Bending.
#Function that calculate necessary rho, with simple approach. And resulting beta and saves this as a dictionary, for a list of beams. Bending.
def save_Beams_inDict_M(Beams):
    Beam_dict={}
    for i,Beam in enumerate(Beams):
        chis,Loadings=create_chi_loading2(Beam,Beam['Gp'])
        rhols=BC.calc_rhols(Beam,Loadings)
        mu_rhol=list(map(lambda x:x['moments'][0], rhols))
        
        t=False
        if True in np.isnan(mu_rhol):
            t=True
       
        
        
        Beam_dict[str(i+1)]={}
        
        Beam_dict[str(i+1)]['chi']=chis
        Beam_dict[str(i+1)]['Loadings']=Loadings
        Beam_dict[str(i+1)]['rhos']=rhols
        
        if np.max(mu_rhol) >= 0.02: #If rhol is greater than 0.02 the beam is excluded and beta is not calculated
            Beam_dict[str(i+1)]['state']='unrealistic' #A it is fair to assume that such a beam would not be created.
        elif t== True: #The beams is excluded if the rho func returns nan, have to check that this corresponds with to little capacity.
          Beam_dict[str(i+1)]['state']='unrealistic'
        else:
            Beam_dict[str(i+1)]['state']='realistic'
            Beam_dict[str(i+1)]['betas']=BC.calc_Betas_M(Beam,Loadings,Beam_dict[str(i+1)]['rhos'])
    
    return Beam_dict
    #print(Beam_dict[str(i+1)]['state'])



#%% Function that calculate necessary rho, with simple approach. And resulting beta and saves this as a dictionary, for a list of beams. Shear.
#Function that calculate necessary rho, with simple approach. And resulting beta and saves this as a dictionary, for a list of beams. Shear.

def save_Beams_inDict_V(Beams):
    Beam_dict={}
    for i,Beam in enumerate(Beams):
        chis,Loadings=create_chi_loading2(Beam,Beam['Gp'],chiLim=[0.05,0.6])
        rhows=BC.calc_rhows(Beam,Loadings)
        mu_rhow=list(map(lambda x:x['moments'][0], rhows))
        
        t=False
        if True in np.isnan(mu_rhow):
            t=True
       
        
        
        Beam_dict[str(i+1)]={}
        
        Beam_dict[str(i+1)]['chi']=chis
        Beam_dict[str(i+1)]['Loadings']=Loadings
        Beam_dict[str(i+1)]['rhos']=rhows
        
        Beam['rho_w']=rhows[-1] #Ikkje vanntett om last ikkje er størst for høgast chi
        
        if np.max(mu_rhow) >= 0.005: #If rhow is greater than 0.01 the beam is excluded and beta is not calculated
            Beam_dict[str(i+1)]['state']='unrealistic' #A it is fair to assume that such a beam would not be created.
        elif t== True:
            Beam_dict[str(i+1)]['state']='unrealistic'
        elif Bj.cot_t(Beam) <2.5:
            Beam_dict[str(i+1)]['state']='unrealistic-cot'

        else:
            Beam_dict[str(i+1)]['state']='realistic'
            Beam_dict[str(i+1)]['betas']=BC.calc_Betas_V(Beam,Loadings,Beam_dict[str(i+1)]['rhos'])
        
    
    return Beam_dict
    #print(Beam_dict[str(i+1)]['state'])


#%% Dictionary with rho and beta are created for the different beams for bending.
Bending_dict=save_Beams_inDict_M(Beams)

#%%Dictionary with rho and beta are created for the different beams for shear.
Shear_dict=save_Beams_inDict_V(Beams)

#%% Betas plotted against chi for the different beams to control that different beams gives similar results.
#Betas plotted against chi for the different beams to control that different beams gives similar results.

def plot_Betas_realistic(chis,a_dict,Bruddform,B_t=B_t):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    
    for metode in a_dict:
        if a_dict[metode]['state'] == 'realistic':
            plt.plot(chis,a_dict[metode]['betas'],label=metode)
   
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    
    plt.grid()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    plt.ylim([2.7,4.7])
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,ncol=8) 
    plt.show() 
    
#%% Plotting necessary rho for the different beam, her also "unrealistic" beams are included.
#Plotting necessary rho for the different beam, her also "unrealistic" beams are included.
BC.plot_rho_d(Bending_dict['1']['chi'],Bending_dict)

#%%
plot_Betas_realistic(Bending_dict['1']['chi'],Bending_dict,'bending',B_t=B_t)
    
#%% Plotting necessary rho for the different beam, her also "unrealistic" beams are included.
#Plotting necessary rho for the different beam, her also "unrealistic" beams are included.
#BC.plot_rho_d(Shear_dict['1']['chi'],Shear_dict)

#%%
plot_Betas_realistic(Shear_dict['1']['chi'],Shear_dict,'bending',B_t=B_t)

#%% To get a overview of which beams that have been excluded histograms of the value of different variables for the excluded beams are plotted.
# To get a overview of which beams that have been excluded histograms of the value of different variables for the excluded beams are plotted.

def control_unreal(Beams,Beam_dict):
    uhs=[]
    ubs=[]
    ufcs=[]
    ufs=[]
    uGps=[]
    utypes=[]
    uVfcs=[]
    uVfss=[]
    
    
    n_unreal=0
    n_cot=0
    for i,Beam in enumerate(Beams):
        if Beam_dict[str(i+1)]['state']=='unrealistic':
            uhs+=[Beam['h']]
            ubs+=[Beam['b']]
            ufcs+=[Beam['fc']['moments'][0]]
            ufs+=[Beam['fs']['moments'][0]]
            uGps+=[Beam['Gp']]
            utypes+=[Beam['type']]
            uVfcs+=[Beam['fc']['moments'][2]]
            uVfss+=[Beam['fs']['moments'][2]]
            
            n_unreal+=1
        if Beam_dict[str(i+1)]['state']=='unrealistic-cot':
            uhs+=[Beam['h']]
            ubs+=[Beam['b']]
            ufcs+=[Beam['fc']['moments'][0]]
            ufs+=[Beam['fs']['moments'][0]]
            uGps+=[Beam['Gp']]
            utypes+=[Beam['type']]
            uVfcs+=[Beam['fc']['moments'][2]]
            uVfss+=[Beam['fs']['moments'][2]]
            
            n_cot+=1
            n_unreal+=1
    
    print(n_unreal) #Number of excluded beams
    print(n_cot)
    
    #plt.title('h') 
    plt.hist(uhs)
    plt.show()
    
    #plt.title('b') 
    plt.hist(ubs)
    plt.show()
    
    #plt.title('fc')       
    plt.hist(ufcs)
    plt.show()
    
    #plt.title('fs') 
    plt.hist(ufs)
    plt.show()
    
    plt.hist(uGps)
    plt.show()
    
    plt.hist(utypes)
    plt.show()
    
    plt.hist(uVfcs)
    plt.show()
    
    plt.hist(uVfss)
    plt.show()
    
    

    return

#Kan med fordel lete etter betre måte å vurdere discarded beams.
#%% Plotting histograms for bending.
control_unreal(Beams,Bending_dict)

#For bending the histogram indicates that to low height is the main reason that a beam can not provide sufficient strength with realistic rho.
#%% Plotting histograms for shear.
control_unreal(Beams,Shear_dict)
#For shear the histogram indicates that both to low fs, height and width is important reasons that a beam can not provide sufficient strength with realistic rho.

#%% In addition to the simple method one makes the same plots for the best alpha approximations, bending CoV-fc0 and shear Sc3
# Saving betas for the different beams with cov-fc0 bending alpha aproxx.

def save_Beams_inDict_M_SD(Beams):
    Beam_dict={}
    for i,Beam in enumerate(Beams):
        chis,Loadings=create_chi_loading2(Beam,Beam['Gp'])
        rhols=AL.calc_rhols_alphas(Beam,Loadings,AL.alphas_M_SD2)
        mu_rhol=list(map(lambda x:x['moments'][0], rhols))
        
        t=False
        if True in np.isnan(mu_rhol):
            t=True
       
        
        
        Beam_dict[str(i+1)]={}
        
        Beam_dict[str(i+1)]['chi']=chis
        Beam_dict[str(i+1)]['Loadings']=Loadings
        Beam_dict[str(i+1)]['rhos']=rhols
        
        if np.max(mu_rhol) >= 0.02: #If rhol is greater than 0.05 the beam is excluded and beta is not calculated
            Beam_dict[str(i+1)]['state']='unrealistic' #A it is fair to assume that such a beam would not be created.
        elif t== True: #The beams is excluded if the rho func returns nan, have to check that this corresponds with to little capacity.
          Beam_dict[str(i+1)]['state']='unrealistic'
        
        else:
            Beam_dict[str(i+1)]['state']='realistic'
            Beam_dict[str(i+1)]['betas']=BC.calc_Betas_M(Beam,Loadings,Beam_dict[str(i+1)]['rhos'])
    
    return Beam_dict

#%% Saving betas for the different beams with SC3 shear alpha aproxx.
def save_Beams_inDict_V_SD(Beams):
    Beam_dict={}
    for i,Beam in enumerate(Beams):
        chis,Loadings=create_chi_loading2(Beam,Beam['Gp'],chiLim=[0.05,0.6])
        rhows=AL.calc_rhows_alphas(Beam,Loadings,AL.alphas_V_SD2)
        mu_rhow=list(map(lambda x:x['moments'][0], rhows))
        
        t=False
        if True in np.isnan(mu_rhow):
            t=True
       
        
        
        Beam_dict[str(i+1)]={}
        
        Beam_dict[str(i+1)]['chi']=chis
        Beam_dict[str(i+1)]['Loadings']=Loadings
        Beam_dict[str(i+1)]['rhos']=rhows
        
        Beam['rho_w']=rhows[-1] #Ikkje vanntett om last ikkje er størst for høgast chi
        
        if np.max(mu_rhow) >= 0.005: #If rhow is greater than 0.01 the beam is excluded and beta is not calculated
            Beam_dict[str(i+1)]['state']='unrealistic' #A it is fair to assume that such a beam would not be created.
        elif t== True:
            Beam_dict[str(i+1)]['state']='unrealistic'
        elif Bj.cot_t(Beam) <2.5:
            Beam_dict[str(i+1)]['state']='unrealistic-cot'
        else:
            Beam_dict[str(i+1)]['state']='realistic'
            Beam_dict[str(i+1)]['betas']=BC.calc_Betas_V(Beam,Loadings,Beam_dict[str(i+1)]['rhos'])
    
    return Beam_dict
    #print(Beam_dict[str(i+1)]['state'])
#%% Saving and plotting betas with approximated alphas for bending and shear.
MSD_dict=save_Beams_inDict_M_SD(Beams)

#%%

plot_Betas_realistic(MSD_dict['1']['chi'],MSD_dict,'bending',B_t=B_t)

#%%
VSD_dict=save_Beams_inDict_V_SD(Beams)
#%%
plot_Betas_realistic(VSD_dict['1']['chi'],VSD_dict,'bending',B_t=B_t)

#%%
#BC.plot_rho_d(VSD_dict['1']['chi'],VSD_dict)

#%% Plotting histograms.
control_unreal(Beams,VSD_dict)

#%% Prøver å finne ut kvifor nokre bjelke som gir låge beta ved bending SD2:
#MSD_dict2=MSD_dict
#%%    
for i,k in enumerate(MSD_dict):
    if MSD_dict[k]['state']=='realistic':
        if np.min(MSD_dict[k]['betas'][30:])<2.5:
            print(np.min(MSD_dict[k]['betas'][30:]))
            print(Beams[i])
            print()
            
            

