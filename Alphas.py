#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:02:20 2024

@author: perivar
"""

#import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#%% import necessary modules:
#import necessary modules:
import Bjelker as Bj #importerer globale parametre og funksjoner


#Importing FORM analysis modules
from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
from FORM_V_beam_module import FORM_V_beam

#Partialfactors methods is imported from Partial_safety_factors.py
import Partial_safety_factors as PF


import B_chi_plot as BC

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

#%% Importing example beams
#Importing example beams

from B_chi_plot import BeamM, Loadings, chis,  rhols2, rhols_RA, Betas2,BetasRA
from B_chi_plot import BeamV, LoadingsV, chisV, rhows_RAV,Betas_RAV,rhowsV_Dmodel,Betas_V2_model
from B_chi_plot import plot_Betas_d, plot_rho_d
#%% Creating a_dict for beam M with standardized and alpha values.

alphasM={}

alphasM['Standardized']={}
alphasM['Standardized']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['Standardized']['rhos']=rhols2
alphasM['Standardized']['betas']=Betas2

#alphasM['Standardized']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['FORM']={}
alphasM['FORM']['rhos']=rhols_RA
alphasM['FORM']['betas']=BetasRA



#%% Creating a_dict for beam V with standardized and alpha values.
alphasV={}


alphasV['Standardized']={}
alphasV['Standardized']['alpha']=[alpha_R2, alpha_R,alpha_R, alpha_R, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Standardized']['rhos']=rhowsV_Dmodel
alphasV['Standardized']['betas']=Betas_V2_model

#alphasM['Standardized']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['FORM']={}
alphasV['FORM']['rhos']=rhows_RAV
alphasV['FORM']['betas']=Betas_RAV


    




#%% Estimating alphas SD/Sum(SD) M  a_fcd=0

def alphas_M_SD(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_l']['moments'][2]**2#+Beam['fc']['moments'][2]**2
    ScR=theta_R_bending['moments'][0]*Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]*(1-0.5+Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]/
                                                                                            (alpha_cc*Beam['fc']['moments'][0]))*Beam['b']*Beam['d']**2
    
    
    SumE=(Beam['L']**2/8)**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumE+SumR*ScR**2)
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_l']['moments'][2]/Sum
    alphas[2]=0#ScR*Beam['fc']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_bending['moments'][2]/Sum
    
    
    alphas[4]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-Beam['L']**2/8*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    #alphas=alphas/np.linalg.norm(alphas)*1.05
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_SD(BeamM,theta_R_bending,theta_E,Print_results=True)
#USE THIS EVT with different scaling, and excluded fc0

#%% Scaled version of SD
def alphas_M_SD2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_l']['moments'][2]**2#+Beam['fc']['moments'][2]**2
    ScR=theta_R_bending['moments'][0]*Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]*(1-0.5+Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]/
                                                                                            (alpha_cc*Beam['fc']['moments'][0]))*Beam['b']*Beam['d']**2
    
    
    SumE=(Beam['L']**2/8)**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumE+SumR*ScR**2)
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_l']['moments'][2]/Sum
    alphas[2]=0#ScR*Beam['fc']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_bending['moments'][2]/Sum
    
    
    alphas[4]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-Beam['L']**2/8*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    alphas=alphas/np.linalg.norm(alphas)*1.1
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_SD2(BeamM,theta_R_bending,theta_E,Print_results=True)











#%% Estimating alphas SD shear as for linear normal USE THIS
def alphas_V_SD(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_w']['moments'][2]**2
    #SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    
    ScE=(Beam['L']/2-Beam['d'])**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+
                                 theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumR*ScR**2+ScE)
   
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_shear['moments'][2]/Sum
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    #alphas=alphas/np.linalg.norm(alphas)*1.1
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_V_SD(BeamV,theta_R_shear,theta_E,Print_results=True)

#Could be tried with different scaling 

#%% Estimating alphas SD2 shear as for linear normal USE THIS

def alphas_V_SD2(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_w']['moments'][2]**2
    #SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    
    ScE=(Beam['L']/2-Beam['d'])**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+
                                 theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumR*ScR**2+ScE)
   
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_shear['moments'][2]/Sum
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    alphas=alphas/np.linalg.norm(alphas)*1.1
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas
    
alphas_V_SD2(BeamV,theta_R_shear,theta_E,Print_results=True)

#Could be tried with different scaling 



#%% Function that calculate the value of rho_w, such that the semi-probabilistic shear control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_w, such that the semi-probabilistic shear control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach, with different alphas for all values from simplification.
#Simply supported beam is assumed.
def calc_rhows_alphas(Beam,Loadings,a_func):
    rhows=[]
    Beam['rho_w']=Bj.Create_dist_variable(0.001, 0, 'Normal') #Guesses initial rho_w
    Beam['cot_t']=Bj.cot_t(Beam,Print_results=False) 
    for loading in Loadings:
        Beam['Loading']=loading
        alphas=a_func(Beam,theta_R_shear,theta_E,Print_results=False)
        PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas,B_t,delta,Print_results=False)
    
        L=Beam['L']
        q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
           +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
           +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
        d=Beam['d']
        bw=Beam['bw']
        
        
        
        fyd=Beam['fs']['fk']/Beam['gamma_M_s'][1]
        Ed=q*(L/2-d)
        
        #rho_w is calculated by eq. 5.15?
        rho_w=Ed/(bw*0.9*d*fyd*2.5) #!!!  Cot(theta)=2.5 is assumed. The Beta-chi plots clearly shows where this assumption becomes wrong. But for normal chi range it seems okay.
        rho_w=Bj.Create_dist_variable(rho_w, 0, 'Normal') #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhows+=[rho_w]
        Beam['rho_w']=rho_w
        Beam['cot_t']=Bj.cot_t(Beam,Print_results=False)
    return np.array(rhows)

#%% Calculating necessary reinforcement for the different methods.


rhows_SD=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_SD)
rhows_SD2=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_SD2)

#%% Function that calculate the value of rho_l, such that the semi-probabilistic bending control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_l, such that the semi-probabilistic bending control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach, with different alphas for all values from simplification.

def calc_rhols_alphas(Beam,Loadings,a_func):
    rhols=[]
    Beam['rho_l']=Bj.Create_dist_variable(0.005, 0, 'Normal') #Guesses initial rho_l
    for loading in Loadings:
        Beam['Loading']=loading
        alphas=a_func(Beam,theta_R_bending,theta_E,Print_results=False)
        PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas,B_t,delta,Print_results=False)
        #Partial factors are update for each loading, as the variation do not change this is actually unecessary but it guarantees that they are correctly defined.
        L=Beam['L']
        q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
           +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
           +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
        d=Beam['d']
        b=Beam['b']
        
        
        
        fcd=alpha_cc*Beam['fc']['fk']/Beam['gamma_M_c'][0]
        fyd=Beam['fs']['fk']/Beam['gamma_M_s'][0]
        Ed=q*L**2/8
        
        rho_l=fcd/fyd*(1-np.sqrt(1-2*Ed/(b*d**2*fcd))) #rho_l is calculated by eq. 5.14?
        rho_l=Bj.Create_dist_variable(rho_l, 0, 'Normal')  #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhols+=[rho_l]
        Beam['rho_l']=rho_l
        
    return np.array(rhols)


#%% Calculating necessary reinforcement for the different methods.

rhols_SD=calc_rhols_alphas(BeamM,Loadings,alphas_M_SD)
rhols_SD2=calc_rhols_alphas(BeamM,Loadings,alphas_M_SD2)


#%% Calculating resulting BETA for the different methods.

Betas_SDM=BC.calc_Betas_M(BeamM, Loadings, rhols_SD)
Betas_SDM2=BC.calc_Betas_M(BeamM, Loadings, rhols_SD2)

#%% Calculating resulting BETA for the different methods.

Betas_SD=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SD)
Betas_SD2=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SD2)



#%% Adding the different methods and their results to the a_dicts M.

alphasM['SD']={}
alphasM['SD']['rhos']=rhols_SD
alphasM['SD']['betas']=Betas_SDM


alphasM['SD-2']={}
alphasM['SD-2']['rhos']=rhols_SD2
alphasM['SD-2']['betas']=Betas_SDM2


#%% Adding the different methods and their results to the a_dicts V.


alphasV['SD']={}
alphasV['SD']['rhos']=rhows_SD
alphasV['SD']['betas']=Betas_SD

alphasV['SD-2']={}
alphasV['SD-2']['rhos']=rhows_SD2
alphasV['SD-2']['betas']=Betas_SD2
#%% Plot rho_l for the separate approach with different alphas against chi
# Plot rho_l for the separate approach with different alphas against chi


plot_rho_d(chisV,alphasV)
#%% The most interesting plot: Betas against chi for the three approaches.
#The most interesting plot: Betas against chi for the three approaches.


plot_Betas_d(chis,alphasM,'bending',B_t=B_t) #Beta-chi BeamM bending



#%%
plot_Betas_d(chisV,alphasV,'shear',B_t=B_t)

#%% Calc and save alphas

def save_alphas(Beam,Loadings,a_func,Bruddform):
    alphas=[]
    gammas=[]
    if Bruddform == 'bending':
        for loading in Loadings:
            Beam['Loading']=loading
            alphas+=[a_func(Beam,theta_R_bending,theta_E,Print_results=False)]
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[-1],B_t,delta,Print_results=False)
            
            gammas+=[[Beam['gamma_M_s'][0],Beam['gamma_M_c'][0],Beam['Loading'][0]['gamma'],Beam['Loading'][1]['gamma'],Beam['Loading'][2]['gamma']]]
            
    if Bruddform == 'shear':
        for loading in Loadings:
            Beam['Loading']=loading
            alphas+=[a_func(Beam,theta_R_shear,theta_E,Print_results=False)]
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[-1],B_t,delta,Print_results=False)
            
            gammas+=[[Beam['gamma_M_s'][1],Beam['gamma_M_c'][1],Beam['Loading'][0]['gamma'],Beam['Loading'][1]['gamma'],Beam['Loading'][2]['gamma']]]
    
    return np.array(alphas),np.array(gammas)
#%% calc and save alphas FORM     
def save_alphas_FORM(Beam,Loadings,Bruddform):
    alphas=[]
    gammas=[]
    if Bruddform == 'bending':
        rhols=[]
        Beam['rho_l']=Bj.Create_dist_variable(0.005, 0, 'Normal') #Guesses initial rho_l
        for loading in Loadings:
            Beam['Loading']=loading
            a_FORM,beta=FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=alpha_cc,Print_results=False)
            alphas+=[a_FORM*-1]
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[-1],B_t,delta,Print_results=False)
            
            gammas+=[[Beam['gamma_M_s'][0],Beam['gamma_M_c'][0],Beam['Loading'][0]['gamma'],Beam['Loading'][1]['gamma'],Beam['Loading'][2]['gamma']]]
            
            L=Beam['L']
            q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
               +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
               +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
            d=Beam['d']
            b=Beam['b']
            
            
            
            fcd=alpha_cc*Beam['fc']['fk']/Beam['gamma_M_c'][0]
            fyd=Beam['fs']['fk']/Beam['gamma_M_s'][0]
            Ed=q*L**2/8
            
            rho_l=fcd/fyd*(1-np.sqrt(1-2*Ed/(b*d**2*fcd))) #rho_l is calculated by eq. 5.14?
            rho_l=Bj.Create_dist_variable(rho_l, 0, 'Normal')  #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
            rhols+=[rho_l]
            Beam['rho_l']=rho_l
    if Bruddform == 'shear':
        rhows=[]
        Beam['rho_w']=Bj.Create_dist_variable(0.001, 0, 'Normal') #Guesses initial rho_w
        Beam['cot_t']=Bj.cot_t(Beam,Print_results=False) 
        for loading in Loadings:
            Beam['Loading']=loading
            alpha_V,beta=FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=alpha_cc,Print_results=False)
            a_FORM=np.zeros(8)
            a_FORM[:2]=alpha_V[:2]
            a_FORM[-5:]=alpha_V[-5:]
            alphas+=[a_FORM*-1]
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[-1],B_t,delta,Print_results=False)
        
            gammas+=[[Beam['gamma_M_s'][1],Beam['gamma_M_c'][1],Beam['Loading'][0]['gamma'],Beam['Loading'][1]['gamma'],Beam['Loading'][2]['gamma']]]
        
            L=Beam['L']
            q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
               +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
               +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
            d=Beam['d']
            bw=Beam['bw']
            
            
            
            fyd=Beam['fs']['fk']/Beam['gamma_M_s'][1]
            Ed=q*(L/2-d)
            
            #rho_w is calculated by eq. 5.15?
            rho_w=Ed/(bw*0.9*d*fyd*2.5) #!!!  Cot(theta)=2.5 is assumed. The Beta-chi plots clearly shows where this assumption becomes wrong. But for normal chi range it seems okay.
            rho_w=Bj.Create_dist_variable(rho_w, 0, 'Normal') #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
            rhows+=[rho_w]
            Beam['rho_w']=rho_w
            Beam['cot_t']=Bj.cot_t(Beam,Print_results=False)
            
    return np.array(alphas),np.array(gammas)
#%% Adding alphas to the a_dict


alphasV['SD']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SD,'shear')[0]
alphasV['SD-2']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SD2,'shear')[0]
#alphasV['CoV']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_cov,'shear')[1]
#alphasV['CoV-2']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_cov2,'shear')[1]
       
alphasV['FORM']['alphas']=save_alphas_FORM(BeamV,LoadingsV,'shear')[0]
alphasV['FORM']['gammas']=save_alphas_FORM(BeamV,LoadingsV,'shear')[1]

#%%Adding alphas to the a_dict


alphasM['SD']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SD,'bending')[0]
alphasM['SD-2']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SD2,'bending')[0]

alphasM['FORM']['alphas']=save_alphas_FORM(BeamM,Loadings,'bending')[0]
alphasM['FORM']['gammas']=save_alphas_FORM(BeamM,Loadings,'bending')[1]



#%% Function for plotting resulting alphas for a method, with results saved in alpha dict.
#Function for plotting resulting alphas for a method, with results saved in alpha dict.
plt.rcParams.update({'font.size': 20})
def plot_alpha_d(chis,a_dict,metode,Bruddform):
    plt.figure(figsize=(10,7))
    
    
    #for i, alpha in enumerate():
        
    if Bruddform=='bending':
    
        plt.plot(chis,a_dict[metode]['alphas'][:,0],label=r'$f_y$')
        plt.plot(chis,a_dict[metode]['alphas'][:,1],label=r'$\rho_l$')
        plt.plot(chis,a_dict[metode]['alphas'][:,2],label=r'$f_c$')
    if Bruddform=='shear':
    
        plt.plot(chis,a_dict[metode]['alphas'][:,0],label=r'$f_y$')
        plt.plot(chis,a_dict[metode]['alphas'][:,1],label=r'$\rho_w$')
       
    
    plt.plot(chis,a_dict[metode]['alphas'][:,3],label=r'$\theta_R$')
    plt.plot(chis,a_dict[metode]['alphas'][:,4],label=r'$G_s$')
    plt.plot(chis,a_dict[metode]['alphas'][:,5],label=r'$G_p$')
    plt.plot(chis,a_dict[metode]['alphas'][:,6],label=r'$Q$')
    plt.plot(chis,a_dict[metode]['alphas'][:,7],label=r'$\theta_E$')
   
    plt.axhline(alpha_R,color='k',linestyle='dotted', label=r'$\alpha_R$')
    plt.axhline(alpha_R2,color='k',linestyle='dashdot', label=r'$\alpha_{R2}$')
    plt.axhline(alpha_E,color='dimgrey',linestyle='dotted', label=r'$\alpha_E$')
    plt.axhline(alpha_E2,color='dimgrey',linestyle='dashdot', label=r'$\alpha_{E2}$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\alpha$")
    plt.legend()#loc='upper right')
    plt.grid()
    plt.show() 

#%% Plotting alphas for different methods:
#Plotting alphas for different methods:
    
plot_alpha_d(chisV,alphasV,'FORM','shear')
plot_alpha_d(chisV,alphasV,'SD','shear')

#%%
plot_alpha_d(chis,alphasM,'FORM','bending')
plot_alpha_d(chis,alphasM,'SD','bending')

#%% Updating and saving gamma for the separate approach for the two beams.
#Material dominating resistance for bending.
#Resistance model dominating for shear.
PF.Update_psf_beam_2nd(BeamV,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,B_t,delta,Print_results=False)
alphasV['Standardized']['gammas']=[BeamV['gamma_M_s'][1],BeamV['gamma_M_c'][1],BeamV['Loading'][0]['gamma'],BeamV['Loading'][1]['gamma'],BeamV['Loading'][2]['gamma']]

PF.Update_psf_beam_2nd(BeamM,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta,Print_results=False)
alphasM['Standardized']['gammas']=[BeamM['gamma_M_s'][0],BeamM['gamma_M_c'][0],BeamM['Loading'][0]['gamma'],BeamM['Loading'][1]['gamma'],BeamM['Loading'][2]['gamma']]


#%% Function for plotting gammas saved in a a_dict. Most relevant for FORM-alphas or other methods where alphas changes over chi.
#Function for plotting gammas saved in a a_dict. Most relevant for FORM-alphas or other methods where alphas changes over chi.
#and comparing it to gammas with standardized values.
def plot_gamma_d(chis,a_dict,metode,Bruddform):
    plt.figure(figsize=(10,7))
    
    
    #for i, alpha in enumerate():
        
    if Bruddform=='bending':
    
        plt.plot(chis,a_dict[metode]['gammas'][:,0],label=r'$f_y$')
        plt.plot(chis,a_dict[metode]['gammas'][:,1],label=r'$f_c$')
        
        
    
    if Bruddform=='shear':
        plt.plot(chis,a_dict[metode]['gammas'][:,0],label=r'$f_y$')
        
       
    
    plt.plot(chis,a_dict[metode]['gammas'][:,2],label=r'$G_s$')
    plt.plot(chis,a_dict[metode]['gammas'][:,3],label=r'$G_p$')
    plt.plot(chis,a_dict[metode]['gammas'][:,4],label=r'$Q$')
    
    
    plt.axhline(a_dict['Standardized']['gammas'][0],color='k',linestyle='dotted', label=r'$f_y$')
    
    if Bruddform=='bending':
        plt.axhline(a_dict['Standardized']['gammas'][1],color='k',linestyle='dashdot', label=r'$f_c$')
        
    plt.axhline(a_dict['Standardized']['gammas'][2],color='dimgrey',linestyle='dotted', label=r'$G_s$')
    plt.axhline(a_dict['Standardized']['gammas'][3],color='dimgrey',linestyle='dashed', label=r'$G_p$')
    plt.axhline(a_dict['Standardized']['gammas'][4],color='dimgrey',linestyle='dashdot', label=r'$Q$')
    

    
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\gamma$")
    plt.legend(loc='lower left')
    plt.grid()
    plt.show() 
    
plot_gamma_d(chisV,alphasV,'FORM','shear')
plot_gamma_d(chis,alphasM,'FORM','bending')



#%%  Function for plotting and comparing alphas from a method with alphas from FORM
#  Function for plotting and comparing alphas from a method with alphas from FORM
#Can also be used to compare several methods.

plt.rcParams.update({'font.size': 20})
def plot_alpha_compare(chis,a_dict,metods,Bruddform):
    plt.figure(figsize=(10,9))
    ax = plt.subplot(111)
    c=['b','r','g','c','m','y']
    
    #for i, alpha in enumerate():
    for i, metode in enumerate(metods):
        if Bruddform=='bending':
        
            plt.plot(chis,a_dict[metode]['alphas'][:,0],label=r'$f_y$',color=c[i])
            plt.plot(chis,a_dict[metode]['alphas'][:,1],label=r'$\rho_l$',color=c[i],linestyle='dotted')
            plt.plot(chis,a_dict[metode]['alphas'][:,2],label=r'$f_c$',color=c[i],linestyle='dashdot')
        if Bruddform=='shear':
        
            plt.plot(chis,a_dict[metode]['alphas'][:,0],label=r'$f_y$',color=c[i])
            plt.plot(chis,a_dict[metode]['alphas'][:,1],label=r'$\rho_w$',color=c[i],linestyle='dotted')
           
        
        plt.plot(chis,a_dict[metode]['alphas'][:,3],label=r'$\theta_R$',color=c[i],linestyle=(0, (1, 10)))
        plt.plot(chis,a_dict[metode]['alphas'][:,4],label=r'$G_s$',color=c[i],linestyle='dashed')
        plt.plot(chis,a_dict[metode]['alphas'][:,5],label=r'$G_p$',color=c[i],linestyle='dotted')
        plt.plot(chis,a_dict[metode]['alphas'][:,6],label=r'$Q$',color=c[i],linestyle='dashdot')
        plt.plot(chis,a_dict[metode]['alphas'][:,7],label=r'$\theta_E$',color=c[i],linestyle=(0, (1, 10)))
   
    plt.axhline(alpha_R,color='k',linestyle='dotted')#, label=r'$\alpha_R$')
    plt.axhline(alpha_R2,color='k',linestyle='dashdot')#, label=r'$\alpha_{R2}$')
    plt.axhline(alpha_E,color='dimgrey',linestyle='dotted')#, label=r'$\alpha_E$')
    plt.axhline(alpha_E2,color='dimgrey',linestyle='dashdot')#, label=r'$\alpha_{E2}$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\alpha$")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

# Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True,ncol=7) 
    plt.ylim([-1.1,1.1])
    plt.grid()
    plt.show() 


#%% Comparing alphas from different methods with alphas from FORM
#Comparing alphas from different methods with alphas from FORM

plot_alpha_compare(chis,alphasM,['FORM','SD'],'bending')
plot_alpha_compare(chis,alphasM,['FORM','SD-2'],'bending')

#Merk at theta E blir null


#%% Comparing SD-alphas against FORM-alpha for shear beam V
#Comparing SD-alphas against FORM-alpha for shear beam V

plot_alpha_compare(chisV,alphasV,['FORM','SD'],'shear')
plot_alpha_compare(chisV,alphasV,['FORM','SD-2'],'shear')


#%% Saving SD gammas V

alphasV['SD']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_SD,'shear')[1]
alphasV['SD-2']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_SD2,'shear')[1]

#%% Function for plotting deviation in % in alpha
plt.rcParams.update({'font.size': 20})
def plot_adev_d(chis,a_dict,metode,Bruddform):
    plt.figure(figsize=(10,7))
    
    
    #for i, alpha in enumerate():
        
    if Bruddform=='bending':
    
        plt.plot(chis,a_dict[metode]['a-dev'][:,0],label=r'$f_y$')
        plt.plot(chis,a_dict[metode]['a-dev'][:,1],label=r'$\rho_l$')
        plt.plot(chis,a_dict[metode]['a-dev'][:,2],label=r'$f_c$')
    if Bruddform=='shear':
    
        plt.plot(chis,a_dict[metode]['a-dev'][:,0],label=r'$f_y$')
        plt.plot(chis,a_dict[metode]['a-dev'][:,1],label=r'$\rho_w$')
       
    
    plt.plot(chis,a_dict[metode]['a-dev'][:,3],label=r'$\theta_R$')
    plt.plot(chis,a_dict[metode]['a-dev'][:,4],label=r'$G_s$')
    plt.plot(chis,a_dict[metode]['a-dev'][:,5],label=r'$G_p$')
    plt.plot(chis,a_dict[metode]['a-dev'][:,6],label=r'$Q$')
    plt.plot(chis,a_dict[metode]['a-dev'][:,7],label=r'$\theta_E$')
   
    #plt.ylim([-100,100]) 
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\% $")
    plt.legend()#loc='upper right')
    plt.grid()
    plt.show() 
    
#%% Function for plotting deviation in % in gamma
plt.rcParams.update({'font.size': 20})
def plot_gdev_d(chis,a_dict,metode,Bruddform):
    plt.figure(figsize=(10,7))
        
        
    #for i, alpha in enumerate():
            
    if Bruddform=='bending':
        
        plt.plot(chis,a_dict[metode]['g-dev'][:,0],label=r'$\gamma_{S}$')
            
        plt.plot(chis,a_dict[metode]['g-dev'][:,1],label=r'$\gamma_{C}$')
    if Bruddform=='shear':
        
        plt.plot(chis,a_dict[metode]['g-dev'][:,0],label=r'$\gamma_{S}$')
            
           
        
    plt.plot(chis,a_dict[metode]['g-dev'][:,2],label=r'$\gamma_{Gs}$')
    plt.plot(chis,a_dict[metode]['g-dev'][:,3],label=r'$\gamma_{Gp}$')
    plt.plot(chis,a_dict[metode]['g-dev'][:,4],label=r'$\gamma_{Q}$')
        
    plt.ylim([0,50])    
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\% $")
    plt.legend()#loc='upper right')
    plt.grid()
    plt.show() 
#%%

alphasV['SD']['g-dev']=np.abs(alphasV['SD']['gammas']-alphasV['FORM']['gammas'])/np.abs(alphasV['FORM']['gammas'])*100

plot_gdev_d(chisV,alphasV,'SD','shear')

#%%

alphasV['SD']['a-dev']=np.abs(alphasV['SD']['alphas']-alphasV['FORM']['alphas'])/np.abs(alphasV['FORM']['alphas'])*100

plot_adev_d(chisV,alphasV,'SD','shear')


#%%
alphasM['SD']['gammas']=save_alphas(BeamM,Loadings,alphas_M_SD,'bending')[1]
#alphasM['SD-2']['gammas']=save_alphas(BeamM,Loadings,alphas_M_SD2,'bending')[1]

#%%
alphasM['SD']['g-dev']=np.abs(alphasM['SD']['gammas']-alphasM['FORM']['gammas'])/np.abs(alphasM['FORM']['gammas'])*100

plot_gdev_d(chis,alphasM,'SD','bending')

#%%

alphasM['SD']['a-dev']=np.abs(alphasM['SD']['alphas']-alphasM['FORM']['alphas'])/np.abs(alphasM['FORM']['alphas'])*100

plot_adev_d(chis,alphasM,'SD','bending')

#%% Fetching alpha values for specific chi value for FORM and Sc methods.
#Fetching alpha values for specific chi value for FORM and Sc methods.

np.round(alphasM['SD']['alphas'][50],2)
#[0.7,0,0.7,0.28,-0.8,-0.8,-0.8,0.32]
#np.round(np.linalg.norm(alphasM['SD']['alphas'][50]),2)


#%% Trying to estimate alpha for simple approach:


#%% Estimating alphas SD/Sum(SD)

def alphas_M_SD_simple(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_l']['moments'][2]**2#+Beam['fc']['moments'][2]
    ScR=theta_R_bending['moments'][0]*Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]*(1-0.5+Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]/
                                                                                            (alpha_cc*Beam['fc']['moments'][0]))*Beam['b']*Beam['d']**2
    
    
    SumE=(Beam['L']**2/8)**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumE+SumR*ScR**2)
    
    
    
    alphas=np.zeros(2)
    
    alphas[0]=np.sqrt(SumR*ScR**2)/Sum
    alphas[1]=-np.sqrt(SumE)/Sum
    
    #alphas=alphas/np.linalg.norm(alphas)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_SD_simple(BeamM,theta_R_bending,theta_E,Print_results=True)  




#%% Estimating alphas CoV/Sum(CoV)
def alphas_V_SD_simple(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]**2+Beam['fs']['moments'][2]**2+Beam['rho_w']['moments'][2]**2
    
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    
    ScE=(Beam['L']/2-Beam['d'])**2*(theta_E['moments'][0]**2*(Beam['Loading'][0]['moments'][1]**2+Beam['Loading'][1]['moments'][1]**2+Beam['Loading'][2]['moments'][1]**2)+
                                 theta_E['moments'][1]**2*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])**2)
    
    Sum=np.sqrt(SumR*ScR**2+ScE)
    
    alphas=np.zeros(2)
    
    alphas[0]=np.sqrt(SumR*ScR**2)/Sum
    alphas[1]=-np.sqrt(ScE)/Sum
    
                       #(Beam['L']/2-Beam['d'])*(theta_E['moments'][0]*(Beam['Loading'][0]['moments'][1]+Beam['Loading'][1]['moments'][1]+Beam['Loading'][2]['moments'][1])+
                                                    #theta_E['moments'][1]*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0]))
    #alphas=alphas/np.linalg.norm(alphas)

    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_V_SD_simple(BeamV,theta_R_shear,theta_E,Print_results=True) 
#%% Calculating necessary reinforcement for the simple apprroach.
def calc_rhols_alphas_simple(Beam,Loadings,a_func):
    rhols=[]
    Beam['rho_l']=Bj.Create_dist_variable(0.005, 0, 'Normal') #Guesses initial rho_l
    for loading in Loadings:
        Beam['Loading']=loading
        alphas=a_func(Beam,theta_R_bending,theta_E,Print_results=False)
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[1],alphas[0],B_t,delta=1.05,Print_results=False)
        #Partial factors are update for each loading, as the variation do not change this is actually unecessary but it guarantees that they are correctly defined.
        L=Beam['L']
        q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
           +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
           +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
        d=Beam['d']
        b=Beam['b']
        
        
        
        fcd=alpha_cc*Beam['fc']['fk']/Beam['gamma_M_c'][0]
        fyd=Beam['fs']['fk']/Beam['gamma_M_s'][0]
        Ed=q*L**2/8
        
        rho_l=fcd/fyd*(1-np.sqrt(1-2*Ed/(b*d**2*fcd))) #rho_l is calculated by eq. 5.14?
        rho_l=Bj.Create_dist_variable(rho_l, 0, 'Normal')  #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhols+=[rho_l]
        Beam['rho_l']=rho_l
        
    return np.array(rhols)
#%% Calculating necessary reinforcement for the simple apprroach.
def calc_rhows_alphas_simple(Beam,Loadings,a_func):
    rhows=[]
    Beam['rho_w']=Bj.Create_dist_variable(0.001, 0, 'Normal') #Guesses initial rho_w
    Beam['cot_t']=Bj.cot_t(Beam,Print_results=False) 
    for loading in Loadings:
        Beam['Loading']=loading
        alphas=a_func(Beam,theta_R_shear,theta_E,Print_results=False)
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alphas[1],alphas[0],B_t,delta=1.05,Print_results=False)
    
        L=Beam['L']
        q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
           +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
           +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
        d=Beam['d']
        bw=Beam['bw']
        
        
        
        fyd=Beam['fs']['fk']/Beam['gamma_M_s'][1]
        Ed=q*(L/2-d)
        
        #rho_w is calculated by eq. 5.15?
        rho_w=Ed/(bw*0.9*d*fyd*2.5) #!!!  Cot(theta)=2.5 is assumed. The Beta-chi plots clearly shows where this assumption becomes wrong. But for normal chi range it seems okay.
        rho_w=Bj.Create_dist_variable(rho_w, 0, 'Normal') #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhows+=[rho_w]
        Beam['rho_w']=rho_w
        Beam['cot_t']=Bj.cot_t(Beam,Print_results=False)
    return np.array(rhows)

#%% Importing results for simple approach with standardized values
from B_chi_plot import rhols, Betas
from B_chi_plot import rhowsV, Betas_V
#%% Creating new a_dict
alphasM2={}
alphasM2['Simple']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM2['Simple']['rhos']=rhols
alphasM2['Simple']['betas']=Betas

#%% Adding estimations to a_dict
rhols_simple_SD=calc_rhols_alphas_simple(BeamM,Loadings,alphas_M_SD_simple)
Betas_simple_SD=BC.calc_Betas_M(BeamM, Loadings, rhols_simple_SD)


alphasM2['SD']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM2['SD']['rhos']=rhols_simple_SD
alphasM2['SD']['betas']=Betas_simple_SD



#%% Plotting betas for simple method/estimation
plot_Betas_d(chis,alphasM2,'bending',B_t=B_t) 

#%% Creating new a_dict
 
alphasV2={}
alphasV2['Simple']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV2['Simple']['rhos']=rhowsV
alphasV2['Simple']['betas']=Betas_V

#%%Adding estimations to a_dict
rhows_simple_SD=calc_rhows_alphas_simple(BeamV,LoadingsV,alphas_V_SD_simple)
Betas_simple_SDV=BC.calc_Betas_V(BeamV, LoadingsV, rhows_simple_SD)

alphasV2['SD']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV2['SD']['rhos']=rhows_simple_SD
alphasV2['SD']['betas']=Betas_simple_SDV

#%% Plotting betas for simple method/estimation

plot_Betas_d(chisV,alphasV2,'shear',B_t=B_t) 


