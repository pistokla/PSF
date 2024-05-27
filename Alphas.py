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

'''
alphasV['Standardized']={}
alphasV['Standardized']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Standardized']['rhos']=rhows2V
alphasV['Standardized']['betas']=Betas_V2
'''

alphasV['Standardized']={}
alphasV['Standardized']['alpha']=[alpha_R2, alpha_R,alpha_R, alpha_R, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Standardized']['rhos']=rhowsV_Dmodel
alphasV['Standardized']['betas']=Betas_V2_model

#alphasM['Standardized']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['FORM']={}
alphasV['FORM']['rhos']=rhows_RAV
alphasV['FORM']['betas']=Betas_RAV

#%% Scale alpha

def scale_alpha(alphas,LEN):
    
    #Len0=np.linalg.norm(alphas)
    
    
    
    #equation = lambda x: np.linalg.norm(alphas*abs(alphas)*x)-LEN
    #x = fsolve(equation, x0= 1 )
    
    #alphas=alphas*abs(alphas)*x
    '''
    alphas[:4]=alphas[:4]*1.15
    alphas[4:]=alphas[4:]*4
    '''
    aR=alphas[:4]
    aE=alphas[4:]
    
    
    
    equation = lambda x: np.linalg.norm([aR,x*aE])-LEN
    x = fsolve(equation, x0= 1 )
    
    alphas[4:]=x*aE
    
    return alphas
    

#%% Estimating alphas CoV/Sum(CoV) M NOT USE

def alphas_M_cov(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    Sum=SumE+SumR
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/Sum
    alphas[1]=Beam['rho_l']['moments'][2]/Sum
    alphas[2]=Beam['fc']['moments'][2]/Sum
    alphas[3]=theta_R_bending['moments'][2]/Sum
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][2]/Sum
    alphas[5]=-Beam['Loading'][1]['moments'][2]/Sum
    alphas[6]=-Beam['Loading'][2]['moments'][2]/Sum
    alphas[7]=-theta_E['moments'][2]/Sum
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_M_cov(BeamM,theta_R_bending,theta_E,Print_results=True)
#%% Estimating alphas CoV/Sum(CoV_E/R) M NOT USE

def alphas_M_cov2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/SumR
    alphas[1]=Beam['rho_l']['moments'][2]/SumR
    alphas[2]=Beam['fc']['moments'][2]/SumR
    alphas[3]=theta_R_bending['moments'][2]/SumR
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][2]/SumE
    alphas[5]=-Beam['Loading'][1]['moments'][2]/SumE
    alphas[6]=-Beam['Loading'][2]['moments'][2]/SumE
    alphas[7]=-theta_E['moments'][2]/SumE
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas






#%% Estimating alphas SD/Sum(SD) M 

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

#%%
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

#%% Estimating alphas SD/Sum(SD) M 
'''
def alphas_M_SD2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]#+Beam['fc']['moments'][2]
    ScR=theta_R_bending['moments'][0]*Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]*(1-0.5+Beam['fs']['moments'][0]*Beam['rho_l']['moments'][0]/
                                                                                            (alpha_cc*Beam['fc']['moments'][0]))*Beam['b']*Beam['d']**2
    
    
    SumE=Beam['L']**2/8*(theta_E['moments'][0]*(Beam['Loading'][0]['moments'][1]+Beam['Loading'][1]['moments'][1]+Beam['Loading'][2]['moments'][1])+theta_E['moments'][1]*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0]))
    
    Sum=SumE+SumR*ScR
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_l']['moments'][2]/Sum
    alphas[2]=0#ScR*Beam['fc']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_bending['moments'][2]/Sum
    
    
    alphas[4]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-Beam['L']**2/8*theta_E['moments'][0]*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-Beam['L']**2/8*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    aR=alphas[:4]
    aE=alphas[4:]
    
    
    
    equation = lambda x: np.linalg.norm([aR,x*aE])-1.05
    x = fsolve(equation, x0= 1 )
    
    alphas[4:]=x*aE
    
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_SD2(BeamM,theta_R_bending,theta_E,Print_results=True)
#USE THIS EVT with different scaling, and excluded fc0

'''

#%% Estimating alphas SD/Sum(SD) M NOT USE
'''
def alphas_M_SD2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][1]+Beam['fs']['moments'][1]+Beam['rho_l']['moments'][1]+Beam['fc']['moments'][1]
    SumE=theta_E['moments'][1]+Beam['Loading'][0]['moments'][1]+Beam['Loading'][1]['moments'][1]+Beam['Loading'][2]['moments'][1]
    
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][1]/SumR
    alphas[1]=Beam['rho_l']['moments'][1]/SumR
    alphas[2]=Beam['fc']['moments'][1]/SumR
    alphas[3]=theta_R_bending['moments'][1]/SumR
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][1]/SumE
    alphas[5]=-Beam['Loading'][1]['moments'][1]/SumE
    alphas[6]=-Beam['Loading'][2]['moments'][1]/SumE
    alphas[7]=-theta_E['moments'][1]/SumE
    
    alphas=alphas/np.linalg.norm(alphas)*1.05
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_M_SD(BeamM,theta_R_bending,theta_E,Print_results=True)
'''


#%% Estimating alphas CoV/Sum(CoV_E/R) a_fc=0 NOT USE

def alphas_M_cov_fc0(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]#+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/SumR
    alphas[1]=Beam['rho_l']['moments'][2]/SumR
    alphas[2]=0#Beam['fc']['moments'][2]/SumR
    alphas[3]=theta_R_bending['moments'][2]/SumR
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][2]/SumE
    alphas[5]=-Beam['Loading'][1]['moments'][2]/SumE
    alphas[6]=-Beam['Loading'][2]['moments'][2]/SumE
    alphas[7]=-theta_E['moments'][2]/SumE
    #alphas=alphas*1.05 Kan gjøres sikrere med denne
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#%% Estimating alphas CoV/Sum(CoV) shear NOT USE
def alphas_V_cov(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    Sum=SumE+SumR
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/Sum
    alphas[1]=Beam['rho_w']['moments'][2]/Sum
    alphas[3]=theta_R_shear['moments'][2]/Sum
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][2]/Sum
    alphas[5]=-Beam['Loading'][1]['moments'][2]/Sum
    alphas[6]=-Beam['Loading'][2]['moments'][2]/Sum
    alphas[7]=-theta_E['moments'][2]/Sum
    

    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_V_SC(BeamV,theta_R_shear,theta_E,Print_results=True)


#%% Estimating alphas SD/Sum(SD) shear NOT USE
'''
def alphas_V_SD(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][1]*Beam['fs']['moments'][1]*Beam['rho_w']['moments'][1]
    SumE=theta_E['moments'][1]*(Beam['Loading'][0]['moments'][1]*Beam['Loading'][1]['moments'][1]*Beam['Loading'][2]['moments'][1])
    Sum=SumE+SumR
    
    alphas=np.zeros(8)
    
    alphas[0]=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['fs']['moments'][1]/Sum
    alphas[1]=theta_R_shear['moments'][0]*Beam['fs']['moments'][0]*Beam['rho_w']['moments'][1]/Sum
    alphas[3]=Beam['rho_w']['moments'][0]*Beam['fs']['moments'][0]*theta_R_shear['moments'][1]/Sum
    
    
    alphas[4]=-theta_E['moments'][0]*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-theta_E['moments'][0]*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-theta_E['moments'][0]*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    

  
    
    
    alphas=alphas/np.linalg.norm(alphas)*1.05
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_V_SD(BeamV,theta_R_shear,theta_E,Print_results=True)
'''

#%% Estimating alphas CoV/Sum(CoV_E/R) shear
#Fikk betre resultat med SD=mu*V ...
def alphas_V_cov2(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/SumR
    alphas[1]=Beam['rho_w']['moments'][2]/SumR
    alphas[3]=theta_R_shear['moments'][2]/SumR
    
    
    alphas[4]=-Beam['Loading'][0]['moments'][2]/SumE
    alphas[5]=-Beam['Loading'][1]['moments'][2]/SumE
    alphas[6]=-Beam['Loading'][2]['moments'][2]/SumE
    alphas[7]=-theta_E['moments'][2]/SumE
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_V_cov2(BeamV,theta_R_shear,theta_E,Print_results=True)
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
    '''
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    #SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    
    ScE=(Beam['L']/2-Beam['d'])*(theta_E['moments'][0]*(Beam['Loading'][0]['moments'][1]+Beam['Loading'][1]['moments'][1]+Beam['Loading'][2]['moments'][1])+
                                 theta_E['moments'][1]*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0]))
    
    Sum=SumR*ScR+ScE
   
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/Sum
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/Sum
    alphas[3]=ScR*theta_R_shear['moments'][2]/Sum
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/Sum
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/Sum
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/Sum
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/Sum
    
    
    aR=alphas[:4]
    aE=alphas[4:]
    #LenR=np.linalg.norm(aR)
    #LenE=np.linalg.norm(aE)
    
    equation = lambda x: np.linalg.norm([aR,x*aE])-1.05
    x = fsolve(equation, x0= 1 )
    
    alphas[4:]=x*aE
    #alphas[:4]=x/LenE*aR
    
    
    alphas=scale_alpha(alphas, 1.05)
    #print(x)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    
    
    return alphas
'''
alphas_V_SD2(BeamV,theta_R_shear,theta_E,Print_results=True)

#Could be tried with different scaling 

#%% Estimating alphas SC1 shear NOT USE
def alphas_V_SC(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    ScE=theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/(ScR*SumR)
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/(ScR*SumR)
    alphas[3]=ScR*theta_R_shear['moments'][2]/(ScR*SumR)
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/(ScE*SumE)
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/(ScE*SumE)
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/(ScE*SumE)
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/(ScE*SumE)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_V_SC(BeamV,theta_R_shear,theta_E,Print_results=True)


#Fikk betre resultat med SD=mu*V ...
#%% Estimating alphas SC2 shear setting |alpha|=1.05 NOT USE
def alphas_V_SC2(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    ScE=theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/(ScR*SumR)
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/(ScR*SumR)
    alphas[3]=ScR*theta_R_shear['moments'][2]/(ScR*SumR)
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/(ScE*SumE)
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/(ScE*SumE)
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/(ScE*SumE)
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/(ScE*SumE)
    
    alphas=alphas/np.linalg.norm(alphas)*1.05
    
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#alphas_V_SC2(BeamV,theta_R_shear,theta_E,Print_results=True)



#%% Estimating alphas SC3 shear setting |alpha|=1.05 with scaling of load side. NOT USE

def alphas_V_SC3(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    
    ScR=theta_R_shear['moments'][0]*Beam['rho_w']['moments'][0]*Beam['bw']*Beam['d']*0.9*2.5*Beam['fs']['moments'][0]
    ScE=theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])
    
    alphas=np.zeros(8)
    
    alphas[0]=ScR*Beam['fs']['moments'][2]/(ScR*SumR)
    alphas[1]=ScR*Beam['rho_w']['moments'][2]/(ScR*SumR)
    alphas[3]=ScR*theta_R_shear['moments'][2]/(ScR*SumR)
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][0]['moments'][1]/(ScE*SumE)
    alphas[5]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][1]['moments'][1]/(ScE*SumE)
    alphas[6]=-theta_E['moments'][0]*(Beam['L']/2-Beam['d'])*Beam['Loading'][2]['moments'][1]/(ScE*SumE)
    alphas[7]=-(Beam['L']/2-Beam['d'])*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][1]/(ScE*SumE)
    
    aR=alphas[:4]
    aE=alphas[4:]
    
    
    
    equation = lambda x: np.linalg.norm([aR,x*aE])-1.05
    x = fsolve(equation, x0= 1 )
    
    alphas[4:]=x*aE
    
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_V_SC3(BeamV,theta_R_shear,theta_E,Print_results=True)




#%% Estimating alphas by SC1 method for bending. Bare lastside blir skalert. Kva gjer ein med rho og fy i flere ledd på motstandside? NOT USE

def alphas_M_SC(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]#+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    ScE=theta_E['moments'][0]*(Beam['L']**2/8)*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/SumR
    alphas[1]=Beam['rho_l']['moments'][2]/SumR
    alphas[2]=0#Beam['fc']['moments'][2]/Sum
    alphas[3]=theta_R_bending['moments'][2]/SumR
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][0]['moments'][0]*Beam['Loading'][0]['moments'][2]/(ScE*SumE)
    alphas[5]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][1]['moments'][0]*Beam['Loading'][1]['moments'][2]/(ScE*SumE)
    alphas[6]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][2]['moments'][0]*Beam['Loading'][2]['moments'][2]/(ScE*SumE)
    alphas[7]=-theta_E['moments'][0]*(Beam['L']**2/8)*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][2]/(ScE*SumE)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

#%% Estimating alphas by SC method with scaling NOT USE
def alphas_M_SC2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]#+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    ScE=theta_E['moments'][0]*(Beam['L']**2/8)*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])
    
    alphas=np.zeros(8)
    
    alphas[0]=Beam['fs']['moments'][2]/SumR
    alphas[1]=Beam['rho_l']['moments'][2]/SumR
    alphas[2]=0#Beam['fc']['moments'][2]/Sum
    alphas[3]=theta_R_bending['moments'][2]/SumR
    
    
    alphas[4]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][0]['moments'][0]*Beam['Loading'][0]['moments'][2]/(ScE*SumE)
    alphas[5]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][1]['moments'][0]*Beam['Loading'][1]['moments'][2]/(ScE*SumE)
    alphas[6]=-theta_E['moments'][0]*(Beam['L']**2/8)*Beam['Loading'][2]['moments'][0]*Beam['Loading'][2]['moments'][2]/(ScE*SumE)
    alphas[7]=-theta_E['moments'][0]*(Beam['L']**2/8)*(Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0])*theta_E['moments'][2]/(ScE*SumE)
    
    #alphas=alphas/np.linalg.norm(alphas)*1.05
    
    aR=alphas[:4]
    aE=alphas[4:]
    
    equation = lambda x: np.linalg.norm([aR,x*aE])-1.1
    x = fsolve(equation, x0= 1 )
    
    alphas[4:]=x*aE
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_SC2(BeamM,theta_R_bending,theta_E,Print_results=False)

#Justerer ikkje nok...

#Kan kanskje lime inn denne figuren for å vise at det ikkje fungerer 

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
'''
rhows_cov=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_cov)  #Rho_ws is calculated for beam M with the separate PSF approach.
rhows_cov2=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_cov2)

'''

rhows_SC=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_SC)
rhows_SC2=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_SC2)
rhows_SC3=calc_rhows_alphas(BeamV,LoadingsV,alphas_V_SC3)

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
'''
rhols_cov=calc_rhols_alphas(BeamM,Loadings,alphas_M_cov)
rhols_cov2=calc_rhols_alphas(BeamM,Loadings,alphas_M_cov2)
rhols_cov_fc0=calc_rhols_alphas(BeamM,Loadings,alphas_M_cov_fc0)


rhols_SD2=calc_rhols_alphas(BeamM,Loadings,alphas_M_SD2)

rhols_SC=calc_rhols_alphas(BeamM,Loadings,alphas_M_SC)
rhols_SC2=calc_rhols_alphas(BeamM,Loadings,alphas_M_SC2)
'''
rhols_SD=calc_rhols_alphas(BeamM,Loadings,alphas_M_SD)
rhols_SD2=calc_rhols_alphas(BeamM,Loadings,alphas_M_SD2)

#%% Plot rho_l for the separate approach with different alphas against chi
# Plot rho_l for the separate approach with different alphas against chi
def plot_rhol(chis,Rhols,Rhols2,Rhols3,Rhols4):
    RHOLS=[]
    for rho_l in Rhols:
        RHOLS+=[rho_l['moments'][0]]
    RHOLS2=[]
    for rho_l in Rhols2:
        RHOLS2+=[rho_l['moments'][0]]
    RHOLS3=[]
    for rho_l in Rhols3:
        RHOLS3+=[rho_l['moments'][0]]
    RHOLS4=[]
    for rho_l in Rhols4:
        RHOLS4+=[rho_l['moments'][0]]
    
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOLS,label=r'Standarized $\alpha$')
    plt.plot(chis,RHOLS2,label='CoV')
    plt.plot(chis,RHOLS4,label='CoV-2')
    plt.plot(chis,RHOLS3,label=r'$\alpha-FORM$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho$")
    plt.legend()
    plt.grid()
    plt.show() 

#plot_rhol(chis,rhols2,rhols_cov,rhols_RA,rhols_cov2)

#%% Plotting rhow
#plot_rhol(chisV,rhows2V,rhows_cov,rhows_RAV,rhows_cov2)


#%% Calculating resulting BETA for the different methods.
'''
Betas_cov=BC.calc_Betas_M(BeamM, Loadings, rhols_cov)
Betas_cov2=BC.calc_Betas_M(BeamM, Loadings, rhols_cov2)
Betas_cov_fc0=BC.calc_Betas_M(BeamM, Loadings, rhols_cov_fc0)


Betas_SDM2=BC.calc_Betas_M(BeamM, Loadings, rhols_SD2)

Betas_SCM=BC.calc_Betas_M(BeamM, Loadings, rhols_SC)
Betas_SCM2=BC.calc_Betas_M(BeamM, Loadings, rhols_SC2)
'''
Betas_SDM=BC.calc_Betas_M(BeamM, Loadings, rhols_SD)
Betas_SDM2=BC.calc_Betas_M(BeamM, Loadings, rhols_SD2)
#%% The most interesting plot: Betas against chi for the three approaches.
#The most interesting plot: Betas against chi for the three approaches.

def plot_Betas(chis,Betas_1,Betas_2,Betas_3,BetasRA,Bruddform):
   
    #plt.figure(figsize=(4,3)) Smaller fig gives bigger text.
    
    plt.figure(figsize=(10,6))
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    plt.plot(chis,Betas_1,label=r'Standarized $\alpha$')
    plt.plot(chis,Betas_2,label='CoV')
    plt.plot(chis,Betas_3,label='CoV-2')
    plt.plot(chis,BetasRA,label=r'$\alpha-FORM$')
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    plt.legend()#loc='upper right')
    plt.grid()
    plt.show() 

#plot_Betas(chis,Betas2,Betas_cov,Betas_cov2,BetasRA,'bending') #Beta-chi BeamM bending

#%% Calculating resulting BETA for the different methods.
'''
Betas_covV=BC.calc_Betas_V(BeamV, LoadingsV, rhows_cov)
Betas_cov2V=BC.calc_Betas_V(BeamV, LoadingsV, rhows_cov2)



Betas_SC=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SC)
Betas_SC2=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SC2)
Betas_SC3=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SC3)
#plot_Betas(chisV,Betas_V2,Betas_covV,Betas_cov2V,Betas_RAV,'shear') #Beta-chi Beam V shear
'''
Betas_SD=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SD)
Betas_SD2=BC.calc_Betas_V(BeamV, LoadingsV, rhows_SD2)


#%% Plot rho_l for the separate approach with different alphas against chi
# Plot rho_l for the separate approach with different alphas against chi
def plot_rhol5(chis,Rhols,Rhols2,Rhols3,Rhols4,Rhols5):
    RHOLS=[]
    for rho_l in Rhols:
        RHOLS+=[rho_l['moments'][0]]
    RHOLS2=[]
    for rho_l in Rhols2:
        RHOLS2+=[rho_l['moments'][0]]
    RHOLS3=[]
    for rho_l in Rhols3:
        RHOLS3+=[rho_l['moments'][0]]
    RHOLS4=[]
    for rho_l in Rhols4:
        RHOLS4+=[rho_l['moments'][0]]
    RHOLS5=[]
    for rho_l in Rhols5:
        RHOLS5+=[rho_l['moments'][0]]
    
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOLS,label=r'Standarized $\alpha$')
    plt.plot(chis,RHOLS2,label='CoV')
    plt.plot(chis,RHOLS4,label='CoV-2')
    plt.plot(chis,RHOLS5,label=r'$\sigma$')
    plt.plot(chis,RHOLS3,label=r'$\alpha-FORM$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho$")
    plt.legend()
    plt.grid()
    plt.show() 

#plot_rhol5(chis,rhols2,rhols_cov,rhols_RA,rhols_cov2,rhols_SD)
#Betas_SD=BC.calc_Betas_M(BeamM, Loadings, rhols_SD)
#plot_Betas(chis,Betas2,Betas_cov,Betas_SD,BetasRA,'bending') #Beta-chi BeamM bending

#%% Adding the different methods and their results to the a_dicts M.

'''
alphasM['CoV']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['CoV']['rhos']=rhols_cov
alphasM['CoV']['betas']=Betas_cov



alphasM['CoV-2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['CoV-2']['rhos']=rhols_cov2
alphasM['CoV-2']['betas']=Betas_cov2

alphasM['CoV-fc0']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['CoV-fc0']['rhos']=rhols_cov_fc0
alphasM['CoV-fc0']['betas']=Betas_cov_fc0


alphasM['Sc']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['Sc']['rhos']=rhols_SC
alphasM['Sc']['betas']=Betas_SCM


alphasM['Sc2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['Sc2']['rhos']=rhols_SC2
alphasM['Sc2']['betas']=Betas_SCM2

alphasM['SD-2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM['SD-2']['rhos']=rhols_SD2
alphasM['SD-2']['betas']=Betas_SDM2
'''

alphasM['SD']={}
alphasM['SD']['rhos']=rhols_SD
alphasM['SD']['betas']=Betas_SDM


alphasM['SD-2']={}
alphasM['SD-2']['rhos']=rhols_SD2
alphasM['SD-2']['betas']=Betas_SDM2


#%% Adding the different methods and their results to the a_dicts V.
'''
alphasV['CoV']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['CoV']['rhos']=rhows_cov
alphasV['CoV']['betas']=Betas_covV


alphasV['CoV-2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['CoV-2']['rhos']=rhows_cov2
alphasV['CoV-2']['betas']=Betas_cov2V

alphasV['Sc']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Sc']['rhos']=rhows_SC
alphasV['Sc']['betas']=Betas_SC
'''
'''
alphasV['Sc2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Sc2']['rhos']=rhows_SC2
alphasV['Sc2']['betas']=Betas_SC2

alphasV['Sc3']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasV['Sc3']['rhos']=rhows_SC3
alphasV['Sc3']['betas']=Betas_SC3
'''

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
'''
alphasV['CoV']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_cov,'shear')
alphasV['CoV-2']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_cov2,'shear')
alphasV['Sc']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SC,'shear')


alphasV['Sc2']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SC2,'shear')[0]
alphasV['Sc3']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SC3,'shear')[0]
'''

alphasV['SD']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SD,'shear')[0]
alphasV['SD-2']['alphas']=save_alphas(BeamV,LoadingsV,alphas_V_SD2,'shear')[0]
#alphasV['CoV']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_cov,'shear')[1]
#alphasV['CoV-2']['gammas']=save_alphas(BeamV,LoadingsV,alphas_V_cov2,'shear')[1]
       
alphasV['FORM']['alphas']=save_alphas_FORM(BeamV,LoadingsV,'shear')[0]
alphasV['FORM']['gammas']=save_alphas_FORM(BeamV,LoadingsV,'shear')[1]

#%%Adding alphas to the a_dict
'''
alphasM['CoV']['alphas']=save_alphas(BeamM,Loadings,alphas_M_cov,'bending')
alphasM['CoV-2']['alphas']=save_alphas(BeamM,Loadings,alphas_M_cov2,'bending')

alphasM['CoV-fc0']['alphas']=save_alphas(BeamM,Loadings,alphas_M_cov_fc0,'bending')
alphasM['Sc']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SC,'bending')
alphasM['Sc2']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SC2,'bending')
'''

alphasM['SD']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SD,'bending')[0]
alphasM['SD-2']['alphas']=save_alphas(BeamM,Loadings,alphas_M_SD2,'bending')[0]

alphasM['FORM']['alphas']=save_alphas_FORM(BeamM,Loadings,'bending')[0]
alphasM['FORM']['gammas']=save_alphas_FORM(BeamM,Loadings,'bending')[1]

#alphasM['CoV']['gammas']=save_alphas(BeamM,Loadings,alphas_M_cov,'bending')[1]
#alphasM['CoV-2']['gammas']=save_alphas(BeamM,Loadings,alphas_M_cov2,'bending')[1]




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
#plot_alpha_compare(chis,alphasM,['FORM','Sc2'],'bending')

#%%
plot_alpha_compare(chis,alphasM,['FORM','SD'],'bending')
plot_alpha_compare(chis,alphasM,['FORM','SD-2'],'bending')

#Merk at theta E blir null
#%%
#plot_alpha_compare(chisV,alphasV,['FORM','CoV'],'shear')


#%% Comparing Sc-alphas against FORM-alpha for shear beam V
#Comparing Sc-alphas against FORM-alpha for shear beam V

plot_alpha_compare(chisV,alphasV,['FORM','SD'],'shear')
plot_alpha_compare(chisV,alphasV,['FORM','SD-2'],'shear')


#%%

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
'''
alphasV['Sc3']['a-dev']=np.abs(alphasV['Sc3']['alphas']-alphasV['FORM']['alphas'])/np.abs(alphasV['FORM']['alphas'])*100

plot_adev_d(chisV,alphasV,'Sc3','shear')

alphasV['Sc2']['a-dev']=np.abs(alphasV['Sc2']['alphas']-alphasV['FORM']['alphas'])/np.abs(alphasV['FORM']['alphas'])*100

plot_adev_d(chisV,alphasV,'Sc2','shear')
'''
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
    
#%% Estimating alphas CoV/Sum(CoV)

def alphas_M_cov_simple(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    Sum=SumE+SumR
    
    alphas=np.zeros(2)
    
    alphas[0]=SumR/Sum
    alphas[1]=-SumE/Sum
    
    #alphas=alphas/np.linalg.norm(alphas)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_cov_simple(BeamM,theta_R_bending,theta_E,Print_results=True)  

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
#%% Estimating alphas CoV/Sum(CoV) |alpha|=1
def alphas_M_cov_simple2(Beam,theta_R_bending,theta_E,Print_results=False):
    SumR=theta_R_bending['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_l']['moments'][2]#+Beam['fc']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    Sum=SumE+SumR
    
    alphas=np.zeros(2)
    
    alphas[0]=SumR/Sum
    alphas[1]=-SumE/Sum
    
    alphas=alphas/np.linalg.norm(alphas)
    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_M_cov_simple2(BeamM,theta_R_bending,theta_E,Print_results=True)  

#%% Estimating alphas CoV/Sum(CoV)
def alphas_V_cov_simple(Beam,theta_R_shear,theta_E,Print_results=False):
    SumR=theta_R_shear['moments'][2]+Beam['fs']['moments'][2]+Beam['rho_w']['moments'][2]
    SumE=theta_E['moments'][2]+Beam['Loading'][0]['moments'][2]+Beam['Loading'][1]['moments'][2]+Beam['Loading'][2]['moments'][2]
    Sum=SumE+SumR
    
    alphas=np.zeros(2)
    
    alphas[0]=SumR/Sum
    alphas[1]=-SumE/Sum
    
    #alphas=alphas/np.linalg.norm(alphas)

    
    if Print_results==True:
        print(u'\u03B1={}'.format(np.round(alphas,2)))
        print('Vector sum={}'.format(np.round(np.linalg.norm(alphas),2)))
    return alphas

alphas_V_cov_simple(BeamV,theta_R_shear,theta_E,Print_results=True) 

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

#%%Adding estimations to a_dict
'''
rhols_simple_cov2=calc_rhols_alphas_simple(BeamM,Loadings,alphas_M_cov_simple2)
Betas_simple_cov2=BC.calc_Betas_M(BeamM, Loadings, rhols_simple_cov2)


alphasM2['CoV-2']={}
#alphasM['CoV']['alpha']=[alpha_R, alpha_R,alpha_R, alpha_R2, alpha_E,alpha_E,alpha_E,alpha_E2]
alphasM2['CoV-2']['rhos']=rhols_simple_cov2
alphasM2['CoV-2']['betas']=Betas_simple_cov2
'''


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


