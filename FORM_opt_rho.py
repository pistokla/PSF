#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 06:13:52 2024

@author: perivar
"""

#import necessary libraries:
import numpy as np

from scipy.optimize import minimize

#%% import necessary modules:
#import necessary modules:
import Bjelker as Bj #importerer globale parametre og funksjoner


#Importing FORM analysis modules
from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
from FORM_V_beam_module import FORM_V_beam

#Partialfactors methods is imported from Partial_safety_factors.py
#import Partial_safety_factors as PF
#%% Defiing global parameters from Bjelker.py (eller manuelt)
#Defiing global parameters from Bjelker.py (eller manuelt)
alpha_cc=Bj.alpha_cc
B_t=Bj.B_t

alpha_E=Bj.alpha_E
alpha_R=Bj.alpha_R
alpha_E2=Bj.alpha_E2
alpha_R2=Bj.alpha_R2
delta=Bj.delta

#print('alpha_cc={},B_t={},[alpha_E,alpha_R, alpha_E2 ,alpha_R2]={},delta={}'.format(alpha_cc,B_t,[alpha_E,alpha_R, alpha_E2 ,alpha_R2],delta))


theta_R_bending=Bj.theta_R_bending
theta_R_shear=Bj.theta_R_shear
theta_E=Bj.theta_E




#%% Function for optimising rhow by FORM for shear
#Function for optimising rhow by FORM for shear
#The function returns the value of rhol that exactly satifies B_t. 


def opt_rhow(Beam,B_t,Print_results=False):
    equation = lambda SCrho: np.abs(FORM_V_beam(Beam,theta_R_shear,theta_E,Print_results=False,SCrho=SCrho)[1][0]-B_t)
    rhow = minimize(equation, x0= 0.001,bounds=[(0,1)],method='SLSQP')['x']#Finds gamma_M that gives the function value of equation = 0
    if Print_results==True:
        print(u'\u03C1_w=',np.round(rhow[0],4))
    return rhow

#%% Function for optimising rhol by FORM for bending
#Function for optimising rhol by FORM for shear
#The function returns the value of rhol that exactly satifies B_t. 


def opt_rhol(Beam,B_t,Print_results=False):
    equation = lambda SCrho: np.abs(FORM_M_beam(Beam,theta_R_bending,theta_E,Print_results=False,SCrho=SCrho)[1][0]-B_t)
    rhol = minimize(equation, x0= 0.01,bounds=[(0,1)],method='SLSQP')['x']#Finds gamma_M that gives the function value of equation = 0
    if Print_results==True:
        print(u'\u03C1_l=',np.round(rhol[0],3))
    return rhol


#rhol=opt_rhol(Beam4, B_t)
#%%#%% Importing example beams
#Importing example beams

from B_chi_plot import BeamM, Loadings
from B_chi_plot import BeamV, LoadingsV
#from B_chi_plot import plot_Betas_d, plot_rho_d

#%% Setting load equal to chi=0.55
#Setting load equal to chi=0.55
BeamM['Loading']=Loadings[50]
BeamV['Loading']=LoadingsV[50]


#%% Calculating rho for new structure such that the original structure has beta=4.7 (1y.)
#Calculating rho for new structure such that the original structure has beta=4.7 (1y.)
BeamM['rho_l']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamM['rho_l']=Bj.Create_dist_variable(opt_rhol(BeamM,4.7,Print_results=True)[0], 0, 'Normal')
BeamM['rho_w']=Bj.Create_dist_variable(0, 0, 'Normal')

BeamV['rho_w']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamV['rho_w']=Bj.Create_dist_variable(opt_rhow(BeamV,4.7,Print_results=True)[0], 0, 'Normal')
BeamV['rho_l']=Bj.Create_dist_variable(0, 0, 'Normal')


#%% The follwing procedures are not included in the masters thesis.
#%% Creating example beam
#Creating example beam 

Beam4 = Bj.Create_beam_dict(600, 300, 300, 550, 1, 1, Bj.fc, Bj.fs, 'indoor')
  

Beam4['rho_l']=Bj.Create_dist_variable(Beam4['rho_l'], 0, 'Normal')

Beam4['L']=8000 #mm 
Beam4['cot_t']=2.5
Beam4['rho_w']=Bj.Create_dist_variable(Beam4['rho_w'], 0, 'Normal')


#Creating loads with realistic mean values... 
G_s2 = Bj.Create_dist_variable(10, 0.05, 'Normal')
G_p2=Bj.Create_dist_variable(10, 0.1, 'Normal')
Q_imposed2 = Bj.Create_dist_variable(10, 0.5, 'Gumbel')

G_s2['LoadType']='SelfWeight'
G_p2['LoadType']='Permanent'
Q_imposed2['LoadType']='Imposed'

Bj.add_qk(G_s2, 0.5)
Bj.add_qk(G_p2, 0.5)
Bj.add_qk(Q_imposed2, 1.35)

Beam4['Loading']=[G_s2,G_p2,Q_imposed2]

FORM_V_beam(Beam4,theta_R_shear,theta_E,Print_results=False,SCrho=0.0021)[1][0]


#%% Optimizing gamma_M shear such that the optimal value of rho from FORM is the same as the rho by eq.5.15!!
#Optimizing gamma_M shear such that the optimal value of rho from FORM is the same as the rho by eq.5.15!!
def opt_gamma_V(Beam,B_t,gamma_G,gamma_Q,Print_results=False):
    rhow=opt_rhow(Beam,B_t,Print_results=Print_results)
    L=Beam['L']
    q=(Beam['Loading'][0]['qk']*gamma_G
       +Beam['Loading'][1]['qk']*gamma_G
       +Beam['Loading'][2]['qk']*gamma_Q) 
    d=Beam['d']
    bw=Beam['bw']
    
    
    fyk=Beam['fs']['fk']
    Ed=q*(L/2-d)
    
    #rho_w is calculated by eq. 5.15?
     #!!! Cot(theta)=2.5 is assumed. The Beta-chi plots clearly shows where this assumption becomes wrong. But for normal chi range it seems okay.
    
    equation = lambda gamma_M: np.abs(Ed/(bw*0.9*d*fyk/gamma_M*2.5)-rhow)
    gamma_S = minimize(equation, x0= 1.15,bounds=[(0,2)],method='SLSQP')['x']#Finds gamma_M that gives the function value of equation = 0
    if Print_results==True:
        print(u'\u03B3_M_s=',np.round(gamma_S[0],2))
    return gamma_S, rhow

opt_gamma_V(Beam4,3,1.1,1.2,Print_results=False)

#%% Optimizing gamma_M bending such that the optimal value of rho from FORM is the same as the rho by eq.5.14!!
#Optimizing gamma_M bending such that the optimal value of rho from FORM is the same as the rho by eq.5.14!!


def opt_gamma_M(Beam,B_t,gamma_G,gamma_Q,Print_results=False):
    rhol=opt_rhol(Beam,B_t,Print_results=Print_results)
    
    L=Beam['L']
    q=(Beam['Loading'][0]['qk']*gamma_G
       +Beam['Loading'][1]['qk']*gamma_G
       +Beam['Loading'][2]['qk']*gamma_Q) 
    d=Beam['d']
    b=Beam['b']
    
    
    fck=alpha_cc*Beam['fc']['fk']
    fyk=Beam['fs']['fk']
    Ed=q*L**2/8
        
    
    equation = lambda gamma_M: np.abs((fck/gamma_M[0])/(fyk/gamma_M[1])*(1-np.sqrt(1-2*Ed/(b*d**2*(fck/gamma_M[0])))) -rhol)
    gammaM = minimize(equation, x0= [1.5,1.15],bounds=[(0,2),(0,2)],method='SLSQP')['x']#Finds gamma_M that gives the function value of equation = 0
    if Print_results==True:
        print(u'\u03B3_M_c=',np.round(gammaM[0],2))
        print(u'\u03B3_M_s=',np.round(gammaM[1],2))
    return gammaM ,rhol

opt_gamma_M(Beam4,4,1.1,1.2,Print_results=False)



