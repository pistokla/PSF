#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:15:27 2024

@author: perivar
"""
import numpy as np
from scipy import stats
import pandas as pd


# %% Partialfactors by simplified formulas (Caspeele 2023)
#Partialfactors by simplified formulas (Caspeele 2023)
class gamma():
    def __init__(self, value):
        self.value = value


class gamma_M_lognorm(gamma):
    def __init__(self, alfa_R, B, Vf, Vtheta, mu_theta):
        self.value = np.exp(-1.645*Vf)/(mu_theta *
                                        np.exp(-alfa_R*B*np.sqrt(Vtheta**2+Vf**2)))


class gamma_G_norm(gamma):
    def __init__(self, alfa_E, B, VG, Vtheta):
        self.value = (1-alfa_E*B*np.sqrt(Vtheta**2+VG**2))


class gamma_Q_gumbel(gamma):
    def __init__(self, alfa_E, B, mu_Q_tref, V_q_tref,qk, delta):
        self.value = delta*mu_Q_tref/qk*(
            1-V_q_tref*(0.45+0.78*np.log(-np.log(stats.norm.cdf(-alfa_E*B)))))  # Her er det noko galt i beregning av mu_Qtref#!!!


#%% #Gammas for 2nd approach:
#Gammas for 2nd approach:
class gamma_m_lognorm(gamma):
    def __init__(self, alfa_R, B, V):
        self.value = np.exp(-1.645*V)/(np.exp(-alfa_R*B*V))

class gamma_g_norm(gamma): 
    def __init__(self,alfa_E,B,V):
        self.value=(1-alfa_E*B*V)  #4.1-24 #Antar qk=q.mean =>k=0 #4.1-24
        
class gamma_q_gumbel(gamma):
    def __init__(self, alfa_E, B, mu_q_tref, V_q_tref,qk, delta):
        self.value = delta*mu_q_tref/qk*(
            1-V_q_tref*(0.45+0.78*np.log(-np.log(stats.norm.cdf(-alfa_E*B)))))  # Her er det noko galt i beregning av mu_Qtref#!!!

#fjerner theta_k jfr. 4.2-6)
class gamma_Rd_lognorm(gamma): 
    def __init__(self,alfa_R,B,V,mean=1):
        #self.value=1/mean*np.exp(-1.645*V)/np.exp(-alfa_R*B*V) #kommer mean inn på rett plass #!!! #4.1-33
        self.value=1/(mean*np.exp(-alfa_R*B*V))
        
class gamma_Sd_lognorm(gamma):
    def __init__(self,alfa_E,B,V,mean=1):
        self.value=mean*np.exp(-alfa_E*B*V)#/np.exp(1.645*V) #4.1-34


#Geometrical gamma_Rd2 mangler


#%% Partial factors for corroded structures.

class gamma_M_lognorm_corr(gamma):
    def __init__(self, alfa_R, B, Vf, Vtheta, mu_theta,Veta):
        self.value = np.exp(-1.645*Vf)/(mu_theta *
                                        np.exp(-alfa_R*B*np.sqrt(Veta**2+Vtheta**2+Vf**2)))
        
class gamma_Rd_norm(gamma): 
    def __init__(self,alfa_R,B,V,mean=1):
        self.value=1/(mean*(1-alfa_R*B*V)) 

#%% Defining function for calculating gamma_M for a beam:
#Defining function for calculating gamma_M for a beam:
def calc_gamma_M(Beam,theta_R_bending,theta_R_shear,alpha_R,B_t,Print_results=False,corrosion=False):
    if corrosion== False:
        gamma_M_s_bending = gamma_M_lognorm(
            alpha_R, B_t, Beam['fs']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value
        gamma_M_c_bending = gamma_M_lognorm(
            alpha_R, B_t, Beam['fc']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value
        
        gamma_M_s_shear = gamma_M_lognorm(
            alpha_R, B_t, Beam['fs']['moments'][2], theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value
        gamma_M_c_shear = gamma_M_lognorm( #Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
            alpha_R, B_t, Beam['fc']['moments'][2], theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value
    
    if corrosion == True:
        gamma_M_s_bending = gamma_M_lognorm_corr(
            alpha_R, B_t, Beam['fs']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0],Beam['eta']['moments'][2]).value
        gamma_M_c_bending = gamma_M_lognorm(
            alpha_R, B_t, Beam['fc']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value
        
        gamma_M_s_shear = gamma_M_lognorm_corr(
            alpha_R, B_t, Beam['fs']['moments'][2], theta_R_shear['moments'][2], theta_R_shear['moments'][0],Beam['eta']['moments'][2]).value
        gamma_M_c_shear = gamma_M_lognorm( #Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
            alpha_R, B_t, Beam['fc']['moments'][2], theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value
        
    
    if Print_results==True:
        print(pd.DataFrame(np.round([gamma_M_s_bending, gamma_M_s_shear,gamma_M_c_bending], 2), [u'\u03B3_M_s_bending', u'\u03B3_M_s_shear',u'\u03B3_M_c_bending'])) 
       
    
    Beam['gamma_M_c']=[gamma_M_c_bending, gamma_M_c_shear]
    Beam['gamma_M_s']=[gamma_M_s_bending, gamma_M_s_shear] 

#%% Defining function for calculating gamma_load for a load:
#Defining function for calculating gamma_load for a load:
def calc_gamma_load(Load,theta_E,alpha_E,B_t,delta=1.05,Print_results=False):
    LoadType=Load['LoadType']
    if LoadType == 'SelfWeight':
        gamma= gamma_G_norm(
            alpha_E, B_t, Load['moments'][2], theta_E['moments'][2]).value 
        if Print_results==True:
            print(u'\u03B3_Gs=',np.round(gamma,2))
    if LoadType == 'Permanent':
        gamma= gamma_G_norm(
            alpha_E, B_t, Load['moments'][2], theta_E['moments'][2]).value
        if Print_results==True:
            print(u'\u03B3_Gp=',np.round(gamma,2))
    if LoadType == 'Climatic':
        mu_tref = Load['moments'][0]*theta_E['moments'][0]
        qk=Load['qk']*theta_E['moments'][0]
        Vtref = np.sqrt(Load['moments'][2]**2+theta_E['moments'][2]**2)
        gamma = gamma_Q_gumbel(
            alpha_E, B_t, mu_tref, Vtref,qk, delta).value
        if Print_results==True:
            print(u'\u03B3_Q=',np.round(gamma,2))
    if LoadType == 'Imposed':
        mu_tref = Load['moments'][0]*theta_E['moments'][0]
        qk=Load['qk']*theta_E['moments'][0]
        Vtref = np.sqrt(Load['moments'][2]**2+theta_E['moments'][2]**2)
        gamma = gamma_Q_gumbel(
            alpha_E, B_t, mu_tref, Vtref,qk,delta).value
        if Print_results==True:
            print(u'\u03B3_Q=',np.round(gamma,2))
            
    Load['gamma'] = gamma
    
#Also here functions are defined for use several times.

#%% Function for updating, printing and storing PSF for BEAM:
#Function for updating, printing and storing PSF for BEAM:
def Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False,corrosion=False):
    calc_gamma_M(Beam,theta_R_bending,theta_R_shear,alpha_R,B_t,Print_results,corrosion)
    for Load in Beam['Loading']:
        calc_gamma_load(Load,theta_E,alpha_E,B_t,delta,Print_results)
    
#%% Function for Semi-probabilistic control of simply supported beam with uniform distributed load and updated PSF.
#Function for Semi-probabilistic control of simply supported beam with uniform distributed load and updated PSF. 
 
def Controll_psf_beam(Beam,alpha_cc,Bending=True,Shear=True,Print_results=True,corrosion=False):
    L=Beam['L']
    q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
       +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
       +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
    d=Beam['d']
    b=Beam['b']
    bw=Beam['bw']

    if type(Beam['rho_l']) == dict:
        
        rho_l=Beam['rho_l']['moments'][0] #Om rho er fordelt variabel brukes mean value...
    else:
        rho_l=Beam['rho_l']
    if type(Beam['rho_w']) == dict:
        rho_w=Beam['rho_w']['moments'][0] #Kan plasseres under skjær/moment for å kunne kjøre PSF for bare ein
    else:
        rho_w=Beam['rho_w']
   
    
    Beam['utn']=[0,0]

    #Bending:
    if Bending==True:
        fcd_M=alpha_cc*Beam['fc']['fk']/Beam['gamma_M_c'][0]
        fyd_M=Beam['fs']['fk']/Beam['gamma_M_s'][0]
        
        if corrosion==False:
            Mr=rho_l*fyd_M*(1-0.5*rho_l*fyd_M/(fcd_M))*b*d**2
        if corrosion==True:
            Mr=Beam['eta']['moments'][0]*rho_l*fyd_M*(1-0.5*Beam['eta']['moments'][0]*rho_l*fyd_M/(fcd_M))*b*d**2
        
        Me=q*L**2/8
        if Print_results==True:
            print('The moment capacity is Mr=',np.round(Mr/10**6,2),'kNm')
            print('The beam is subject to Me=',np.round(Me/10**6,2),'kNm')
            if Me < Mr:
                print('Design check OK Utn=',np.round(Me/Mr,2))
            else:
                print('Design check not OK Utn=',np.round(Me/Mr,2))
        Beam['Mr']=Mr
        Beam['Me']=Me
        Beam['utn'][0]=Me/Mr#np.round(Me/Mr,2)
    if Shear==True:
        cot_t=Beam['cot_t']
        fyd_V=Beam['fs']['fk']/Beam['gamma_M_s'][1]
        
        if corrosion==False:
            Vr=rho_w*bw*0.9*d*fyd_V*cot_t
        if corrosion==True:
            Vr=Beam['eta']['moments'][0]*rho_w*bw*0.9*d*fyd_V*cot_t
            
        Ve=q*(L/2-d)
        if Print_results==True:
            print('The shear capacity is Vr=',np.round(Vr/10**3,2),'kN')
            print('The beam is subject to Ve=',np.round(Ve/10**3,2),'kN')
            if Ve < Vr:
                print('Design check OK Utn=',np.round(Ve/Vr,2))
            else:
                print('Design check not OK Utn=',np.round(Ve/Vr,2))
        Beam['Vr']=Vr
        Beam['Ve']=Ve
        Beam['utn'][1]=Ve/Vr #np.round(Ve/Vr,2)
    
#%% Defining function for calculating gamma_M for a beam: for second approach
#Defining function for calculating gamma_M for a beam: for second approach
def calc_gamma_M_2nd(Beam,theta_R_bending,theta_R_shear,alpha_R1,alpha_R2,B_t,Print_results=False,corrosion=False):
    gamma_m_s = gamma_m_lognorm(
        alpha_R1, B_t, Beam['fs']['moments'][2]).value
    gamma_m_c = gamma_m_lognorm(
        alpha_R1, B_t, Beam['fc']['moments'][2]).value
    
    gamma_Rd_bending = gamma_Rd_lognorm(
        alpha_R2, B_t, theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value
    gamma_Rd_shear = gamma_Rd_lognorm( 
        alpha_R2, B_t, theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value
    
    if corrosion==False:
        gamma_M_c_bending= gamma_Rd_bending*gamma_m_c
        gamma_M_c_shear= gamma_Rd_shear*gamma_m_c#Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
        gamma_M_s_bending= gamma_Rd_bending*gamma_m_s
        gamma_M_s_shear= gamma_Rd_shear*gamma_m_s
        
        if Print_results==True:
            print(pd.DataFrame(np.round([gamma_m_c,gamma_m_s,gamma_Rd_bending,gamma_Rd_shear,
                                         gamma_M_s_bending,gamma_M_s_shear,gamma_M_c_bending], 2), 
                               [u'\u03B3_m_c', u'\u03B3_m_s',u'\u03B3_Rd_bending',u'\u03B3_Rd_shear',
                                u'\u03B3_M_s_bending',u'\u03B3_M_s_shear',u'\u03B3_M_c_bending'])) 
    if corrosion==True:
        gamma_Rd_eta= gamma_Rd_norm(
            alpha_R2, B_t, Beam['eta']['moments'][2]).value
        gamma_M_c_bending= gamma_Rd_bending*gamma_m_c
        gamma_M_c_shear= gamma_Rd_shear*gamma_m_c#Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
        gamma_M_s_bending= gamma_Rd_bending*gamma_m_s*gamma_Rd_eta
        gamma_M_s_shear= gamma_Rd_shear*gamma_m_s*gamma_Rd_norm(
            alpha_R1, B_t, Beam['eta']['moments'][2]).value
        
        if Print_results==True:
            print(pd.DataFrame(np.round([gamma_m_c,gamma_m_s,gamma_Rd_bending,gamma_Rd_shear,gamma_Rd_eta,
                                         gamma_M_s_bending,gamma_M_s_shear,gamma_M_c_bending], 2), 
                               [u'\u03B3_m_c', u'\u03B3_m_s',u'\u03B3_Rd_bending',u'\u03B3_Rd_shear',u'\u03B3_Rd_\u03B7',
                                u'\u03B3_M_s_bending',u'\u03B3_M_s_shear',u'\u03B3_M_c_bending'])) 
       
    theta_R_bending['gamma']=gamma_Rd_bending
    theta_R_shear['gamma']=gamma_Rd_shear
    Beam['gamma_M_c']=[gamma_M_c_bending, gamma_M_c_shear]
    Beam['gamma_M_s']=[gamma_M_s_bending, gamma_M_s_shear] 

#%% Defining function for calculating gamma_load for a load: for second approach
#Defining function for calculating gamma_load for a load: for second approach

def calc_gamma_load_2nd(Load,theta_E,alpha_E1,alpha_E2,B_t,delta=1.05,Print_results=False):
    gamma_Sd=gamma_Sd_lognorm(alpha_E2,B_t,theta_E['moments'][2],theta_E['moments'][0]).value
    if Print_results==True:
        print(u'\u03B3_Sd=',np.round(gamma_Sd,2))
    LoadType=Load['LoadType']
    if LoadType == 'SelfWeight':
        gamma= gamma_g_norm(
            alpha_E1, B_t, Load['moments'][2]).value 
        if Print_results==True:
            print(u'\u03B3_gs=',np.round(gamma,2))
            print(u'\u03B3_Gs=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Permanent':
        gamma= gamma_g_norm(
            alpha_E1, B_t, Load['moments'][2]).value
        if Print_results==True:
            print(u'\u03B3_gp=',np.round(gamma,2))
            print(u'\u03B3_Gp=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Climatic':
        mu_tref = Load['moments'][0]
        qk=Load['qk']
        Vtref = Load['moments'][2]
        gamma = gamma_q_gumbel(
            alpha_E1, B_t, mu_tref, Vtref,qk, delta).value
        if Print_results==True:
            print(u'\u03B3_q=',np.round(gamma,2))
            print(u'\u03B3_Q=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Imposed':
       mu_tref = Load['moments'][0]
       qk=Load['qk']
       Vtref = Load['moments'][2]
       gamma = gamma_q_gumbel(
           alpha_E1, B_t, mu_tref, Vtref,qk, delta).value
       if Print_results==True:
           print(u'\u03B3_q=',np.round(gamma,2))
           print(u'\u03B3_Q=',np.round(gamma_Sd*gamma,2))
            
    Load['gamma'] = gamma*gamma_Sd
    
    theta_E['gamma']=gamma_Sd
    
#Also here functions are defined for use several times.

#Lite fleskibel med tanke på val av alfa
#%% Function for updating, printing and storing PSF for BEAM 2nd approach:
#Function for updating, printing and storing PSF for BEAM 2nd approach:
def Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E1,alpha_R1,alpha_E2,alpha_R2,B_t,delta=1.05,Print_results=False,corrosion=False):
    calc_gamma_M_2nd(Beam,theta_R_bending,theta_R_shear,alpha_R1,alpha_R2,B_t,Print_results,corrosion)
    for Load in Beam['Loading']:
        calc_gamma_load_2nd(Load,theta_E,alpha_E1,alpha_E2,B_t,delta,Print_results)
        
        
#%% Defining function for calculating gamma_M for a beam: with separate approach for real alphas (or other alphas specified for every parameter)
#Defining function for calculating gamma_M for a beam:  with separate approach for real alphas (or other alphas specified for every parameter)

#Alphas samme organisering som for FORM-bending

def calc_gamma_M_RA(Beam,theta_R_bending,theta_R_shear,alphas,B_t,Print_results=False,corrosion=False):
    gamma_m_s = gamma_m_lognorm(
        alphas[0], B_t, Beam['fs']['moments'][2]).value
    gamma_m_c = gamma_m_lognorm(
        alphas[2], B_t, Beam['fc']['moments'][2]).value
    
    gamma_Rd_bending = gamma_Rd_lognorm(
        alphas[3], B_t, theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value
    gamma_Rd_shear = gamma_Rd_lognorm( 
        alphas[3], B_t, theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value
    
    if corrosion==False:
        gamma_M_c_bending= gamma_Rd_bending*gamma_m_c
        gamma_M_c_shear= gamma_Rd_shear*gamma_m_c#Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
        gamma_M_s_bending= gamma_Rd_bending*gamma_m_s
        gamma_M_s_shear= gamma_Rd_shear*gamma_m_s
        
        if Print_results==True:
            print(pd.DataFrame(np.round([gamma_m_c,gamma_m_s,gamma_Rd_bending,gamma_Rd_shear,
                                         gamma_M_s_bending,gamma_M_s_shear,gamma_M_c_bending], 2), 
                               [u'\u03B3_m_c', u'\u03B3_m_s',u'\u03B3_Rd_bending',u'\u03B3_Rd_shear',
                                u'\u03B3_M_s_bending',u'\u03B3_M_s_shear',u'\u03B3_M_c_bending'])) 
    if corrosion==True:
        gamma_Rd_eta= gamma_Rd_norm( #Bruker lognormal fordi høg alfa gir løgne resultat...
            alphas[1], B_t, Beam['eta']['moments'][2]).value
        gamma_M_c_bending= gamma_Rd_bending*gamma_m_c
        gamma_M_c_shear= gamma_Rd_shear*gamma_m_c#Gamma_M_c_shear er egentlig ubrukelig og trengs ikkje beregnes
        gamma_M_s_bending= gamma_Rd_bending*gamma_m_s*gamma_Rd_eta
        gamma_M_s_shear= gamma_Rd_shear*gamma_m_s*gamma_Rd_eta
        
        if Print_results==True:
            print(pd.DataFrame(np.round([gamma_m_c,gamma_m_s,gamma_Rd_bending,gamma_Rd_shear,gamma_Rd_eta,
                                         gamma_M_s_bending,gamma_M_s_shear,gamma_M_c_bending], 2), 
                               [u'\u03B3_m_c', u'\u03B3_m_s',u'\u03B3_Rd_bending',u'\u03B3_Rd_shear',u'\u03B3_Rd_\u03B7',
                                u'\u03B3_M_s_bending',u'\u03B3_M_s_shear',u'\u03B3_M_c_bending'])) 
        
       
    theta_R_bending['gamma']=gamma_Rd_bending
    theta_R_shear['gamma']=gamma_Rd_shear
    Beam['gamma_M_c']=[gamma_M_c_bending, gamma_M_c_shear]
    Beam['gamma_M_s']=[gamma_M_s_bending, gamma_M_s_shear] 
    
#%% Defining function for calculating gamma_load for a load:  with separate approach  for real alphas (or other alphas specified for every parameter)
#Defining function for calculating gamma_load for a load:  with separate approach for real alphas (or other alphas specified for every parameter)

#alphas = list of load alphas
def calc_gamma_load_RA(Load,theta_E,alphas,B_t,delta=1.05,Print_results=False):
    gamma_Sd=gamma_Sd_lognorm(alphas[3],B_t,theta_E['moments'][2],theta_E['moments'][0]).value
    if Print_results==True:
        print(u'\u03B3_Sd=',np.round(gamma_Sd,2))
    LoadType=Load['LoadType']
    if LoadType == 'SelfWeight':
        gamma= gamma_g_norm(
            alphas[0], B_t, Load['moments'][2]).value 
        if Print_results==True:
            print(u'\u03B3_gs=',np.round(gamma,2))
            print(u'\u03B3_Gs=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Permanent':
        gamma= gamma_g_norm(
            alphas[1], B_t, Load['moments'][2]).value
        if Print_results==True:
            print(u'\u03B3_gp=',np.round(gamma,2))
            print(u'\u03B3_Gp=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Climatic':
        mu_tref = Load['moments'][0]
        qk=Load['qk']
        Vtref = Load['moments'][2]
        gamma = gamma_q_gumbel(
            alphas[2], B_t, mu_tref, Vtref,qk, delta).value
        if Print_results==True:
            print(u'\u03B3_q=',np.round(gamma,2))
            print(u'\u03B3_Q=',np.round(gamma_Sd*gamma,2))
    if LoadType == 'Imposed':
       mu_tref = Load['moments'][0]
       qk=Load['qk']
       Vtref = Load['moments'][2]
       gamma = gamma_q_gumbel(
           alphas[2], B_t, mu_tref, Vtref,qk, delta).value
       if Print_results==True:
           print(u'\u03B3_q=',np.round(gamma,2))
           print(u'\u03B3_Q=',np.round(gamma_Sd*gamma,2))
            
    Load['gamma'] = gamma*gamma_Sd
    
    theta_E['gamma']=gamma_Sd
    
#Also here functions are defined for use several times.

#%% Function for updating, printing and storing PSF for BEAM individual alphas:
#Function for updating, printing and storing PSF for BEAM individual alphas:

#alphas stored as FORM bending Especially for shear this has to be controlled.
def Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas,B_t,delta=1.05,Print_results=False,corrosion=False):
    calc_gamma_M_RA(Beam,theta_R_bending,theta_R_shear,alphas,B_t,Print_results,corrosion)
    for Load in Beam['Loading']:
        calc_gamma_load_RA(Load,theta_E,alphas[-4:],B_t,delta,Print_results)
        


    

        