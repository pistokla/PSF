#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:29:14 2024

@author: perivar
"""

#import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import fsolve

#%% import necessary modules:
#import necessary modules:
import Bjelker as Bj #importerer globale parametre og funksjoner


#Importing FORM analysis modules
from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
from FORM_V_beam_module import FORM_V_beam

#Partialfactors methods is imported from Partial_safety_factors.py
import Partial_safety_factors as PF


#import B_chi_plot as BC

import FORM_opt_rho as opt

import Alphas as AL

import corrosion_estimation as CE


#%% Printing global parameters for control:
#Printing global parameters for control:
alpha_cc=Bj.alpha_cc
B_t=Bj.B_t
alpha_E=Bj.alpha_E
alpha_R=Bj.alpha_R
alpha_E2=Bj.alpha_E2
alpha_R2=Bj.alpha_R2
delta=Bj.delta
#print('alpha_cc={},B_t={},[alpha_E,alpha_R, alpha_E2 ,alpha_R2]={},delta={}'.format(alpha_cc,B_t,np.round([alpha_E,alpha_R, alpha_E2 ,alpha_R2],2),delta))


theta_R_bending=Bj.theta_R_bending
theta_R_shear=Bj.theta_R_shear
theta_E=Bj.theta_E

#%% Importing example beams
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
BeamM['rho_l']=Bj.Create_dist_variable(opt.opt_rhol(BeamM,4.7,Print_results=True)[0], 0, 'Normal')
BeamM['rho_w']=Bj.Create_dist_variable(0, 0, 'Normal')

BeamV['rho_w']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamV['rho_w']=Bj.Create_dist_variable(opt.opt_rhow(BeamV,4.7,Print_results=True)[0], 0, 'Normal')
BeamV['rho_l']=Bj.Create_dist_variable(0, 0, 'Normal')

#%% Example values for rho these are chosen such that resulting beta is aprox. 3 (around critical)
#Example values for rho these are chosen such that resulting beta is aprox. 3 (around critical)
BeamM['eta']=Bj.Create_dist_variable(0.9, 0.2, 'Normal')
BeamV['eta']=Bj.Create_dist_variable(0.75, 0.2, 'Normal')

#BeamM['eta']=Bj.Create_dist_variable(0.99, 3.5, 'Lognormal')
#BeamV['eta']=Bj.Create_dist_variable(0.99, 3.5, 'Lognormal')
#%% Updating partial factors and controlling Beam M with and without corrosion: simple approach
#Updating partial factors and controlling Beam M with and without corrosion: simple approach
print('BeamM simple')
print('Without corrosion:')
PF.Update_psf_beam(BeamM,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,3,delta=1.05,Print_results=True,corrosion=False)

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam(BeamM,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=True)
print()


#%% Updating partial factors and controlling Beam M with and without corrosion: separate approach
print('BeamM separate')
print('Without corrosion:')
PF.Update_psf_beam_2nd(BeamM,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,3,delta=1.05,Print_results=True,corrosion=False)

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam_2nd(BeamM,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=True)


print()
#%% Updating partial factors and controlling Beam V with and without corrosion: simple approach
#Updating partial factors and controlling Beam V with and without corrosion: simple approach
print('Beam V simple')
print('Without corrosion:')
PF.Update_psf_beam(BeamV,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,3,delta=1.05,Print_results=True,corrosion=False)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam(BeamV,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=True)


#%% Updating partial factors and controlling Beam V with and without corrosion: separate approach
#Updating partial factors and controlling Beam V with and without corrosion: separate approach

print('Beam V separate')
print('Without corrosion:')
PF.Update_psf_beam_2nd(BeamV,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,3,delta=1.05,Print_results=True,corrosion=False)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam_2nd(BeamV,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=True)
print()
#Per nå blir automatisk eta og modell dominant eller ikkje 

#%% Kjører FORM-analyse av beamM med sammenslått rho og eta
#Kjører FORM-analyse av beamM med sammenslått rho og eta

#Denne cella må ikkje kjøres fleire gonger...
print('BeamM FORM:')
BeamM['rho_l']=Bj.Create_dist_variable(BeamM['rho_l']['moments'][0]*BeamM['eta']['moments'][0],
                                       np.sqrt(BeamM['rho_l']['moments'][2]**2+BeamM['eta']['moments'][2]**2), 'Normal')

FORM_M_beam(BeamM,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
print()

BeamM['rho_l']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamM['rho_l']=Bj.Create_dist_variable(opt.opt_rhol(BeamM,4.7,Print_results=False)[0], 0, 'Normal')
#%% Kjører FORM-analyse av beamV med sammenslått rho og eta
#Kjører FORM-analyse av beamV med sammenslått rho og eta


print('BeamV FORM:')
BeamV['rho_w']=Bj.Create_dist_variable(BeamV['rho_w']['moments'][0]*BeamV['eta']['moments'][0],
                                       np.sqrt(BeamV['rho_w']['moments'][2]**2+BeamV['eta']['moments'][2]**2), 'Normal')

FORM_V_beam(BeamV,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
print()

BeamV['rho_w']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamV['rho_w']=Bj.Create_dist_variable(opt.opt_rhow(BeamV,4.7,Print_results=False)[0], 0, 'Normal')

#%% approksimerer alfa av beamM med sammenslått rho og eta


print('BeamM alfa:')
BeamM['rho_l']=Bj.Create_dist_variable(BeamM['rho_l']['moments'][0]*BeamM['eta']['moments'][0],
                                       np.sqrt(BeamM['rho_l']['moments'][2]**2+BeamM['eta']['moments'][2]**2), 'Normal')

alphasM=AL.alphas_M_SD2(BeamM,theta_R_bending,theta_E,Print_results=True)


print()

BeamM['rho_l']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamM['rho_l']=Bj.Create_dist_variable(opt.opt_rhol(BeamM,4.7,Print_results=False)[0], 0, 'Normal')

#%% Updating partial factors and controlling Beam M with and without corrosion: approx alfa approach
print('BeamM approx a')
print('Without corrosion:')
PF.Update_psf_beam_RA(BeamM,theta_R_bending,theta_R_shear,theta_E,alphasM,3,delta=1.05,Print_results=True,corrosion=False)
#Bruker lognorm gamma_eta for å unngå negativ pga. høg alfa

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam_RA(BeamM,theta_R_bending,theta_R_shear,theta_E,alphasM,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamM,alpha_cc,Bending=True,Shear=False,Print_results=True,corrosion=True)


print()

#%% approksimerer alfa av beamV med sammenslått rho og eta


print('BeamV alfa:')
BeamV['rho_w']=Bj.Create_dist_variable(BeamV['rho_w']['moments'][0]*BeamV['eta']['moments'][0],
                                       np.sqrt(BeamV['rho_w']['moments'][2]**2+BeamV['eta']['moments'][2]**2), 'Normal')

alphasV=AL.alphas_V_SD2(BeamV,theta_R_shear,theta_E,Print_results=True)
print()

BeamV['rho_w']=Bj.Create_dist_variable(1, 0, 'Normal')
BeamV['rho_w']=Bj.Create_dist_variable(opt.opt_rhow(BeamV,4.7,Print_results=False)[0], 0, 'Normal')


#%% Updating partial factors and controlling Beam V with and without corrosion: approx alfa approach
print('BeamV approx a')
print('Without corrosion:')
PF.Update_psf_beam_RA(BeamV,theta_R_bending,theta_R_shear,theta_E,alphasV,3,delta=1.05,Print_results=True,corrosion=False)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=False)
print()
print('With corrosion:')
PF.Update_psf_beam_RA(BeamV,theta_R_bending,theta_R_shear,theta_E,alphasV,3,delta=1.05,Print_results=True,corrosion=True)

PF.Controll_psf_beam(BeamV,alpha_cc,Bending=False,Shear=True,Print_results=True,corrosion=True)


print()
#%% Plot UTN over eta
#Interessante plot? gamma over eta  

#Lager array over gjennomsnitt og CoV for eta.
mu_etas=np.arange(0.5,1.01,0.01)
Cov_etas=np.arange(0.05,0.5,0.01)

#%% Berekner utnyttelse for ulike verdier av eta  med simple method. (Og beta frå FORM for gitt eta)
#Her er eta antatt normalfordelt#!!!
def calc_UTNs_eta(Beam,etas,MC,B_t,Bending=True,Shear=False):
    UTNs=[]
    Bs_FORM=[]
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    for  eta in etas:
        if MC=='mu':
            Beam['eta']=Bj.Create_dist_variable(eta, 0.2, 'Normal')
        if MC=='CoV':
            Beam['eta']=Bj.Create_dist_variable(0.9, eta, 'Normal')
        
        #Simple method:
    
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False,corrosion=True)
        PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
        if Bending==True:
            UTNs+=[Beam['utn'][0]]
            
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_l']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')
            alpha,B= FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            Beam['rho_l']=rhol 
        if Shear==True:
            UTNs+=[Beam['utn'][1]]
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_w']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')
            alpha,B= FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            
            Beam['rho_w']=rhow
        Bs_FORM+=[B[0]]
    return UTNs,Bs_FORM

#%%

#UTNs_M,Bs_M=calc_UTNs_eta(BeamM,Cov_etas,'CoV',3,Bending=True,Shear=False) 



#%% funskjon for å Plotte UTN og beta i samme plot

def plot_UB(etas,UTNs,Bs,MC):

    fig, ax1 = plt.subplots()
    
    if MC=='mu':
        ax1.set_xlabel(r'$\mu_{\eta}$')
    if MC=='CoV':
        ax1.set_xlabel(r'$V_{\eta}$')
        
    
    color = 'tab:red'
    ax1.plot(etas,UTNs,color = color)
    #ax1.plot(etas,UTNs_V,color = color)
    ax1.set_ylabel('UTN',color=color)
    
    
    ax1.set_ylim([0,2])
    ax1.axhline(1,color='k',linestyle='dotted')
    
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\beta_{FORM}$',color=color)
    #ax2.axhline(3,color='r', label=r'$\beta_t$')
    ax2.set_ylim([1,5])
    ax2.plot(etas,Bs,color=color)
    
    #plt.grid()
    plt.show()
    
#%% Plotter UTN/B for bøying, over forskjellige mu_eta og V_eta. Simple
mu_UTNs_M,mu_Bs_M=calc_UTNs_eta(BeamM,mu_etas,'mu',3,Bending=True,Shear=False) 
plot_UB(mu_etas,mu_UTNs_M,mu_Bs_M,'mu')

cov_UTNs_M,cov_Bs_M=calc_UTNs_eta(BeamM,Cov_etas,'CoV',3,Bending=True,Shear=False) 
plot_UB(Cov_etas,cov_UTNs_M,cov_Bs_M,'CoV')

#%% Plotter UTN/B for skjær, over forskjellige mu_eta og V_eta. Simple
mu_UTNs_V,mu_Bs_V=calc_UTNs_eta(BeamV,mu_etas,'mu',3,Bending=False,Shear=True) 
plot_UB(mu_etas,mu_UTNs_V,mu_Bs_V,'mu')

cov_UTNs_V,cov_Bs_V=calc_UTNs_eta(BeamV,Cov_etas,'CoV',3,Bending=False,Shear=True) 
plot_UB(Cov_etas,cov_UTNs_V,cov_Bs_V,'CoV')

#%% Berekner utnyttelse for ulike verdier av eta med separate method. (Og beta frå FORM for gitt eta)
#Her er eta antatt normalfordelt#!!!

def calc_UTNs_eta_2nd(Beam,etas,MC,B_t,Bending=True,Shear=False):
    UTNs=[]
    Bs_FORM=[]
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    for  eta in etas:
        if MC=='mu':
            Beam['eta']=Bj.Create_dist_variable(eta, 0.2, 'Normal')
        if MC=='CoV':
            Beam['eta']=Bj.Create_dist_variable(0.9, eta, 'Normal')
        
        #Separate method:
    
        
        if Bending==True:
        
            PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta=1.05,Print_results=False,corrosion=True)
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            
            UTNs+=[Beam['utn'][0]]
            
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_l']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')
            alpha,B= FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            Beam['rho_l']=rhol 
        if Shear==True:
            PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,B_t,delta=1.05,Print_results=False,corrosion=True)
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            
            UTNs+=[Beam['utn'][1]]
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_w']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')
            alpha,B= FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            
            Beam['rho_w']=rhow
        Bs_FORM+=[B[0]]
    return UTNs,Bs_FORM

#%% #%% Berekner utnyttelse for ulike verdier av eta med separate method og approximerte alpha. (Og beta frå FORM for gitt eta)
#Her er eta antatt normalfordelt#!!!


def calc_UTNs_eta_approx(Beam,etas,MC,B_t,Bending=True,Shear=False):
    UTNs=[]
    Bs_FORM=[]
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    for  eta in etas:
        if MC=='mu':
            Beam['eta']=Bj.Create_dist_variable(eta, 0.2, 'Normal')
        if MC=='CoV':
            Beam['eta']=Bj.Create_dist_variable(0.9, eta, 'Normal')
        
        #Separate method:
    
        
        if Bending==True:
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_l']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')

            alphasM=AL.alphas_M_SD2(Beam,theta_R_bending,theta_E,Print_results=False)
            
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphasM,B_t,delta=1.05,Print_results=False,corrosion=True)
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            
            UTNs+=[Beam['utn'][0]]
            
           
            alpha,B= FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            Beam['rho_l']=rhol 
            
        if Shear==True:
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_w']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')

            alphasV=AL.alphas_V_SD2(Beam,theta_R_shear,theta_E,Print_results=False)
            
            PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphasV,B_t,delta=1.05,Print_results=False,corrosion=True)
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            
            UTNs+=[Beam['utn'][1]]
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*Beam['eta']['moments'][0],
                                                   np.sqrt(Beam['rho_w']['moments'][2]**2+Beam['eta']['moments'][2]**2), 'Normal')
            alpha,B= FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            
            Beam['rho_w']=rhow
        Bs_FORM+=[B[0]]
    return UTNs,Bs_FORM

#%% Plotter UTN/B for bøying, over forskjellige mu_eta og V_eta. Separate
mu_UTNs_M2,mu_Bs_M2=calc_UTNs_eta_2nd(BeamM,mu_etas,'mu',3,Bending=True,Shear=False) 
plot_UB(mu_etas,mu_UTNs_M2,mu_Bs_M2,'mu')

cov_UTNs_M2,cov_Bs_M2=calc_UTNs_eta_2nd(BeamM,Cov_etas,'CoV',3,Bending=True,Shear=False) 
plot_UB(Cov_etas,cov_UTNs_M2,cov_Bs_M2,'CoV')

#%% Plotter UTN/B for skjær, over forskjellige mu_eta og V_eta. Separate
mu_UTNs_V2,mu_Bs_V2=calc_UTNs_eta_2nd(BeamV,mu_etas,'mu',3,Bending=False,Shear=True) 
plot_UB(mu_etas,mu_UTNs_V2,mu_Bs_V2,'mu')

cov_UTNs_V2,cov_Bs_V2=calc_UTNs_eta_2nd(BeamV,Cov_etas,'CoV',3,Bending=False,Shear=True) 
plot_UB(Cov_etas,cov_UTNs_V2,cov_Bs_V2,'CoV')


#%% Plotter UTN/B for ulike mu/V eta for begge metoder.
def plot_UB2(etas,UTNs1,UTNs2,Bs,MC):

    fig, ax1 = plt.subplots()
    
    if MC=='mu':
        ax1.set_xlabel(r'$\mu_{\eta}$')
    if MC=='CoV':
        ax1.set_xlabel(r'$V_{\eta}$')
        
    
    color = 'tab:red'
    ax1.plot(etas,UTNs1,color = color,label='Simple')
    ax1.plot(etas,UTNs2,linestyle='dashed',color = color,label='Separate')
    ax1.set_ylabel('UTN',color=color)
    plt.legend()
    
    ax1.set_ylim([0,2])
    ax1.axhline(1,color='k',linestyle='dotted')
    
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\beta_{FORM}$',color=color)
    #ax2.axhline(3,color='r', label=r'$\beta_t$')
    ax2.set_ylim([1,5])
    ax2.plot(etas,Bs,color=color)
    
    
    #plt.grid()
    plt.show()

#%% Plotter begge metoder for bøying
plot_UB2(mu_etas,mu_UTNs_M,mu_UTNs_M2,mu_Bs_M,'mu')
plot_UB2(Cov_etas,cov_UTNs_M,cov_UTNs_M2,cov_Bs_M,'CoV')

#%% Plotter begge metoder for skjær

plot_UB2(mu_etas,mu_UTNs_V,mu_UTNs_V2,mu_Bs_V,'mu')
plot_UB2(Cov_etas,cov_UTNs_V,cov_UTNs_V2,cov_Bs_V,'CoV')

#%% Plotter med approx alfa og simple bøying
mu_UTNs_M3,mu_Bs_M3=calc_UTNs_eta_approx(BeamM,mu_etas,'mu',3,Bending=True,Shear=False) 
plot_UB(mu_etas,mu_UTNs_M3,mu_Bs_M3,'mu')

cov_UTNs_M3,cov_Bs_M3=calc_UTNs_eta_approx(BeamM,Cov_etas,'CoV',3,Bending=True,Shear=False) 
plot_UB(Cov_etas,cov_UTNs_M3,cov_Bs_M3,'CoV')

#Får svært rare resultat
#%% Plotter med approx alfa og simple skjær
mu_UTNs_V3,mu_Bs_V3=calc_UTNs_eta_approx(BeamV,mu_etas,'mu',3,Bending=False,Shear=True) 
plot_UB(mu_etas,mu_UTNs_V3,mu_Bs_V3,'mu')

cov_UTNs_V3,cov_Bs_V3=calc_UTNs_eta_approx(BeamV,Cov_etas,'CoV',3,Bending=False,Shear=True) 
plot_UB(Cov_etas,cov_UTNs_V3,cov_Bs_V3,'CoV')

#%% Importerer eksponeringsklasser.
from corrosion_estimation import Ex

#%% Definerer relevante parametre

ts=np.arange(0,80,1)
BeamM['Ex']='XS1' 
alfa_eta=0.7
#%%  Beregner utn for bjelke i gitt eksponeringsklasse over tid. DVM separate partialfaktorer
def calc_UTN_etad(Beam,ts,B_t,alfa_eta,Bending=True,Shear=False):
    UTNs=[] 
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    if Bending==True:
        PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta=1.05,Print_results=True,corrosion=False)
        
        
        etads=CE.calc_etad(CE.calc_eta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),CE.calc_Veta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),alfa_eta,B_t)
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*etads[i],0,'Normal')
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            UTNs+=[Beam['utn'][0]]
            Beam['rho_l']=rhol
    if Shear==True:
        PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,B_t,delta=1.05,Print_results=True,corrosion=False)
        
        
        etads=CE.calc_etad(CE.calc_eta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),CE.calc_Veta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),alfa_eta,B_t)
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*etads[i],0,'Normal')
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=False) 
            UTNs+=[Beam['utn'][1]]
            Beam['rho_w']=rhow
    return UTNs

#%%  Beregner utn for bjelke i gitt eksponeringsklasse over tid. Med MC2020 assessment partialfaktorer
def calc_UTN_etad_MC2020(Beam,ts,B_t,alfa_eta,Bending=True,Shear=False):
    UTNs=[] 
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    
    Beam['gamma_M_c']=[1.15, 1.15]
    Beam['gamma_M_s']=[1.075, 1.075] 
    
    Beam['Loading'][0]['gamma'] = 1.225
    Beam['Loading'][1]['gamma'] = 1.325
    Beam['Loading'][2]['gamma'] = 1.075
    
    if Bending==True:
        
        
        
        
        
        
        
        etads=CE.calc_etad(CE.calc_eta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),CE.calc_Veta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),alfa_eta,B_t)
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*etads[i],0,'Normal')
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=True) 
            UTNs+=[Beam['utn'][0]]
            Beam['rho_l']=rhol
    if Shear==True:
        
        
        
        etads=CE.calc_etad(CE.calc_eta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),CE.calc_Veta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20),alfa_eta,B_t)
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*etads[i],0,'Normal')
            PF.Controll_psf_beam(Beam,alpha_cc,Bending=Bending,Shear=Shear,Print_results=False,corrosion=False) 
            UTNs+=[Beam['utn'][1]]
            Beam['rho_w']=rhow
    return UTNs


#%%Plot Beam M UTN over tid.
#plt.plot(ts,calc_UTN_etad(BeamM,ts,B_t,alfa_eta,Bending=True,Shear=False))

plt.figure()
for i in Ex:
    BeamM['Ex']=i
    plt.plot(ts,calc_UTN_etad(BeamM,ts,B_t,alfa_eta,Bending=True,Shear=False),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'UTN')
plt.ylim([0,4])
plt.legend()
plt.grid()
plt.show()

#%% Plot Beam V utn over tid.

plt.figure()
for i in Ex:
    BeamV['Ex']=i
    plt.plot(ts,calc_UTN_etad(BeamV,ts,B_t,alfa_eta,Bending=False,Shear=True),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'UTN')
plt.ylim([0,4])
plt.legend()
plt.grid()
plt.show()

#%%Plot Beam M UTN over tid. MC2020 PSF
#plt.plot(ts,calc_UTN_etad(BeamM,ts,B_t,alfa_eta,Bending=True,Shear=False))

plt.figure()
for i in Ex:
    BeamM['Ex']=i
    plt.plot(ts,calc_UTN_etad_MC2020(BeamM,ts,B_t,alfa_eta,Bending=True,Shear=False),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'UTN')
plt.ylim([0,4])
plt.legend()
plt.grid()
plt.show()

#%% Plot Beam V utn over tid.

plt.figure()
for i in Ex:
    BeamV['Ex']=i
    plt.plot(ts,calc_UTN_etad_MC2020(BeamV,ts,B_t,alfa_eta,Bending=False,Shear=True),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'UTN')
plt.ylim([0,4])
plt.legend()
plt.grid()
plt.show()



#%% Beregner beta for ein bjelke etter t i ein viss eksponeringsklasse

def calc_B_eta(Beam,ts,Bending=True,Shear=False):
    B_FORMs=[] 
    rhol=Beam['rho_l']
    rhow=Beam['rho_w']
    
    mus_eta=CE.calc_eta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20)
    Vs_eta=CE.calc_Veta_ti(ts,Ex[Beam['Ex']]['ti'],Ex[Beam['Ex']]['Vcorr'],CE.pit_factor,20)
    if Bending==True:
        
        
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_l']=Bj.Create_dist_variable(Beam['rho_l']['moments'][0]*mus_eta[i],Vs_eta[i],'Lognormal')
            
            alpha,B= FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            
            Beam['rho_l']=rhol
            
            B_FORMs+=[B[0]]
    if Shear==True:
        
         
                   
        for i,t in enumerate(ts):
            
            Beam['rho_w']=Bj.Create_dist_variable(Beam['rho_w']['moments'][0]*mus_eta[i],Vs_eta[i],'Lognormal')
            alpha,B= FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            Beam['rho_w']=rhow
            B_FORMs+=[B[0]]
    return B_FORMs


#%% Plotter beta beam M over t i eksponering

plt.figure()
for i in Ex:
    BeamM['Ex']=i
    plt.plot(ts,calc_B_eta(BeamM,ts,Bending=True,Shear=False),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'$\beta$')
plt.ylim([1,5])
plt.legend()
plt.grid()
plt.show()

#%% Plotter beta beamV over t i eksponering
plt.figure()
for i in Ex:
    BeamV['Ex']=i
    plt.plot(ts,calc_B_eta(BeamV,ts,Bending=False,Shear=True),label=i)
plt.xlabel('t [y]')
plt.ylabel(r'$\beta$')
plt.ylim([1,5])
plt.legend()
plt.grid()
plt.show()


