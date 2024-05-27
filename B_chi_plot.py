#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 08:31:46 2024

@author: perivar
"""
#import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt

#%% import necessary modules:
#import necessary modules:
import Bjelker as Bj #importerer globale parametre og funksjoner


#Importing FORM analysis modules
from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
from FORM_V_beam_module import FORM_V_beam

#Partialfactors methods is imported from Partial_safety_factors.py
import Partial_safety_factors as PF
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

#%% Creating example beam for bending
#Creating example beam for bending
BeamM = Bj.Create_beam_dict(600, 200, 200, 550, 0, 0, Bj.fc, Bj.fs, 'indoor',10000)

#rhos are irrelevant/gone be calculated later



#Finding self-weight 
Gs=Bj.calc_Gs(BeamM,25,0.05)

#Creating and adding necessary parameters to the loads:
Gs['moments'][0]
Gs['LoadType']='SelfWeight'
Gp=Bj.Create_dist_variable(2, 0.1, 'Normal') #!!! 
Gp['LoadType']='Permanent'

Bj.add_qk(Gs, 0.5)
Bj.add_qk(Gp, 0.5)
#%% Function that create a array of load ratios (chi) and a corresponding array with loads corresponding to the chis:
#Function that create a array of load ratios (chi) and a corresponding array with loads corresponding to the chis:

def create_chi_loading(Beam,Gs,Gp,chiLim=[0.05,0.7]):
    chis=np.arange(chiLim[0],chiLim[1],0.01) 
    Loadings=[]

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
            mS = 0.39*0.81*Qk/0.82 #1 year mean value is calculated from characteristic load,, JRC table values
            vS = np.sqrt(0.51**2+0.26**2)  #CoV_Q from JRC table

            Q = Bj.Create_dist_variable(mS, vS, 'Gumbel')
            Q['LoadType']='Climatic'
            Q['qk']=Qk  
        Loadings+=[[Gs,Gp,Q]]
    Loadings=np.array(Loadings)
    
    return chis, Loadings

#%%
def create_scQ_loading(Beam,Qmax,Gs,Gp):
    scQs=np.arange(0.05,1,0.01) 
    Loadings=[]

    for i, scQ in enumerate(scQs):
        mQ=scQ*Qmax #Qk is calculated from the definition of chi.
        #Depending on the beam type, either imposed load or snowload is added to the beam in addition to self-weight and permanent load.
        if Beam['type']=='indoor':
            Qk = mQ*1.35/(0.53*1) #5 year mean value is calculated from characteristic load, JRC table values
            vQ = np.sqrt(0.49**2+0.10**2) #!!! CoV_Q from JRC table
        
            Q = Bj.Create_dist_variable(mQ, vQ, 'Gumbel')
            Q['LoadType']='Imposed'
            Q['qk']=Qk
        
        if Beam['type']=='roof':    
            Qk =0.82*mQ/(0.39*0.81) #1 year mean value is calculated from characteristic load,, JRC table values
            vS = np.sqrt(0.51**2+0.26**2)  #CoV_Q from JRC table

            Q = Bj.Create_dist_variable(mQ, vS, 'Gumbel')
            Q['LoadType']='Climatic'
            Q['qk']=Qk  
        Loadings+=[[Gs,Gp,Q]]
    Loadings=np.array(Loadings)
    
    return scQs, Loadings

#%% Function creating a_g and a_q
def create_aQ_loading(Beam,Gs,Gp):
    aGs=np.arange(0.6,1.01,0.01) 
    aQs=np.arange(0.1,0.7,0.01) 
    Loadings=[]

    for i, aQ in enumerate(aQs):
        LoadingsQ=[]
        for aG in(aGs):
            G=aG*Gs['moments'][0]+(1-aG)*Gp['moments'][0]
            mQ=aQ/(1-aQ)*(G) #Qk is calculated from the definition of chi. Er vel ikkje gitt at dei skal vere like?
            
            #Depending on the beam type, either imposed load or snowload is added to the beam in addition to self-weight and permanent load.
            
            if Beam['type']=='indoor':
                Qk = mQ*1.35/(0.53*1) #5 year mean value is calculated from characteristic load, JRC table values
                vQ = np.sqrt(0.49**2+0.10**2) #!!! CoV_Q from JRC table
            
                Q = Bj.Create_dist_variable(mQ, vQ, 'Gumbel')
                Q['LoadType']='Imposed'
                Q['qk']=Qk
            
            if Beam['type']=='roof':    
                Qk =0.82*mQ/(0.39*0.81) #1 year mean value is calculated from characteristic load,, JRC table values
                vS = np.sqrt(0.51**2+0.26**2)  #CoV_Q from JRC table
    
                Q = Bj.Create_dist_variable(mQ, vS, 'Gumbel')
                Q['LoadType']='Climatic'
                Q['qk']=Qk  
            LoadingsQ+=[[Gs,Gp,Q]]
        Loadings+=[LoadingsQ]
            
    Loadings=np.array(Loadings)
    
    return aQs, Loadings


testc,testl=create_aQ_loading(BeamM,Gs,Gp)

#Dette blir vel kanskje feil? 


#%% Chis and corresponding loading is calculated for beam M
#Chis and corresponding loading is calculated for beam M
chis,Loadings=create_chi_loading(BeamM,Gs,Gp)
#chis,Loadings=create_scQ_loading(BeamM,4.5,Gs,Gp) #Creates loading with ScQ instead of chi 

#%% Function that calculate the value of rho_l, such that the semi-probabilistic bending control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_l, such that the semi-probabilistic bending control result in UTN=1 for the different loadings.
#This function uses the simple partialfactor approach.
#Simply supported beam is assumed.
def calc_rhols(Beam,Loadings):
    rhols=[]
    for loading in Loadings:
        Beam['Loading']=loading
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta,Print_results=False) 
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
        rho_l=Bj.Create_dist_variable(rho_l, 0, 'Normal')#Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhols+=[rho_l]
    return np.array(rhols)

rhols=calc_rhols(BeamM,Loadings) #Rho_ls is calculated for beam M.

#%% Function that calculate the value of rho_w, such that the semi-probabilistic shear control result in UTN=1 for the different loadings.
#Function that calculate the value of rho_w, such that the semi-probabilistic shear control result in UTN=1 for the different loadings.
#This function uses the simple partialfactor approach.
#Simply supported beam is assumed.

def calc_rhows(Beam,Loadings):
    rhows=[]
    for loading in Loadings:
        Beam['Loading']=loading
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta,Print_results=False)
        #Partial factors are update for each loading, as the variation do not change this is actually unecessary but it guarantees that they are correctly defined.
        L=Beam['L']
        q=(Beam['Loading'][0]['qk']*Beam['Loading'][0]['gamma']
           +Beam['Loading'][1]['qk']*Beam['Loading'][1]['gamma']
           +Beam['Loading'][2]['qk']*Beam['Loading'][2]['gamma']) 
        d=Beam['d']
        bw=Beam['bw']
        
        
        fyd=Beam['fs']['fk']/Beam['gamma_M_s'][1]
        Ed=q*(L/2-d)
        
        #rho_w is calculated by eq. 5.15?
        rho_w=Ed/(bw*0.9*d*fyd*2.5) #!!! Cot(theta)=2.5 is assumed. The Beta-chi plots clearly shows where this assumption becomes wrong. But for normal chi range it seems okay.
        rho_w=Bj.Create_dist_variable(rho_w, 0, 'Normal')  #Rho is defined without any variation, but as a distributed variabel for the FORM-analysis.
        rhows+=[rho_w]
    return np.array(rhows)

rhows=calc_rhows(BeamM,Loadings) #Rho_ws is calculated for beam M.



#%% Function that calculate the value of rho_l, such that the semi-probabilistic bending control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_l, such that the semi-probabilistic bending control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach.
#Simply supported beam is assumed.

def calc_rhols_2nd(Beam,Loadings):
    rhols=[]
    for loading in Loadings:
        Beam['Loading']=loading
        PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta,Print_results=False)
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
    return np.array(rhols)

rhols2=calc_rhols_2nd(BeamM,Loadings)#Rho_ls is calculated for beam M with the separate PSF approach.

#%% Function that calculate the value of rho_w, such that the semi-probabilistic shear control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_w, such that the semi-probabilistic shear control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach.
#Simply supported beam is assumed.
def calc_rhows_2nd(Beam,Loadings, Dom='material'):
    rhows=[]
    for loading in Loadings:
        Beam['Loading']=loading
        
        if Dom == 'material':
            PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta,Print_results=False) #Shear model non-dominant material dominant.
        if Dom == 'model':
            PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R2,alpha_E2,alpha_R,B_t,delta,Print_results=False) #Shear model dominant material non-dominant.
        if Dom == 'all':
            PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R,B_t,delta,Print_results=False) #All resistance dominant
        
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
    return np.array(rhows)

rhows2=calc_rhows_2nd(BeamM,Loadings)  #Rho_ws is calculated for beam M with the separate PSF approach.

#%% Function that calculate the value of rho_l, such that the semi-probabilistic bending control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_l, such that the semi-probabilistic bending control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach, with different alphas for all values from simplification or td FORM, used here.
#Simply supported beam is assumed.

def calc_rhols_RA(Beam,Loadings):
    rhols=[]
    Beam['rho_l']=Bj.Create_dist_variable(0.005, 0, 'Normal') #Guesses initial rho_l
    for loading in Loadings:
        Beam['Loading']=loading
        alphas,beta=FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc=alpha_cc,Print_results=False)
        alphas=alphas*-1
        PF.Update_psf_beam_RA(Beam,theta_R_bending,theta_R_shear,theta_E,alphas,B_t,delta,Print_results=False)
        
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

rhols_RA=calc_rhols_RA(BeamM,Loadings)#Rho_ls is calculated for beam M with the separate PSF approach, with real alphas.

#%% Function that calculate the value of rho_w, such that the semi-probabilistic shear control  result in UTN=1 for the different loadings.
#Function that calculate the value of rho_w, such that the semi-probabilistic shear control result in UTN=1 for the different loadings.
#This function uses the separate partialfactor approach, with alphas from FORM. 
#Simply supported beam is assumed.
def calc_rhows_RA(Beam,Loadings):
    rhows=[]
    Beam['rho_w']=Bj.Create_dist_variable(0.001, 0, 'Normal') #Guesses initial rho_w
    Beam['cot_t']=Bj.cot_t(Beam,Print_results=False) 
    for loading in Loadings:
        Beam['Loading']=loading
        alpha_V,beta=FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc=alpha_cc,Print_results=False)
        alphas=np.zeros(8)
        alphas[:2]=alpha_V[:2]
        alphas[-5:]=alpha_V[-5:]
        alphas=alphas*-1
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

rhows_RA=calc_rhows_RA(BeamM,Loadings)  #Rho_ws is calculated for beam M with the separate PSF approach.

#%% Calculate betas with FORM the different rho and loadings for bending
#Calculate betas with FORM the different rho and loadings for bending
def calc_Betas_M(Beam,Loadings,rhols):
    Betas=[]
    for i, rho_l in enumerate(rhols):
        Beam['rho_l']=rho_l
        Beam['Loading']=Loadings[i]
        alpha1,beta1=FORM_M_beam(Beam,theta_R_bending,theta_E,alpha_cc,Print_results=False,B_plots=False,MonteCarlo=False)
        Betas+=[beta1[0]]
    return np.array(Betas)

Betas=calc_Betas_M(BeamM,Loadings,rhols) #Calc beta for the two approaches for Beam M bending.
Betas2=calc_Betas_M(BeamM,Loadings,rhols2)
BetasRA=calc_Betas_M(BeamM,Loadings,rhols_RA)

#%% Calculate betas with FORM the different rho and loadings for shear
#Calculate betas with FORM the different rho and loadings for shear
def calc_Betas_V(Beam,Loadings,rhows):
    Betas=[]
    for i, rho_w in enumerate(rhows):
        Beam['rho_w']=rho_w
        Beam['cot_t']=Bj.cot_t(Beam, Print_results=False) #Sett denne True for å sjekke cot_t
        Beam['Loading']=Loadings[i]
        alpha1,beta1=FORM_V_beam(Beam,theta_R_shear,theta_E,alpha_cc,Print_results=False,B_plots=False,MonteCarlo=False)
        Betas+=[beta1[0]]
    return np.array(Betas)

BetasV=calc_Betas_V(BeamM,Loadings,rhows) #Calc beta for the two approaches for Beam M shear.
BetasV2=calc_Betas_V(BeamM,Loadings,rhows2)
#%% Plot rho_l for the two approaches against chi
# Plot rho_l for the two approaches against chi
def plot_rhol(chis,rhols,rhols2):
    RHOLS=[]
    for rho_l in rhols:
        RHOLS+=[rho_l['moments'][0]]
    RHOLS2=[]
    for rho_l in rhols2:
        RHOLS2+=[rho_l['moments'][0]]
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOLS,label='Simple approach')
    plt.plot(chis,RHOLS2,label='Separate approach')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho_l$")
    plt.legend()
    plt.grid()
    plt.show() 
    
#plot_rhol(chis,rhols,rhols2)
#plot_rhol(chis,rhols,rhols_RA)

#%% Plot rho_l for the three approaches against chi
# Plot rho_l for the three approaches against chi
def plot_rhol3(chis,rhols,rhols2,rhols3):
    RHOLS=[]
    for rho_l in rhols:
        RHOLS+=[rho_l['moments'][0]]
    RHOLS2=[]
    for rho_l in rhols2:
        RHOLS2+=[rho_l['moments'][0]]
    RHOLS3=[]
    for rho_l in rhols3:
        RHOLS3+=[rho_l['moments'][0]]
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOLS,label='Simple approach')
    plt.plot(chis,RHOLS2,label='Separate approach')
    plt.plot(chis,RHOLS3,label=r'$\alpha-FORM$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho_l$")
    plt.legend()
    plt.grid()
    plt.show() 
    
#plot_rhol3(chis,rhols,rhols2,rhols_RA)
#%% Plot rho_w for the two approaches against chi
#Plot rho_w for the two approaches against chi
def plot_rhow(chis,rhows,rhows2):
    RHOWS=[]
    for rho_w in rhows:
        RHOWS+=[rho_w['moments'][0]]
    RHOWS2=[]
    for rho_w in rhows2:
        RHOWS2+=[rho_w['moments'][0]]
    
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOWS,label='Simple approach')
    plt.plot(chis,RHOWS2,label='Separate approach')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho_w$")
    plt.legend()
    plt.grid()
    plt.show() 
    
#plot_rhow(chis,rhows,rhows2)

#%% Plot rho_w for the three approaches against chi
#Plot rho_w for the three approaches against chi
def plot_rhow3(chis,rhows,rhows2,rhows3):
    RHOWS=[]
    for rho_w in rhows:
        RHOWS+=[rho_w['moments'][0]]
    RHOWS2=[]
    for rho_w in rhows2:
        RHOWS2+=[rho_w['moments'][0]]
    RHOWS3=[]
    for rho_w in rhows3:
        RHOWS3+=[rho_w['moments'][0]]
    
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,RHOWS,label='Simple approach')
    plt.plot(chis,RHOWS2,label='Separate approach')
    plt.plot(chis,RHOWS3,label=r'$\alpha-FORM$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho_w$")
    plt.legend()
    plt.grid()
    plt.show() 
#%% Plot loading against chi
# Plot loading against chi
#Possible to compare the loading of two different beams.
def plot_loading(chis,Loadings):#,Loadings2):
    LOADS=[]
    #LOADS2=[]
    for load in Loadings:
        q=load[0]['moments'][0]+load[1]['moments'][0]+load[2]['moments'][0]
        LOADS+=[q]
    #for load in Loadings2:
     #   q=load[0]['moments'][0]+load[1]['moments'][0]+load[2]['moments'][0]
      #  LOADS2+=[q]
    
    
    plt.figure(figsize=(6,4))
    plt.plot(chis,LOADS)
    #plt.plot(chis,LOADS2)
    
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"Sum of mean load kN/m")
    #plt.legend()
    plt.grid()
    plt.show() 

#plot_loading(chis,Loadings)

#%% The most interesting plot: Betas against chi for the two approaches.
#The most interesting plot: Betas against chi for the two approaches.
def plot_Betas(chis,Betas,Betas2,Bruddform):
   
    #plt.figure(figsize=(4,3)) Smaller fig gives bigger text.
    
    plt.figure(figsize=(6,4))
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    plt.plot(chis,Betas,label='Simple approach')
    plt.plot(chis,Betas2,label='Separate approach')
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    plt.legend()
    plt.grid()
    plt.show() 
    
#plot_Betas(chis,Betas,Betas2,'bending') #Beta-chi BeamM bending
#plot_Betas(chis,Betas,BetasRA,'bending') #Beta-chi BeamM bending
#plot_Betas(chis,BetasV,BetasV2,'shear') ##Beta-chi BeamM shear

#%% The most interesting plot: Betas against chi for the three approaches.
#The most interesting plot: Betas against chi for the three approaches.

def plot_Betas3(chis,Betas,Betas2,BetasRA,Bruddform):
   
    #plt.figure(figsize=(4,3)) Smaller fig gives bigger text.
    
    plt.figure(figsize=(8,5))
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    plt.plot(chis,Betas,label='Simple approach')
    plt.plot(chis,Betas2,label='Separate approach')
    plt.plot(chis,BetasRA,label=r'$\alpha-FORM$')
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    plt.ylim([2,4.6])
    plt.legend()#loc='upper right')
    plt.grid()
    plt.show() 

plot_Betas3(chis,Betas,Betas2,BetasRA,'bending') #Beta-chi BeamM bending
#%% To compare the results of Snow load VS imposed load they to equal beam, one with snow load and one with imposed load are plotted together.
#To compare the results of Snow load VS imposed load they to equal beam, one with snow load and one with imposed load are plotted together.
BeamS = Bj.Create_beam_dict(600, 200, 200, 550, 0, 0, Bj.fc, Bj.fs, 'roof',10000) #BeamS is equal BeamM except the type witch is 'roof', and therefore implies snowload instead of imposed load.
chisS,LoadingsS=create_chi_loading(BeamS,Gs,Gp) #New loadings are created for beam S
#Simple approach:
rhols_S=calc_rhols(BeamS,LoadingsS)   #New rho_ls is calculated for beam S
Betas_S=calc_Betas_M(BeamS,LoadingsS,rhols_S)

#Separate approach:
rhols_S2=calc_rhols_2nd(BeamS,LoadingsS)
Betas_S2=calc_Betas_M(BeamS,LoadingsS,rhols_S2)

def plot_Betas_SI(chis,Betas,Betas2,Betas_S,Betas_S2,Bruddform):
   
    #plt.figure(figsize=(4,3)) Smaller fig gives bigger text.
    
    plt.figure(figsize=(8,5))
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    plt.plot(chis,Betas,label='Imposed simple')
    plt.plot(chis,Betas2,label='Imposeds separate')
    plt.plot(chis,Betas_S,label='Snow simple')
    plt.plot(chis,Betas_S2,label='Snow separate')
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    plt.legend()
    plt.grid()
    plt.show() 
    
#plot_Betas_SI(chis,Betas,Betas2,Betas_S,Betas_S2,'bending')
#%% Comparing rho_l for snow and imposed load.
# Comparing rho_l for snow and imposed load.
#plot_rhol(chis,rhols,rhols_S)
#Mark legend in plot is wrong, should be imposed and snow not simple and separate.

#%% This cell shows how another beam can be created and controlled.

'''
BeamA = Bj.Create_beam_dict(600, 300, 300, 550, 0, 0, Bj.fc, Bj.fs, 'roof',15000)
chisA,LoadingsA=create_chi_loading(BeamA,Gs,Gp) #Må strengt tatt lage nye laster
rhols_A=calc_rhols(BeamA,LoadingsA)  
Betas_A=calc_Betas_M(BeamA,LoadingsA,rhols_A)
plot_Betas(chis,Betas_A,Betas2,'bending')
plot_rhol(chis,rhols_A,rhols)
'''


#%% To control the algorithm the utilization is calculated for the different beams with corresponding rho and loadings.
# To control the algorithm the utilization is calculated for the different beams with corresponding rho and loadings.
# For bending simple approach:
def calc_UTN_M(Beam,Loadings,rhols):
    UTN=[]
    for i, rho_l in enumerate(rhols):
        Beam['rho_l']=rho_l
        Beam['Loading']=Loadings[i]
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta,Print_results=False)
        PF.Controll_psf_beam(Beam,alpha_cc,Bending=True,Shear=False,Print_results=False)
        UTN+=[Beam['utn'][0]]
    return np.array(UTN)

UTNs_M=calc_UTN_M(BeamM,Loadings,rhols) #Utn is calculated for bending simple approach for Beam M

#UTNs_A=calc_UTN_M(BeamA,LoadingsA,rhols_A)

#%% Calculate UTN for bending separate approach:
# Calculate UTN for bending separate approach:
def calc_UTN_M_2nd(Beam,Loadings,rhols):
    UTN=[]
    for i, rho_l in enumerate(rhols):
        Beam['rho_l']=rho_l
        Beam['Loading']=Loadings[i]
        PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta,Print_results=False)
        PF.Controll_psf_beam(Beam,alpha_cc,Bending=True,Shear=False,Print_results=False)
        UTN+=[Beam['utn'][0]]
    return np.array(UTN)

UTNs_M2=calc_UTN_M_2nd(BeamM,Loadings,rhols2) #Utn is calculated for bending separate approach for Beam M
#%% Calculate UTN for shear simple approach:
# Calculate UTN for shear simple approach:
def calc_UTN_V(Beam,Loadings,rhows):
    UTN=[]
    for i, rho_w in enumerate(rhows):
        Beam['rho_w']=rho_w
        Beam['Loading']=Loadings[i]
        PF.Update_psf_beam(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta,Print_results=False)
        PF.Controll_psf_beam(Beam,alpha_cc,Bending=False,Shear=True,Print_results=False)
        UTN+=[Beam['utn'][1]]
    return np.array(UTN)


UTNs_M_V=calc_UTN_V(BeamM,Loadings,rhows) #Utn is calculated for shear simple approach for Beam M

#%% Calculate UTN for shear separate approach:
# Calculate UTN for shear separate approach:
def calc_UTN_V_2nd(Beam,Loadings,rhows):
    UTN=[]
    for i, rho_w in enumerate(rhows):
        Beam['rho_w']=rho_w
        Beam['Loading']=Loadings[i]
        PF.Update_psf_beam_2nd(Beam,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta,Print_results=False)
        PF.Controll_psf_beam(Beam,alpha_cc,Bending=False,Shear=True,Print_results=False)
        UTN+=[Beam['utn'][1]]
    return np.array(UTN)


UTNs_M_V2=calc_UTN_V_2nd(BeamM,Loadings,rhows2) #Utn is calculated for shear separate approach for Beam M

#%% Plot UTN against chi, to control that UTN = 1
#Plot UTN against chi, to control that UTN = 1 

def plot_UTN(chis,UTNs_M,UTNs_V,labels):
    plt.figure(figsize=(6,4))
    plt.plot(chis,UTNs_M,label=labels[0])
    plt.plot(chis,UTNs_V,label=labels[1])
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"UTNs")
    plt.legend()
    plt.grid()
    plt.show() 
    
#Plot of Utn for beam M for the two approaches:
#plot_UTN(chis,UTNs_M,UTNs_M_V,['bending 1. approach','shear 1. approach'])
#plot_UTN(chis,UTNs_M2,UTNs_M_V2,['bending 2. approach','shear 2. approach'])

#Bending seems okay for both approaches
#Shear seems okay for both approach. 


#plot_UTN(chis,UTNs_A,UTNs_V)
#%% Creates a new beamm, Beam V. That is more realistic in the case of shear failure:
#Creates a new beamm, Beam V. That is more realistic in the case of shear failure:
BeamV = Bj.Create_beam_dict(400, 400, 400, 350, 0, 0, Bj.fc, Bj.fs, 'indoor',10000)

#Creates corresponding loading:
GsV=Bj.calc_Gs(BeamV,25,0.05)
GsV['moments'][0]
GsV['LoadType']='SelfWeight'
GpV=Bj.Create_dist_variable(20, 0.1, 'Normal') #To get "normal" rhow values the permanent load is much higher for this beam. #!!!
GpV['LoadType']='Permanent'

Bj.add_qk(GsV, 0.5)
Bj.add_qk(GpV, 0.5)

chisV,LoadingsV=create_chi_loading(BeamV,GsV,GpV,chiLim=[0.05,0.6]) #Limits chi to 0.6 to avoid cot_t<2.5. However I do not think load ratios over 0.6 is very relevant, should be controlled#!!!
#chisV,LoadingsV=create_scQ_loading(BeamV, 13, GsV, GpV) #Creates loading with ScQ instead of chi 

#Then rhows and betas is calculated and plotted for the two approaches for beam V:
rhowsV=calc_rhows(BeamV,LoadingsV)  
rhows2V=calc_rhows_2nd(BeamV,LoadingsV)  
#plot_rhow(chisV,rhowsV,rhows2V)
Betas_V=calc_Betas_V(BeamV,LoadingsV,rhowsV)
Betas_V2=calc_Betas_V(BeamV,LoadingsV,rhows2V)
#plot_Betas(chisV,Betas_V,Betas_V2,'shear')


#plot_loading(chisV,LoadingsV)
#%% Adding results with alphas estimated by FORM
# Adding results with alphas estimated by FORM
rhows_RAV=calc_rhows_RA(BeamV,LoadingsV)
Betas_RAV=calc_Betas_V(BeamV,LoadingsV,rhows_RAV)
#plot_rhow3(chisV,rhowsV,rhows2V,rhows_RAV)
#plot_Betas3(chisV,Betas_V,Betas_V2,Betas_RAV,'shear')

#%% Evaluating different chioces of dominating alfa

rhowsV_Dmodel=calc_rhows_2nd(BeamV,LoadingsV,Dom='model')  
#%%
rhowsV_Dall=calc_rhows_2nd(BeamV,LoadingsV,Dom='all')  
Betas_V2_model=calc_Betas_V(BeamV,LoadingsV,rhowsV_Dmodel)
Betas_V2_all=calc_Betas_V(BeamV,LoadingsV,rhowsV_Dall)

#%%
Vdict={}
Vdict['Simple approach']={}
Vdict['Simple approach']['rhos']=rhowsV
Vdict['Simple approach']['betas']=Betas_V


Vdict['FORM']={}
Vdict['FORM']['rhos']=rhows_RAV
Vdict['FORM']['betas']=Betas_RAV


Vdict['Material dominating']={}
Vdict['Material dominating']['rhos']=rhows2V
Vdict['Material dominating']['betas']=Betas_V2

Vdict['Model dominating']={}
Vdict['Model dominating']['rhos']=rhowsV_Dmodel
Vdict['Model dominating']['betas']=Betas_V2_model

Vdict['All dominating']={}
Vdict['All dominating']['rhos']=rhowsV_Dall
Vdict['All dominating']['betas']=Betas_V2_all
#%%
#%% Plot rho_l for the separate approach with different alphas against chi
# Plot rho_l for the separate approach with different alphas against chi
def plot_rho_d(chis,a_dict):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10,6))
    for metode in a_dict:
        
        RHOS=[]
        for rho in a_dict[metode]['rhos']:
            RHOS+=[rho['moments'][0]]
        plt.plot(chis,RHOS,label=metode)

    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\rho$")
    plt.legend()
    plt.grid()
    plt.show() 
    

#%% The most interesting plot: Betas against chi for the three approaches.
#The most interesting plot: Betas against chi for the three approaches.

def plot_Betas_d(chis,a_dict,Bruddform,B_t=B_t):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10,6))
    #plt.title(r'$\beta-{}$'.format(Bruddform))
    
    for metode in a_dict:
        plt.plot(chis,a_dict[metode]['betas'],label=metode)
   
    plt.axhline(B_t,color='r', label=r'$\beta_t$')
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\beta$")
    plt.legend()#loc='upper right')
    plt.ylim([2,4.6])
    plt.grid()
    plt.show() 




#%%
#plot_rho_d(chisV,Vdict)
#plot_Betas_d(chisV,Vdict,'shear')


#%% A beam equal to beam V but with snowload instead of imposed load is created to compare the two loadtypes.
#A beam equal to beam V but with snowload instead of imposed load is created to compare the two loadtypes.
BeamVS = Bj.Create_beam_dict(400, 400, 400, 350, 0, 0, Bj.fc, Bj.fs, 'roof',10000)
chisVS,LoadingsVS=create_chi_loading(BeamVS,GsV,GpV,chiLim=[0.05,0.6])
rhows_VS=calc_rhows(BeamVS,LoadingsVS)  
Betas_VS=calc_Betas_V(BeamVS,LoadingsVS,rhows_VS)

rhows_VS2=calc_rhows_2nd(BeamVS,LoadingsVS)  
Betas_VS2=calc_Betas_V(BeamVS,LoadingsVS,rhows_VS2)

    
#plot_Betas_SI(chisV,Betas_V,Betas_V2,Betas_VS,Betas_VS2,'shear') #The results for the two load types with both approaches for beam V is plotted together.
#%% Checks utilization for BeamV, it seems okay
# Checks utilization for BeamV, it seems okay
UTNs_V=calc_UTN_V(BeamV,LoadingsV,rhowsV)
UTNs_V2=calc_UTN_V_2nd(BeamV,LoadingsV,rhows2V)
#plot_UTN(chisV,UTNs_V,UTNs_V2,['Shear 1. approach', 'Shear 2.approach']) 

#%% Checks utilization for BeamV with snowload, this gives same results as for imposed load.
#Checks utilization for BeamV with snowload, this gives same results as for imposed load.
UTNs_VS=calc_UTN_V(BeamVS,LoadingsVS,rhows_VS)
UTNs_VS2=calc_UTN_V_2nd(BeamVS,LoadingsVS,rhows_VS2)
#plot_UTN(chisV,UTNs_V,UTNs_V2,['Snow shear 1. approach', 'Snow shear 2.approach'])
