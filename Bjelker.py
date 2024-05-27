#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jan 23 10:20:05 2024

@author: perivar
'''
# Import relevant libraies:

import numpy as np
from scipy import stats as st
import pandas as pd




# %% Defining global parameters:
#Defining global parameters:
alpha_cc = 0.85

# Defining alfa for 1year ref period:
alpha_E = -0.8
alpha_R = 0.7
alpha_E2 =0.4*alpha_E
alpha_R2=0.4*alpha_R

B_t = 3.3

delta = 1.05  # for asssessment delta=1 for design

#These parameters can easily be added to individual beams, if one also wants to differate in this when analysing.

# %% Defining distributed part of material
#Defining distributed part of material
# Creating one example material (concrete):
mfc = 20.0  # MPa mean example value
vfc = 0.1  # CoV from JRC
sfc = vfc*mfc  # stdev
tfc = 'Lognormal'  # Either 'Normal', 'Lognormal', 'Gumbel'


# Defining dictionary:
fc = {}  # initiate a empty dictionary
fc['type'] = tfc  # define and specify the property type = lognormal
# define and specify the property moments = mean, stdev, CoV
fc['moments'] = [mfc, sfc, vfc]

# %% Adds the characteristic value to distributed variable (material)
#Adds the characteristic value to distributed variable 
def add_fk(X):  # Definerer den karakteristiske verdien som 5% persentilen for ein fordelt styrkeparameter
    moments = X['moments']
    if X['type'] == 'Normal':
        fk = st.norm(loc=moments[0], scale=moments[1]).ppf(0.05)
    if X['type'] == 'Lognormal':
        fk = moments[0]/np.exp(1.645*moments[2]) #Assume this is a approximation, should maybe be calculated more exactly?
        #Fetched from JRC Table A.14
    X['fk']=fk

add_fk(fc)

fc['fk']

#Charateristic value of loads (variable) and other relevant parameters should be calculated as well
#Dette hadde eg misforstått, bruker verdier frå tabell 1.1
def add_qk(X,p): #p (percentil for permanent loads and climatic loads, scaling parameter for imposed loads)
    moments = X['moments'] #Gjennomsnitt for 1års last
    if X['type'] == 'Normal':
        qk = st.norm(loc=moments[0], scale=moments[1]).ppf(p)
    
    if X['LoadType'] =='Climatic':
        if X['type'] == 'Gumbel':
           # a = np.pi/np.sqrt(6)/moments[1] #Basic parameters for gumbel load...
            #b = moments[0] - 0.5772/a
            #qk = b+1/a*np.log(1/p) #Aprox. by compendium eq.4.52
            qk=p*moments[0]/(0.81*0.39) #p normalt lik 0.82
    if X['LoadType']=='Imposed':
        qk=p*moments[0]/0.53 #p normalt lik 1.35
    X['qk']=qk


# %% Function for defining variable dictionary.
#Function for defining variable dictionary.

def Create_dist_variable(mean, CoV, Type):
    stdev = mean*CoV
    material = {}  # initiate a empty dictionary
  
    material['type'] = Type  # define and specify the property type = lognormal, normal, gumble
    
    material['moments'] = [mean, stdev, CoV] # define and specify the property moments = mean, stdev, CoV
    
    return material  # Returns material dictionary


fs = Create_dist_variable(246, 0.045, 'Lognormal')  # Creating example steel
add_fk(fs)

# %% Defining deictionary for example beam1
#Defining deictionary for example beam1

# Example beam 1
h = 400
b = 200
d = 350

rho_l = 0.02
rho_w = 0.002

Beam1 = {}
Beam1['h'] = h
Beam1['b'] = b
Beam1['bw'] = b
Beam1['d'] = d
Beam1['rho_l'] = rho_l
Beam1['rho_w'] = rho_w
Beam1['alpha_cw'] = 1
Beam1['fc'] = fc
Beam1['fs'] = fs
Beam1['type'] = 'roof'


#print('Beam1 fck=', Beam1['fc']['fk'])
#print()

# %% Function for creating dict example beams:
#Function for creating dict example beams:

def Create_beam_dict(h, b, bw, d, rho_l, rho_w, fc, fs,Type,L=0,alpha_cw=1):
    Beam = {}
    Beam['h'] = h
    Beam['b'] = b
    Beam['bw'] = bw
    Beam['d'] = d
    Beam['rho_l'] = rho_l
    Beam['rho_w'] = rho_w
    Beam['alpha_cw'] = alpha_cw
    Beam['fc'] = fc
    Beam['fs'] = fs
    Beam['type'] = Type
    Beam['L']=L

    return Beam


Beam2 = Create_beam_dict(600, 300, 300, 550, 0.015,
                         0.005, fc, fs, 'indoor') #Testing function and define new example beam.

#print('Beam 2 alpha_cw=', Beam2['alpha_cw'])
#print('Beam 2 bw=', Beam2['bw'])
#print()
# %% Defining cross section capacity (From mean value, should be modified for design capacity.)
#Defining cross section capacity (From mean value, should be modified for design capacity.)
# Defining moment capacity: (Eq.1.4)


def Mr(Beam):
    rho_l = Beam['rho_l']
    fy = Beam['fs']['moments'][0]
    b = Beam['b']
    d = Beam['d']
    fc = Beam['fc']['moments'][0]
    return rho_l*fy*(1-0.5*rho_l*fy/(alpha_cc*fc))*b*d**2

# Defining shear capacity: (Eq.1.9)


def Vr(Beam):
    rho_w = Beam['rho_w']
    fyw = Beam['fs']['moments'][0]
    b_w = Beam['bw']
    d = Beam['d']
    cot_t = Beam['cot_t']

    z = 0.9*d
    return rho_w*b_w*z*fyw*cot_t

#%% Defining load effect from mean values, for simply supported beam
#Defining load effect from mean values, for simply supported beam
def LoadEffect(Beam):
    L=Beam['L']
    q=Beam['Loading'][0]['moments'][0]+Beam['Loading'][1]['moments'][0]+Beam['Loading'][2]['moments'][0]
    d=Beam['d']
    M=q*L**2/8
    V=q*(L-d)/2
    return M/10**6,V/10**3 #Returns values in kNm/kN for lesbarhet
    

# %% Function for calculating cot(theta) (Eq.1.8)
#Function for calculating cot(theta) (Eq.1.8)

def cot_t(Beam,Print_results=False):  # (Eq.1.8)
    
    if type(Beam['rho_w']) == dict:
        rho_w=Beam['rho_w']['moments'][0] #Kan plasseres under skjær/moment for å kunne kjøre PSF for bare ein
    else:
        rho_w=Beam['rho_w']
    fyw = Beam['fs']['moments'][0]
    alpha_cw = Beam['alpha_cw']
    nu_1 = 0.6*(1-Beam['fc']['fk']/250)
    fc = Beam['fc']['moments'][0]
    theta = np.arcsin(np.sqrt(rho_w*fyw/(alpha_cw*nu_1*alpha_cc*fc)))
    cot = 1/np.tan(theta)
    if Print_results==True:
        print('cot_t= ', np.round(cot,2), 'før den settes 1<=cot_t<=2.5')
    if cot < 1:
        return 1
    elif cot > 2.5:
        return 2.5
    else:
        return cot


# %% Some example results
#Some example results
Beam1['cot_t'] = cot_t(Beam1)

'''
print('Beam1 cot(theta)=',Beam1['cot_t'])

print(f'Based on mean values Vr={np.round(Vr(Beam1)/1000,2)} kN')
print(f'Based on mean values Mr={np.round(Mr(Beam1)/10**6,2)} kNm')
print()
'''
Beam2['cot_t'] = cot_t(Beam2)
'''
print('Beam2 cot(theta)=',Beam2['cot_t'])

print(f'Based on mean values Vr={np.round(Vr(Beam2)/1000,2)} kN')
print(f'Based on mean values Mr={np.round(Mr(Beam2)/10**6,2)} kNm')
print()
'''
# %% Defining example loads and model uncertianty
#Defining example loads and model uncertianty
# Self-weight
G_s = Create_dist_variable(2, 0.05, 'Normal') #Load model included
G_s['LoadType']='SelfWeight'
# Permanent load (assume small V)
G_p = Create_dist_variable(2, 0.1, 'Normal')
G_p['LoadType']='Permanent'

# Snowload 1 year reference period
S50 = 4  # kN/m #Karakteristisk
mS = 0.39*0.81*S50/0.82
vS = np.sqrt(0.51**2+0.26**2)



Q_snow = Create_dist_variable(mS, vS, 'Gumbel')
Q_snow['LoadType']='Climatic'
Q_snow['qk']=S50

# Imposed load 1 (5) year reference period
Q50 = 3 # kN/m #!!! karakteristisk
mQ = 0.53*1*Q50/1.35 #5års gjennomsnitt
vQ = np.sqrt(0.49**2+0.10**2)


Q_imposed = Create_dist_variable(mQ, vQ, 'Gumbel')
Q_imposed['LoadType']='Imposed'
Q_imposed['qk']=Q50

# Load model
# assuming statistically determined frame #!!!
theta_E = Create_dist_variable(1, 0.05, 'Lognormal') #from JRC

# Resistance model
theta_R_bending = Create_dist_variable(1.03, 0.07, 'Lognormal') #from JRC
theta_R_shear = Create_dist_variable(1.4, 0.25, 'Lognormal') #from JRC

#%% Calculate and adds charcteristic values to the loads.
#Calculate and adds charcteristic values to the loads.

add_qk(G_s,0.5) #Mean values are assumed as charcteristic value
add_qk(G_p,0.5)
#add_qk(Q_snow,0.02) #Charcteristic value is defined as value that has annual probability of exceedance equal 2%
#add_qk(Q_imposed,1.35) #Charcteristic value is defined as value that has annual probability of exceedance equal 2%

#print(u'\u03BC(Q_snow)={} and qk(Q_snow)={}'.format(np.round(Q_snow['moments'][0],2),np.round(Q_snow['qk'],2)))
#print(u'\u03BC(Q_imposed)={} and qk(Q_imposed)={}'.format(np.round(Q_imposed['moments'][0],2),np.round(Q_imposed['qk'],2)))

# %% Partialfactors by simplified formulas from (Caspeele 2023) is imported from Partial_safety_factors.py
#Partialfactors by simplified formulas from (Caspeele 2023) is imported from Partial_safety_factors.py
import Partial_safety_factors as PF
# %% Calculating load gammas for example: All alphas the same
#Calculating load gammas for example: All alphas the same
gamma_Gs = PF.gamma_G_norm(
    alpha_E, B_t, G_s['moments'][2], theta_E['moments'][2]).value 
gamma_Gp = PF.gamma_G_norm(
    alpha_E, B_t, G_p['moments'][2], theta_E['moments'][2]).value


mu_tref_snow = Q_snow['moments'][0]*theta_E['moments'][0]
Vtref_snow = np.sqrt(Q_snow['moments'][2]**2+theta_E['moments'][2]**2)
qk_snow=Q_snow['qk']*theta_E['moments'][0]
gamma_snow = PF.gamma_Q_gumbel(
    alpha_E, B_t, mu_tref_snow, Vtref_snow,qk_snow, delta).value


mu_tref_imposed = Q_imposed['moments'][0]*theta_E['moments'][0]
Vtref_imposed = np.sqrt(Q_snow['moments'][2]**2+theta_E['moments'][2]**2)
qk_imposed=Q_imposed['qk']*theta_E['moments'][0]
gamma_imposed = PF.gamma_Q_gumbel(
    alpha_E, B_t, mu_tref_imposed, Vtref_imposed,qk_imposed, delta).value

# Heilt klart noko feil med håndteringa av mu_variabel #!!!
#Men tenker eg må få klarhet i definisjon av karakteristiske laster før eg ser på dette.
# %% Calculating gamma_M for example: All alphas the same
#Calculating gamma_M for example: All alphas the same
gamma_M_c_bending = PF.gamma_M_lognorm(
    alpha_R, B_t, Beam1['fc']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value

gamma_M_s_bending = PF.gamma_M_lognorm(
    alpha_R, B_t, Beam1['fs']['moments'][2], theta_R_bending['moments'][2], theta_R_bending['moments'][0]).value

gamma_M_s_shear = PF.gamma_M_lognorm(
    alpha_R, B_t, Beam1['fs']['moments'][2], theta_R_shear['moments'][2], theta_R_shear['moments'][0]).value



# %% Printing gammas
#Printing gammas
gamma_load = np.round([gamma_Gs, gamma_Gp, gamma_snow,
                      gamma_imposed, gamma_M_c_bending,gamma_M_s_bending, gamma_M_s_shear], 2)
gamma_load_names = [u'\u03B3_Gs', u'\u03B3_Gp', u'\u03B3_snow',
                    u'\u03B3_imposed', u'\u03B3_M_c_bending', u'\u03B3_M_s_bending',u'\u03B3_M_shear']

#print(pd.DataFrame(gamma_load, gamma_load_names))
#print()

#%% Calculating gamma_M for a beam:
#Calculating gamma_M for a beam:
   
PF.calc_gamma_M(Beam2,theta_R_bending,theta_R_shear,alpha_R,B_t,Print_results=False) #Test for beam2

#%% Calculating gamma_load for the different loads:
#Calculating gamma_load for the different loads:

PF.calc_gamma_load(G_s,theta_E,alpha_E,B_t,Print_results=False)
PF.calc_gamma_load(G_p,theta_E,alpha_E,B_t,Print_results=False)
PF.calc_gamma_load(Q_snow,theta_E,alpha_E,B_t,Print_results=False)
#print()

#%% Creating beam for FORM-analysis. Including distributed rho. And distributed load.
#Creating beam for FORM-analysis. Including distributed rho. And distributed load.
#print('FORM-example Beam3:')
Beam3 = Create_beam_dict(600, 300, 300, 550, 0.015, 0.002, fc, fs, 'indoor')

Beam3['rho_l']=Create_dist_variable(Beam3['rho_l'], 0.02, 'Normal')


#Creating loads with realistic mean values... 
G_s2 = Create_dist_variable(10, 0.05, 'Normal')
G_p2=Create_dist_variable(10, 0.1, 'Normal')
Q_imposed2 = Create_dist_variable(10, vQ, 'Gumbel')

Beam3['Loading']=[G_s2,G_p2,Q_imposed2]
Beam3['L']=8000 #mm 

#FORM-V
#cot må defineres før fordelt rho_w pga. funksjon
Beam3['cot_t']=cot_t(Beam3)
#print()
Beam3['rho_w']=Create_dist_variable(Beam3['rho_w'], 0.02, 'Normal')


#%% Importing FORM analysis modules
#Importing FORM analysis modules
from FORM_M_beam_module import FORM_M_beam #FORM_M_beam(Beam3,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
from FORM_V_beam_module import FORM_V_beam
#%% Bending FORM analysis of Beam3
#Bending FORM analysis of Beam3
alpha_M,beta_M=FORM_M_beam(Beam3,theta_R_bending,theta_E,Print_results=False, B_plots=False,MonteCarlo=False)
#print(alpha_M)
#print()
#%% Shear FORM analysis of Beam3
#Shear FORM analysis of Beam3
alpha_V,beta_V=FORM_V_beam(Beam3,theta_R_shear,theta_E,Print_results=False,B_plots=False)
#print(u'\u03B1_V=',alpha_V)
#print(u'\u03B2_V=',beta_V)
#print()


#%% Updating all PSF for Beam2 by use of the function in Partial_safety_factors.py
#Updating all PSF for Beam2 by use of the function in Partial_safety_factors.py
Beam2['Loading']=[G_s,G_p,Q_imposed]
Beam2['L']=8000 #mm 

PF.Update_psf_beam(Beam2,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False)
#print()
#%% Semi-probabilistic control of simply supported beam with uniform distributed load and updated PSF.
#Semi-probabilistic control of simply supported beam with uniform distributed load and updated PSF.
PF.Controll_psf_beam(Beam2,alpha_cc,Bending=True,Shear=True,Print_results=False)
#print()

#%% PSF update and controll for Beam 1
#PSF update and controll for Beam 1
#print('Beam1 and loads1:')
Beam1['Loading']=[G_s,G_p,Q_snow]
Beam1['cot_t']=cot_t(Beam1)
Beam1['L']=10000 #mm 

PF.Update_psf_beam(Beam1,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False)
PF.Controll_psf_beam(Beam1,alpha_cc,Bending=True,Shear=True,Print_results=False)
print()

#%% Runs the PSF for beam 3 to show that it works for distributed rho 
#Runs the PSF for beam 3 to show that it works for distributed rho 
#print('Beam3 and loads2:')
G_s2['LoadType']='SelfWeight'
G_p2['LoadType']='Permanent'
Q_imposed2['LoadType']='Imposed'

add_qk(G_s2, 0.5)
add_qk(G_p2, 0.5)
add_qk(Q_imposed2, 1.35)
#add_qk(Q_snow2, 0.82)

PF.Update_psf_beam(Beam3,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False)
PF.Controll_psf_beam(Beam3,alpha_cc,Bending=True,Shear=True,Print_results=False)

#%% Printing dictionary as pd.DataFrame
#Printing dictionary as pd.DataFrame
def print_dict_as_df(Dict):
    print(pd.DataFrame(Dict.values(),Dict.keys()))
    
#print_dict_as_df(Beam2)

#%% Creating Beam with lower shear capacity than moment
#print('Beam4:')
Beam4 = Create_beam_dict(600, 300, 300, 550, 0.02, 0.002, fc, fs, 'indoor')
fs = Create_dist_variable(246, 0.045, 'Lognormal')  # Creating example steel
add_fk(fs)

Beam4['rho_l']=Create_dist_variable(Beam4['rho_l'], 0.02, 'Normal')
Beam4['Loading']=[G_s2,G_p2,Q_imposed2]
Beam4['L']=8000 #mm 
Beam4['cot_t']=cot_t(Beam4)
Beam4['rho_w']=Create_dist_variable(Beam4['rho_w'], 0.02, 'Normal')


PF.Update_psf_beam(Beam4,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,B_t,delta=1.05,Print_results=False)
PF.Controll_psf_beam(Beam4,alpha_cc,Bending=True,Shear=True,Print_results=False)

Beam4['fs']['moments'][0]=Beam4['fs']['moments'][0]*Beam4['gamma_M_s'][1]
FORM_V_beam(Beam4,theta_R_shear,theta_E,Print_results=False)
#FORM_M_beam(Beam4,theta_R_bending,theta_E,Print_results=False)

#%% Controlling Beam4 with separate PSF approach 
#Controlling Beam4 with separate PSF approach 

PF.Update_psf_beam_2nd(Beam4,theta_R_bending,theta_R_shear,theta_E,alpha_E,alpha_R,alpha_E2,alpha_R2,B_t,delta=1.05,Print_results=False)
PF.Controll_psf_beam(Beam4,alpha_cc,Bending=True,Shear=True,Print_results=False)

#%% Caluculating chi from loading:
    #Caluculating chi from loading:
def calc_chi(Beam,Print_results=False):
    G=Beam['Loading'][0]['qk']+Beam['Loading'][1]['qk']
    Q=Beam['Loading'][2]['qk']
    
    Chi=Q/(Q+G)
    Beam['chi']=Chi
    
    if Print_results==True:
        print(u'Load ratio \u03C7=', np.round(Chi,2))
        
calc_chi(Beam2,Print_results=False)   
calc_chi(Beam3,Print_results=False)    

#%% Calculate self-weight of beam
#Calculate self-weight of beam

def calc_Gs(Beam,rho_c,V): #rho_c[kN/m^3]
    return Create_dist_variable(Beam['b']*Beam['h']*rho_c/10**6, V,'Normal') #kN/m
#Blei i tvil om enheter her... men stemmer rho_c stemmer ut eining

calc_Gs(Beam1,25,0.05)

#%% Updating partial safety factors with alphas from FORM.
#Updating partial safety factors with alphas from FORM.

alpha_M,beta_M=FORM_M_beam(Beam3,theta_R_bending,theta_E,B_plots=False)
#print(alpha_M)

alpha_M=alpha_M*-1

PF.Update_psf_beam_RA(Beam3,theta_R_bending,theta_R_shear,theta_E,alpha_M,B_t,Print_results=False)
PF.Controll_psf_beam(Beam3,alpha_cc,Bending=True,Shear=False,Print_results=False)
   

#%% Updating partial safety factors with alphas from FORM.
#Updating partial safety factors with alphas from FORM.
alpha_M,beta_M=FORM_M_beam(Beam4,theta_R_bending,theta_E,B_plots=False)
#print(alpha_M)

alpha_M=alpha_M*-1

PF.Update_psf_beam_RA(Beam4,theta_R_bending,theta_R_shear,theta_E,alpha_M,B_t,Print_results=False)
PF.Controll_psf_beam(Beam4,alpha_cc,Bending=True,Shear=False,Print_results=False)
    



#%% Updating partial safety factors with alphas from FORM. Shear
#Updating partial safety factors with alphas from FORM. Shear


alpha_V,beta_V=FORM_V_beam(Beam3,theta_R_shear,theta_E,B_plots=False)
#print(alpha_V)

alphas=np.zeros(8)
alphas[:2]=alpha_V[:2]
alphas[-5:]=alpha_V[-5:]

alphas=alphas*-1
         
PF.Update_psf_beam_RA(Beam3,theta_R_bending,theta_R_shear,theta_E,alphas,B_t,Print_results=False)
PF.Controll_psf_beam(Beam3,alpha_cc,Bending=False,Shear=True,Print_results=False)




