#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:42:36 2024

@author: perivar
"""

#Importerer nødvendige bibliotek

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from scipy.optimize import fsolve
#from scipy.optimize import minimize


#%%importerer globale parametre og funksjoner


import Bjelker as Bj #importerer globale parametre og funksjoner

#%% Funksjon for å kalkulere fordelingspesifikke parametre

def addparams(X):
    moments = X['moments']
    if X['type'] == 'Normal':
        theta = X['moments']
    if X['type'] == 'Lognormal':
        sigma_ln = np.sqrt(np.log(moments[1]**2/moments[0]**2+1))
        mu_ln = np.log(moments[0])-0.5*sigma_ln**2
        theta = [mu_ln,sigma_ln]
    if X['type'] == 'Gumbel':
        a = np.pi/np.sqrt(6)/moments[1]
        b = moments[0] - 0.5772/a
        theta = [a,b]
    if X['type'] == 'Weibull': 
        f = lambda k:moments[1]**2/moments[0]**2-sp.special.gamma(1+2/k)/sp.special.gamma(1+1/k)**2+1
        k0 = 2              #Initial guess
        k = fsolve(f, x0=k0)       # Solve for k
        lam = moments[0]/sp.special.gamma(1+1/k) #Substitue to find lambda
        theta=[k[0],lam[0]]
    if X['type'] == 'Beta':
        bounds=X['bounds']
        mu=(moments[0]-bounds[0])/(bounds[1]-bounds[0])
        sigma=moments[1]/(bounds[1]-bounds[0])
        
        def get_ab(mean, var):
            assert(var < mean*(1-mean))
            var = (mean*(1-mean)/var - 1)
            return mean*var, (1-mean)*var
        #a = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
        #b = a * (1 / mu - 1)
        a,b=get_ab(mu, sigma**2)
        theta = [a,b]

    X['theta']=theta

#%% Lager dictionary med ulike eksponeringsklasser
Ex={}
Ex['XS1']={}
#Ex['XS2']={} #Dropper XS2 si det ikkje er forventa korrosjon hær, og Vcorr=(0) gir problem...
Ex['XS3']={}

#Ex['XD1']={}
#Ex['XD2']={} #Dropper XD i første omgang pga. manglende data...
#Ex['XD3']={}

Ex['XS1']['description']='Atmospheric'
#Ex['XS2']['description']='Submerged' #Droppe XS2 si ein ikkje forventer korrosjon?
Ex['XS3']['description']='Tida-Splash'

#Ex['XD3']['description']='svømmebasseng'
#Ex['XD3']['description']='road-splash'
#%% Definerer prior info om klorid intrengning.

#Marine env. frå DuraCrete:
#Overflate kloridinnhold:
Csurf={} #Error term?
Csurf['Submerged']=Bj.Create_dist_variable(10.348*0.5, np.sqrt(0.714**2*0.5**2+0.58**2)/(0.5*10.348), 'Normal') #0.5 er w/c tall #Bj.Create_dist_variable(mean, CoV,Type )
Csurf['Splash']=Bj.Create_dist_variable(7.758*0.5,  np.sqrt(1.36**2*0.5**2+1.059**2)/(0.5*7.758), 'Normal')
Csurf['Tidal']=Bj.Create_dist_variable(7.758*0.5,np.sqrt(1.36**2*0.5**2+1.105**2)/(7.758*0.5), 'Normal')
Csurf['Atmospheric']=Bj.Create_dist_variable(2.565*0.5,np.sqrt(0.356**2*0.5**2+0.405**2)/(2.565*0.5), 'Normal')
Csurf['frida']=Bj.Create_dist_variable(1.6,1.1/1.6, 'Lognormal') #Fordelingen brukt av Liljefors på herøysundet bru.


Csurf['dry-road']=Bj.Create_dist_variable(0.24,0.16/0.24, 'Lognormal') #Usikker om Xd3 stemmer for dette, kanskje heller xd1
Csurf['swiss-splash']=Bj.Create_dist_variable(2.2,0.16/0.24, 'Lognormal') #Angst 2019

D=Bj.Create_dist_variable(16, 0.2, 'Normal') #fib34 Kloriddiffusjonskoeffisient

ec=Bj.Create_dist_variable(1, 0.1, 'Normal') #Liljefors. Modellusikkerhet 


Ccrit=Bj.Create_dist_variable(0.6, 0.15/0.6, 'Beta') #Kritisk kloridinnhold
Ccrit['bounds']=[0.2,2]

#%% Legger til fordelingspesifikke parametre
addparams(ec)
addparams(D)
addparams(Ccrit)

#%% Sjekker at D og ec fordelinger ser fornuftige ut.

x = np.linspace(0, 30, 100)
y = sp.stats.norm(D['theta'][0], D['theta'][1]).pdf(x)

plt.plot(x,y)
plt.show()


x = np.linspace(0.5, 1.5, 100)
y = sp.stats.norm(ec['theta'][0], ec['theta'][1]).pdf(x)

plt.plot(x,y)
plt.show()



#%% Legger til fordelingspesifikke parametre
addparams(Csurf['Submerged'])
addparams(Csurf['Splash'])
addparams(Csurf['Tidal'])
addparams(Csurf['Atmospheric'])
addparams(Csurf['dry-road'])
addparams(Csurf['swiss-splash'])
addparams(Csurf['frida'])

#%% Sjekker at Ccrit fordeling ser fornuftig ut

x = np.linspace(0.2, 2, 100)
y = sp.stats.beta(Ccrit['theta'][0], Ccrit['theta'][0], loc = Ccrit['bounds'][0], scale =(Ccrit['bounds'][1]-Ccrit['bounds'][0]) ).pdf(x)

plt.plot(x,y)
plt.show()


#%% Genererer tilfeldige tall for å sjekke at lognormal fordelinger ble rett.
#C_tidal=np.random.lognormal(Csurf['Tidal']['theta'][0], Csurf['Tidal']['theta'][1], size=10000)
C_tidal=np.random.normal(Csurf['Tidal']['theta'][0], Csurf['Tidal']['theta'][1], size=10000)
#%% Sjekker ar Csurf pdf ser OK ut.
x = np.linspace(0, 8, 100)
#y = sp.stats.lognorm(Csurf['Tidal']['theta'][1],scale=np.exp(Csurf['Tidal']['theta'][0])).pdf(x)
y = sp.stats.norm(Csurf['Tidal']['theta'][0],Csurf['Tidal']['theta'][1]).pdf(x)
h = sp.stats.norm(Csurf['Splash']['theta'][0],Csurf['Splash']['theta'][1]).pdf(x)
#z = sp.stats.lognorm(Csurf['Atmospheric']['theta'][1],scale=np.exp(Csurf['Atmospheric']['theta'][0])).pdf(x)
z = sp.stats.norm(Csurf['Atmospheric']['theta'][0],Csurf['Atmospheric']['theta'][1]).pdf(x)
k = sp.stats.lognorm(Csurf['frida']['theta'][1],scale=np.exp(Csurf['frida']['theta'][0])).pdf(x)

plt.hist(C_tidal,bins=x,density=True)
plt.plot(x,y,label='XS3')
plt.plot(x,z,label='XS1')
plt.plot(x,h,label='Splash')
plt.plot(x,k,label='Frida XD3?')
plt.legend()
plt.show()


#%% Sjekker Csurf  gjennomsnitt
sp.stats.norm.mean(Csurf['Tidal']['theta'][0],Csurf['Tidal']['theta'][1])
#sp.stats.lognorm.mean(Csurf['Atmospheric']['theta'][1],scale=np.exp(Csurf['Atmospheric']['theta'][0]))

#%% Kontrollerer Ccrit pdf og at gjennomsnitt og standardavvik ser rett ut.
x=np.arange(0,3,0.01)
sp.stats.beta.mean(Ccrit['theta'][0],Ccrit['theta'][1],loc=Ccrit['bounds'][0],scale=(Ccrit['bounds'][1]-Ccrit['bounds'][0]))
sp.stats.beta.std(Ccrit['theta'][0],Ccrit['theta'][1],loc=Ccrit['bounds'][0],scale=(Ccrit['bounds'][1]-Ccrit['bounds'][0]))
plt.plot(x,sp.stats.beta.pdf(x,Ccrit['theta'][0],Ccrit['theta'][1],loc=Ccrit['bounds'][0],scale=(Ccrit['bounds'][1]-Ccrit['bounds'][0])))
plt.show()

#%% Legger til overflate kloridinnhold til eksponeringsklasse

Ex['XS1']['Csurf']=Csurf['Atmospheric']
#Ex['XS2']['Csurf']=Csurf['Submerged'] 
Ex['XS3']['Csurf']=Csurf['Tidal']
#Ex['XS3']['Csurf']=Csurf['frida']

#Ex['XD3']['Csurf']=Csurf['swiss-splash']
#Ex['XD2']['Csurf']=Csurf['swiss-splash'] #denne stemmer nok ikkje har ikkje peiling på kva denne skal vere.



#%% Beregnerer gjennomsnittlig kloridinnhold på armeringsoverflate for dei ulike klassene

for i in Ex:
    Ex[i]['Cmean']=lambda x,t: Ex[i]['Csurf']['moments'][0]*(1-sp.special.erf(x/(2*np.sqrt(D['moments'][0]*t))))*ec['moments'][0]
    
 



#%%  Plotter gjennomsnittlig kloridinnhold for gitt betongdybde for ulike eksponeringsklasser

x=50 #Dybde inn i betongen (Overdekning)

t=np.arange(1,50,1)

plt.figure(figsize=(6,4))


for i in Ex:
    plt.plot(t,Ex[i]['Cmean'](x,t),label=i)
    
plt.xlabel('t [y]')
plt.ylabel(r'$\mu_{C} [wt\%c]$')
plt.grid()
plt.legend()
plt.show()

# Ett t-x plot er kanskje meir interesant for kontroll

#%% Plotter gjennomsnittlig kloridinnhold for ulike tidspunkt, dybder og eksponeringsklasser.
ts=[25,50,75]

x=np.arange(0,80,1)

measures=np.array([[10,0.8],[30,0.5],[50,0.1],[70,0.01]])


plt.figure(figsize=(6,4))
linestyles=['solid','dashed','dotted']
colors=['r','b','g']
for k,t in enumerate(ts):
    
    
    
    for n,i in enumerate( Ex):
        if n==0:
            plt.plot(x,Ex[i]['Cmean'](x,t),linestyle=linestyles[k],color=colors[n],label=t)
        else:
            plt.plot(x,Ex[i]['Cmean'](x,t),linestyle=linestyles[k],color=colors[n])
plt.scatter(measures[:,0], measures[:,1],marker='X')
plt.xlabel('Depth into concrete [mm]')
plt.ylabel(r'$\mu_{C} [wt\%c]$')
plt.grid()
plt.legend()
plt.show()
    


#%% Definerer variabler for propagering


pit_factor=Bj.Create_dist_variable(9.28, 4.04/9.28, 'Normal') #Liljefors


#Definerer korrosjonsrate for ulike miljø. Kjelde DuraCrete
Vcorr1={}

Vcorr1['wet_rarely_dry']=Bj.Create_dist_variable(0.004, 0.006/0.004, 'Weibull') 
Vcorr1['cyclic_wet_dry']=Bj.Create_dist_variable(0.03, 0.04/0.03, 'Weibull')
Vcorr1['Airborne_sea']=Bj.Create_dist_variable(0.03, 0.04/0.03, 'Weibull')
Vcorr1['Tidal_zone']=Bj.Create_dist_variable(0.07, 0.07/0.07, 'Weibull')
#Vcorr1['Submerged']=Bj.Create_dist_variable(0.0, 0.0, 'Weibull')



#%% Legger til parametre

addparams(pit_factor)
addparams(Vcorr1['wet_rarely_dry'])
addparams(Vcorr1['cyclic_wet_dry'])
addparams(Vcorr1['Airborne_sea'])
#addparams(Vcorr1['Submerged'])
addparams(Vcorr1['Tidal_zone'])
#Får ein varsel om invalid value divide #!!!

#%% Knytter sammen eksponeringsklasser og propagering
Ex['XS1']['Vcorr']=Vcorr1['Airborne_sea']
#Ex['XS2']['Vcorr']=Vcorr1['Submerged']
Ex['XS3']['Vcorr']=Vcorr1['Tidal_zone']

#Ex['XD3']['Vcorr']=Vcorr1['cyclic_wet_dry']
#Ex['XD2']['Vcorr']=Vcorr1['wet_rarely_dry']


#%% Definere parameter for del av året med aktiv korrosjon.

wt={}
wt['Airborne_sea']=Bj.Create_dist_variable(0.5, 0.12/0.5, 'Normal') #Kjelde DuraCrete
wt['Tidal_zone']=Bj.Create_dist_variable(1, 0.25, 'Normal') #Kjelde DuraCrete

#Legger til parametre og knytter saman med eksponeringsklasser
addparams(wt['Airborne_sea'])
addparams(wt['Tidal_zone'])

Ex['XS1']['wt']=wt['Airborne_sea']
#Ex['XS2']['Vcorr']=Vcorr1['Submerged']
Ex['XS3']['wt']=wt['Tidal_zone']


#%% Plotter pdf for Vcorr for å sjekke at det ser greit ut.
x = np.linspace(0, 2, 100)
y = sp.stats.weibull_min(Ex['XS3']['Vcorr']['theta'][0],scale= Ex['XS3']['Vcorr']['theta'][1]).pdf(x)
z = sp.stats.weibull_min(Ex['XS1']['Vcorr']['theta'][0],scale= Ex['XS1']['Vcorr']['theta'][1]).pdf(x)

plt.plot(x,z)
plt.plot(x,y)

plt.show()





#%% Funksjon for å generere tilfeldige tall frå fordelingene
#We use Monte Carlo as a way to find a approximated distribution for eta

  
def genraterandom(X,n_sim):
    if X['type'] == 'Normal':
        real = np.random.normal(X['theta'][0], X['theta'][1], size=n_sim)
    if X['type'] == 'Lognormal':
        real = np.random.lognormal(X['theta'][0], X['theta'][1], size=n_sim)
    if X['type'] == 'Gumbel':
        real = np.random.gumbel(X['theta'][1], 1/X['theta'][0], size=n_sim)
    if X['type'] == 'Beta':
        real = sp.stats.beta.rvs(X['theta'][0], X['theta'][1],loc=X['bounds'][0],scale= (X['bounds'][1]-X['bounds'][0]), size=n_sim) #Fortsatt noko rart med mean...
    if X['type'] == 'Weibull':
        real = X['theta'][1]*np.random.weibull(X['theta'][0], size=n_sim)   
    X['realisations']=real

#%% Sjekker at generering gir rett mean for Ccrit/beta fordeling
np.mean(sp.stats.beta.rvs(Ccrit['theta'][0], Ccrit['theta'][1],loc=Ccrit['bounds'][0],scale= Ccrit['bounds'][1], size=10000))

#%% Funksjon som beregner sannsynlighet for initiert korrosjon for gitt t med hjelp av montecarlo sim.

def pf_MCS(EX_class,ec,D,x,ti,n_sim): #Input=(Eksponeringsklasse_dict,kloridintrenging_modelusikkerhet,Kloriddiffusjonskoeffisient,x_distanse inn i betong, kontrolltidspunkt, antall simuleringer)
      
    n_sim = int(n_sim)
  
    # Defining arrays with realisations for each randomly distributed variable
    genraterandom(EX_class['Csurf'],n_sim)
    genraterandom(ec,n_sim)
    genraterandom(D,n_sim)
    genraterandom(Ccrit,n_sim)
    
   
    #LSF for initiering:
    g=Ccrit['realisations']-EX_class['Csurf']['realisations']*(1-sp.special.erf(x/(2*np.sqrt(D['realisations']*ti))))*ec['realisations']
    
    fails = np.zeros(n_sim);
    fails[np.nonzero(g<=0)[0]] = 1;
    pf = np.sum(fails)/n_sim
    betamc = -sp.stats.norm.ppf(pf)
    return betamc, pf


#%% Beregner kumulativ fordeling og sansynlighetstetthet punktvis for initiering over t, med montecarlosim. f(ti)

def calc_p_ti(EX_class,tmax,n_sim):
    Fs=[] #Kumulativ fordeling
    pfs=[] #sannsynlighetstetthet
   
    
    betamc,pf=pf_MCS(EX_class,ec,D,50,1,n_sim) #x=50
    Fs+=[pf]
    pfs+=[pf]
    
    
    for t in range(2,tmax):
        betamc,pf=pf_MCS(EX_class,ec,D,50,t,n_sim) #Beregner sannsynlighet for initiert korrosjon ved kvart år frå t=0 til t=tmax.
        
        pfs+=[pf-Fs[-1]] 
        Fs+=[pf]
        
    return np.array(pfs), np.array(Fs)
    
#%% Kjører MCS initiering for XS3 (Celle med litt lang kjøretid) #!!!
pfs_XS3,Fs_XS3=calc_p_ti(Ex['XS3'],100,1e6)

#%% Kjører MCS initiering for XS1(Celle med litt lang kjøretid) #!!!
pfs_XS1,Fs_XS1=calc_p_ti(Ex['XS1'],100,1e6)

#%% Plotter kumulativ sannsynsfunksjon og sannsynlighetstetthet for initiering over t.
ts=np.arange(1,100,1)

#pdf:
plt.scatter(ts,pfs_XS1,marker='x',label='XS1')
plt.scatter(ts,pfs_XS3,marker='x',label='XS3')
plt.ylabel(r'$f_{t_i}$')
plt.xlabel(r'$t_i$')
plt.legend()
plt.grid()
plt.show()


#Kumulativ fordeling:
plt.plot(ts,Fs_XS1,label='XS1')
plt.plot(ts,Fs_XS3,label='XS3')
plt.ylabel(r'$F_{t_i}$')
plt.xlabel(r'$t_i$')
plt.legend()
plt.show()

#%%
'''
ts2,lt2,scalet2=sp.stats.lognorm.fit(pfs_XS3,loc=0,method='MM') 


#%%

plt.scatter(ts,pfs_XS1,marker='x',label='XS1')
plt.scatter(ts,pfs_XS3,marker='x',label='XS3')
plt.plot(ts,sp.stats.lognorm.pdf(ts,ts2,loc=lt2,scale=scalet2))
plt.ylabel(r'$f_{t_i}$')
plt.xlabel(r'$t_i$')
plt.legend()
plt.grid()
plt.show()
'''
#%% #Sjekker at integral under fordelingstetthet = 1 
print(np.sum(calc_p_ti(Ex['XS3'],200,1e4)[0])) #Stemmer bra for XS3

print(np.sum(calc_p_ti(Ex['XS1'],1000,1e4)[0])) #Ser ikkje ut som dette blir lik 1 selv for svært høg tmax. 
#Men dette er knytta til at initiering aldri skjer for mange bjelker.

#%%
#Har videre berre knytta sannsynlighet for initiering for kvart enkelt med propagering. 
#Trur det er i dei neste cellene det variansen i ti ikkje blir ivaretatt. 
#Skulle tru at det burde være varians i også eta før initiering.

#%% Funksjon for å finne fordelingsparametere til lognorm fordeling frå gitt pdf.

def fit_log_to_pdf(pdf,y,x0,bounds):
    func=lambda x: np.sum((pdf-sp.stats.lognorm.pdf(y,x[0],loc=0,scale=x[1]))**2)
    #x=minimize(func, x0= x0,bounds=[(0,np.inf),(0,np.inf)],method='SLSQP')['x'] #bounds=[(0,1)]
    x=sp.optimize.differential_evolution(func, x0= x0,bounds=bounds)['x'] #bounds=[(0,1)]

    return x


#%% Tilpasser parametre for initieringsfordeling og printer desse.
x_t_XS3=fit_log_to_pdf(pfs_XS3,ts,[1,30],[(0,1),(0,200)])
print(x_t_XS3)

x_t_XS1=fit_log_to_pdf(pfs_XS1,ts,[1,30],[(0,1),(0,200)])
print(x_t_XS1)

#%% Sjekker mean og std for initiering.
print(sp.stats.lognorm.mean(x_t_XS3[0],loc=0,scale=x_t_XS3[1]))
print(sp.stats.lognorm.std(x_t_XS3[0],loc=0,scale=x_t_XS3[1]))

#%% Plotter initiering, scatter for kvart år og pdf for fordeling.

#pdf:
plt.scatter(ts,pfs_XS1,marker='x',label='XS1')
plt.plot(ts,sp.stats.lognorm.pdf(ts,x_t_XS1[0],loc=0,scale=x_t_XS1[1]),label='XS1')
plt.plot(ts,sp.stats.lognorm.pdf(ts,x_t_XS3[0],loc=0,scale=x_t_XS3[1]),label='XS3')
plt.scatter(ts,pfs_XS3,marker='x',label='XS3')
plt.ylabel(r'$f_{t_i}$')
plt.xlabel(r'$t_i$')
plt.legend()
plt.grid()
plt.show()

#%% Bergener realisations av eta* for gitt ti.
'''
def eta_ti(EX_class,ti,d0,t,n_sim):
     
     Pcorr_av_t=EX_class['Vcorr']['realisations']*(t-ti)*EX_class['wt']['realisations']
     
     Aloss=(pit_factor['realisations']*Pcorr_av_t)**2/4 #Sircular pit 
     
     A0=np.pi*d0**2/4
     
     
     eta=Aloss/A0
     return eta


#%% Plotter histogram over eta*(50) for initiering etter 20 år

genraterandom(Ex['XS3']['Vcorr'],10000)
genraterandom(Ex['XS3']['wt'],10000)
genraterandom(pit_factor,10000)
eta=eta_ti(Ex['XS3'],20,20,50,1e5)

plt.figure()
plt.hist(eta,density=True,bins=np.arange(0,1,0.01))
#plt.plot(x,sp.stats.lognorm.pdf(x,s,loc=loc1,scale=scale1))
#plt.plot(x,sp.stats.lognorm.pdf(x,2.99,loc=0,scale=0.03))
plt.show()

'''


#%% Beregner lognormal fordeling av eta* og returnerer pdf for denne for gitt ti f(eta,t|ti)
def fordeling_eta_ti(r,EX_class,ti,d0,t):
     
     Pcorr_av_t=EX_class['Vcorr']['realisations']*(t-ti)*EX_class['wt']['realisations']
     
     Aloss=(pit_factor['realisations']*Pcorr_av_t)**2/4 #Sircular pit 
     
     A0=np.pi*d0**2/4
     
     eta= Aloss/A0
 
     s,loc1,scale1=sp.stats.lognorm.fit(eta,floc=0,method='MLE')
     
     #if sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s,loc=loc1,scale=scale1), x=r) == 0:
      #  pdf=sp.stats.lognorm.pdf(r,s,loc=loc1,scale=scale1)
     #else:
     pdf=sp.stats.lognorm.pdf(r,s,loc=loc1,scale=scale1)/sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s,loc=loc1,scale=scale1), x=r)
     
     return pdf
 


#%% Simulerer propagering og beregner summen (numerisk integral) for å finne f(eta,t) for ein bestemt t 
def feta_t(EX_class,pfs,d0,t,r,n_sim):
    n_sim = int(n_sim)
   
    # Defining arrays with realisations for each randomly distributed variable
    
    genraterandom(EX_class['Vcorr'],n_sim)
    genraterandom(EX_class['wt'],n_sim)
    genraterandom(pit_factor,n_sim)

    
    feta=np.zeros_like(r)
    for i in range(1,t):
        feta=feta+fordeling_eta_ti(r,EX_class,i,d0,t)*pfs[i-1]
    
    if sp.integrate.trapezoid(feta, x=r)>=0.000001: #Klarer ikkje normalisere der sjansen for initiering er tilnærma lik 0. (Høgare n_sim for ti vil kanskje løyse dette?)
        feta=feta/sp.integrate.trapezoid(feta, x=r) #Normaliserer slik at integral under fordelingstetthet = 0 Neglisjerer areal under hale (eta>rmax).
    
    return feta #Dei første åra før initiering er vel kanskje ikkje så viktig uansett.

#%% Beregner f(eta,50) for ulike eksponeringsklasser
r=np.arange(0,1,0.001) #Definerer r som parameter eta* er fordelt over
#r=np.logspace(-4,0,100) #Burde prøve å omskalere r

eta50_XS1=feta_t(Ex['XS1'],pfs_XS1,20,50,r,1e6)
eta50_XS3=feta_t(Ex['XS3'],pfs_XS3,20,50,r,1e6)

#%% Finner f(eta,50) basert på fordeling for initiering
eta50_XS1_f=feta_t(Ex['XS1'],sp.stats.lognorm.pdf(ts,x_t_XS1[0],loc=0,scale=x_t_XS1[1]),20,50,r,1e6)
eta50_XS3_f=feta_t(Ex['XS3'],sp.stats.lognorm.pdf(ts,x_t_XS3[0],loc=0,scale=x_t_XS3[1]),20,50,r,1e6)
#%% Finner og printer lognorm fordelingsparametre for f(eta,50)

x_eta50_XS1=fit_log_to_pdf(eta50_XS1_f, r, [1,0.1], [(0,10),(0,1)])
print(x_eta50_XS1)

x_eta50_XS3=fit_log_to_pdf(eta50_XS3_f, r, [1,0.1], [(0,10),(0,1)])

print(x_eta50_XS3)

#I praksis gir dette samme funksjon for XS1 of XS3...
#%%
sp.integrate.trapezoid(eta50_XS3, x=r) #Sjekker at integral under f(eta,50) er lik 1. 


#%% plotter f(eta,50)
#plt.title('Sannsynlighetstetthet for eta ved t=50')
#plt.plot(r[:500],eta50_XS1[:500],label='XS1') #Plotter bare til eta*=0.4 for å sjå for
#plt.plot(r,eta50_XS1_f,label='XS1-f')  #samanlikner med utrekna fordeling - i praksis like
plt.plot(r[:100],sp.stats.lognorm.pdf(r[:100],x_eta50_XS1[0],loc=0,scale=x_eta50_XS1[1]),label='XS1')
#plt.plot(r[:500],eta50_XS3[:500],label='XS3')
#plt.plot(r,eta50_XS3_f,label='XS3-f')  #Større forskjell på XS3
plt.plot(r[:100],sp.stats.lognorm.pdf(r[:100],x_eta50_XS3[0],loc=0,scale=x_eta50_XS3[1]),label='XS3')
plt.ylabel(r'$f_{\eta}$')
plt.xlabel(r'$\eta^*$')
plt.legend()
#plt.hist(eta50,density=True,bins=np.arange(0,2,0.01))
plt.show()
#%% funksjon som beregner f(eta,t) for alle t
def calc_eta_over_t(EX_class,pfs,ts,r,n_sim):
    eta_over_t=[]
    #param_over_t=[]
    
    for t in ts:
        etat=feta_t(EX_class,pfs,20,t,r,n_sim)
        eta_over_t+=[etat]
        #param_over_t+=param
        
    return np.array(eta_over_t)


#%% Beregner f(eta,t) for XS1 (Celle med lang kjøretid) #!!!
ts=np.arange(1,100,1)
eta_over_t_XS1=calc_eta_over_t(Ex['XS1'],sp.stats.lognorm.pdf(ts,x_t_XS1[0],loc=0,scale=x_t_XS1[1]),ts,r,1e6)

#%% Beregner f(eta,t) for XS3 (Celle med lang kjøretid) #!!!
ts=np.arange(1,100,1)
eta_over_t_XS3=calc_eta_over_t(Ex['XS3'],sp.stats.lognorm.pdf(ts,x_t_XS3[0],loc=0,scale=x_t_XS3[1]),ts,r,1e6)
#%% Ser at dei 40 første fordelingene f(eta,t<40) ser ok ud

plt.figure()
for eta in eta_over_t_XS3[15:80]:
    plt.plot(r[:300],eta[:300])
    plt.ylabel(r'$f_{\eta^*}$')
    plt.xlabel(r'$\eta*$')
plt.show()
#Ved initiering ser det OK ut, for dei første kor det ikkje er initiering er det verre.


#%% 3D-plot av fordelinger av eta over tid.
#ts=np.arange(1,100,1

AA, SS = np.meshgrid(r[:300],ts[15:])
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(SS,AA,eta_over_t_XS3[15:][:,:300],cmap=plt.cm.coolwarm)
            
            # Enhance the plot
            
ax1.set_ylim(r[0],r[300])
ax1.set_xlim(ts[15],ts[-1])
            
            
ax1.set_xlabel(r't',fontsize=15)
ax1.set_ylabel(r'1-$\eta$',fontsize=15)
ax1.set_title(r'$f_{eta}$',fontsize=20)
            
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.zaxis.set_tick_params(labelsize=12)
            
plt.figure()
plt.show()

#%%
AA, SS = np.meshgrid(r[:300],ts[27:])
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(SS,AA,eta_over_t_XS1[27:][:,:300],cmap=plt.cm.coolwarm)
            
            # Enhance the plot
            
ax1.set_ylim(r[0],r[300])
ax1.set_xlim(ts[27],ts[-1])
            
            
ax1.set_xlabel(r't',fontsize=15)
ax1.set_ylabel(r'1-$\eta$',fontsize=15)
ax1.set_title(r'$f_{eta}$',fontsize=20)
            
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.zaxis.set_tick_params(labelsize=12)
            
plt.figure()
plt.show()
        

#%% funksjon som finner fordelingsparametre, mean og std over tid. 
'''
#Her er det eit problem. #!!!
def find_params_eta(eta_over_t):
    params=[]
    logmean_over_t=[]
    ppf05_over_t=[]
    npmean_over_t=[]
    std_over_t=[]
    npstd_over_t=[]
    for eta in eta_over_t:
        s2,loc2,scale2=sp.stats.lognorm.fit(eta,floc=0,method='MM') #Tilpasser fordeling til kvar eta. Desse fordelingene ser ikkje ut til å stemme heilt.
        
        
        #Kontrollerer skalering av ppf:
        #SCp=sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s2,loc=loc2,scale=scale2),x=r)
        #SC=sp.integrate.trapezoid(eta,x=r)
        #print(SC,SCp)
        #print(sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s2,loc=loc2,scale=scale2)/SCp,x=r))
        
        
        #Plotter pdf for fordelingene for kontroll:
        plt.plot(r,sp.stats.lognorm.pdf(r,s2,loc=loc2,scale=scale2)) # Ser greit ut (utenom tidligere nevnt problem med eta=0 ved pf_initering=0.)
        #plt.plot(r,sp.stats.lognorm.cdf(r,s2,loc=loc2,scale=scale2)) #Sjekker kumulative fordelinger
        
        
        #npmean_over_t+=[np.mean(eta)] #Denne gir gjennomsnittlig sannsynlighetstetthet ikkje gjennomsnittlig eta
        
        #Første alternativ finne mean og eta uten å tilpasse ny fordeling:
        #npstd_over_t+=[np.std(eta)] Trur denne gir std i f(eta).
        
        #Finner mean value for eta med weighted average av f(eta,t):
        if np.sum(eta)<0.0001:
            npmean_over_t+=[0]
        else:
            npmean_over_t+=[np.average(r,weights=eta)] #Dette blir ikke heilt rett siden ein ikkje tar med r>1. 
            
        npstd_over_t+=[np.sqrt(eta@(r-npmean_over_t[-1])**2)] #Prøver å berekne standardavvik i eta manuelt...
        
        
        #Andre alternativ er å bruke fordelingen tilpassa:
        ppf05_over_t+=[sp.stats.lognorm.ppf(0.5, s2,loc=loc2,scale=scale2)] #Denne gir fornuftig resultat for mean value
        logmean_over_t+=[sp.stats.lognorm.mean(s2,loc=loc2,scale=scale2)]  #Denne gir heilt feil resultat..., virker som den gir gjennomsnittlig sannsynlighetstetthet. Men det er jo desse som gir fornuftige plott av sansylighetstetthet.
        std_over_t+=[sp.stats.lognorm.std(s2,loc=loc2,scale=scale2)] #Gir samme resultat som np.std(), standardavvik for sannsynlighetstetthet 
  
        params+=[[s2,loc2,scale2]] #Lagrer parametrene for fordelingen tilpassa f(eta)
    return np.array(logmean_over_t),np.array(std_over_t),params, np.array(npmean_over_t),np.array(ppf05_over_t),np.array(npstd_over_t)
'''
#%% Finner fordelingsparametre for dei ulike f(eta,t)
def find_params_eta(eta_over_t):
    params=[]
    logmean_over_t=[]
    std_over_t=[]
    #means=[]
    #stds=[]
    
    for eta in eta_over_t:
        x=fit_log_to_pdf(eta, r, [1,0.1], [(0,3),(0,10)]) #Tilpasser fordeling til kvar eta. Desse fordelingene ser ikkje ut til å stemme heilt.
        
        
        #Kontrollerer skalering av ppf:
        #SCp=sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s2,loc=loc2,scale=scale2),x=r)
        #SC=sp.integrate.trapezoid(eta,x=r)
        #print(SC,SCp)
        #print(sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,s2,loc=loc2,scale=scale2)/SCp,x=r))
        
        
        
        logmean_over_t+=[sp.stats.lognorm.mean(x[0],loc=0,scale=x[1])]  
        std_over_t+=[sp.stats.lognorm.std(x[0],loc=0,scale=x[1])]
        
        print(x,[sp.stats.lognorm.mean(x[0],loc=0,scale=x[1]),sp.stats.lognorm.std(x[0],loc=0,scale=x[1])])
        
        
        #means+=[sp.integrate.trapezoid(r*sp.stats.lognorm.pdf(r,x[0],loc=0,scale=x[1]),x=r)]
        #stds+=[np.sqrt(sp.integrate.trapezoid((r-means[-1])**2*sp.stats.lognorm.pdf(r,x[0],loc=0,scale=x[1]),x=r))]
        #print(means[-1],stds[-1])
        plt.plot(r,sp.stats.lognorm.pdf(r,x[0],loc=0,scale=x[1])) 
  
        params+=[[x[0],0,x[1]]] #Lagrer parametrene for fordelingen tilpassa f(eta)
    return np.array(logmean_over_t),np.array(std_over_t),np.array(params)#,np.array(means),np.array(stds)
#%%
'''
def find_m_std(eta_dict):
    means=[]
    stds=[]
    for i in range(len(eta_dict['mean'])):
        eta_i=Bj.Create_dist_variable(eta_dict['mean'][i], eta_dict['std'][i], 'Lognormal')
        addparams(eta_i)
        genraterandom(eta_i,1000000)
        means+=[np.mean(eta_i['realisations'])]
        stds+=[np.std(eta_i['realisations'])]
    return means , stds
'''
#%% Finner og lagrer parametre for eta over tid for XS1
m,s,p=find_params_eta(eta_over_t_XS1)
Ex['XS1']['eta']={}
Ex['XS1']['eta']['mean']=m
#Ex['XS1']['eta']['n-mean']=nm
Ex['XS1']['eta']['std']=s

Ex['XS1']['eta']['params']=p
#%% Finner og lagrer parametre for eta over tid for XS3
m,s,p=find_params_eta(eta_over_t_XS3)
Ex['XS3']['eta']={}
Ex['XS3']['eta']['mean']=m

Ex['XS3']['eta']['std']=s

Ex['XS3']['eta']['params']=p

#%%
'''
n=np.array(find_m_std(Ex['XS3']['eta']))
Ex['XS3']['eta']['nmean']=n[0]
Ex['XS3']['eta']['nstd']=n[1]
'''

#%% Finner og lagrer fordelingsparametre for eta for XS1
'''
m,s,p,nm,ppf05,ns=find_params_eta(eta_over_t_XS1)

Ex['XS1']['eta']={}
Ex['XS1']['eta']['nmean']=nm
Ex['XS1']['eta']['mean']=m
Ex['XS1']['eta']['ppf05']=ppf05
Ex['XS1']['eta']['std']=s
Ex['XS1']['eta']['nstd']=ns
Ex['XS1']['eta']['params']=p

#%% Finner og lagrer fordelingsparametre for eta for XS3
m,s,p,nm,ppf05,ns=find_params_eta(eta_over_t_XS3)

Ex['XS3']['eta']={}
Ex['XS3']['eta']['nmean']=nm
Ex['XS3']['eta']['mean']=m
Ex['XS3']['eta']['ppf05']=ppf05
Ex['XS3']['eta']['std']=s
Ex['XS3']['eta']['nstd']=ns
Ex['XS3']['eta']['params']=p

#Fordelingene for høg t ser alt for flate ut...

'''
#%% Sjekker integral under eta fordeling, 
sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,Ex['XS3']['eta']['params'][80][0],loc=Ex['XS3']['eta']['params'][80][1],scale=Ex['XS3']['eta']['params'][80][2]), x=r)
#Denne er ikkje lik 1 og må skaleres.
#%% Kontroll av fordelingene:
#r=np.arange(0,1,0.001)
#x=np.arange(0,1000,1)
'''
x = np.linspace(sp.stats.lognorm.ppf(0.001, Ex['XS3']['eta']['params'][98][0],loc=Ex['XS3']['eta']['params'][98][1],scale=Ex['XS3']['eta']['params'][98][2]),sp.stats.lognorm.ppf(0.95, Ex['XS3']['eta']['params'][98][0],loc=Ex['XS3']['eta']['params'][98][1],scale=Ex['XS3']['eta']['params'][98][2]), 100)
#plt.plot(x,sp.stats.lognorm.pdf(x,Ex['XS3']['eta']['params'][20][0],loc=Ex['XS3']['eta']['params'][20][1],scale=Ex['XS3']['eta']['params'][20][2]))
#plt.plot(x,sp.stats.lognorm.pdf(x,Ex['XS3']['eta']['params'][40][0],loc=Ex['XS3']['eta']['params'][40][1],scale=Ex['XS3']['eta']['params'][40][2]))
plt.plot(x,sp.stats.lognorm.pdf(x,Ex['XS3']['eta']['params'][98][0],loc=Ex['XS3']['eta']['params'][98][1],scale=Ex['XS3']['eta']['params'][98][2]))
sp.stats.lognorm.mean(Ex['XS3']['eta']['params'][98][0],loc=Ex['XS3']['eta']['params'][98][1],scale=Ex['XS3']['eta']['params'][98][2])

#Dette ser ikkje rett ut
'''

#%% Plotter std for eta over tid.
#ts=np.arange(1,100,1)


plt.figure()
plt.plot(ts[27:],Ex['XS1']['eta']['std'][27:],label='XS1')
#plt.plot(ts[16:],Ex['XS3']['eta']['std'][16:],label='XS3')
plt.plot(ts[16:],Ex['XS3']['eta']['std'][16:],label='XS3')
#plt.plot(ts,Ex['XS1']['eta']['nstd'],linestyle='dotted',label='XS1-np')
#plt.plot(ts,Ex['XS3']['eta']['nstd'],linestyle='dotted',label='XS3-np')
plt.ylabel(r'$\sigma$')
plt.xlabel('t')
plt.xlim([0,100])
plt.grid()
plt.legend()
plt.show()

#Stor sjanse for at noko av feilen ligger i beregninga av std.
#Dei to ulike versjonene gir like std, men trur begge er feil/gir standardavvik i sannsynlighetstetthet.
#%% Plotter dei to ulike versjonene av mean for eta
plt.figure()
#plt.plot(ts,1-Ex['XS1']['eta']['nmean'],label='XS1-np')
#plt.plot(ts[15:],1-Ex['XS3']['eta']['nmean'][15:],label='XS3-np') #Kan forklare økende i forskjell, med effekten av å ikkje ta med r over 2.  
plt.plot(ts[27:],1-Ex['XS1']['eta']['mean'][27:],label='XS1') 
plt.plot(ts[15:],1-Ex['XS3']['eta']['mean'][15:],label='XS3')
#plt.plot(ts,1-Ex['XS1']['eta']['ppf05'],label='XS1-ppf05') 
#plt.plot(ts,1-Ex['XS3']['eta']['ppf05'],label='XS3-ppf05')
#plt.ylim([0.3,1.1])
#plt.scatter(ts,mean_over_t,marker='x')
plt.xlabel('t')
plt.ylabel(r'$\mu_{\eta}$')
plt.xlim([0,100])
plt.legend()
plt.grid()
plt.show()
#Får eit hopp i mu sidan eta er skalert bare for verdier med initiering.
#Må finne ut kva som blir best resultat av ppf og np.average #!!! 

#%% Plotter dei to ulike versjonene av mean for eta*
plt.figure()
#plt.plot(ts,1-Ex['XS1']['eta']['nmean'],label='XS1-np')
#plt.plot(ts[15:],Ex['XS3']['eta']['nmean'][15:],label='XS3-np') #Kan forklare økende i forskjell, med effekten av å ikkje ta med r over 2.  
#plt.plot(ts[27:],Ex['XS1']['eta']['n-mean'][27:],label='XS1') 
plt.plot(ts[27:],Ex['XS1']['eta']['mean'][27:],label='XS1') 
#plt.plot(ts[15:],Ex['XS3']['eta']['mean'][15:],label='XS3')
plt.plot(ts[15:],Ex['XS3']['eta']['mean'][15:],label='XS3')
#plt.plot(ts,1-Ex['XS1']['eta']['ppf05'],label='XS1-ppf05') 
#plt.plot(ts,1-Ex['XS3']['eta']['ppf05'],label='XS3-ppf05')
#plt.ylim([0,1.1])
#plt.scatter(ts,mean_over_t,marker='x')
plt.xlabel('t')
plt.ylabel(r'$\mu_{\eta^*}$')
plt.xlim([0,100])
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.show()



#%% Plotter CoV for eta

plt.figure()
plt.plot(ts[27:],Ex['XS1']['eta']['std'][27:]/(1-Ex['XS1']['eta']['mean'][27:]),label='XS1')
plt.plot(ts[15:],Ex['XS3']['eta']['std'][15:]/(1-Ex['XS3']['eta']['mean'][15:]),label='XS3')
#plt.plot(ts,Ex['XS1']['eta']['nstd']/(1-Ex['XS1']['eta']['nmean']),label='XS1-np')
#plt.plot(ts,Ex['XS3']['eta']['nstd']/(1-Ex['XS3']['eta']['nmean']),label='XS3-np')
#plt.scatter(ts,mean_over_t,marker='x')
plt.xlabel('t')
plt.ylabel(r'$V_{\eta}$')
plt.legend()
plt.grid()
plt.show()


#Denne gir unaturlig høge verdier.
#%% Plotter V eta*
plt.figure()
#plt.plot(ts,Ex['XS1']['eta']['nstd']/(np.round(Ex['XS1']['eta']['ppf05'],4)),label='XS1')
#plt.plot(ts,Ex['XS3']['eta']['nstd']/(np.round(Ex['XS3']['eta']['ppf05'],4)),label='XS3')
plt.plot(ts[27:],Ex['XS1']['eta']['std'][27:]/(Ex['XS1']['eta']['mean'][27:]),label='XS1')
plt.plot(ts[15:],Ex['XS3']['eta']['std'][15:]/(Ex['XS3']['eta']['mean'][15:]),label='XS3')
#plt.scatter(ts,mean_over_t,marker='x')
plt.xlabel('t')
plt.ylabel(r'$V_{\eta^*}$')
plt.xlim([0,100])
plt.legend()
plt.grid()
plt.show()

#Denne gir ekstremt høge verdier

#%% Plotter eta_over_t, skalert fordeling basert på parametre og fordeling basert på mean/std for ulike t


plt.figure()
for i in range(40,60):
    eta_i=Bj.Create_dist_variable(Ex['XS3']['eta']['mean'][i], Ex['XS3']['eta']['std'][i]/Ex['XS3']['eta']['mean'][i], 'Lognormal')
    addparams(eta_i)
    #eta_i2=Bj.Create_dist_variable(Ex['XS3']['eta']['ppf05'][i], Ex['XS3']['eta']['nstd'][i]/Ex['XS3']['eta']['ppf05'][i], 'Lognormal')
    #addparams(eta_i2)
    #eta_i3=Bj.Create_dist_variable(means[i], stds[i]/means[i], 'Lognormal')
    #addparams(eta_i3)
    
    plt.plot(r,sp.stats.lognorm.pdf(r,eta_i['theta'][1],scale=np.exp(eta_i['theta'][0])),label='m-s') #Denne stemmer bra med eta_over_t
    

    #plt.plot(r,sp.stats.lognorm.pdf(r,Ex['XS3']['eta']['params'][i][0],loc=Ex['XS3']['eta']['params'][i][1],scale=Ex['XS3']['eta']['params'][i][2]))#/sp.integrate.trapezoid(sp.stats.lognorm.pdf(r,Ex['XS3']['eta']['params'][i][0],loc=Ex['XS3']['eta']['params'][i][1],scale=Ex['XS3']['eta']['params'][i][2]),x=r),label='log-param') 
    plt.plot(r,eta_over_t_XS3[i],linestyle='dotted',label='eta_over_t')
    
    #Plotter ulike mean/ppf-er:
    #plt.axvline(Ex['XS3']['eta']['ppf05'][i],color='grey') #Ser bra ut, om enn noko høg?
    #plt.axvline(Ex['XS3']['eta']['nmean'][i],color='k') #Ser OK ut
    plt.axvline(Ex['XS3']['eta']['mean'][i]*0.1,color='g') #Ser dårlig ut, kvifor?
    #plt.axvline(sp.stats.lognorm.ppf(0.05,Ex['XS3']['eta']['params'][i][0],loc=Ex['XS3']['eta']['params'][i][1],scale=Ex['XS3']['eta']['params'][i][2]),color='yellow')
    #plt.axvline(sp.stats.lognorm.ppf(0.90,Ex['XS3']['eta']['params'][i][0],loc=Ex['XS3']['eta']['params'][i][1],scale=Ex['XS3']['eta']['params'][i][2]),color='yellow') 
    plt.ylabel(r'$f_{\eta}$')
    plt.xlabel(r'$\eta*$')
    plt.grid()
    plt.legend()
    plt.show()


#%% Sjekker om mean og std gir lik fordeling som lognormal parametre for t lik 40
eta40=Bj.Create_dist_variable(Ex['XS3']['eta']['mean'][40], Ex['XS3']['eta']['std'][40]/Ex['XS3']['eta']['mean'][40], 'Lognormal') #Definerer ein fordeling heilt uskalert mean og CoV som eta_t40
addparams(eta40)

#eta40_2=Bj.Create_dist_variable(1, Ex['XS3']['eta']['nstd'][40]/Ex['XS3']['eta']['ppf05'][40], 'Lognormal') #Definerer ein variabel heilt skalert mean=1 og CoV som eta_t40
#addparams(eta40_2)
#addparams(eta40)

genraterandom(eta40,10000)
#genraterandom(eta40_2,10000)


#plt.plot(r,eta_over_t_XS3[40],label='XS3')
#plt.plot(r,eta_over_t_XS3[40],label='XS3-40')

#plt.hist(eta40['realisations'],density=True,bins=r) #Begge desse stemmer bare fordelingsmessig med eta_over_t_XS3[40]. Men skalering vanskelig å seie noko om.
#plt.hist(eta40_2['realisations'],density=True,bins=r) 
plt.plot(r,sp.stats.lognorm.pdf(r,eta40['theta'][1],scale=np.exp(eta40['theta'][0])),label='m-s') #Denne treffer dårlig, for høg topp.
#plt.plot(r,np.exp(Ex['XS3']['eta']['ppf05'][40])*sp.stats.lognorm.pdf(r,eta40_2['theta'][1],scale=np.exp(eta40_2['theta'][0])),label='1-s') #Denne treffer betre, burde ikkje denne gi likt resultat som den forrige.
plt.plot(r,sp.stats.lognorm.pdf(r,Ex['XS3']['eta']['params'][40][0],loc=Ex['XS3']['eta']['params'][40][1],scale=Ex['XS3']['eta']['params'][40][2]),linestyle='dotted',label='log-param') #Denne ser ut til å treffe ganske dårlig, for låg topp...
plt.ylabel(r'$f_{\eta}$')
plt.xlabel(r'$\eta^*$')
plt.legend()
plt.grid()
plt.show()

#Mean/std og params gir like fordelinger... Trenger ikkje ta med fordeling inn i FORM... 

#Korleis skal det skaleres?







#%%
#Herfra begynner prossessen med å estimere design verdi basert på form og partialfaktor forenkling.
#Dette er avhengig av at estimeringa før er rett. Så dette er det ikkje nødvendig å sjå på i første omgang.


#Lagrer dei føregående resultata som ser best ut:
'''    
Ex['XS1']['eta']['mean']=Ex['XS1']['eta']['ppf05']
Ex['XS3']['eta']['mean']=Ex['XS3']['eta']['ppf05']

Ex['XS1']['eta']['std']=Ex['XS1']['eta']['nstd']
Ex['XS3']['eta']['std']=Ex['XS3']['eta']['nstd']
'''

#%% importerer eksempelbjelker

from B_chi_plot import BeamM, Loadings
from B_chi_plot import BeamV, LoadingsV
#from B_chi_plot import plot_Betas_d, plot_rho_d

#%% importerer FORM analyser med eta 
from FORM_M_beam_eta import FORM_M_beam_eta #funksjon for FORM analyse som tar in bjelke med bla. Beam['eta'] som iknluderer info om eta*, og returnerer alfa,beta
from FORM_V_beam_eta import FORM_V_beam_eta

#%% Lagrer modelusikkerhet
theta_R_bending=Bj.Create_dist_variable(1.01, 0.2, 'Lognormal')#Bj.theta_R_bending
theta_R_shear=Bj.Create_dist_variable(1.4, 0.3, 'Lognormal')#Bj.theta_R_shear
theta_E=Bj.theta_E


#%% Setting load equal to chi=0.55
#Setting load equal to chi=0.55
BeamM['Loading']=Loadings[50]
BeamV['Loading']=LoadingsV[50]

#%% Definerer rho som gir beta=4.7 ved eta=1 funnet i Reinforcement_corrosion.py
#BeamM['rho_l']=0.02
#BeamV['rho_w']=0.007

BeamM['rho_l']=Bj.Create_dist_variable(0.02, 0, 'Normal')
BeamV['rho_w']=Bj.Create_dist_variable(0.007, 0, 'Normal')

#BeamM['rho_l']=Bj.Create_dist_variable(BeamM['rho_l'],0,'Lognormal')
#FORM_M_beam(BeamM,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=False)
#%% Funksjon for å berekne alfa og beta med FORM for ulike eta. Denne gir ikkje fornuftige resultat, utenom Montecarloanalyse.
def calc_alfas_eta(Beam,eta,t,bruddform='bending',MonteCarlo=False,alpha=False):
    alfas_eta=[]
    betas=[]
    betas_msc=[]
    alphas_msc=[]
    if bruddform=='bending':
        
        for i in range (len(t)):
            Beam['eta']=Bj.Create_dist_variable(eta['mean'][i], eta['std'][i]/(eta['mean'][i]), 'Lognormal') 
            if MonteCarlo==True:
                alfas,beta,beta_msc=FORM_M_beam_eta(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=MonteCarlo)
                betas_msc+=[beta_msc]
            else:
                alfas,beta=FORM_M_beam_eta(Beam,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=MonteCarlo)
                
            alfas_eta+=[alfas[-1]]
            betas+=[beta]
            
        
        
    if bruddform=='shear':
         
        for i in range (len(t)):
            Beam['eta']=Bj.Create_dist_variable(eta['mean'][i], eta['std'][i]/(eta['mean'][i]), 'Lognormal') 
            if MonteCarlo==True:
                alfas,beta,beta_msc,alpha_msc=FORM_V_beam_eta(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=MonteCarlo,alpha=alpha)
                betas_msc+=[beta_msc]
                alphas_msc+=[alpha_msc]
            else:
                alfas,beta=FORM_V_beam_eta(Beam,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=False,B_plots=False,MonteCarlo=False)
            alfas_eta+=[alfas[-1]]
            betas+=[beta]
    
    if MonteCarlo==True:    
        return np.array(alfas_eta),np.array(betas),np.array(betas_msc),np.array(alphas_msc)
    else:
        return np.array(alfas_eta),np.array(betas)

#%% Berekner alfa og beta for bjelke M for t=40
BeamM['eta']=Bj.Create_dist_variable(Ex['XS3']['eta']['mean'][40], Ex['XS3']['eta']['std'][40]/(Ex['XS3']['eta']['mean'][40]), 'Lognormal') 
alfas,beta,beta_msc,a_msc=FORM_M_beam_eta(BeamM,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=True,alpha=True)
#På montecarlo ser det ud som FORM fungerer etter "knekk"


#%% Berekner alfa og beta for bjelke V for t=40
BeamV['eta']=Bj.Create_dist_variable(Ex['XS3']['eta']['mean'][40], Ex['XS3']['eta']['std'][40]/(Ex['XS3']['eta']['mean'][40]), 'Lognormal') 
alfas,beta,beta_msc,a_msc=FORM_V_beam_eta(BeamV,theta_R_shear,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=True,alpha=True)
#På montecarlo ser det ud som FORM fungerer etter "knekk"
#%%
a_msc
#%% Berekner alfa og beta for bjelke M for t=[1,100]. På grunn av dårlig resultat frå FORM er MSC-beta mest viktig.
t=np.arange(1,100,1)

Ex['XS3']['eta']['Malfa'],Ex['XS3']['eta']['Mbeta'],Ex['XS3']['eta']['Mbeta-msc']=calc_alfas_eta(BeamM,Ex['XS3']['eta'],t,MonteCarlo=True)
Ex['XS1']['eta']['Malfa'],Ex['XS1']['eta']['Mbeta'],Ex['XS1']['eta']['Mbeta-msc']=calc_alfas_eta(BeamM,Ex['XS1']['eta'],t,MonteCarlo=True)


#%%  Berekner alfa og beta for bjelke V for t=[1,100] På grunn av dårlig resultat frå FORM er MSC-beta mest viktig.
Ex['XS3']['eta']['Valfa'],Ex['XS3']['eta']['Vbeta'],Ex['XS3']['eta']['Vbeta-msc'],Ex['XS3']['eta']['Valfa-msc']=calc_alfas_eta(BeamV,Ex['XS3']['eta'],t,bruddform='shear',MonteCarlo=True,alpha=True)
#Ex['XS1']['eta']['Valfa'],Ex['XS1']['eta']['Vbeta'],Ex['XS1']['eta']['Vbeta-msc']=calc_alfas_eta(BeamV,Ex['XS1']['eta'],t,bruddform='shear',MonteCarlo=True)


#%% Plotter alfa for eta for ulike XS for bjelke M
''' 
plt.figure()
#for k,i in enumerate(Ex):
plt.plot(t[27:],-Ex['XS1']['eta']['Malfa'][27:],label='XS1') #Alfa blir positiv siden høgare eta* gir lågare kapasitet. #!!!
plt.plot(t[15:],-Ex['XS3']['eta']['Malfa'][15:],label='XS3') 
#plt.axhline(-0.7,color='g')
plt.xlabel('t [y]')
plt.ylabel(r'$\alpha_{\eta}$')
#plt.ylim([-0.2,0])
plt.legend()
#plt.xlim(76,80)
plt.grid()
plt.show()
#Burde ikkje alfa eta være negativ...
'''
#%% Plotter alfa for ulike XS for bjelke V

plt.figure()
#for k,i in enumerate(Ex):
plt.plot(t[27:],-Ex['XS3']['eta']['Valfa'][27:],label='XS1') #Alfa blir positiv siden høgare eta* gir lågare kapasitet. #!!!
plt.plot(t[27:],-np.array(Ex['XS3']['eta']['Valfa-msc'][27:])[:,1],label='XS1-mcs')
    
#plt.axhline(-0.7,color='g')
plt.xlabel('t [y]')
plt.ylabel(r'$\alpha_{\eta}$')
plt.ylim([-0.05,0])

plt.legend()
plt.grid()
plt.show()

#%% Funksjon som beregner Design verdi for eta, med partialfaktor approximasjon
'''
def calc_etad(mu_eta,V_eta,alfa,B_t,distr='Lognormal'): #Antar lognormalfordelt...
    if distr=='Normal':
        return mu_eta/(1-alfa*B_t*V_eta) #Antar normalfordelt
    if distr=='Lognormal':
        return mu_eta/np.exp(-alfa*B_t*V_eta)

'''

#%% Funksjon for å berekne design verdi for eta*, med partial approximasjon
def calc_etad_star(mu_eta,V_eta,alfa,B_t,distr='Lognormal'): #Antar lognormalfordelt...
    if distr=='Normal':
        return mu_eta*(1-alfa*B_t*V_eta) #Antar normalfordelt
    if distr=='Lognormal':
        return mu_eta*np.exp(-alfa*B_t*V_eta)
  
#%%
def calc_gamma_star(V_eta,alfa,B_t,distr='Lognormal'): #Antar lognormalfordelt...
    if distr=='Lognormal':
        return np.exp(-alfa*B_t*V_eta)
#%% Beregner gamma for modellusikkerhet for korrodert og ikkje korrodert modell.
class gamma_Rd_lognorm(): 
    def __init__(self,alfa_R,B,V,mean=1):
        #self.value=1/mean*np.exp(-1.645*V)/np.exp(-alfa_R*B*V) #kommer mean inn på rett plass #!!! #4.1-33
        self.value=1/(mean*np.exp(-alfa_R*B*V))
        
gamma_Rd_M_0=gamma_Rd_lognorm(-0.28,3.3,Bj.theta_R_bending['moments'][2],mean=Bj.theta_R_bending['moments'][0]).value

gamma_Rd_M_1=gamma_Rd_lognorm(-0.28,3.3,0.2,mean=1.01).value

gamma_Rd_V_0=gamma_Rd_lognorm(-0.7,3.3,Bj.theta_R_shear['moments'][2],mean=Bj.theta_R_shear['moments'][0]).value

gamma_Rd_V_1=gamma_Rd_lognorm(-0.7,3.3,0.3,mean=1.4).value


print(u'M0={},M1={},M1/0={},V0={},V1={},V1/0={}'.format(gamma_Rd_M_0,gamma_Rd_M_1,gamma_Rd_M_1/gamma_Rd_M_0,gamma_Rd_V_0,gamma_Rd_V_1,gamma_Rd_V_1/gamma_Rd_V_0))
    
#%% Plot etad over tid with values from nedenfor, for begge eksponeringsklasser
'''
t=np.arange(1,100,1)
alfa=0.7
B_t=Bj.B_t


plt.figure()
#for i in Ex:
    #plt.plot(t,calc_etad((Ex[i]['eta']['mean']),Ex[i]['eta']['std'],alfa,B_t),label=i)
plt.plot(t[27:],calc_etad(1-Ex['XS1']['eta']['mean'][27:],Ex['XS1']['eta']['std'][27:]/(1-Ex['XS1']['eta']['mean'][27:]),alfa,B_t),label='XS1') #1-eta* har vel ikkje lognormal fordeling? etad berekning antar at eta er lognormal fordelt.
plt.plot(t[15:],calc_etad(1-Ex['XS3']['eta']['mean'][15:],Ex['XS3']['eta']['std'][15:]/(1-Ex['XS3']['eta']['mean'][15:]),alfa,B_t),label='XS1')
plt.xlabel('t [y]')
plt.ylabel(r'$\eta_{d}$')
plt.ylim([0.6,1.1])
plt.legend()
plt.grid()
plt.show()

'''
#%% Plotter etad for ulike alpha, men her er det eit problem med at V blir for høg til at lognorm gir fornuftig gamma
#t=np.arange(1,100,1)
alfa=-0.7
alfa2=-0.28
B_t=Bj.B_t

plt.figure()

line=['solid','dashed']


    
plt.plot(t[27:],gamma_Rd_M_1/gamma_Rd_M_0*(1-calc_etad_star((Ex['XS1']['eta']['mean'][27:]),Ex['XS1']['eta']['std'][27:]/Ex['XS1']['eta']['mean'][27:],-0.3,B_t)),color='b',linestyle=line[0],label='XS1')
plt.plot(t[15:],gamma_Rd_M_1/gamma_Rd_M_0*(1-calc_etad_star((Ex['XS3']['eta']['mean'][15:]),Ex['XS3']['eta']['std'][15:]/Ex['XS1']['eta']['mean'][15:],-0.1,B_t)),color='b',linestyle=line[1],label='XS3') 

plt.plot(t[27:],gamma_Rd_V_1/gamma_Rd_V_0*(1-calc_etad_star((Ex['XS1']['eta']['mean'][27:]),Ex['XS1']['eta']['std'][27:]/Ex['XS1']['eta']['mean'][27:],-0.3,B_t)),color='r',linestyle=line[0])
plt.plot(t[15:],gamma_Rd_V_1/gamma_Rd_V_0*(1-calc_etad_star((Ex['XS3']['eta']['mean'][15:]),Ex['XS3']['eta']['std'][15:]/Ex['XS1']['eta']['mean'][15:],-0.1,B_t)),color='r',linestyle=line[1])  
     
plt.xlabel('t [y]')
plt.ylabel(r'$\eta_{d}$')
plt.ylim([0,1.1])
plt.legend()
plt.grid()
plt.show()

#Deler alfa på 10 for å få fornuftige svar, gir ikkje meining... #!!!
#%% plotter etad*
'''
plt.plot(t[27:57],1-calc_etad_star((Ex['XS1']['eta']['mean'][27:57]),Ex['XS1']['eta']['std'][27:57]/Ex['XS1']['eta']['mean'][27:57],-Ex['XS1']['eta']['Malfa'][27:57],B_t),color='b',linestyle=line[0],label=r'$\alpha=FORM-M$')
plt.plot(t[15:27],1-calc_etad_star((Ex['XS3']['eta']['mean'][15:27]),Ex['XS3']['eta']['std'][15:27]/Ex['XS1']['eta']['mean'][15:27],-Ex['XS3']['eta']['Malfa'][15:27],B_t),color='b',linestyle=line[1],label=r'$\alpha=FORM-M$')      
plt.xlabel('t [y]')
plt.ylabel(r'$\eta_{d}$')
#plt.ylim([0,1.1])
#plt.legend()
plt.grid()
'''
#%% Plotter eta_d over t, ved å beregne designverdi for eta* først og finne etad=1-etad_star
'''
B_t=Bj.B_t

plt.figure()

line=['solid','dashed']
plt.figure()
plt.plot(t[27:57],calc_gamma_star(Ex['XS1']['eta']['std'][27:57]/Ex['XS1']['eta']['mean'][27:57],-Ex['XS1']['eta']['Malfa'][27:57],B_t),color='b',linestyle=line[0],label=r'$\gamma=FORM-M$')
plt.plot(t[15:27],calc_gamma_star(Ex['XS3']['eta']['std'][15:27]/Ex['XS3']['eta']['mean'][15:27],-Ex['XS3']['eta']['Malfa'][15:27],B_t),color='b',linestyle=line[1],label=r'$\gamma=FORM-M$')
plt.plot(t[27:],calc_gamma_star(Ex['XS1']['eta']['std'][27:]/Ex['XS1']['eta']['mean'][27:],-0.1,B_t),color='r',linestyle=line[0],label=r'$\gamma=FORM-M$')
plt.plot(t[15:],calc_gamma_star(Ex['XS3']['eta']['std'][15:]/Ex['XS3']['eta']['mean'][15:],-0.1,B_t),color='r',linestyle=line[1],label=r'$\gamma=FORM-M$')
plt.xlabel('t [y]')
plt.ylabel(r'$\gamma_{\eta^*}$')
plt.legend()
#plt.ylim([1,1.5])
plt.grid()
plt.show()
'''

#%% Plotter beta kalkulertt med FORM for bjelke M for eta_over_t
#Ser forskjell på FORM og MSC

plt.figure()
#for k,i in enumerate(Ex):
plt.plot(t[27:],Ex['XS1']['eta']['Mbeta'][27:],label='XS1-FORM')
plt.plot(t[15:],Ex['XS3']['eta']['Mbeta'][15:],label='XS3-FORM')
plt.plot(t[27:],Ex['XS1']['eta']['Mbeta-msc'][27:],label='XS1-msc')
plt.plot(t[15:],Ex['XS3']['eta']['Mbeta-msc'][15:],label='XS3-msc')
#plt.axhline(-0.7,color='g')
plt.xlabel('t [y]')
plt.ylabel(r'$\beta$')
#plt.ylim([0,1])
plt.ylim([0.5,4.1])

plt.legend()
plt.grid()
plt.show()

#%%Plotter beta kalkulert med FORM for bjelke V for eta_over
#Ser forskjell på FORM og MSC
plt.figure()
#for k,i in enumerate(Ex):
plt.plot(t[27:],Ex['XS1']['eta']['Vbeta'][27:],label='XS1')
plt.plot(t[15:],Ex['XS3']['eta']['Vbeta'][15:],label='XS3')  
plt.plot(t[27:],Ex['XS1']['eta']['Vbeta-msc'][27:],label='XS1-msc')
plt.plot(t[15:],Ex['XS3']['eta']['Vbeta-msc'][15:],label='XS3-msc') 
#plt.axhline(-0.7,color='g')
plt.xlabel('t [y]')
plt.ylabel(r'$\beta$')
plt.ylim([0.5,4.1])
plt.legend()
plt.grid()
plt.show()


#%% Prøver å finne designverdi av eta med FORM
#Dette blir vel feil med sikkerhetsfaktorer og, kva med karakteristiske verdier
'''
def calc_etad_form(Beam,eta,t,B_t,bruddform='bending',Print_results=False):
    eta_ds=[]
    gammas=[]
    if bruddform=='bending':
        
        for i in range (len(t)):
            #,gammas=[1.15,1.075,1.225,1.325,1.15] #Har åpnet for å skalere inn andre sikkerhetsfaktorer i FORM analysen, for å få ut design verdien til eta som gir ønska sikkerhetsnivå, ved gitte sikkerhetsfaktorer.
            Beam['eta']=Bj.Create_dist_variable(eta['mean'][i], eta['std'][i]/(eta['mean'][i]), 'Lognormal') 
            equation = lambda SC_ETA: np.abs(FORM_M_beam_eta(Beam,theta_R_shear,theta_E,Print_results=False,SC_ETA=SC_ETA,gammas=[1.15,1.075,1.225,1.325,1.15])[1][0]-B_t)
            SC_ETA= minimize(equation, x0= 1,bounds=[(0,1)],method='SLSQP')['x']#Finner sikkerthetsfaktor som gir etad slik at ein får ønska beta 
            if Print_results==True:
                print(u'\u03B7_d*=',np.round(eta['mean'][i]*SC_ETA[0],4))
                print(u'\u03B3=',np.round(SC_ETA[0],4))
            
            gammas+=[SC_ETA]
            eta_ds+=[eta['mean'][i]*SC_ETA]
            
            
        
        
    if bruddform=='shear':
         
        for i in range (len(t)):
            
            Beam['eta']=Bj.Create_dist_variable(eta['mean'][i], eta['std'][i]/(eta['mean'][i]), 'Lognormal') 
            equation = lambda SC_ETA: np.abs(FORM_V_beam_eta(Beam,theta_R_shear,theta_E,Print_results=False,SC_ETA=SC_ETA,gammas=[1.075,1.225,1.325,1.15])[1][0]-B_t)
            SC_ETA= minimize(equation, x0= 1,bounds=[(0,1)],method='SLSQP')['x']#Finds etad  #Kan jo fort være at g_eta<1
            if Print_results==True:
                print(u'\u03B7_d*=',np.round(eta['mean'][i]*SC_ETA[0],4))
                print(u'\u03B3=',np.round(SC_ETA[0],4))
            
            gammas+=[SC_ETA]
            eta_ds+=[eta['mean'][i]*SC_ETA]
            
    return np.array(eta_ds),np.array(gammas)
#%% Berekner og lagrer etad_form og gamma_eta for beamM XS3
eta_ds_XS3,gamma_eta_XS3=calc_etad_form(BeamM,Ex['XS3']['eta'],t,B_t,bruddform='bending',Print_results=True)


#%% Berekner og lagrer etad*_form og gamma_eta for beamM XS3. med MC2020 gammas=[1.15,1.075,1.225,1.325,1.15]
#eta_ds_XS3_2,gamma_eta_XS3_2=calc_etad_form(BeamM,Ex['XS3']['eta'],t,B_t,bruddform='bending',gammas=[1.15,1.075,1.225,1.325,1.15],Print_results=True)
#%% Berekner og lagrer etad*_form og gamma_eta for beamM XS1.
eta_ds_XS1,gamma_eta_XS1=calc_etad_form(BeamM,Ex['XS1']['eta'],t,B_t,bruddform='bending',Print_results=True)

#%%Berekner og lagrer etad*_form og gamma_eta for beamV XS3.
eta_ds_XS3_V,gamma_eta_XS3_V=calc_etad_form(BeamV,Ex['XS3']['eta'],t,B_t,bruddform='shear',Print_results=True)
#%% Berekner og lagrer etad*_form og gamma_eta for beamV XS1.
eta_ds_XS1_V,gamma_eta_XS1_V=calc_etad_form(BeamV,Ex['XS1']['eta'],t,B_t,bruddform='shear',Print_results=True)
#%% Plotter eta_d_star_form, bjelke M.
plt.figure()
plt.plot(t,eta_ds_XS1,label='XS1')
plt.plot(t,eta_ds_XS3,label='XS3')
#plt.plot(t,1-eta_ds_XS3_2,label='XS3') Med sikkerhetsfaktorer på dei andre parametrene
#plt.plot(t,1-eta_ds_XS1_V,linestyle='dotted',label='XS1')
#plt.plot(t,1-eta_ds_XS3_V,linestyle='dotted',label='XS3')
plt.grid()
plt.xlabel('t [y]')
plt.legend()
plt.ylabel(r'$\eta_{d}$')
plt.show()

#gamma stabiliserer seg fort rundt 0, noko galt.
#%%  Plotter gamma_eta_star_form, bjelke M/V. Ser at gamma_eta er uavhengig av bjelken.
plt.figure()
plt.plot(t,gamma_eta_XS1,label='XS1')
plt.plot(t,gamma_eta_XS3,label='XS3')
plt.plot(t,gamma_eta_XS1_V,linestyle='dotted',label='XS1')
plt.plot(t,gamma_eta_XS3_V,linestyle='dotted',label='XS3')
plt.grid()
plt.xlabel('t [y]')
plt.ylabel(r'$\gamma_{\eta^*}$')
plt.legend()
plt.show()

#%% Plotter eta_d_form, bjelke V.
plt.figure()
plt.plot(t,1-eta_ds_XS1_V,label='XS1')
plt.plot(t,1-eta_ds_XS3_V,label='XS3')
plt.grid()
plt.xlabel('t [y]')
plt.legend()
plt.ylabel(r'$\eta_{d}$')
plt.show()
#%%  Plotter gamma_eta_form, bjelke V.
plt.figure()
plt.plot(t,gamma_eta_XS1_V,label='XS1')
plt.plot(t,gamma_eta_XS3_V,label='XS3')
plt.grid()
plt.xlabel('t [y]')
plt.ylabel(r'$\gamma_{\eta^*}$')
plt.legend()
plt.show()

#%% Sjekker at minimeringa fungerer, at den oppjusterte eta* gir beta=3.3, med MC2020 partialfaktorer.
BeamM['eta']=Bj.Create_dist_variable(Ex['XS3']['eta']['mean'][40]*gamma_eta_XS3[40], Ex['XS3']['eta']['std'][40]/(Ex['XS3']['eta']['mean'][40]), 'Lognormal') 
alfas,beta=FORM_M_beam_eta(BeamM,theta_R_bending,theta_E,alpha_cc=0.85,Print_results=True,B_plots=False,MonteCarlo=True,gammas=[1.15,1.075,1.225,1.325,1.15])
#Dette blir feil om gamma er berekna uten andre sikkerhetsfaktorer.
'''


#%% Legger til MC2020 assessment gammas på eksempelbjelkene.
BeamM['Loading'][0]['gamma']=1.225
BeamM['Loading'][1]['gamma']=1.325
BeamM['Loading'][2]['gamma']=1.15

BeamM['gamma_M_c']=[1.15,1.15]
BeamM['gamma_M_s']=[1.075,1.075]

BeamV['Loading'][0]['gamma']=1.225
BeamV['Loading'][1]['gamma']=1.325
BeamV['Loading'][2]['gamma']=1.15

BeamV['gamma_M_c']=[1.15,1.15]
BeamV['gamma_M_s']=[1.075,1.075]

#%% Beregner etad' (uten endra modellusikkerhet.) for begge eksponeringsklasser.
 
Ex['XS1']['etad']=1-calc_etad_star((Ex['XS1']['eta']['mean'][27:]),Ex['XS1']['eta']['std'][27:]/Ex['XS1']['eta']['mean'][27:],-0.3,B_t)
Ex['XS3']['etad']=1-calc_etad_star((Ex['XS3']['eta']['mean'][15:]),Ex['XS3']['eta']['std'][15:]/Ex['XS1']['eta']['mean'][15:],-0.1,B_t)


#%% Controll_psf_beam_etad funksjon for å kjøre PSF kontroll inkludert eta.
def Controll_psf_beam_etad(Beam,etad,alpha_cc,Bending=True,Shear=True,Print_results=True,corrosion=False):
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
            Mr=gamma_Rd_M_1/gamma_Rd_M_0*etad*rho_l*fyd_M*(1-0.5*gamma_Rd_M_1/gamma_Rd_M_0*etad*rho_l*fyd_M/(fcd_M))*b*d**2
        
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
        
        return Me/Mr
    if Shear==True:
        cot_t=Beam['cot_t']
        fyd_V=Beam['fs']['fk']/Beam['gamma_M_s'][1]
        
        if corrosion==False:
            Vr=rho_w*bw*0.9*d*fyd_V*cot_t
        if corrosion==True:
            Vr=gamma_Rd_V_1/gamma_Rd_V_0*etad*rho_w*bw*0.9*d*fyd_V*cot_t
            
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
        
        return Ve/Vr
        
        
#%% Kjører PSF kontroll for BeamM t=40
Controll_psf_beam_etad(BeamM,Ex['XS1']['etad'][40],alpha_cc=0.85,Bending=True,Shear=False,Print_results=True,corrosion=True)
#%% Kjører  PSF kontroll for BeamM over t
UTNs_M_XS1=np.array(list(map(lambda etad: Controll_psf_beam_etad(BeamM,etad,alpha_cc=0.85,Bending=True,Shear=False,Print_results=False,corrosion=True),Ex['XS1']['etad'])))
UTNs_M_XS3=np.array(list(map(lambda etad: Controll_psf_beam_etad(BeamM,etad,alpha_cc=0.85,Bending=True,Shear=False,Print_results=False,corrosion=True),Ex['XS3']['etad'])))

#%% Plotter UTN for beam M
plt.plot(ts[27:],UTNs_M_XS1)
plt.plot(ts[15:],UTNs_M_XS3)
plt.ylim(0.5,1.5)
plt.grid()
plt.show()

#%% Kjører PSF kontroll for BeamV t=40
Controll_psf_beam_etad(BeamV,Ex['XS1']['etad'][40],alpha_cc=0.85,Bending=False,Shear=True,Print_results=True,corrosion=True)
#%% Kjører  PSF kontroll for BeamV over t
UTNs_V_XS1=np.array(list(map(lambda etad: Controll_psf_beam_etad(BeamV,etad,alpha_cc=0.85,Bending=False,Shear=True,Print_results=False,corrosion=True),Ex['XS1']['etad'])))
UTNs_V_XS3=np.array(list(map(lambda etad: Controll_psf_beam_etad(BeamV,etad,alpha_cc=0.85,Bending=False,Shear=True,Print_results=False,corrosion=True),Ex['XS3']['etad'])))

#%% Plotter UTN for beam V
plt.plot(ts[27:],UTNs_V_XS1)
plt.plot(ts[15:],UTNs_V_XS3)
plt.ylim(0.5,1.5)
plt.grid()
plt.show()

#%% Funksjon for å plotte UTN og Beta_MSC i samme plott.
def plot_UB(ts,UTNs,Bs,UTN_lim):

    fig, ax1 = plt.subplots()
    
    
    ax1.set_xlabel('t')
        
    
    color = 'tab:red'
    ax1.plot(ts[:UTN_lim],UTNs[:UTN_lim],color = color)
    #ax1.plot(etas,UTNs_V,color = color)
    ax1.set_ylabel('UTN',color=color)
    
    ax1.set_ylim([0.4,1.6])
    ax1.axhline(1,color='k',linestyle='dotted')
    
    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\beta_{MCS}$',color=color)
    #ax2.axhline(3,color='r', label=r'$\beta_t$')
    ax2.set_ylim([2.5,4.1])
    ax2.plot(ts,Bs,color=color)
    
    #plt.grid()
    plt.show()
#%% Plotter UTN og Beta_msc for beam V

plot_UB(ts[27:],UTNs_V_XS1,Ex['XS1']['eta']['Vbeta-msc'][27:],68)

plot_UB(ts[15:],UTNs_V_XS3,Ex['XS3']['eta']['Vbeta-msc'][15:],14)

#%% Plotter UTN og Beta_msc for beam M
plot_UB(ts[27:],UTNs_M_XS1,Ex['XS1']['eta']['Mbeta-msc'][27:],68)

plot_UB(ts[15:],UTNs_M_XS3,Ex['XS3']['eta']['Mbeta-msc'][15:],14)
