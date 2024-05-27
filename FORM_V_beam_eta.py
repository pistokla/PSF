#!/usr/bin/env python
# coding: utf-8



#%%

#Importerer bib og definerer references
import numpy as np
import scipy as sp
#from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import scipy.stats
#import time
#import sympy as sym

import pprint
# Create a PrettyPrinter object
pp = pprint.PrettyPrinter(indent=4)


from scipy.optimize import fsolve
    

fontsizes=18
plt.rcParams.update({'font.size': fontsizes})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"mathtext.fontset" : "cm"})
plt.rcParams.update({'font.serif': 'Times New Roman'})
plt.close('all')



#%% Function for defining dictionary for distributed variable

def Create_dist_variable(mean, CoV, Type):
    stdev = mean*CoV
    material = {}  # initiate a empty dictionary
  
    material['type'] = Type  # define and specify the property type = lognormal, normal, gumble
    
    material['moments'] = [mean, stdev, CoV] # define and specify the property moments = mean, stdev, CoV
    
    return material  # Returns material dictionary

#%% Defining input

#Lar heile form-analysen vere ein funksjon for å enkelt kunne kalle på den...
#Nødvendig input: Bjelke(dict. som definert i Bjelker.py)
#last og motstandsmodelusikkerhet (definert som fordelt variabel dict.) 
#MonteCarlo=True gir kontroll mot montecarlo
#B_plots=True gir 3D-plot av beta mot nokre variabler
def FORM_V_beam_eta(Beam,theta_R_M,theta_E,alpha_cc=0.85, Print_results=False, MonteCarlo=False,B_plots=False,SCrho=1,SC_ETA=1,gammas=[1,1,1,1],alpha=False):

    #Henter ut input frå bjelken, model og last som skal analyseres.
    fy=Create_dist_variable(1,Beam['fs']['moments'][2],Beam['fs']['type'])
    scfy = Beam['fs']['moments'][0]*gammas[0]
    
    
    
    rho_w=Create_dist_variable(1,Beam['rho_w']['moments'][2],Beam['rho_w']['type'])
    scrho_w=Beam['rho_w']['moments'][0]*SCrho
    
    eta=Create_dist_variable(1,Beam['eta']['moments'][2],Beam['eta']['type'])
    sceta=Beam['eta']['moments'][0]*SC_ETA
    
    #Geometriske størrelser er definert som deteministiske, må ta stilling til korleis geometrisk usikkerhet skal behandles.
    #Per nå i modellusikkerhet?
    bw=Beam['bw'] #Deterministic mm
    d=Beam['d'] #Deterministic mm
    L=Beam['L'] #Deterministic mm

    cot_theta=Beam['cot_t']
   

    
    eR=Create_dist_variable(1,theta_R_M['moments'][2],theta_R_M['type'])#Motstands-modellusikkerhet 
    sceR=theta_R_M['moments'][0]
    
    
    eS=Create_dist_variable(1,theta_E['moments'][2],theta_R_M['type']) #Lastmodellusikkerhet 
    sceS=theta_E['moments'][0]
    
   
    Gs=Create_dist_variable(1,Beam['Loading'][0]['moments'][2],Beam['Loading'][0]['type'])  #Egenvekt
    scGs=Beam['Loading'][0]['moments'][0]/gammas[1]

    Gp=Create_dist_variable(1,Beam['Loading'][1]['moments'][2],Beam['Loading'][1]['type'])  #Permanent last
    scGp=Beam['Loading'][1]['moments'][0]/gammas[2]
    
    Q=Create_dist_variable(1,Beam['Loading'][2]['moments'][2],Beam['Loading'][2]['type'])   #Variabel last
    scQ=Beam['Loading'][2]['moments'][0]/gammas[3]
   

        
   
    #%% Now we can define a function that is calculating parameters from moments based on the equation above and adds them (as the vector $\theta$ into the dictionary):
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
        X['theta']=theta
    
    
    # %%... and we can use the function to add the parameters to the corresponding dictionaries (by replacing $X$ with $R,A,S$):
    
    addparams(fy)
    addparams(rho_w)
    addparams(eR)
    addparams(Gs)
    addparams(Gp)
    addparams(Q)
    addparams(eS)
    addparams(eta)
    
    
 
    
#%%  #The limit state as a function of the scaling factors:
    gX = lambda x,scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta: sceR*x[2]*scrho_w*x[1]*scfy*x[0]*(1-sceta*x[7])*bw*0.9*d*cot_theta-sceS*x[6]*(scGs*x[3]+scGp*x[4]+scQ*x[5])*(L/2-d)
    
    #%%Transform the limit state from x2u
    def x2u(X):
        #X is here a dictionary, which describes the type and stores its parameters
        if X['type'] == 'Normal':
            # If X is normal distributed, use Eq. (3.8) (isolated x) 
            x = lambda u: u*X['theta'][1] + X['theta'][0]
            # X['theta'][1] = sigma
            # X['theta'][0] = mu
        if X['type'] == 'Lognormal':
            # If X is lognormal distributed, use Eq. (3.26)
            x = lambda u: np.exp(X['theta'][1]*u+X['theta'][0])
            # X['theta'][1] = sigma_L
            # X['theta'][0] = mu_L
        if X['type'] == 'Gumbel':
            # If X is Gumbel distributed, use Eq. (3.27)
            x = lambda u: X['theta'][1]-1/X['theta'][0]*np.log(-np.log(sp.stats.norm.cdf(u)))
            # X['theta'][1] = b
            # X['theta'][0] = a
        return x

    
    
    
    
    #%% Define the differentiation 
    
    def dxdufun(X):
        if X['type'] == "Normal":
            dxdu = lambda u: X['theta'][1]
        if X['type'] == "Lognormal":
            dxdu = lambda u: X['theta'][1] * np.exp(X['theta'][0]+X['theta'][1] * u)
        if X['type'] == "Gumbel":
            dxdu = lambda u: - sp.stats.norm.pdf(u)/(X['theta'][0] * sp.stats.norm.cdf(u)*np.log(sp.stats.norm.cdf(u)))
        return dxdu
    
    
    #%% We create a list of the transformed function objects for each variable to make the code easier to look at.
    
    
    U = [x2u(fy),x2u(rho_w),x2u(eR),x2u(Gs),x2u(Gp),x2u(Q),x2u(eS),x2u(eta)] 
    
    #%% We can now write our LSF as a function of u:
    
    
    #LSF as a function of u, scX:
    gU = lambda u,scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta: gX([U[0](u[0]),
                                          U[1](u[1]),
                                          U[2](u[2]),U[3](u[3]),U[4](u[4]),U[5](u[5]),U[6](u[6]),U[7](u[7])],
                                                         scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta) 
    #%%Next, we define the function that gives us the next, improved $\alpha$-vector
    
    #Compute a normal vector on point $u$ on the limit state.
    # An improved $\alpha$-vector is computed based on the partial derivatives (see Equation 3.10 in the compendium).
    # Note that this is a function that is problem specific, i.e. depending on the specific limit state. Here, u[0], u[1], u[2] refer to the variables R,A,S. 
    
    
    def alpha_next(u,scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta):
        dgdu = np.zeros(8)
        
        #Resistance
        dgdu[0] = sceR*U[2](u[2])*scrho_w*U[1](u[1])*scfy*dxdufun(fy)(u[0])*(1-sceta*U[7](u[7]))*bw*0.9*d*cot_theta #fy
        dgdu[1] = sceR*U[2](u[2])*scrho_w*dxdufun(rho_w)(u[1])*scfy*U[0](u[0])*(1-sceta*U[7](u[7]))*bw*0.9*d*cot_theta #rho_l
        dgdu[2] = sceR*dxdufun(eR)(u[2])*scrho_w*U[1](u[1])*scfy*U[0](u[0])*(1-sceta*U[7](u[7]))*bw*0.9*d*cot_theta #eR
       
        #Load
        dgdu[3] = ( -scGs*dxdufun(Gs)(u[3])*sceS*U[6](u[6])*(L/2-d) )#Gs
        dgdu[4] = ( -scGp*dxdufun(Gp)(u[4])*sceS*U[6](u[6])*(L/2-d)) #Gp
        dgdu[5] = ( -scQ*dxdufun(Q)(u[5])*sceS*U[6](u[6])*(L/2-d) )#Q
        dgdu[6] = ( -sceS*dxdufun(eS)(u[6])*(scGs*U[3](u[3])+scGp*U[4](u[4])+scQ*U[5](u[5]))*(L/2-d) )#eS
       
        dgdu[7] = -sceR*U[2](u[2])*scrho_w*U[1](u[1])*scfy*U[0](u[0])*sceta*dxdufun(eta)(u[7])*bw*0.9*d*cot_theta #fy
        
        k = np.sqrt(sum(dgdu**2))                                  #Normalisation factor, makes ||alpha|| = 1
        
        alpha = -dgdu/k
        
        return alpha
    
    #%% Recall that this function does not correspond exactly to eq. (3.10) as this equation requires us to insert the estimated values of $\beta$ and $\mathrm{\alpha}$. Once we have started the iteration we use the values from previous iteration steps. Initially, however, we have to define starting values, $\mathrm{\alpha}_0$ and $\beta_0$.
    
    
    
    alpha0 = np.array([-1/np.sqrt(8),-1/np.sqrt(8),-1/np.sqrt(8),1/np.sqrt(8),1/np.sqrt(8),1/np.sqrt(8),1/np.sqrt(8),1/np.sqrt(8)])

    beta0=4
    
    
    #%%Now we need a way to solve $g_u(\beta \cdot \mathbf{\alpha}) = 0$ with respect to $\beta$. This can be done in Python by utilizing `fsolve`. This is a numerical solver that depends on a staring value for $\beta$. We use initially $\beta_0$, and later the $\beta$ from the previous iteration step.
    
    
    #%%
    
   #Løyser g(u)=0 med hensynn til beta, basert på forrige beta og gamma
    def beta_next(alpha, beta_prev,scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta):
        equation = lambda beta: gU(beta*alpha,scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta) #this defines the fsolve should work with.
        beta_new = fsolve(equation, x0=beta_prev) #Finds beta that gives the function value of equation = 0
        return beta_new
        
    
    #%% Now we are all set and we can define a function for the FORM algorithm!
    
    
    
    def FORM_tol(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta,alpha_start, beta_start, tol=1e-5, itreturn=False):
        alphas = [alpha_start]
        beta1 = beta_next(alpha_start, beta_start, scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta)
        betas = [beta_start, beta1]                          #Need at least two values of beta for the next line to work, first beta is therefore calculated manually
        while abs((betas[-1] - betas[-2])/betas[-1]) >= tol:
            alpha_new = alpha_next(betas[-1]*alphas[-1],scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta) #calculates new, improved alpha, using the last calculated values (by using [-1])
            alphas.append(alpha_new)                         #adds the new, improved alpha to the list of alphas
            beta_new = beta_next(alphas[-1],betas[-1],scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta)   #calculates new, improved beta
            betas.append(beta_new)                           #adds the new, improved beta to the list of betas
        if itreturn:
            O = {}
            O['betas'] = betas
            O['alphas'] = alphas
            
            return O     #returns all values, organised in a dictionary O
        else:
            return alphas[-1], betas[-1]                     #returns only the final values

    #%% 
    # We use Monte Carlo as a way to test the quality of our results
   #Monte Carlo simulation 
    
    def genraterandom(X,n_sim):
        if X['type'] == 'Normal':
            real = np.random.normal(X['theta'][0], X['theta'][1], size=n_sim)
        if X['type'] == 'Lognormal':
            real = np.random.lognormal(X['theta'][0], X['theta'][1], size=n_sim)
        if X['type'] == 'Gumbel':
            real = np.random.gumbel(X['theta'][1], 1/X['theta'][0], size=n_sim)
        X['realisations']=real
    
    
    def MCS(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta,n_sim,alpha=alpha):
        
        n_sim = int(n_sim)
    
        # Defining arrays with realisations for each randomly distributed variable
        genraterandom(fy,n_sim)
        genraterandom(rho_w,n_sim)
        genraterandom(eR,n_sim)
        genraterandom(Gs,n_sim)
        genraterandom(Gp,n_sim)
        genraterandom(Q,n_sim)
        genraterandom(eS,n_sim)
        genraterandom(eta,n_sim)
        
        # LSF for MCS:
        g=sceR*eR['realisations']*scrho_w*rho_w['realisations']*(1-sceta*eta['realisations'])*scfy*fy['realisations']*bw*0.9*d*cot_theta-sceS*eS['realisations']*(scGs*Gs['realisations']+scGp*Gp['realisations']+scQ*Q['realisations'])*(L/2-d)
        
        fails = np.zeros(n_sim);
        fails[np.nonzero(g<=0)[0]] = 1;
        pf = np.sum(fails)/n_sim
        betamc = -sp.stats.norm.ppf(pf)
        
        
        
        if alpha==True:
            arg=np.argwhere(np.abs(g)<=20)
            print(len(arg))
            
            a=[-np.mean(fy['realisations'][arg]),-np.mean(rho_w['realisations'][arg]),-np.mean(eR['realisations'][arg]),np.mean(Gs['realisations'][arg]),np.mean(Gp['realisations'][arg]),np.mean(Q['realisations'][arg]),np.mean(eS['realisations'][arg]),np.mean(eta['realisations'][arg])]
            
            a=a/np.linalg.norm(a)
        #np.argwhere(np.abs(g)<=0.001)
            return betamc, pf,a
        
        else:
            return betamc, pf
        
        
        
        

    
    #%% Hint to Task 1 and 2
    
    # We test our FORM algortithm by comparing the results with Monte Carlo simulation.
    
    
    
    alpha1, beta1 = FORM_tol(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta,alpha_start=alpha0, beta_start=beta0)
    if MonteCarlo == True:
        MCS_save=MCS(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta, 1e7)
        beta1_mcs = MCS_save[0]#MCS(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta, 1e6)[0]
        if alpha==True:
            a_msc=MCS_save[2]#MCS(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta, 1e6)[3]
            if Print_results ==True:
                print('alpha=',a_msc )
    
    
    
    
    if Print_results ==True:
        print(u'\u03BC(fy) = {} MPa,\u03BC(rho_w) = {},\u03BC(eta) = {}   ,\u03BC(eR) = {}'.format(scfy,scrho_w,(1-sceta),sceR))     #write mu in unicode(\u03BC). {} acts as placeholder for the variables given in .format()
        print(u'\u03BC(Gs) = {} kN/m,\u03BC(Gp) = {} kN/m,\u03BC(Q) = {} kN/m, \u03BC(eS) = {} :'.format(scGs,scGp,scQ,sceS))
        print(u'\u03B2-FORM = %.5f' % beta1[0])
        if MonteCarlo == True:
            print(u'\u03B2-MCS = %.5f' % beta1_mcs)
   
        
        

    #%%#Printer heile iterasjonsprosessen Liljefors_input
    if Print_results ==True:
        RES = FORM_tol(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,sceta,alpha_start=alpha0, beta_start=beta0, itreturn=True)
        
        
        
        # Print the dictionary using the PrettyPrinter
        pp.pprint(RES)
            
    #%%
    if B_plots==True:
        f1 = lambda scfy,scrho_w,sceR,scGs,scGp,scQ,sceS: FORM_tol(scfy,scrho_w,sceR,scGs,scGp,scQ,sceS,alpha_start=alpha0, beta_start=beta0)[1] #Only returns the beta-value
        #%%
        #fs/rho_w
        n=20                                               # Resolution of the plot
        
                       # Arrays used for plotting
        scfs_dum = np.linspace(200, 400,n)
        scrho_w_dum = np.linspace(0.004, 0.01,n) 
        
        AA, SS = np.meshgrid(scfs_dum,scrho_w_dum)
        
        beta_dum = np.zeros_like(AA)
        
        for i in range(n):
            for j in range(n):
                beta_dum[i, j] = f1(AA[i, j],SS[i, j],sceR, scGs,scGp,scQ,sceS)[0]

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(AA,SS, beta_dum,cmap=plt.cm.coolwarm)
        
        # Enhance the plot
        
        ax1.set_xlim(scfs_dum[0],scfs_dum[-1])
        ax1.set_ylim(scrho_w_dum[0],scrho_w_dum[-1])
        
        
        ax1.set_xlabel(r'$\mu_{fy}$ [MPa]',fontsize=15)
        ax1.set_ylabel(r'$\mu_{rho,w}$',fontsize=15)
        ax1.set_title(r'$\beta(\mu_{fy},\mu_{rho,w})$',fontsize=20)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax1.zaxis.set_tick_params(labelsize=12)
        
        plt.figure()
        plt.show()
        #%% Plot rho_l/last

        n=20                                               # Resolution of the plot
        
        scrho_w_dum = np.linspace(0.002, 0.007,n)                  # Arrays used for plotting  
        scV_dum = np.linspace(40, 70,n)
        
        
        AA, SS = np.meshgrid(scrho_w_dum,scV_dum)
        
        
        beta_dum = np.zeros_like(AA)
        
        for i in range(n):
            for j in range(n):
                beta_dum[i, j] = f1(scfy,AA[i, j],sceR, 0,SS[i, j],0,sceS)[0]
                
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(SS,AA, beta_dum,cmap=plt.cm.coolwarm)
        
        # Enhance the plot
        ax1.set_ylim(scrho_w_dum[0],scrho_w_dum[-1])
        ax1.set_xlim(scV_dum[0],scV_dum[-1])
        
        ax1.set_ylabel(r'$\mu_{rho,w}$',fontsize=15)
        ax1.set_xlabel(r'$Fordelt last$ [kN/m]',fontsize=15)
        ax1.set_title(r'$\beta(\mu_{rho,w},\mu_q)$',fontsize=20)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax1.zaxis.set_tick_params(labelsize=12)
        
        plt.figure()
        plt.show()

 
        #%% #Plot eR/eS

        n=20                                               # Resolution of the plot
        
                       # Arrays used for plotting
        sceR_dum = np.linspace(1, 1.7,n)   
        sceS_dum = np.linspace(0.9, 1.2,n)
        
        
        AA, SS = np.meshgrid(sceR_dum,sceS_dum)
        
        beta_dum = np.zeros_like(AA)
        
        for i in range(n):
            for j in range(n):
               
                beta_dum[i, j] = f1(scfy,scrho_w,AA[i, j], scGs,scGp,scQ,SS[i, j])[0]

        
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(SS,AA, beta_dum,cmap=plt.cm.coolwarm)
        
        # Enhance the plot
        
        ax1.set_ylim(sceR_dum[0],sceR_dum[-1])
        ax1.set_xlim(sceS_dum[0],sceS_dum[-1])
        
        
        ax1.set_ylabel(r'$\mu_{eR}$',fontsize=15)
        ax1.set_xlabel(r'$\mu_{eS}$',fontsize=15)
        ax1.set_title(r'$\beta(\mu_{eR},\mu_{eS})$',fontsize=20)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax1.zaxis.set_tick_params(labelsize=12)
        
        plt.figure()
        plt.show()
    
#%%
    if MonteCarlo==False:
        return alpha1, beta1 
    if MonteCarlo==True:
        
        if alpha==True:
            return alpha1, beta1, beta1_mcs, a_msc
        else:
            return alpha1, beta1, beta1_mcs
            
            
#%%
'''
g=np.array([1,2,0.0001,-1,-0.00001])

g0=g[np.argwhere(np.abs(g)<=0.001)]

g0/np.linalg.norm(g0)

'''