#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from cosmoTransitions import generic_potential_1
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import scipy.integrate as integrate
import random
from scipy import interpolate, special
import seaborn as sns
from scipy import misc




####Some definitions##
v2 = 246.2**2
mh=125.09
v=246.2
alpha=1/137
sinthw=np.sqrt(0.223)
g1=np.sqrt(4*np.pi*alpha/(1-sinthw**2))
g=np.sqrt(4*np.pi*alpha)/sinthw
Mplanck=2.4*10**18
cs=1/3**0.5 ##Sound speed constant


####This code uses an interpoaltion function for the number of degrees of freedom as function of temperature
###Data is obtained from https://member.ipmu.jp/satoshi.shirai/EOS2018
data = np.loadtxt( 'satoshi_dof.dat' )[500:3900]
Temperature_d=(data.T)[0]
dof_d=(data.T)[1]#relativistic degrees of freedom
dof_s=(data.T)[3]#entropic degrees of freedom
g_star = interpolate.interp1d(Temperature_d, dof_d, kind='cubic')
g_star_s = interpolate.interp1d(Temperature_d, dof_s, kind='cubic')

def my_fun(modind):
    class model1(generic_potential_1.generic_potential):
        def init(self, ms = 50, theta = 0, muhs = 0, u = 100, mu3 = 0,Lam=500):
            self.Lam=Lam
            self.yt=1/(1+u**2/Lam**2)**.5
            self.Ndim = 2
            self.renormScaleSq = v2
            self.ms = ms
            self.theta = theta
            self.muhs = muhs
            self.u = u
            self.mu3 = mu3
            self.lamh = 1/(4*v2)*(mh**2+self.ms**2 + (mh**2 - ms**2)*np.cos(2*self.theta))
            #self.lams = 1/(2*self.u**2)*(mh**2*np.sin(self.theta)**2+self.ms**2*np.cos(self.theta)**2 + self.mu3*self.u + self.muhs*v**2/(2*self.u))
            self.lams = 1/(4*self.u**3)*(mh**2*self.u + ms**2*self.u + 2*self.u**2*self.mu3 + v**2*self.muhs - (mh**2-ms**2)*self.u*np.cos(2*self.theta))
            self.lammix = 1/(v*self.u)*(-(self.ms**2-mh**2)*np.sin(self.theta)*np.cos(self.theta) - self.muhs*v)
            self.muh2 = self.lamh*v2 + self.muhs*self.u + self.lammix/2*self.u**2
            self.mus2 = -self.mu3*self.u + self.lams*self.u**2 + self.muhs*v2/(2*self.u) + self.lammix/2*v2

        def forbidPhaseCrit(self, X):
            return any([np.array([X])[...,0] < -5.0])
            #return any([np.array([X])[...,0] < -5.0, np.array([X])[...,1] < -5.0])

        def V0(self, X):
            X = np.asanyarray(X)
            h, s = X[...,0], X[...,1]
            pot = -1/2*self.muh2*h**2 + 1/4*self.lamh*h**4 - 1/2*self.mus2*s**2 - 1/3*self.mu3*s**3 + 1/4*self.lams*s**4 + 1/2*self.muhs*h**2*s + 1/4*self.lammix*h**2*s**2
            return pot

        def boson_massSq(self, X, T):
            X = np.array(X)
            h, s = X[...,0], X[...,1]

           #####Scalar thermal masses, obtained from appendix of 1702.06124
            Pi_h = T**2*(g1**2/16 + 3*g**2/16 + self.lamh/2 + self.yt**2/4 + self.lammix/24)
            Pi_s= T**2*(self.lammix/6 + self.lams/4)

            ##Scalar mass matrix##
            a=3*h**2*self.lamh + s**2*self.lammix/2 - self.muh2 + s*self.muhs + Pi_h
            b=h**2*self.lammix/2 + 3*s**2*self.lams - 2*s*self.mu3 - self.mus2 + Pi_s
            cc=h*s*self.lammix  + h*self.muhs
            A=(a+b)/2
            B=1/2*np.sqrt((a-b)**2+4*cc**2)
            m1=A+B
            m2=A-B

            ####Gauge boson masses (Longitudinal)
            mWL = g**2*h**2/4 + 11/6*g**2*T**2
            ag=g**2*h**2/4 + 11/6*g**2*T**2
            bg=1/4*g1**2*h**2 + 11/6*g1**2*T**2
            ccg=-1/4*g1*g*h**2
            Ag=(ag+bg)/2
            Bg=1/2*np.sqrt((ag-bg)**2+4*ccg**2)
            mZL=Ag+Bg
            mPh=Ag-Bg


            M = np.array([m1,m2,g**2*h**2/4,h**2/4*(g**2+g1**2),mWL,mZL])
            if self.ms<mh:
                Mphys = np.array([mh**2,self.ms**2,g**2*v**2/4,v**2/4*(g**2+g1**2),g**2*v**2/4,v**2/4*(g**2+g1**2)])
            else:
                Mphys = np.array([self.ms**2,mh**2,g**2*v**2/4,v**2/4*(g**2+g1**2),g**2*v**2/4,v**2/4*(g**2+g1**2)])

            # At this point, we have an array of boson masses, but each entry might
            # be an array itself. This happens if the input X is an array of points.
            # The generic_potential class requires that the output of this function
            # have the different masses lie along the last axis, just like the
            # different fields lie along the last axis of X, so we need to reorder
            # the axes. The next line does this, and should probably be included in
            # all subclasses.
            M = np.rollaxis(M, 0, len(M.shape))
            Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))

            # The number of degrees of freedom for the masses. This should be a
            # one-dimensional array with the same number of entries as there are
            # masses.

            dof = np.array([1,1,4,2 , 2,1]) ##Longitudinal at the end


            # c is a constant for each particle used in the Coleman-Weinberg
            # potential using MS-bar renormalization. It equals 1.5 for all scalars
            # and the longitudinal polarizations of the gauge bosons, and 0.5 for
            # transverse gauge bosons.
            #c = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
            c = np.array([1.5,1.5,1.5,1.5,1.5,1.5])

            return M, dof, c, Mphys




        def fermion_massSq(self, X):
            X = np.array(X)
            h,s = X[...,0], X[...,1]
            mt=self.yt**2*h**2/2*(1+s**2/self.Lam**2)
            #mt=self.yt**2*h**2/2
            M = np.array([mt])
            Mphys = np.array([v**2/2])

            # At this point, we have an array of boson masses, but each entry might
            # be an array itself. This happens if the input X is an array of points.
            # The generic_potential class requires that the output of this function
            # have the different masses lie along the last axis, just like the
            # different fields lie along the last axis of X, so we need to reorder
            # the axes. The next line does this, and should probably be included in
            # all subclasses.
            M = np.rollaxis(M, 0, len(M.shape))
            Mphys = np.rollaxis(Mphys, 0, len(Mphys.shape))

            dof = np.array([12])
            return M, dof, Mphys


        def approxZeroTMin(self):
            # There are generically two minima at zero temperature in this model,
            # and we want to include both of them.

            return [np.array([v,self.u])]

        def theory_consistent(self):
            perturbative_limit=4*np.pi
            perturbativity=self.lamh<=perturbative_limit and self.lams<=perturbative_limit and abs(self.lammix)<=perturbative_limit
            positivity=(self.lamh>0) and (self.lams>0) and (self.lammix>-2*(self.lamh*self.lams)**.5)
            if perturbativity and positivity:
                #print("Model is theoretically consistent \n")
                return True
            else:
                #print("Model is NOT theoretically consistent \n")
                return False


        def print_couplings(self):
            print("Potential parameters are given by \n ")
            print("mus2=",self.mus2, "muh2=",self.muh2,"lamh=",self.lamh,"lams=",self.lams,"lammix=",self.lammix,"\n")
            print("Model parameters are \n")
            print("ms=",self.ms,"theta=",self.theta,"muhs=",self.muhs,"u=",self.u,"mu3=",self.mu3,"\n")

        def isEWSB_old(self):
            """Method to find the deepest minima of the potential at T=0.
            Doesn't work for Z_2 symmetric potential!!!"""
            n=100
            X_EW=np.array([v,self.u])
            minima=[]
            if self.muhs==0 and self.mu3==0:
                #print("Model has a Z2 symmetry in the potential \n")
                #print("isEWSB=True \n")
                return True,X_EW
            #------------
            X0=self.findMinimum([0,100],0)
            if self.Vtot(X0,0)<=self.Vtot(X_EW,0) and abs(abs(X0[0])-v)>10 and abs(self.Vtot(X0,0)-self.Vtot(X_EW,0))>1:
                #print("Global minimum found at X=",X0,"\n")
                #print("isEWSB=False \n")
                return False, X0
            X0=self.findMinimum([0,-100],0)
            if self.Vtot(X0,0)<=self.Vtot(X_EW,0) and abs(abs(X0[0])-v)>10 and abs(self.Vtot(X0,0)-self.Vtot(X_EW,0))>1:
                #print("Global minimum found at X=",X0,"\n")
                #print("isEWSB=False \n")
                return False, X0

            ###This loop search for a global minima randomly
            for i in range(n):
                #x1=np.random.uniform(-100,4*self.Tmax)
                #x2=np.random.uniform(-4*self.Tmax,4*self.Tmax)
                x1=np.random.uniform(-100,self.Tmax)
                x2=np.random.uniform(self.Tmax,self.Tmax)
                X0=self.findMinimum([x1,x2], T=0.0)
                if self.Vtot(X0,0)<=self.Vtot(X_EW,0) and abs(X0[0])-v>10 and abs(self.Vtot(X0,0)-self.Vtot(X_EW,0))>1e2:
                    #print("Global minimum found at X=",X0,"\n")
                    #print("isEWSB=False \n")
                    return False, X0
            #print("isEWSB=True \n")
            return True,X_EW
        def isEWSB(self):
            """Method to find the deepest minima of the potential at T=0.
            Doesn't work for Z_2 symmetric potential!!!"""
            n=100
            X_EW=np.array([v,self.u])
            minima=[]
            if self.muhs==0 and self.mu3==0:
                #print("Model has a Z2 symmetry in the potential \n")
                #print("isEWSB=True \n")
                return True,X_EW
            #------------
            X0=self.findMinimum([0,100],0)
            if self.V0(X0)<=self.V0(X_EW) and abs(abs(X0[0])-v)>10 and abs(self.V0(X0)-self.V0(X_EW))>1:
                #print("Global minimum found at X=",X0,"\n")
                #print("isEWSB=False \n")
                return False, X0
            X0=self.findMinimum([0,-100],0)
            if self.V0(X0)<self.V0(X_EW) and abs(abs(X0[0])-v)>10 and abs(self.V0(X0)-self.V0(X_EW))>1:
                #print("Global minimum found at X=",X0,"\n")
                #print("isEWSB=False \n")
                return False, X0

            ###This loop search for a global minima randomly
            for i in range(n):
                #x1=np.random.uniform(-100,4*self.Tmax)
                #x2=np.random.uniform(-4*self.Tmax,4*self.Tmax)
                x1=np.random.uniform(-100,self.Tmax)
                x2=np.random.uniform(self.Tmax,self.Tmax)
                X0=self.findMinimum([x1,x2], T=0.0)
                if self.V0(X0)<=self.V0(X_EW) and abs(X0[0])-v>10 and abs(self.V0(X0)-self.V0(X_EW))>1e2:
                    #print("Global minimum found at X=",X0,"\n")
                    #print("isEWSB=False \n")
                    return False, X0
            #print("isEWSB=True \n")
            return True,X_EW

    #######HERE ARE MY OWN FUNCTIONS
    #######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
    #######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
    #######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTION
    #######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
    #######HERE ARE MY OWN FUNCTIONS

    def g_loop(z):
        """Loop integral for EDM. Extracted from (A.2) of 1712.09613"""
        g_integrand=lambda x: np.log(x*(1-x)/z)/(x*(1-x)-z)
        integral=integrate.quad(g_integrand,0,1)[0]
        return z*0.5*integral


    def d_eEDM(X):
        X=np.array(X)
        theta,ms,Lam=X[...,0],X[...,1],X[...,2]
        G_f=1/2**.5/v**2
        ee=g1
        me=0.5*1e-3
        mt=172.9
        alpha=1/137
        numeric=ee/3/np.pi**2*alpha*G_f*v/2**.5/np.pi/mt*me*(v/2**.5/Lam)
        out=np.sin(theta)*np.cos(theta)*(-g_loop(mt**2/mh**2) + g_loop(mt**2/ms**2))
        return np.abs(numeric*out)

    d_eEDM_bound=1.89*10**(-16)

    def alpha_GW(Tnuc,Drho):
        ####This code gives the parameter alpha relevant for stochastic GW spectrum
        ##AS APPEAR IN FORMULA (8.2) OF 1912.12634
        num_dof=g_star(Tnuc)
        radiationDensity=np.pi**2/30*num_dof*Tnuc**4
        latentHeat=Drho
        return latentHeat/radiationDensity


    def trans_class(SymNR):
        """Classify the transition according to the following characteristics:
        ---------
        phi-sym: transition happens in the s-field direction
        phi-symNR: transition happens in the s-field direction (symmetry is not restored at T=1000)
        """
        SNR="sym"
        if SymNR==True:
            SNR="SNR"

        if dh>10 and ds>10:
            return "hs-"+SNR
        elif dh>10 and ds<1:
            return "h-"+SNR
        elif ds>10 and dh<1:
            return "s-"+SNR
        else:
            return "none"+SNR

    def beta_GW(Tnuc,dS_TdT):
        ###This code defines the parameter beta relevant for stochastic GW spectrum
        beta=Tnuc*dS_TdT
        return beta

    def S_profile(T,elem):
        """This function calculates the Euclidean action from a model m at temperature T
        after knowing its phase history. If more than one FOPT is found it uses "elem" from the list.
        """
        profile=elem["instanton"].profile1D
        alpha_ode=2
        temp=T
        r, phi, dphi, phivector = profile.R, profile.Phi, profile.dPhi, elem["instanton"].Phi
        phi_meta=elem["high_vev"]
        # Find the area of an n-sphere (alpha=n):
        d = alpha_ode+1  # Number of dimensions in the integration
        area = r**alpha_ode * 2*np.pi**(d*.5)/special.gamma(d*.5) ##4 pi r^2 the surface of a sphere
        # And integrate the profile
        integrand = 0.5 * dphi**2 + m.Vtot(phivector,temp) - m.Vtot(phi_meta,temp)
        integrand *= area
        S = integrate.simps(integrand, r)
        # Find the bulk term in the bubble interior
        volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
        S += volume * (m.Vtot(phivector[0],temp) - m.Vtot(phi_meta,temp))

        return S/T

    #####
    def my_getPhases(m):
        myexps=[(-5,-3),(-5,-5),(-5,-4),(-3,-3)]
        for nord in myexps:
            print("doing",nord)
            try:
                m.getPhases(tracingArgs={"dtstart":10**(nord[0]), "tjump":10**(nord[1])})
                phases_out=m.phases
            except:
                phases_out={}
            finally:
                if len(phases_out)>1:
                    break
        return phases_out




    ##DEFINE Nucleation FUNCTION (TO BE OPTIMIZED)
    def nucleation_temp(T):
        """DEFINE Nucleation FUNCTION (TO BE OPTIMIZED)
        """
        from cosmoTransitions import transitionFinder as tf
        try:
            S=tf._tunnelFromPhaseAtT(T=T, phases=m.phases, start_phase=m.phases[nuc_dict["high_phase"]], V=m.Vtot,
                                     dV=m.gradV,phitol=1e-15, overlapAngle=45, nuclCriterion=lambda S,T: Gamma_Hubble4(S,T)-Hubble_total(T)**4,
                                     fullTunneling_params={}, verbose=True, outdict={})
        except:
            return 1e+100

        if np.isnan(S):
            return 1e+100
        else:
            return S**2

    def Hubble_vacuum(T):
        """Hubble parameter for matter density alone. Second term in formula 2.6 of 1809.08242 """
        if m.phases is None:
            phases_dict=m.getPhases()
        else:
            phases_dict=m.phases
        if T<nuc_dict["Tmin"] or T>nuc_dict["Tmax"]:
            return 0
        from cosmoTransitions import transitionFinder as tf
        crit_temps=tf.findCriticalTemperatures(phases_dict, m.Vtot)
        Delta_V=0.0
        for tran_barrier in crit_temps:
            if tran_barrier["trantype"] ==2:
                continue
            ###LOOKS FOR TWO-STEP TRANSITIONS
            #elif T>tran_barrier["Tcrit"] or ((abs(tran_barrier["high_vev"][0]-tran_barrier["low_vev"][0])<1) and (abs(tran_barrier["high_vev"][1]-tran_barrier["low_vev"][1])<1)):
            elif T>tran_barrier["Tcrit"]: ##This is redundant
                continue
            else:
                V_high=m.Vtot(phases_dict[tran_barrier["high_phase"]].valAt(T),T)
                V_low=m.Vtot(phases_dict[tran_barrier["low_phase"]].valAt(T),T)
                if V_high>V_low:
                    Delta_V += V_high - V_low

        return (Delta_V/3)**0.5/Mplanck

    def Hubble_radiation(T):
        num_dof=g_star(T)
        Hubble_rad_squared=num_dof*np.pi**2/90*T**4/Mplanck**2
        return Hubble_rad_squared**.5

    def Hubble_total(T):
        num_dof=g_star(T)
        Hubble_rad_squared=num_dof*np.pi**2/90*T**4/Mplanck**2
        return (Hubble_rad_squared+Hubble_vacuum(T)**2)**0.5


    def Gamma_Hubble4(S,T):
        """Nucleation probability per Hubble volume, including matter contribution.
        Integrand in formula (2.2) 1809.08242 """
        Gamma1=T**4*np.exp(-S/T)*np.sqrt((S/2/np.pi/T))**3
        return Gamma1

    ##DEFINE ACTION FUNCTION
    def my_Action(T):
        """Calculates S/T at T"""
        from cosmoTransitions import transitionFinder as tf
        try:
            S=tf._tunnelFromPhaseAtT(T=T, phases=m.phases, start_phase=m.phases[nuc_dict["high_phase"]], V=m.Vtot,
                                 dV=m.gradV, phitol=1e-15, overlapAngle=45, nuclCriterion=lambda S,T: S/T ,
                                 fullTunneling_params={}, verbose=True, outdict={})
        except:
            S=np.inf
        return S



    ####This codes the GW signal and SNR given T,alpha, beta and vw.
    LISA_data = np.loadtxt( 'PLS_ESACallv1-2_04yr.txt' )
    LISA_data=LISA_data[::20]
    LISA_noise=LISA_data[::,0::2]
    LISA_data=LISA_data[::,0::3]


    def GW_signal_old(Temp,alpha,beta,vel):
        f_redshift=1.65*10**(-5)*(Temp/100)*(g_star(Temp)/100)**(1/6)
        Omega_redshift=1.67*10**(-5)*(100/g_star(Temp))**(1/3)
        kappa_sw=alpha/(0.73+0.083*alpha**0.5+alpha)
        Uf=(3/4*alpha/(1+alpha)*kappa_sw)**0.5
        HR=(8*np.pi)**(1/3)*max(vel,cs)/beta
        HRb=(vel-cs)/vel*HR
        Htau_sw=HR/Uf
        S_fun=lambda s:s**3*(7/(4+3*s**2))**(7/2)
        Omega_sw=3*0.687*Omega_redshift*(1-1/(1+2*HR/Uf)**0.5)*(kappa_sw*alpha/(1+alpha))**2*0.012*HR/cs
        f_sw=f_redshift*(2.6/1.65)*(1/HR)
        GW_tab=[Omega_sw*S_fun(f/f_sw) for f in LISA_noise[::,0]]
        return np.array([LISA_noise[::,0],GW_tab])

    def SNR_GW_old(Temp,alpha,beta,vel):
        time=4
        f_redshift=1.65*10**(-5)*(Temp/100)*(g_star(Temp)/100)**(1/6)
        Omega_redshift=1.67*10**(-5)*(100/g_star(Temp))**(1/3)
        kappa_sw=alpha/(0.73+0.083*alpha**0.5+alpha)
        Uf=(3/4*alpha/(1+alpha)*kappa_sw)**0.5
        HR=(8*np.pi)**(1/3)*max(vel,cs)/beta
        HRb=(vel-cs)/vel*HR
        Htau_sw=HR/Uf
        S_fun=lambda s:s**3*(7/(4+3*s**2))**(7/2)
        Omega_sw=3*0.687*Omega_redshift*(1-1/(1+2*HR/Uf)**0.5)*(kappa_sw*alpha/(1+alpha))**2*0.012*HR/cs
        f_sw=f_redshift*(2.6/1.65)*(1/HR)
        integral=np.sum([(LISA_noise[i+1][0]-LISA_noise[i][0])/2*(Omega_sw*S_fun(LISA_noise[i][0]/f_sw)/(LISA_noise[i][1]))**2 for i in range(0,len(LISA_noise)-1)])
        return (time*3.15*10**7*integral)**0.5


    def r_comoving(T,T1):
        """Comoving size of a bubble nucleated at T' after growing until T"""
        vw=1
        T_range=np.linspace(T,T1)
        T_half=(T_range[1:]+T_range[:-1])/2
        T_diff=(T_range[1:]-T_range[:-1])
        Hubble_range=[1/Hubble_total(Temp) for Temp in T_half]
        integral=np.sum(np.array(Hubble_range)*T_diff)
        return vw*integral

    def Gamma(T):
        """This function gives the nucleation rate at finite temperature """
        try:
            S_action=my_Action(T)*T
        except:
            S_action=np.inf
        if S_action==np.inf:
            return 0
        Gamma1=T**4*np.exp(-S_action/T)*np.sqrt((S_action/2/np.pi/T))**3
        return Gamma1


    def I_volume_fraction_integrand(T,T1):
        """For given temperatures T1>T, it calculates the integrand of the volume fraction.
        Includes the factor 4pi/3."""
        if T>T1:
            print("incorrect setting of temperatures")
            return
        Gam=Gamma(T1)/Hubble_total(T1)/T1**4
        if Gam==0:
            return 0
        r_vol=r_comoving(T,T1)
        if r_vol==0:
            return 0
        output=Gam*r_vol**3
        return 4*np.pi/3*output

    def I_volume_fraction(T,vw):
        """For given temperature T. It calculates the volume fraction doing the integral of I_volume_fraction_integrand
        starting from T until the critial temperature found by nucleation_temp ("Tmax")"""
        n=20 ###This number should be changed on a case by case basis
        t_range=np.linspace(T,nuc_dict["Tmax"],n)
        I_list=[]
        for tempi in t_range:
            x=I_volume_fraction_integrand(T,tempi)
            I_list.append(x)
        fun_inter=interpolate.interp1d(t_range,I_list)
        result = integrate.quad(lambda Tx: fun_inter(Tx),t_range[0], t_range[-1])
        return vw**3*result[0]-0.34




    def kappa(xi_w,alpha):
        """
        Fit for the efficiency factor
        """
        c_s         = 1./np.sqrt(3.)
        #kappa_A     = xi_w**1.2 * 6.9*alpha/(1.39-0.0037*np.sqrt(alpha)+alpha)
        kappa_A     = xi_w**1.2 * 6.9*alpha/(1.36-0.037*np.sqrt(alpha)+alpha)
        #kappa_B     = alpha**0.4/(0.0017+(0.997+alpha)**0.4)
        kappa_B     = alpha**0.4/(0.017+(0.997+alpha)**0.4)
        #kappa_C     = np.sqrt(alpha)/(0.135+np.sqrt(0.98)+alpha)
        kappa_C     = np.sqrt(alpha)/(0.135+np.sqrt(0.98+alpha))
        #kappa_D     = alpha/(0.75+0.085*np.sqrt(alpha)+alpha)
        kappa_D     = alpha/(0.73 + 0.083*np.sqrt(alpha)+alpha)
        delta_kappa = -0.9*np.log(np.sqrt(alpha)/(1.+np.sqrt(alpha)))
        xi_w_J      = (np.sqrt(2./3.*alpha+alpha**2)+1./np.sqrt(3.))/(1.+alpha)

        if xi_w < c_s:
        # deflagration
            return c_s**2.2 * kappa_A * kappa_B / ( (c_s**2.2 - xi_w**2.2 )*kappa_B + xi_w * c_s**1.2 * kappa_A )
        elif xi_w < xi_w_J:
        # hybrid
            return (
                kappa_B
                + (xi_w -c_s)*delta_kappa
                + (xi_w-c_s)**3/(xi_w_J-c_s)**3 * ( kappa_C - kappa_B - (xi_w_J-c_s)*delta_kappa )
            )
        #else:
        # detonation
        return (xi_w_J-1.)**3 * (xi_w_J/xi_w)**2.5 * kappa_C * kappa_D / (
            ( (xi_w_J-1.)**3 - (xi_w-1.)**3 ) * xi_w_J**2.5 * kappa_C
            + (xi_w-1.)**3 * kappa_D
        )




    def T_percolation(n,vw):
        """Having Tnuc it computes the percolation temperature using I=0.34 as optimization criteria."""
        t_range=np.linspace(nuc_dict["Tmin"],Tnuc,n)[::-1]
        I_list=[]
        t_range_1=[]
        for temp in t_range:
            x=I_volume_fraction(temp,vw)
            I_list.append(x)
            t_range_1.append(temp)
            if x>10:
                break
        fun_inter=interpolate.interp1d(t_range_1,I_list)
        def f_opt(t):
            try:
                return fun_inter(t)**2
            except:
                return 1e+100
        Tp = optimize.fmin(f_opt,(t_range_1[0]+t_range_1[-1])*0.5,xtol=0.0001)[0]
        volume_shrinks=Tp*misc.derivative(fun_inter, x0=Tp, dx=1e-5, n=1)<-3
        print("The physical volunme diminishes at Tp:",volume_shrinks)
        # plt.plot(t_range_1,fun_inter(t_range_1))
        # plt.ylim(-1,1)
        # plt.axhline(y=0)
        # plt.xlabel("$T (GeV)$")
        # plt.title("$I(T)-0.34$")
        # plt.show()
        return Tp,volume_shrinks


    LISA_curve=pd.read_csv("LISA_integrated_curve.csv",index_col=[0])
    def GW_signal(Temp,alpha,beta,vel):
        HR=(8*np.pi)**(1/3)*max(vel,cs)/beta ##eqn. (7.5)
        f_sw=2.6*1e-5/HR*(Temp/100)*(g_star(Temp)/100)**(1/6) ##eqn. (7.3)
        Sw = lambda f: (f/f_sw)**3*(4/7 + 3/7*(f/f_sw)**2)**(-7/2) ##eqn. (7.2)
        Uf=np.sqrt(3/4*alpha/(1+alpha)*kappa(vel,alpha))##eqn. (7.4)
        tauH = HR/Uf ##eqn. (7.4)
        prefactor=4.13*1e-7*HR*(1-1/np.sqrt(1+2*tauH))*(kappa(vel,alpha)*alpha/(1+alpha))**2 ##eqn. (7.1)
        Omega=lambda f: prefactor*(100/g_star(Temp))**(1/3)*Sw(f) ##eqn. (7.1)
        f_range=LISA_curve.f
        GW_tab=[Omega(f) for f in f_range]
        return np.array([list(f_range), list(GW_tab)])

    def SNR_GW(signal):
        """Computes SNR given signal (f,Omega)"""
        time=4
        f,Om=signal
        Om=np.array(Om)
        On=np.array(LISA_curve.Omega_noise)
        integral=integrate.simps(Om**2/On**2,f)
        return (time*3.15*10**7*integral)**0.5


    def find_nucleation(m):
        """Find min and max temperatures to search for nucleation. IT will be used by bisection method.
        Parameters
            ----------
            m: a model instance. In this case m=model1(kk=1/600**2) for example.
        Returns
            -------
            nuc_dict: a dictionary containing the relevant temperatures and phases indexes.
                    It will be used by the other methods to find the nucleation and percolation parameters
        """
        if m.phases is None:
            try:
                #phases_dict=m.getPhases()
                #phases_dict=m.getPhases(tracingArgs={"dtstart":1e-3, "tjump":1e-3})
                #phases_dict=m.getPhases(tracingArgs={"dtstart":1e-5, "tjump":1e-4})
                phases_dict=my_getPhases(m)
            except:
                return {}
        else:
            phases_dict=m.phases
        if len(phases_dict)<=1:
            return {}
        from cosmoTransitions import transitionFinder as tf
        crit_temps=tf.findCriticalTemperatures(phases_dict, m.Vtot)
        Num_relevant_trans=0
        ###DETERMINE IF THERE COULD BE TWO-STEP FOPTs
        my_dicts=[]
        for elem in crit_temps:
            if elem["trantype"]==1 and (abs(elem["low_vev"][0]-elem["high_vev"][0])>10 or abs(elem["low_vev"][1]-elem["high_vev"][1])>10):
                print("Tunneling is relevant from phase " + str(elem["high_phase"])+ " to " + str(elem["low_phase"])  )
                Tmax=elem["Tcrit"]
                Tmin=phases_dict[elem["high_phase"]].T[0]
                print("high_vev", elem["high_vev"])
                print("low_vev", elem["low_vev"])
                print("max temperature is", Tmax)
                print("min temperature is", Tmin)
                Num_relevant_trans+=1
                high_phase_key=elem["high_phase"]
                low_phase_key=elem["low_phase"]
                dict_output= {"Tmin":Tmin, "Tmax":Tmax, "high_phase": high_phase_key,"low_phase": low_phase_key}
                #my_dicts.append(dict_output)

                X0=m.phases[dict_output["high_phase"]].X[0]
                T0=m.phases[dict_output["high_phase"]].T[0]
                stable=not np.any(np.linalg.eig(m.d2V(X0,T0))[0]<=0)

                def findminT(T):
                    """Function to find the minimum temperature at which the high_vev coexists.
                    Written in form for optimization"""
                    Xmin=m.findMinimum(X0,T)
                    dx=np.sum((Xmin-X0)**2)**0.5
                    stable=not np.any(np.linalg.eig(m.d2V(Xmin,T))[0]<=0)
                    if stable==False or (dx<1) == False or T<0:
                        return 5000
                    else:
                        return  T
                Tmin_opt=optimize.fminbound(findminT,0,T0)
                dict_output["Tmin"]=Tmin_opt
                my_dicts.append(dict_output)
            else:
                continue

        return my_dicts




    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Compute the transition of model(s)

    # In[2]:


    def findTnuc():
        """Finds the nucleation temperature and checks if it actually satisfies the condition Gamma/H^4==1"""
        fun_nucleation=lambda T: (Gamma(T)/Hubble_total(T)**4-1.)**4
        if nuc_dict["Tmin"]<1:
            Tnuc=optimize.fminbound(fun_nucleation,np.mean([nuc_dict["Tmin"]+1e-3,nuc_dict["Tmax"]]),nuc_dict["Tmax"])
        else:
            Tnuc=optimize.fminbound(fun_nucleation,nuc_dict["Tmin"]+1e-3,nuc_dict["Tmax"])
        S_nuc=my_Action(Tnuc)
        check_nucleation=abs(Gamma_Hubble4(S_nuc*Tnuc,Tnuc)/Hubble_total(Tnuc)**4-1)<1e-3
        if check_nucleation==True:
            return Tnuc, S_nuc, True
        else:
            Tnuc=optimize.fminbound(nucleation_temp,nuc_dict["Tmin"],nuc_dict["Tmax"])
        S_nuc=my_Action(Tnuc)
        check_nucleation=abs(Gamma_Hubble4(S_nuc*Tnuc,Tnuc)/Hubble_total(Tnuc)**4-1)<1e-3
        if check_nucleation==True:
            return Tnuc, S_nuc, True
        else:
            return None, None, False


    # In[4]:


    hydrocolumns=['vw', 'Lh', 'dh', 'h0', 'Ls', 'ds', 'shigh', 'slow',
                  'Type', 'alpha_p', 'vm', 'vp', 'xi_s', 'Tp/TN', 'vel_converged']

    modi=modind


    num_points=1
    for l in range(num_points):
        ms_val=df.iloc[modi]["ms"]
        theta_val=df.iloc[modi]["theta"]
        u_val=df.iloc[modi]["u"]
        mu3_val=df.iloc[modi]["mu3"]
        muhs_val=df.iloc[modi]["muhs"]
        Lam_val=df.iloc[modi]["Lam_CP"]
        m=model1(ms = ms_val, theta = theta_val,muhs= muhs_val ,u = u_val,mu3 = mu3_val,Lam=Lam_val)

        edm_Bool=d_eEDM([m.theta,m.ms,m.Lam])<d_eEDM_bound
        thbool=m.theory_consistent()
        EWSBbool=m.isEWSB()
        EWSB_new=EWSBbool[0]==True  or (sum(EWSBbool[1]**2)**.5>Mplanck)
        if not (edm_Bool==True and thbool==True and EWSB_new):
            break
        Pih=g1**2/16 + 3*g**2/16 + m.lamh/2 + m.yt**2/4 + m.lammix/24
        Pis=m.lammix/6 + m.lams/4
        lamh_tilde=m.lamh - m.lammix**2/4/m.lams
        dict_out={'ms':m.ms,'theta':m.theta, 'u':m.u,"muhs":m.muhs,"mu3":m.mu3,"yt":m.yt,
              "lamh":m.lamh,"lams":m.lams,"lammix":m.lammix,
              "muh2":m.muh2,"mus2":m.mus2,
              "Pih":Pih,"Pis":Pis,"lamh_tilde":lamh_tilde,"Lam_CP":m.Lam}
        dict_out.update({ "th_bool":thbool,"isEWSB": EWSBbool[0],"edm_Bool":edm_Bool})

        nuc_dicts=find_nucleation(m)
        phases_copy=m.phases.copy()
        if len(nuc_dicts)==0:
            continue
        else:
            num_dicts=len(nuc_dicts)
        dict_out.update({"num_FOPT":num_dicts})

        for nuc_dict in nuc_dicts:
            try:
                ###Parameters at nucleation temperature------------------------------------------------------------
                dict_out.update({"Tc_"+str(nuc_dicts.index(nuc_dict)):nuc_dict["Tmax"],
                                 "Tmin_"+str(nuc_dicts.index(nuc_dict)):nuc_dict["Tmin"]})

                relevant_phases={nuc_dict["high_phase"]:m.phases[nuc_dict["high_phase"]],nuc_dict["low_phase"]:m.phases[nuc_dict["low_phase"]]}
                m.phases=relevant_phases
                nucleation= findTnuc()
                if nucleation[2]==False:
                    m.phases=phases_copy
                    continue
                else:
                    Tnuc, S_nuc = nucleation[0], nucleation[1]
                    dict_out.update({'Tnuc_'+str(nuc_dicts.index(nuc_dict)): Tnuc})
                    dict_out.update({"action_Tn_"+str(nuc_dicts.index(nuc_dict)):S_nuc*Tnuc})


                m.findAllTransitions(tunnelFromPhase_args={"nuclCriterion":lambda S,T: S/(T+1e-100)-S_nuc})
                alltrans_Tnuc=m.TnTrans
                elem=alltrans_Tnuc[0] ###How many transitions??
                if abs(elem["action"]-S_nuc*Tnuc)>1 and abs(elem["Tnuc"]-Tnuc)>1:
                    """check if all transitions indeed find the same S and T values"""
                    m.phases=phases_copy
                    continue
                dS_TdT_Tnuc=misc.derivative(S_profile, x0=Tnuc, dx=.01, n=1, args=(elem,),order=7)

                #--------vevs, energy density, pressure and wall velocity at Tnuc---------------------------------------------------------
                phi_stable_Tnuc=elem["low_vev"]
                dict_out.update({"h_low_Tnuc_"+str(nuc_dicts.index(nuc_dict)):phi_stable_Tnuc[0],
                                 "s_low_Tnuc_"+str(nuc_dicts.index(nuc_dict)):phi_stable_Tnuc[1]})

                phi_meta_Tnuc=elem["high_vev"]
                dict_out.update({"h_high_Tnuc_"+str(nuc_dicts.index(nuc_dict)):phi_meta_Tnuc[0],
                                 "s_high_Tnuc_"+str(nuc_dicts.index(nuc_dict)):phi_meta_Tnuc[1]})

                Delta_rho_Tnuc=m.energyDensity(phi_meta_Tnuc,Tnuc,include_radiation=True)-m.energyDensity(phi_stable_Tnuc,Tnuc,include_radiation=True)
                dict_out.update({"Delta_rho_Tnuc_"+str(nuc_dicts.index(nuc_dict)): Delta_rho_Tnuc})

                Delta_p_Tnuc=m.Vtot(phi_meta_Tnuc,Tnuc)-m.Vtot(phi_stable_Tnuc,Tnuc)
                dict_out.update({"Delta_p_Tnuc_"+str(nuc_dicts.index(nuc_dict)): Delta_p_Tnuc})

                alpha_Tnuc=alpha_GW(Tnuc,Delta_rho_Tnuc)
                dict_out.update({"alpha_Tnuc_"+str(nuc_dicts.index(nuc_dict)):alpha_Tnuc})

                beta_Tnuc=beta_GW(Tnuc,dS_TdT_Tnuc)
                dict_out.update({"beta_Tnuc_"+str(nuc_dicts.index(nuc_dict)):beta_Tnuc})

                xi_Jouguet=((alpha_Tnuc*(2+3*alpha_Tnuc))**0.5+1)/(3**0.5*(1+alpha_Tnuc))
                dV = m.Vtot(phi_meta_Tnuc,Tnuc)-m.Vtot(phi_stable_Tnuc,Tnuc)
                radiationDensity=np.pi**2/30*g_star(Tnuc)*Tnuc**4
                vwall=(dV/alpha_Tnuc/radiationDensity)**0.5 ##Analytic formula
                dict_out.update({"vwall_"+str(nuc_dicts.index(nuc_dict)): vwall,
                                 "xi_Jouguet_"+str(nuc_dicts.index(nuc_dict)):xi_Jouguet})

                my_signal_Tnuc=GW_signal(Tnuc*(1+alpha_Tnuc)**0.25,alpha_Tnuc,beta_Tnuc,1)
                peak_vals_Tnuc=my_signal_Tnuc.T[my_signal_Tnuc[1]==max(my_signal_Tnuc[1])][0] ##Extract values at peak
                f_peak_Tnuc=peak_vals_Tnuc[0]
                Omega_peak_Tnuc=peak_vals_Tnuc[1]
                dict_out.update({"f_peak_Tnuc_"+str(nuc_dicts.index(nuc_dict)): f_peak_Tnuc,
                            "Omega_peak_Tnuc_"+str(nuc_dicts.index(nuc_dict)):Omega_peak_Tnuc})
                SNR_Tnuc=SNR_GW(my_signal_Tnuc)
                dict_out.update({"SNR_Tnuc_"+str(nuc_dicts.index(nuc_dict)): SNR_Tnuc})

                ##Parameters at PERCOLATION temperature Tp--------------------------------------
                Tp,volume_shrinks=T_percolation(20,df.iloc[modi]["vw"])
                dict_out.update({"Tp_"+str(nuc_dicts.index(nuc_dict)):Tp,
                                 "volume_shrinks_"+str(nuc_dicts.index(nuc_dict)):volume_shrinks})

                S_p=my_Action(Tp)
                dict_out.update({"action_Tp_"+str(nuc_dicts.index(nuc_dict)):S_p*Tp})

                m.findAllTransitions(tunnelFromPhase_args={"nuclCriterion":lambda S,T: S/(T+1e-100)-S_p})
                alltrans_Tp=m.TnTrans
                elem=alltrans_Tp[0] ###How many transitions??
                if abs(elem["action"]-S_p*Tp)>1 and abs(elem["Tnuc"]-Tp)>1:
                    """check if all transitions indeed finds the same S and T values"""
                    m.phases=phases_copy
                    continue
                dS_TdT_Tp=misc.derivative(S_profile, x0=Tp, dx=.01, n=1, args=(elem,),order=7)


                #-------vevs, energy density and pressure at Tp--------------------------------
                phi_stable_Tp=elem["low_vev"]
                dict_out.update({"s_low_Tp_"+str(nuc_dicts.index(nuc_dict)):phi_stable_Tp[1],
                                 "h_low_Tp_"+str(nuc_dicts.index(nuc_dict)):phi_stable_Tp[0]})

                phi_meta_Tp=elem["high_vev"]
                dict_out.update({"s_high_Tp_"+str(nuc_dicts.index(nuc_dict)):phi_meta_Tp[1],
                                 "h_high_Tp_"+str(nuc_dicts.index(nuc_dict)):phi_meta_Tp[0]})

                Delta_rho_Tp=m.energyDensity(phi_meta_Tp,Tp,include_radiation=True)-m.energyDensity(phi_stable_Tp,Tp,include_radiation=True)
                dict_out.update({"Delta_rho_Tp_"+str(nuc_dicts.index(nuc_dict)): Delta_rho_Tp})

                Delta_p_Tp=m.Vtot(phi_meta_Tp,Tp)-m.Vtot(phi_stable_Tp,Tp)
                dict_out.update({"Delta_p_Tp_"+str(nuc_dicts.index(nuc_dict)): Delta_p_Tp})

                alpha_Tp=alpha_GW(Tp,Delta_rho_Tp)
                dict_out.update({"alpha_Tp_"+str(nuc_dicts.index(nuc_dict)):alpha_GW(Tp,Delta_rho_Tp)})

                beta_Tp=beta_GW(Tp,dS_TdT_Tp)
                dict_out.update({"beta_Tp_"+str(nuc_dicts.index(nuc_dict)):beta_Tp})

                xi_Jouguet_Tp=((alpha_Tp*(2+3*alpha_Tp))**0.5+1)/(3**0.5*(1+alpha_Tp))
                dV_Tp = m.Vtot(phi_meta_Tp,Tp)-m.Vtot(phi_stable_Tp,Tp)
                radiationDensity_Tp=np.pi**2/30*g_star(Tp)*Tp**4
                vwall_Tp=(dV_Tp/alpha_Tp/radiationDensity_Tp)**0.5 ##Analytic formula
                dict_out.update({"vwall_Tp_"+str(nuc_dicts.index(nuc_dict)): vwall_Tp,
                                 "xi_Jouguet_Tp_"+str(nuc_dicts.index(nuc_dict)):xi_Jouguet_Tp})

                my_signal_Tp=GW_signal(Tp*(1+alpha_Tp)**0.25,alpha_Tp,beta_Tp,df.iloc[modi]["vw"])
                peak_vals_Tp=my_signal_Tp.T[my_signal_Tp[1]==max(my_signal_Tp[1])][0] ##Extract values at peak
                f_peak_Tp=peak_vals_Tp[0]
                Omega_peak_Tp=peak_vals_Tp[1]
                dict_out.update({"f_peak_Tp_"+str(nuc_dicts.index(nuc_dict)): f_peak_Tp,
                                 "Omega_peak_Tp_"+str(nuc_dicts.index(nuc_dict)):Omega_peak_Tp})
                SNR_Tp=SNR_GW(my_signal_Tp)
                dict_out.update({"SNR_Tp_"+str(nuc_dicts.index(nuc_dict)): SNR_Tp})
                dict_out.update(dict(df[hydrocolumns].iloc[modi]))####Comment out not for BAU

                #-----Fill dictionary------------------------------------------------------------
                print("\n ..........\n Current dictionary is: \n")
                print(dict_out)
            except:
                print("error ocurred")
                m.phases=phases_copy
                continue

    return dict_out





##------INSERT PANDAS:


df=pd.read_csv("SCANS/BAU/sols_fullmodel_All.csv",index_col=[0])
df=df[df["vel_converged"]==True]
df=df[df.Lam_CP>df.ms]
df=df[df.Lam_CP>v]
df=df[df.Lam_CP>abs(df.mu3)]
df=df[df.Lam_CP>abs(df.muhs)]
df=df[df.alpha_max>1e-3]
df=df.sort_values("alpha_max")




###Do parallelization

from multiprocessing import Pool
import time
start = time.time()

###The Multiprocessing package provides a Pool class,
##which allows the parallel execution of a function on the multiple input values.
##Pool divides the multiple inputs among the multiple processes which can be run parallelly.
f= my_fun
if __name__ == '__main__':
    with Pool() as p:
        df_pool=p.map(f, range(len(df)))

print(df_pool)
pd.DataFrame(df_pool).to_csv("./SCANS/PERCOLATION/scan_new.csv")



end = time.time()
print("The time of execution of above program is :", end-start)
