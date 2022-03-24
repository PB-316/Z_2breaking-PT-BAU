#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:27:05 2021

@author: MarcoAntonio
"""

import numpy as np
from cosmoTransitions import generic_potential_1
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import scipy.integrate as integrate
from scipy import interpolate
from scipy.interpolate import interp1d






####Some definitions##
v2 = 246.**2 
mh=125
v=246
alpha=1/137 #fine structure constant
sinthw=np.sqrt(0.223) #sine of Weinberg angle
g1=np.sqrt(4*np.pi*alpha/(1-sinthw**2)) ## U(1)_Y gauge coupling constant
g=np.sqrt(4*np.pi*alpha)/sinthw #SU(2)_L gauge coupling constant
Mplanck=2.4*10**18 #reduced planck mass
cs=1/3**0.5 ##Sound speed constant


    



class model1(generic_potential_1.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self, ms=50, lammix=0.1, lams=1):
        """
          m1 - tree-level mass of first singlet when mu = 0.
          m2 - tree-level mass of second singlet when mu = 0.
          mu - mass coefficient for the mixing term.
          Y1 - Coupling of the extra boson to the two scalars individually
          Y2 - Coupling to the two scalars together: m^2 = Y2*s1*s2
          n - degrees of freedom of the boson that is coupling.
        """
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
        self.ms = ms
        self.lammix = lammix
        self.lams = lams
 
       

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return any([np.array([X])[...,0] < -5.0, np.array([X])[...,1] < -5.0])
    


    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        h, s = X[...,0], X[...,1]
        pot =-1/4*h**2*mh**2 + 1/2*self.ms**2*s**2 + 1/8*h**4*mh**2/v**2 + 1/4*h**2*s**2*self.lammix
        pot+= -1/4*s**2*v**2*self.lammix + 1/4*s**4*self.lams


        return pot

    
    def boson_massSq(self, X, T):
        X = np.array(X)
        h, s = X[...,0], X[...,1]

    
        #####Scalar thermal masses, obtained from appendix of 1702.06124
        Pi_h = T**2*(g1**2/16 + 3*g**2/16 + 1/4*mh**2/v**2 + 1/4 + self.lammix/24)
        Pi_s= T**2*(self.lammix/6 + self.lams/4)
     
        ##Scalar mass matrix##
        a=-1/2*mh**2 + 3/2*h**2*mh**2/v**2 + 1/2*s**2*self.lammix + Pi_h
        b=self.ms**2 + 1/2*(h**2-v**2)*self.lammix + 3*s**2*self.lams + Pi_s
        cc=h*s*self.lammix
        A=(a+b)/2
        B=1/2*np.sqrt((a-b)**2+4*cc**2)
        m1=A+B
        m2=A-B
        
        ####Gauge boson masses
        mW = g**2*h**2/4 + 11/6*g**2*T**2
        ag=g**2*h**2/4 + 11/6*g**2*T**2
        bg=1/4*g1**2*h**2 + 11/6*g1**2*T**2
        ccg=-1/4*g1*g*h**2
        Ag=(ag+bg)/2
        Bg=1/2*np.sqrt((ag-bg)**2+4*ccg**2)
        mZ=Ag+Bg
        mPh=Ag-Bg


        M = np.array([m1,m2,mW,mZ])
        if self.ms<mh:
            Mphys = np.array([mh**2,self.ms**2,g**2*v**2/4,v**2/4*(g**2+g1**2)])
        else:
            Mphys = np.array([self.ms**2,mh**2,g**2*v**2/4,v**2/4*(g**2+g1**2)])

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

        dof = np.array([1,1,6,3])


        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        #c = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        c = np.array([1.5,1.5,1.5,1.5])
        


        
        return M, dof, c, Mphys


    
    def fermion_massSq(self, X):
        X = np.array(X)
        h,s = X[...,0], X[...,1]

        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        mt=h**2/2
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

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([12])
        return M, dof, Mphys
 
 
    def approxZeroTMin(self):
        #There are generically two minima at zero temperature in this model,
        #and we want to include both of them.
        return [np.array([v,0]), np.array([-v,0])]
    
    

    
#######HERE ARE MY OWN FUNCTIONS
#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTION
#######HERE ARE MY OWN FUNCTIONS#######HERE ARE MY OWN FUNCTIONS
#######HERE ARE MY OWN FUNCTIONS
def sample_model(panda,i):
    """this function creates an instance 
    of model1 with the paramaters extracted 
    from row i of a panda dataframe.
    """
    m=model1(ms=panda["ms"].iloc[i],lammix=panda["lammix"].iloc[i],lams=panda["lams"].iloc[i])
    
    return m

def EWSB_tree(m):
    """This function checks if the EW vacuum is 
    the global one assuming there is minima at some 
    non-zero s-value. The formula uses only the tree-level
    formula of the potential
    
    Parameters
    ----------
    m: is instance of model1
            

    Returns
    -------
    boolean :True or False"""
    return (-2*m.ms**2+v2*m.lammix)**2<2*mh**2*v2*m.lams 

def EWSB_tree_panda(panda):
    """This loop checks the validity of EWSB at tree-level for a panda 
    frame with many points of parameter space
    
    Parameters
    ----------
    panda: a pandaframe
            

    Returns
    -------
    which_one : A list. Contains the index of the parameters 
    for which EWSB is not satisfied
    """
    total=0
    which_one=[]
    for i in range(len(panda)):
        m=sample_model(panda,i)
        if EWSB_tree(m):
             total+=EWSB_tree(m)
        else:
            which_one.append(i)
    return which_one


def isEWSB(m):
    """This function will test if a given model is consistent with succesful EWSB
    It includes one-loop correction
    
    Parameters
        ----------
        m : is a model1
    Returns
        -------
        boolean: True or False """
    
    EWSB=[v,0]
    if m.ms**2<=v**2*m.lammix/2:
        omega=((v**2*m.lammix/2-m.ms**2)/m.lams)**0.5
        minima=m.findMinimum([0,omega],0)
        bosons_minima=m.boson_massSq(minima,0.)
        fermions_minima=m.fermion_massSq(minima)
        bosons_EWSB=m.boson_massSq(EWSB,0.)
        fermions_EWSB=m.fermion_massSq(EWSB)
        if (m.V0(EWSB)+m.V1(bosons_EWSB,fermions_EWSB)<m.V0(minima)+m.V1(bosons_minima,fermions_minima)) and (not np.any(np.linalg.eig(m.d2V(minima,0))[0]<=0)):
            return True
        else: 
            return False
    else:
        return True

def S_cosmoT(m,T):
    """This function calculates the action of a given model m at temperature T.
    It can be used for any temperature T.
    
    Parameters
    ----------
    m : is a model1
    T: (float) is the temperature 
    Returns
    -------
    float: S_3/T, minimum action solution divided by temperature """
    from cosmoTransitions import transitionFinder as tf
    myaction=[]
    for i in range(len(m.phases)):
        s=tf._tunnelFromPhaseAtT(T, m.phases, m.phases[list(m.phases.keys())[i]], m.Vtot,
                                               m.gradV,phitol=1e-15, overlapAngle=45.0,
                                               nuclCriterion=lambda S,T: S/T,
                                               fullTunneling_params = {}, verbose =True, outdict={})
        if s !=np.inf:
            break
        else: continue

    return s



def DS_DT(m, T, S, eps=0.001):
    """This function calculates the derivative of S_3/T using finite differences
    Parameters
        ----------
        m : is a model1
        T: (float) is the temperature 
        S: S/T, a function of m, it can be called by S_cosmoT or S_profile
    Returns
        -------
        float: d/dT(S_3/T), minimum action solution divided by temperature
    """
    from cosmoTransitions import helper_functions
    Tn=m.TnTrans[1]["Tnuc"]
    action = []
    Temp=[]
    n=3
    def Saction(T):
        return S(m,T)
    for i in range(-n,n+1):
        action.append(Saction(Tn+i*eps))
        Temp.append(Tn+i*eps)
    action=np.array(action)
    Temp=np.array(Temp)
    return helper_functions.deriv14(action,Temp)[3]



def S_profile(m,T):
    """This function calculates the Euclidean action from a model m at temperature T
    after knowing its phase history. If more than one FOPT is found, it uses the last 
    transition to compute the action"""
    profile=m.TnTrans[-1]["instanton"].profile1D
    alpha_ode=2
    temp=T
    r, phi, dphi, phivector = profile.R, profile.Phi, profile.dPhi, m.TnTrans[-1]["instanton"].Phi
    phi_meta=m.TnTrans[-1]["high_vev"]
    # Find the area of an n-sphere (alpha=n):
    d = alpha_ode+1  # Number of dimensions in the integration
    area = r**alpha_ode * 2*np.pi**(d*.5)/special.gamma(d*.5)
    # And integrate the profile
    integrand = 0.5 * dphi**2 + m.Vtot(phivector,temp) - m.Vtot(phi_meta,temp)
    integrand *= area
    S = integrate.simps(integrand, r)
    # Find the bulk term in the bubble interior
    volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
    S += volume * (m.Vtot(phivector[0],temp) - m.Vtot(phi_meta,temp))

    return S/T

      
def gradAT(m,T,S,eps=0.001):
    """This function calculates the derivative of S_3/T using finite differences.
    It should only be used for T close to the nucleation tmeperature.
    Parameters
        ----------
        m : is a model1
        T: (float) is the temperature 
        S: S/T, a function of m, it can be called by S_cosmoT or S_profile
    Returns
        -------
        float: d/dT(S_3/T), minimum action solution divided by temperature"""
    
    dT = np.array([-2., -1., 1., 2.])*eps
    
    coef = np.array([1., -8., 8., -1. ])/(12.*eps)
    
    action = []
    
    for i in dT:
        action.append(S(m,T+i))
    
    action = np.array(action)
    
    return np.sum(action*coef)

####This code uses an interpoaltion function for the number of degrees of freedom as function of temperature
###Data is obtained from https://member.ipmu.jp/satoshi.shirai/EOS2018
data = np.loadtxt( 'satoshi_dof.dat' )
Temperature_d=(data.T)[0][900:3900]
dof_d=(data.T)[1][900:3900]
#f = interp1d(Temperature_d, dof_d)###"""the function works from T=[10e-4,1000]"""
g_star = interp1d(Temperature_d, dof_d, kind='cubic')


    
def alpha_GW(Tnuc,Drho):
    ####This code gives the parameter alpha relevant for stochastic GW spectrum 
 ##AS APPEAR IN FORMULA (8.2) OF 1912.12634
    num_dof=g_star(Tnuc)
    radiationDensity=np.pi**2/30*num_dof*Tnuc**4
    latentHeat=Drho
    return latentHeat/radiationDensity


def beta_GW(Tnuc,dS_TdT):
    ###This code defines the parameter beta relevant for stochastic GW spectrum
    num_dof=g_star(Tnuc)
    Hubble=np.sqrt(num_dof*np.pi**2/90)*Tnuc**2/Mplanck
    beta=Hubble*Tnuc*dS_TdT
    return beta/Hubble


##########HERE STARTS TRANSPORT EQUATIONS
##########HERE STARTS TRANSPORT EQUATIONS
##########HERE STARTS TRANSPORT EQUATIONS
##########HERE STARTS TRANSPORT EQUATIONS
def c_dist(x,m,n,p):
    """
    From 2007.10935, formula (8).
    Integrates the derivative of the Bose-Einstein or Fermi-Dirac distribution
    in the fluid frame (vw=0) and weighted by momentum^n/energy^m.  

    Parameters
    ----------
    x : float (mass divided by temperature)
    m: integer, power of energy in denominator
    n: integer, power of momenta in numerator
    p: integer /p=0 for Bosons and p=1 for Fermions
    
    Returns
    -------
    float, the result of integration
    """
    if (n%2)==0:
        integ=integrate.quad(lambda ee: \
                             (ee**2-x**2)**((n+1)/2)/ee**(m-1)*np.e**(-ee)/(1-(-1)**p*np.e**(-ee))**2,\
                             x, np.inf)
        integ_1=1/(4*np.pi**2)*(2/(1+n))*integ[0]
        return integ_1
    else:
        return 0
    


def d_dist(x,m,n,p):
    """
    From 2007.10935, formula (8).
    Integrates the Bose-Einstein or Fermi-Dirac distribution
    weighted by momentum^n/energy^m.  

    Parameters
    ----------
    x : float (mass divided by temperature)
    m: integer 
    n: integer 
    p: integer /p=0 for Bosons and p=1 for Fermions
    
    Returns
    -------
    float, the result of integration
    """
    if (n%2)==0:
        integ=integrate.quad(lambda ee: \
                             (ee**2-x**2)**((n+1)/2)/ee**(m-1)*np.e**(-ee)/(1-(-1)**p*np.e**(-ee)),\
                             x, np.inf)
        integ_1=1/(4*np.pi**2)*(2/(1+n))*integ[0]
        return integ_1
    else:
        return 0

def A_matrix(vw,x,p):
    """
    This function defines the matrix of coefficients for the
    ODE, see eq. (7) of 2007.10935.  

    Parameters
    ----------
    vw : float (the bubble wall velocity)
    x : float (mass divided by temperature)
    p: integer| n=0 for Bosons and n=1 for Fermions
    
    Returns
    -------
    A: np.array| 3x3 matrix of coefficients.
    """
    gamma=1/(1-vw**2)**0.5
    row1=[gamma*vw*c_dist(x,0,0,p),gamma*vw*c_dist(x,-1,0,p),gamma*d_dist(x,0,0,p)]
    row2=[gamma**2*vw*(c_dist(x,-1,0,p)+c_dist(x,1,2,p)),gamma**2*vw*(c_dist(x,-2,0,p)+c_dist(x,0,2,p)),
          gamma**2*(d_dist(x,-1,0,p)+vw**2*d_dist(x,1,2,p))]
    row3=[gamma*vw**2*c_dist(x,0,0,p)+1/gamma**3*sum(vw**(2*n-2)*c_dist(x,2*n,2*n,p) for n in range(1, 12)),
          c_dist(x,1,2,p)/gamma**3+gamma*vw**2*c_dist(x,-1,0,p)+vw/gamma**3*sum(vw**(2*n-3)*c_dist(x,2*n-1,2*n,p) for n in range(2, 12)),
          gamma*vw*d_dist(x,0,0,p)]

              
    Amat=np.array([row1,row2,row3])
    return Amat


def h_profile(z,Lh,h0):
    return h0/2*(np.tanh(z/Lh)+1)
def Dh_profile(z,Lh,h0):
    return h0/2/np.cosh(z/Lh)**2/Lh
def DDh_profile(z,Lh,h0):
    return -h0/Lh**2/np.cosh(z/Lh)**2*np.tanh(z/Lh)

def s_profile(z,Ls,delta,s0):
    return s0/2*(1-np.tanh(z/Ls-delta))
def Ds_profile(z,Ls,delta,s0):
    return -s0/2/Ls/np.cosh(z/Ls-delta)**2
def DDs_profile(z,Ls,delta,s0):
    return s0/Ls**2*np.tanh(z/Ls-delta)/np.cosh(z/Ls-delta)**2


def tanh_profh(z,L,d,h):
    return h/2*(1-np.tanh(z/L-d))
def tanh_profs(z,L,d,h):
    return h/2*(1+np.tanh(z/L-d))


def A_fluid(h0,vw,Lh,z,T):
    """
    This function defines the FULL matrix of coefficients for the
    ODE, see eq. (15) of 2007.10935.  

    Parameters
    ----------
    vw : float (the bubble wall velocity)
    z : float (direction transverse to the wall)
    T: float (Temperature)
    
    Returns
    -------
    A: np.array| 6x6 matrix of coefficients.
    """
    xw=(h_profile(z,Lh,h0)**2/6*(3*g**2/2+g1**2))**0.5/T
    xt=h_profile(z,Lh,h0)/2**0.5/T
    A_fl1=np.concatenate((A_matrix(vw,xw,0),np.zeros((3,3))),axis=1)
    A_fl2=np.concatenate((np.zeros((3,3)),A_matrix(vw,xt,1)),axis=1)
    A_fl=np.concatenate((A_fl1,A_fl2),axis=0)
    return A_fl


def Gamma_W(vw,T):
    """
    This function defines the matrix coming from the collision
    term, see eq.(B6)

    Parameters
    ----------
    vw : float (the bubble wall velocity)
    T: float (Temperature)
    
    Returns
    -------
    Gamma: np.array| 3x3 matrix of coefficients.
    """
    gamma=1/(1-vw**2)**0.5
    G11=0.00239
    G21=0.00512*gamma
    G12=0.00512
    G22=0.0174*gamma
    G13=(4.10*vw-3.28*vw**2+5.51*vw**3-4.47*vw**4)*10**(-3)
    G23=gamma*(1.36*vw+0.610*vw**2-2.90*vw**3+1.36*vw**4)*10**(-2)
    G33=(2.42-1.33*vw**2+3.14*vw**3-2.43*vw**4)*10**(-3)
    G31=(1.18*vw+2.79*vw**2-5.31*vw**3+3.66*vw**4)*10**(-3)
    G32=(2.48*vw+6.27*vw**2-11.9*vw**3+8.12*vw**4)*10**(-3)
    
    return T*np.array([[G11,G12,G13],[G21,G22,G23],[G31,G32,G33]])


def Gamma_t(vw,T):
    """
    This function defines the matrix coming from the collision
    term, see eq.(B6)

    Parameters
    ----------
    vw : float (the bubble wall velocity)
    T: float (Temperature)
    
    Returns
    -------
    Gamza: np.array| 3x3 matrix of coefficients.
    """
    gamma=1/(1-vw**2)**0.5
    G11=0.00196
    G21=0.00445*gamma
    G12=0.00445
    G22=0.0177*gamma
    G13=(5.36*vw-4.49*vw**2+7.44*vw**3-5.90*vw**4)*10**(-3)
    G23=gamma*(1.67*vw+1.38*vw**2-5.46*vw**3+2.85*vw**4)*10**(-2)
    G33=(4.07-2.14*vw**2+4.76*vw**3-4.37*vw**4)*10**(-3)
    G31=(0.948*vw+2.38*vw**2-4.51*vw**3+3.07*vw**4)*10**(-3)
    G32=(2.26*vw+4.82*vw**2-9.32*vw**3+6.54*vw**4)*10**(-3)
    
    return T*np.array([[G11,G12,G13],[G21,G22,G23],[G31,G32,G33]])



def my_Gamma(h0,vw,Lh,z,T):
    """
    This function defines the FULL Gamma matrix of collision terms for the
    ODE, see eq. (16) of 2007.10935 and returns the multiplication of the inverse
    of A_fluid times Gamma.  

    Parameters
    ----------
    vw : float (the bubble wall velocity)
    z : ndarray (direction transverse to the wall)
    T: float (Temperature)
    
    Returns
        -------
    Gamma: np.array| 6x6 matrix, the multiplication of the inverse of A_fluid times Gamma
    
    """
    ####The following code defines the diagonal part of the gamma
    ## Matrix as defined in eq. (16)
    gam__diagupper=np.concatenate((Gamma_W(vw,T),np.zeros((3,3))),axis=1)
    gam__diaglower=np.concatenate((np.zeros((3,3)),Gamma_t(vw,T)),axis=1)
    gamma_diag=np.concatenate((gam__diagupper,gam__diaglower),axis=0)
    ###The following matrices correspond to projection matrices. Defined in the paper.
    P_1=np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    P_2=np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]])
    P_3=np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]])
    ###The following matrices correspond to background collision terms
    Gamma_bgW=-9*Gamma_W(vw,T)
    Gamma_bgt=-12*Gamma_t(vw,T)
    ###The following matrices correspond to eqn. (11)
    Nb=19 #number of bosons
    Nf=78 #number of fermions
    Abg=Nb*A_matrix(vw,0,0)+Nf*A_matrix(vw,0,1)
    ###The following code defines the matrix written in eq. (13) 
    ## where the bottom right block of Abg is inverted and 
    # the rest of the matrix elements are zero
    Abg_inv=np.dot(np.dot(P_2,Abg),P_1)+P_3
    Abg_inv=np.linalg.inv(Abg_inv)-np.transpose(P_3)
    ####The following code defines the non-diagonal part of the gamma
    ## Matrix as defined in the second term of eq. (16)
    gam__nondiagupper=np.concatenate((np.dot(Abg_inv,Gamma_bgW),\
                                      np.dot(Abg_inv,Gamma_bgt)),axis=1)
    gam__nondiaglower=np.concatenate((np.dot(Abg_inv,Gamma_bgW),\
                                      np.dot(Abg_inv,Gamma_bgt)),axis=1)
    gamma_nondiag=np.concatenate((gam__nondiagupper,gam__nondiaglower),axis=0)
    ##We define the total gamma matrix below
    gamma_tot= np.array([(np.dot(np.linalg.inv(A_fluid(h0,vw,Lh,i,T)),gamma_diag)-gamma_nondiag) for i in z])
    return gamma_tot 


def source_A(h0,vw,Lh,z,T):
    """
    This function calculates the source term, see eq. (9) and multplies it by 
    the inverse of A_fluid from the left.
    
    Parameters
    ----------
    vw : float (the bubble wall velocity)
    z : ndarray (direction transverse to the wall)
    T: float (Temperature)
    
    Returns
    -------
    Gamma: np.array| 6x1 matrix, the multiplication of the inverse of A_fluid times S
        
    """
    wholesource=[]
    for i in z:
        gamma=1/(1-vw**2)**0.5
        xw=(h_profile(i,Lh,h0)**2/6*(3*g**2/2+g1**2))**0.5/T
        xt=h_profile(i,Lh,h0)/2**0.5/T
        mwprime=Dh_profile(i,Lh,h0)*(1/6*(3*g**2/2+g1**2))**0.5
        mtprime=Dh_profile(i,Lh,h0)/2**0.5
        Source=np.array([xw*mwprime*(c_dist(xw,1,0,0)),\
                         xw*mwprime*(gamma*c_dist(xw,0,0,0)),\
                         xw*mwprime*(vw*c_dist(xw,1,0,0)-1/gamma**2*sum(vw**(2*n-1)*c_dist(xw,2*n+1,2*n,0) for n in range(1, 12))),\
                         xt*mtprime*(c_dist(xt,1,0,1)),\
                         xt*mtprime*(gamma*c_dist(xt,0,0,1)),\
                         xt*mtprime*(vw*c_dist(xt,1,0,1)-1/gamma**2*sum(vw**(2*n-1)*c_dist(xt,2*n+1,2*n,1) for n in range(1, 12)))])
        Source*=vw*gamma/T
        wholesource.append(np.dot(np.linalg.inv(A_fluid(h0,vw,Lh,i,T)),Source))
    return np.array(wholesource)

def Tranport_eqs(z, q,damping,source):
    """
    This function defines the differential equation to be solved. 
    It corresponds to eq. (14)
    
    Parameters
    ----------
    z : ndarray, the size of the integration region
    q: array (6D), the value of the perturbations
    
    Returns
    -------
    dq/dz, array    
    """
    rows=[]
    #damping=inter_gamma_matrix
    #source=inter_source

    for i in range(6):
        elem=[]
        for j in range(6):
            elem.append(-damping[i][j](z)*q[j])
        rows.append(sum(elem)+source[i](z))
    rows=tuple(rows)
    return np.vstack(rows)
    
def bc(ya, yb):
    """
    This code defines the boundary conditions to be used by BVP
    """
    return np.array([ya[0],yb[0],ya[3],yb[3],ya[1], ya[4]])
    #return np.array([ya[0],yb[0],ya[3],yb[3],ya[1]-ya[2], ya[4]-ya[5]])

    #return np.array([ya[0],yb[0],ya[3],yb[3],ya[1]**2+ya[2]**2 ,ya[4]**2+ya[5]**2])


def background_eqs(vw,z,q,T,sol):
    """
    This function defines the differential equation to be solved. 
    It corresponds to eq. (12)
    
    Parameters
    ----------
    z : ndarray, the size of the integration region
    q: array (6D), the value of the perturbations
    
    Returns
    -------
    dq/dz, array    
    """
    Gamma_bgW=-9*Gamma_W(vw,T)
    Gamma_bgt=-12*Gamma_t(vw,T)
    Nb=20 #number of bosons
    Nf=78 #number of fermions
    Abg=Nb*A_matrix(vw,0,0)+Nf*A_matrix(vw,0,1)
    P_1=np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    P_2=np.array([[1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]])
    P_3=np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]])
    Abg_inv=np.dot(np.dot(P_2,Abg),P_1)+P_3
    Abg_inv=np.linalg.inv(Abg_inv)-np.transpose(P_3)
    At=np.dot(Abg_inv,Gamma_bgt)
    Aw=np.dot(Abg_inv,Gamma_bgW)
    qw=sol.sol(z)[0:3]
    qt=sol.sol(z)[3:6]
    output=-np.dot(At,qt)-np.dot(Aw,qw)
    return output[1:]
    
def bc_background(ya, yb):
    """
    This code defines the boundary conditions to be used by BVP
    """
    return np.array([ya[0], ya[1]])
    

def d_dist_v(x,vw,p):
    """
    Inegral of distribution function in the wall frame divided by pz. 
    Given by eq. (8) with m=0 and n=-1.
    
    Parameters
    ----------
    x : float (mass divided by temperature)
    p: integer /p=0 for Bosons and p=1 for Fermions
    
    Returns
     -------
    float, the result of integration
    """
    ff=lambda x,v,e:2*v*(e**2-x**2)**0.5 + 2*e*(1-v**2)*np.arctanh((e**2-x**2)**0.5/e/v) if (e**2-x**2)**0.5/e/v<1 \
    else 2*v*(e**2-x**2)**0.5 + 2*e*(1-v**2)*np.log((e**2-x**2)**0.5/e/v)
    integ=integrate.quad(lambda e: np.exp(-e)/(1-(-1)**p*np.exp(-e))*ff(x,vw,e), x, np.infty)
    return integ[0]/4/np.pi**2





######HERE STARTS HYDRODYNAMIC EQUATIONS FROM REF.1004.4187
########################
########################
##############################
##############################
##############
##3######3
############## MORE FORMULAS RELEVANT 
 ##############FOR HYDRODYNAMICS
 ############## OF THE TEMPERATURE

def Lorentz_mu(v1,v2):
    """Lorentz transformation. Usually to the fluid frame"""
    return (v1-v2)/(1-v1*v2)

def hydrodynamic_eq(xi,v):
    """Parametrizes the equation for the fluid velocity. 
    Given by eqn. (2.27) in 1004.4187"""
    gamma=lambda vv:1/(1-vv**2)**0.5
    dv_dxi = 2*v/xi*(gamma(v)**2*(1-v*xi)*(Lorentz_mu(xi,v)**2/cs**2-1))**(-1)
    return dv_dxi



def Tp_exp_equation(xi,v_fun):
    """This function is the inegrand of the hydrodynamic relation log(T)/T. """
    v=v_fun(xi)
    return -2*cs**2*v*(xi-v)/xi/((xi-v)**2-cs**2*(1-v*xi)**2)
    
def sol_detonation_odeint(xi,alpha_p):
    """This function solves the hydrodynamic eqns. for detonations.
    For detonation the fluid in front of the wall is at rest (for the fluid rest frame)"""
    vp=xi
    if vp<=cs:
        print("Not a detonation. Velocity too low. ")
        return 
    det=-12*vp**2+(-1-3*vp**2+3*alpha_p*(1-vp**2))**2
    if det<0:
        print("Negative determinant")
        return 
    vm=(1+3*vp**2-3*alpha_p*(1-vp**2))/6/vp
    vm1=vm-1/6/vp*(det)**(0.5)
    vm2=vm+1/6/vp*(det)**(0.5)
    ##" strong detonations (v− < cs−) are not consistent solutions of the fluid equations,"
    vm=max([vm1,vm2])  ##Always choose the maximum value. 
    if vm==cs:
        print("Jouguet detonation")
    ###Now solve it:
    v_init=Lorentz_mu(vp,vm)
    v_range=np.linspace(cs,vp)[::-1]
    solut=integrate.odeint(hydrodynamic_eq, v_init, v_range,tfirst=True)
    return np.vstack((v_range,solut.T))


    
    
def sol_deflagration_odeint(xi,alpha_p):
    """This function solves the hydrodynamic eqns. for deflagrations.
    For deflagration the fluid behind the wall is at rest (for the fluid rest frame)"""
    vm=xi
    det=-1/3+(1/6/vm+vm/2)**2+2*alpha_p/3+alpha_p**2
    if det<0:
        print("Negative determinant")
        return 
    vp=(1/6/vm+vm/2)/(1+alpha_p)
    vp1=vp-det**0.5/(1+alpha_p)
    vp2=vp+det**0.5/(1+alpha_p)
    ###Now solve it:
    v_init=Lorentz_mu(vm,vp1)
    v_range=np.linspace(vm,1)
    solut=integrate.odeint(hydrodynamic_eq,v_init, v_range,tfirst=True)
    return np.vstack((v_range,solut.T))





def sol_deflagration_solve_ivp(xi,alpha_p):
    """This function solves the hydrodynamic eqns. for deflagrations.
    For deflagration the fluid behind the wall is at rest (in the fluid rest frame)
    
    Parameters
    ----------------
    xi: fluid velocity, self-similar variable
    alpha_p: The strength of the phase transition.
    """
    if xi>cs:
        print("Velocity is supersonic. Not a deflagration")
        return 
    vm=xi
    det=-1/3+(1/6/vm+vm/2)**2+2*alpha_p/3+alpha_p**2
    if det<0:
        print("Negative determinant")
        return 
    vp=(1/6/vm+vm/2)/(1+alpha_p)
    vp1=vp-det**0.5/(1+alpha_p)
    vp2=vp+det**0.5/(1+alpha_p)
    ###Now solve it:
    v_init=Lorentz_mu(vm,vp1)
    v_range=(vm,1)
    def myevent(xi,v):
        """Position of the shock-front"""
        return Lorentz_mu(xi,v)*xi-cs**2
    myevent.terminal=True
    solut=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]),method="BDF",events=[myevent])
    return [solut.t,solut.y[0]]

#def sol_detonations_solve_ivp(xi,alpha_p):
 #   """This function solves the hydrodynamic eqns. for detonations:
 #      BOTH strong detonations and hybrids.
 #   For strong detonations the fluid in front of the wall is at rest (for the fluid rest frame).
 #   Hybrids are Jouguet detonations v-=cs"""
    # if xi<cs:
    #     print("Not a detonation. Velocity too low. ")
    #     return 
    # vp=xi
    # det=-12*vp**2+(-1-3*vp**2+3*alpha_p*(1-vp**2))**2
    # if det<0:
    #     print("Negative determinant")
    #     print("The solution could be a hybrid")
    #     vm=cs
    #     xi_Jouguet=((alpha_p*(2+3*alpha_p))**0.5+1)/(3**0.5*(1+alpha_p))
    #     if xi_Jouguet>=vm:
    #         xi_Jouguet=(-(alpha_p*(2+3*alpha_p))**0.5+1)/(3**0.5*(1+alpha_p))
    #     vp=xi_Jouguet 
    #     ###solve detonation component
    #     v_init=Lorentz_mu(xi,vm)
    #     v_range=(xi+0.00001,cs)
    #     solut_low=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]))
    #     ###solve deflagration component
    #     v_init=Lorentz_mu(xi,vp)
    #     v_range=(xi,1)
    #     def myevent(xi,v):
    #         """Position of the shock-front"""
    #         return Lorentz_mu(xi,v)*xi-cs**2
    #     myevent.terminal=True
    #     solut_high=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]),method="BDF",events=[myevent])
    #     xi_range=solut_low.t[1:][::-1],solut_high.t
    #     v_range=solut_low.y[0][1:][::-1],solut_high.y[0]
    #     deton_solution=[np.concatenate(xi_range),np.concatenate(v_range)]
    # elif det>0:
    #     print("Positive determinant. \n.........")
    #     print("Solving strong detonation")
    #     vm=(1+3*vp**2-3*alpha_p*(1-vp**2))/6/vp
    #     vm1=vm-1/6/vp*(det)**(0.5)
    #     vm2=vm+1/6/vp*(det)**(0.5)
    #     vm=max([vm1,vm2])  ##Always choose the maximum value. 
    #     v_init=Lorentz_mu(vp,vm)
    #     v_range=(vp,cs)
    #     solut=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]),method="BDF")
    #     deton_solution=[solut.t,solut.y[0]]
    # return deton_solution 


def sol_detonations_solve_ivp(xi,alpha_p):
    """This function solves the hydrodynamic eqns. for detonations:
       BOTH strong detonations and hybrids.
    For strong detonations the fluid in front of the wall is at rest (for the fluid rest frame).
    Hybrids are Jouguet detonations v-=cs"""
    xi_Jouguet=((alpha_p*(2+3*alpha_p))**0.5+1)/(3**0.5*(1+alpha_p))
    if xi<cs:
        print("Not a detonation. Velocity too low. ")
        return 
    elif (xi<=xi_Jouguet) :
        print("The solution is a hybrid")
        vm=cs
        vp=(-(alpha_p*(2+3*alpha_p))**0.5+1)/(3**0.5*(1+alpha_p))
        ###solve detonation component
        v_init=Lorentz_mu(xi,cs)
        v_range=(xi+10e-5,cs)
        solut_low=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]))
        ###solve deflagration component
        v_init=Lorentz_mu(xi,vp)
        v_range=(xi,1)
        def myevent(xi,v):
            """Position of the shock-front"""
            return Lorentz_mu(xi,v)*xi-cs**2
        myevent.terminal=True
        solut_high=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]),method="BDF",events=[myevent])
        xi_range=solut_low.t[1:][::-1],solut_high.t
        v_range=solut_low.y[0][1:][::-1],solut_high.y[0]
        deton_solution=[np.concatenate(xi_range),np.concatenate(v_range)]
    elif xi>xi_Jouguet:
        vp=xi
        print("The solution is a strong detonation")
        vm=(1+3*vp**2-3*alpha_p*(1-vp**2))/6/vp
        det=-12*vp**2+(-1-3*vp**2+3*alpha_p*(1-vp**2))**2
        vm1=vm-1/6/vp*(det)**(0.5)
        vm2=vm+1/6/vp*(det)**(0.5)
        vm=max([vm1,vm2])  ##Always choose the maximum value. 
        print(vm)
        v_init=Lorentz_mu(vp,vm)
        v_range=(vp,cs)
        solut=integrate.solve_ivp(hydrodynamic_eq, v_range,np.array([v_init]),method="BDF")
        deton_solution=[solut.t,solut.y[0]]
    return deton_solution 


def plot_Temperature_Deflagration(vm,alpha_p):
    """This function makes a plot of the logT/T"""
    det=-1/3+(1/6/vm+vm/2)**2+2*alpha_p/3+alpha_p**2
    if det>0:
        vp1=(1/6/vm+vm/2)/(1+alpha_p)-det**0.5/(1+alpha_p)
        defla=sol_deflagration_solve_ivp(vm,alpha_p)
        xi_range=defla[0]
        v_fun_inter=interp1d(xi_range,defla[1])
        logT_T=integrate.quad(Tp_exp_equation,xi_range[0],xi_range[-1],args=(v_fun_inter))[0]
        plt.plot(xi_range,[Tp_exp_equation(xi,v_fun_inter) for xi in xi_range])
        plt.xlabel("$\\xi$",size=15)
        plt.ylabel("$\\log{T}/T$",rotation=0,size=15)
        plt.title("$\\log{T}/T$")
        plt.show()
    
    


