#!/usr/bin/env python

from cosmoTransitions import generic_potential_1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
import scipy.integrate as integrate
from scipy import interpolate, special
import seaborn as sns
from scipy import misc


v2 = 246.**2
mh=125
v=246
alpha=1/137 #fine structure constant
sinthw=np.sqrt(0.223) #sine of Weinberg angle
g1=np.sqrt(4*np.pi*alpha/(1-sinthw**2)) ## U(1)_Y gauge coupling constant
g=np.sqrt(4*np.pi*alpha)/sinthw #SU(2)_L gauge coupling constant
Mplanck=2.4*10**18 #reduced planck mass
cs=1/3**0.5 ##Sound speed constant
del alpha


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
                x1=np.random.uniform(-100,4*self.Tmax)
                x2=np.random.uniform(-4*self.Tmax,4*self.Tmax)
                #x1=np.random.uniform(-100,4*self.Tmax)
                #x2=np.random.uniform(self.Tmax,self.Tmax)
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





    modi=modind
    EWSB_tree=[]
    dict_out=dict(df.iloc[modi])
    for modi in range(0,len(df)):
        m=model1(ms = df.iloc[modi].ms, theta =df.iloc[modi].theta, muhs = df.iloc[modi].muhs,
                 u = df.iloc[modi].u, mu3 =df.iloc[modi].mu3,Lam=df.iloc[modi].Lam_CP)
        EWSB=m.isEWSB()
        dict_out.update({"EWSBtree":EWSB[0],"htrue":EWSB[1][0],"htrue":EWSB[1][1]})
    return dict_out

###------PANDAS


df=pd.read_csv("SCANS/isEWSB.csv",index_col=[0])

###------Do parallelization
from multiprocessing import Pool
import time
start = time.time()

###The Multiprocessing package provides a Pool class,
##which allows the parallel execution of a function on the multiple input values.
##Pool divides the multiple inputs among the multiple processes which can be run parallelly.
#num_points=4
f= my_fun
if __name__ == '__main__':
    with Pool() as p:
        df_pool=p.map(f, range(len(df)))

print(df_pool)
pd.DataFrame(df_pool).to_csv("/Users/marcoantoniomerchandmedina/results_local.csv")
end = time.time()
print("The time of execution of above program is :", end-start)
