#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 09:44:59 2022

@author: ningyi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import *
import pylab
import math
import tt
import tt.ksl
from numpy import linalg as LA
from time import process_time
t_start=process_time()
EYE=1j
qmodes=52              # number of nuclear DoFs
occ=10                 # number of excited states
eps=1e-14              # truncation parameter for tt.round
nsc = 5100             # number of propagation steps
tau = 1              # propagation time step in au. Everything below are in a.u. For unit of time, 41.34 a.u. = 1 fs. 
dim = qmodes         # number of coords
nstates = 2              # number of surfaces
beta = 1052.520787285549 #Corresponds to T = 300 K
gamma = 24 * 0.00003674930882476 #V_DA = 24 meV
gp = 20 * 0.00003674930882476 #g_p = 20 meV
delG = 959 * 0.00003674930882476 #energy bias \Delta G = 959 meV
wp = 600 * 0.00003674930882476 #w_p=600 meV
cmn1toau=4.5563353e-6 # Conversion of wavenumbers to atomic units
wc=53.*cmn1toau      #max freq for Ohmic bath discretization
wmax=5*wc
om=wc/qmodes*(1-np.exp(-wmax/wc))
lam=0.0195506 #532meV
alpha=lam/(2*wc*np.pi)

freq=np.zeros((qmodes-1)) #frequency
cj=np.zeros((qmodes))   #linear electron-phonon coupling constant
gj=np.zeros((qmodes))   #ck in occupation number representation
thetak=np.zeros((qmodes)) #temperature-dependent mixing parameter in TFD
sinhthetak=np.zeros((qmodes)) #sinh(theta)
coshthetak=np.zeros((qmodes)) #cosh(theta)
for i in range(qmodes-1):                   
    freq[i]=-wc*np.log(1-(i+1)*om/(wc)) # Ohmic frequency
    gj[i]=np.sqrt(np.pi*alpha*freq[i]*om/2) #Transfer ck to occ. num. representation
    thetak[i]=np.arctanh(np.exp(-beta*freq[i]/2)) #theta, defined for harmonic models
wj=np.append(freq,wp)
#initialize arrays for parameters
thetak=np.zeros((qmodes)) #temperature-dependent mixing parameter in TFD
sinhthetak=np.zeros((qmodes)) #sinh(theta)
coshthetak=np.zeros((qmodes)) #cosh(theta)

for i in range(qmodes):    
    if i < qmodes-1:               
        thetak[i]=np.arctanh(np.exp(-beta*wj[i]/2)) #theta, defined for harmonic models
        sinhthetak[i]=np.sinh(thetak[i]) #sinh(theta)
        coshthetak[i]=np.cosh(thetak[i]) #cosh(theta)
    else:
        thetak[i]=np.arctanh(np.exp(-beta*wp/2)) #theta, defined for harmonic models
        sinhthetak[i]=np.sinh(thetak[i]) #sinh(theta)
        coshthetak[i]=np.cosh(thetak[i]) #cosh(theta)
        
au2ps=0.00002418884254
su=np.array([1,0]) 
sd=np.array([0,1])
tt_su=tt.tensor(su)
tt_sd=tt.tensor(sd)
tt_Ie=tt.eye(2,1)
gs=np.zeros((occ))
gs[0]=1.
tt_gs=tt.tensor(gs)
tt_psi0=tt_su
for k in range(2*qmodes):#double space formation
    tt_psi0=tt.kron(tt_psi0,tt_gs)
#constructing Pauli operators
px=np.array([[0,1],[1,0]])
pz=np.array([[1,0],[0,-1]])
pd=np.array([[0,0],[0,1]])
#Build electronic site energy matrix
He=np.array([[delG,gamma],[gamma,lam]])
#TT-ize that energy matrix
tt_He=tt.matrix(He)
tt_He=tt.kron(tt_He,tt.eye(occ,qmodes*2))
#Build number operator, corresponds to harmonic oscillator Hamiltonian, see notesc
numoc=np.diag(np.arange(0,occ,1))
tt_numoc=tt.eye(occ,qmodes)*0.
#Build displacement operator, corresponds to x operator in real space
eneroc=np.zeros((occ,occ))
for i in range(occ-1):
    eneroc[i,i+1]=np.sqrt(i+1)
    eneroc[i+1,i]=eneroc[i,i+1]
#Construct number operator as TT
for k in range(qmodes-1):
    if k==0:
        tmp=tt.kron(tt.matrix(numoc)*wj[k],tt.eye(occ,qmodes-1))
    else:
        tmp=tt.kron(tt.eye(occ,k-1),tt.matrix(numoc)*wj[k])
        tmp=tt.kron(tmp,tt.eye(occ,qmodes-k))
    tt_numoc=tt_numoc+tmp
    tt_numoc=tt_numoc.round(eps)
tt_numoc=tt_numoc+tt.kron(tt.eye(occ,qmodes-1),tt.matrix(numoc)*wp)
tt_systemnumoc=tt.kron(tt_Ie,tt_numoc)
tt_systemnumoc=tt.kron(tt_systemnumoc,tt.eye(occ,qmodes))
#create a duplicate of number operator for the ficticious system
tt_tildenumoc=tt.kron(tt_Ie,tt.eye(occ,qmodes))
tt_tildenumoc=tt.kron(tt_tildenumoc,tt_numoc)
#initialize displacement operator
tt_energy=tt.eye(occ,qmodes)*0.
tt_tilenergy=tt.eye(occ,qmodes)*0.
for k in range(qmodes-1):
    if k==0:
#coshtheta takes account for energy flow from real to ficticious system
#thus takes account for temperature effect
        tmp=tt.kron(tt.matrix(eneroc)*gj[k]*coshthetak[k],tt.eye(occ,qmodes-1))
    else:
        tmp=tt.kron(tt.eye(occ,k-1),tt.matrix(eneroc)*gj[k]*coshthetak[k])
        tmp=tt.kron(tmp,tt.eye(occ,qmodes-k))
    tt_energy=tt_energy+tmp
    tt_energy=tt_energy.round(eps)
tt_systemenergy=tt.kron(tt.matrix(pd),tt_energy)
tt_systemenergy=tt.kron(tt_systemenergy,tt.eye(occ,qmodes))
for k in range(qmodes-1):
    if k==0:
        tmp=tt.kron(tt.matrix(eneroc)*gj[k]*sinhthetak[k],tt.eye(occ,qmodes-1))
    else:
        tmp=tt.kron(tt.eye(occ,k-1),tt.matrix(eneroc)*gj[k]*sinhthetak[k])
        tmp=tt.kron(tmp,tt.eye(occ,qmodes-k))
    tt_tilenergy=tt_tilenergy+tmp
    tt_tilenergy=tt_tilenergy.round(eps)
tt_tildeenergy=tt.kron(tt.matrix(pd),tt.eye(occ,qmodes))
tt_tildeenergy=tt.kron(tt_tildeenergy,tt_tilenergy)
tt_vc=tt.kron(tt.eye(occ,qmodes-1),tt.matrix(eneroc)*gp*coshthetak[qmodes-1])
tt_systemvc=tt.kron(tt.matrix(px),tt_vc)
tt_systemvc=tt.kron(tt_systemvc,tt.eye(occ,qmodes))
tt_tvc=tt.kron(tt.eye(occ,qmodes-1),tt.matrix(eneroc)*gp*sinhthetak[qmodes-1])
tt_tildevc=tt.kron(tt.matrix(px),tt.eye(occ,qmodes))
tt_tildevc=tt.kron(tt_tildevc,tt_tvc)
#Note that ficticious Harmonic oscillators carry negative sign
H=tt_He+tt_systemnumoc-tt_tildenumoc+tt_systemenergy+tt_tildeenergy+tt_systemvc+tt_tildevc
H=H.round(eps)
#Construct propagation operator, d/dt psi(t0)=A psi(t0) 
A=-EYE*H
y0=tt_psi0 #Initialize wavefunction
#Heaviside functions, for selecting electronic states from overall wavefunction
tt_heavu=tt.kron(tt_su,tt.ones(occ,dim*2))
tt_heavd=tt.kron(tt_sd,tt.ones(occ,dim*2))
#Propagation time step and range
t=np.arange(0,nsc*tau,tau)
#t=t*au2ps
#Add noise, for higher rank KSL propagation
radd = 9
#radd = np.array([1,9,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,9,1]) 
#radd=np.array([1,9])
#radd=np.append(radd,np.repeat(9,qmodes-3))
#radd=np.append(radd,np.array([9,1]))
#if ( radd > 0 ):
tt_rand=tt.rand(occ,qmodes,radd)
#tt_rand=tt_rand*10**(-140)
#for i in range(radd-1):
#    tt_rand=tt_rand+tt.rand(occ,qmodes*2,1)
wxw=tt_rand.to_list(tt_rand)
for i in range(qmodes):
    wxw[i]=wxw[i]*0.005
tt_rand=tt_rand.from_list(wxw)
#tt_rand=tt_rand*tt_rand.norm()**(-1) #Renormalize noise
tt_rand=tt.kron(tt.ones(2,1),tt_rand)
#tt_rand=tt.kron(tmp,tt_rand)
#y0 = y0+tt_rand*1e-10 #Ensure noise is small
#Initalize population arrays
psu=np.zeros((nsc))
psd=np.zeros((nsc))
ptot=np.zeros((nsc))
#Propagation loop
for ii in range(nsc):
    if ii>0:
        if y0.r[2]<5:#cap max rank
            #rank adaptive
            yold=y0
            tt_rand=tt.rand(yold.n,yold.d,1)
            tt_rand=tt_rand*tt_rand.norm()**(-1)
            tt_rand=tt_rand*1e-10
            ynew=yold+tt_rand
            ynew=ynew*ynew.norm()**(-1)
#            ynew=ynew.round(1e-14)
            yold=tt.ksl.ksl(A,yold,tau)
            ynew=tt.ksl.ksl(A,ynew,tau)
#            yold=tt_soft_ksl(yold,Vm,T)
#            ynew=tt_soft_ksl(ynew,Vm,T)
            overlap=np.abs(tt.dot(yold,ynew))
            if np.abs(overlap-1)<1e-13:
                y0=yold
            else:
                y0=ynew
            print(y0.r)
            #print(k*tau*0.00002418884254)
            psu[ii]=np.abs(tt.dot(tt_heavu*y0,tt_heavu*y0))
            psd[ii]=np.abs(tt.dot(tt_heavd*y0,tt_heavd*y0))
            print(t[ii])
            print('psu=',psu[ii])
        else:
            y0=tt.ksl.ksl(A,y0,tau)
            #print(y0.r)
            #print(k*tau*0.00002418884254)
            psu[ii]=np.abs(tt.dot(tt_heavu*y0,tt_heavu*y0))
            psd[ii]=np.abs(tt.dot(tt_heavd*y0,tt_heavd*y0))
            print(t[ii])
            print('psu=',psu[ii])            


#Plot population difference    
plt.figure(dpi=600)
plt.xlim(0.,5001.)    
plt.ylim(0.,1.)             
plt.xlabel('time(ps)')
plt.ylabel('Populations')
plt.plot(t,psu,label='HEOM3b,T=300K,r=30,tau=10,occ=10')
plt.legend()                     
    








