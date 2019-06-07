''' Main code to calculate expanded envelopes '''

from scipy.optimize import brentq
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import numpy as np
from numpy import linspace, sqrt, log10, array, pi
from IO import load_params
import os
from lsoda_remove import stdout_redirected

# --------------------------------------- Constants and parameters --------------------------------------------

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2
sigmarad = 0.25*arad*c

# Parameters
M, RNS, y_inner, comp, mode, save, img = load_params()

if comp == 'He':
    mu = 4.0/3.0  
    # mu=4
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0

ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

T_inner = 10**8.5
rg = 2*GM/c**2 # gravitationnal radius


# -------------------------------------------- Microphysics ----------------------------------------------------

def kappa(T):
    return kappa0/(1.0+(T/4.5e8)**0.86)    

def cs2(T):  # ideal gas sound speed  c_s^2
    return kB*T/(mu*mp)

# (To replace using electron gas later)

def pressure(rho, T): # ideal gas + radiation pressure (eq 2c)}  PLUS(new)  electron pressure (non-degen + degen)
    return rho*cs2(T) + arad*T**4/3.0 

def internal_energy(rho, T):  # ideal gas + radiation + electrons 
    return 1.5*cs2(T)*rho + arad*T**4 

def Beta(rho, T):  # pressure ratio
    Pg = rho*cs2(T)
    Pr = arad*T**4/3.0
    return Pg/(Pg+Pr)

def del_ad(rho,T):
    b = Beta(rho,T)
    return (8-6*b)/(32 -24*b - 3*b**2)

def del_rad(rho, T, r):
    term1 = kappa(T)*Linf/(16*pi*c*GM*(1-Beta(rho,T)))*Swz(r) + pressure(rho,T)/(rho*c**2)  
    term2 = (1 + (4-1.5*Beta(rho,T)) * pressure(rho,T)/(rho*c**2) )**(-1)    
    return term1*term2    # (term1 is completely dominating, term2 could be removed)
    # return kappa(T)*Linf*Swz(r)/(16*pi*c*GM*(1-Beta(rho,T)))

def Del(rho,T, r):
    return min((del_ad(rho,T) , del_rad(rho,T, r)))

# ----------------------------------------- General Relativity ------------------------------------------------

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)**(-0.5)     # NOTE : unlike wind_GR code, there is -1/2 exponent here

def grav(r): # local gravity
    return GM/r**2 * Swz(r)

# ----------------------------------------- Initial conditions ------------------------------------------------

def photosphere(f0):

    ''' Finds photospheric density and temperature (eq 9a-b) for a given luminosity-like parameter f0 
        Also sets Linf, the luminosity seen by observers at infinity '''

    def Teff_eq(T): # for root-solving
        # return T**4*kappa(T) - GM*c/(Rphot**2*sigmarad) * Swz(Rphot) * (1-10**f0)
        return kappa(T) - (GM*c/(Rphot**2*sigmarad) * Swz(Rphot) * (1-10**f0))/T**4

    Tkeep1, Tkeep2 = 0.0, 0.0
    npoints = 10
    while Tkeep1 == 0 or Tkeep2 == 0:
        logT = linspace(6, 8, npoints)
        for T in 10**logT:
            foo = Teff_eq(T)
            if foo < 0.0:
                Tkeep1 = T
            if foo > 0.0 and Tkeep2 == 0:
                Tkeep2 = T
        npoints += 10

    T = brentq(Teff_eq, Tkeep1, Tkeep2, xtol=1e-10, maxiter=10000)
    rho = 2/3 * mu*mp/(kB*T) * grav(Rphot)/kappa(T) * 10**f0
    Linf = 4*pi*Rphot**2*sigmarad*T**4* (Swz(Rphot)**(-2))

    return rho,T,Linf

# -------------------------------------------- Calculate derivatives ---------------------------------------

def derivs(r, Y):
    ''' Calculates the derivatives of rho and T with r as the independent variable '''

    rho,T = Y[:2]
    P,b = pressure(rho,T) , Beta(rho,T)
    delr = del_rad(rho, T, r)

    dP_dr = -GM*rho/r**2 * Swz(r)**2 * (1 + 4*(4-1.5*b)*P/(rho*c**2))

    drho_dr = rho/P * dP_dr * (1 - (4-3*b)*delr)/b
    dT_dr   = T/P   * dP_dr * delr

    if Del(rho,T,r) == del_ad: print('WARNING : CONVECTIVE')

    return [drho_dr, dT_dr]

# ---------------------------------------------- Integration -----------------------------------------------

def Shoot(rspan, rhophot, Tphot, returnResult=False):
    ''' Integrates in from the photosphere, using r as the independent variable, until T=Tbase
        We want to match the location of p=p_inner to the NS radius '''

    inic = [rhophot, Tphot]

    def hit_innerTemp(t, y): return y[1]-T_inner
    hit_innerTemp.terminal = True # stop integrating at this point 

    sol = solve_ivp(derivs, rspan, inic, method='Radau',events = hit_innerTemp)

    r = sol.t
    r,rho,T = sol.t , sol.y[0] , sol.y[1]

    return r,rho,T


# ------------------------------------------------- Envelope ---------------------------------------------------

def MakeEnvelope0(f0, Rphot_km, Verbose=0):   # just get envelope for provided f0 value

    global Rphot,Linf,verbose
    Rphot,verbose = Rphot_km*1e5 , Verbose

    rho_phot,T_phot,Linf = photosphere(f0)
    rspan = (Rphot , 1.01*rg)
    r,rho,T = Shoot(rspan,rho_phot,T_phot)    
    RNS_calc = r[-1]
    error = (RNS_calc/1e5-RNS)/RNS
    print(error)

    # return error
    return r,rho,T


def MakeEnvelope(Rphot_km, Verbose=0):    # setup for relaxation method

    global Rphot,Linf,verbose
    Rphot,verbose = Rphot_km*1e5 , Verbose
    
    ## Sketch an initial solution 
    # Span f-values in the interval [-3.7,-4].  Find the minimum value that gives a physical looking solution (Rbase>RNS)
    # Extrapolate that solution to RNS+epsilon and attach a nearly-vertical line to T=Tb, rho=1e3, at RNS

    rspan = (Rphot , 1.01*rg)
    fvalues = linspace(-3.7,-4,100)
    for i,f0 in enumerate(fvalues):
        rho_phot,T_phot,Linf = photosphere(f0)
        r,rho,T = Shoot(rspan,rho_phot,T_phot)    
        if r[-1]<RNS*1e5:
            ilast = i-1
            break
            
    rho_phot,T_phot,Linf = photosphere(fvalues[ilast])
    r,rho,T = Shoot(rspan,rho_phot,T_phot)  


    ## Making inital solution

    # m=20 point grid
    zone1 = np.linspace(RNS*1e5 , RNS*1e5+200 , 15) # surface to 200m above
    zone2 = np.linspace(RNS*1e5+250 , Rphot, 5)     # 250m above to photosphere
    r0 = np.concatenate((zone1,zone2))

    # Fit a line in log space, seperately in the inner and extended part
    fT2 = np.poly1d(np.polyfit(log10(r[10::-1]),log10(T[10::-1]), deg=1))
    fT1 = np.poly1d(np.polyfit(log10([zone1[0] , zone1[-1]]), [log10(T_inner) , fT2(log10(zone1[-1]))], deg=1))

    frho2 = np.poly1d(np.polyfit(log10(r[10::-1]),log10(rho[10::-1]), deg=1))
    frho1 = np.poly1d(np.polyfit(log10([zone1[0] , zone1[-1]]), [3 , frho2(log10(zone1[-1]))], deg=1))

    T0   = 10**np.concatenate((fT1(log10(zone1))   , fT2(log10(zone2))))
    rho0 = 10**np.concatenate((frho1(log10(zone1)) , frho2(log10(zone2))))

    return r,rho,T,r0,T0,rho0


r,rho,T,r0,T0,rho0=MakeEnvelope(20)

import matplotlib.pyplot as plt
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
ax1.set_xlabel('log r')
ax1.set_ylabel('log T')
ax2.set_xlabel('log r')
ax2.set_ylabel('log rho')
ax1.plot(log10(r/1e5),log10(T),'k-')
ax1.plot(log10(r0/1e5),log10(T0),'r.-')
ax2.plot(log10(r/1e5),log10(rho),'k-')
ax2.plot(log10(r0/1e5),log10(rho0),'r.-')

plt.show()


#########################################################################################################
Rphot_km=20

# import matplotlib.pyplot as plt

# fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,6))
# ax.set_xlabel('log r')
# ax.set_ylabel('log T')
# ax2.set_xlabel('log r')
# ax2.set_ylabel('log rho')
# ax.set_ylim([7,9])
# ax.axvline(6,linestyle='--')
# ax2.axvline(6,linestyle='--')
# plt.plot()
# # plt.close()
# fvals = linspace(-3.8399,-3.8401,500)
# count=0
# for i,f0 in enumerate(fvals):
    
#     r,rho,T = MakeEnvelope0(f0,Rphot_km)
#     l1 = ax.plot(log10(r),log10(T),'k-',linewidth=1.5,label=('f=%.10f'%f0))
#     l2 = ax.plot(log10(r),log10(T),'k-',linewidth=0.2)
#     l3 = ax2.plot(log10(r),log10(rho),'k-',linewidth=1.5,label=('f=%.10f'%f0))
#     l4 = ax2.plot(log10(r),log10(rho),'k-',linewidth=0.2)
#     ax.legend()
#     ax2.legend()
#     plt.pause(0.01)
#     fig.savefig('png/%06d.png'%i)
#     l1.pop(0).remove()
#     l3.pop(0).remove()

#     if log10(r[-1])<6:
#         count+=1
#         if count==50:
#             break

# f0 = -3.84
# r,rho,T = MakeEnvelope(f0,Rphot_km,Verbose=1)
# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('log r')
# ax.set_ylabel('log T')
# ax.loglog(r,T,'b.-')
# plt.show()