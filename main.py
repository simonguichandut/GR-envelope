''' Main code to calculate expanded envelopes '''

from scipy.optimize import brentq
from scipy.integrate import odeint
from scipy.interpolate import interp1d
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
rmin = 1e5

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
    # print(Teff_eq(T))
    rho = 2/3 * mu*mp/(kB*T) * grav(Rphot)/kappa(T) * 10**f0
    Linf = 4*pi*Rphot**2*sigmarad*T**4* (Swz(Rphot)**(-2))
    # Linf = (1-10**f0) * 4*pi*c*GM/kappa(T) * Swz(Rphot)**(-1)
    # Linf = 4*pi*c*GM/kappa(T)*Swz(Rphot)**(-1)
    # Linf = LEdd*Swz(Rphot)**(-1)

    return rho,T,Linf

# -------------------------------------------- Calculate derivatives ---------------------------------------

def dr(inic, r):
    ''' Calculates the derivatives of rho and T with r as the independent variable '''

    rho,T = inic[:2]
    P,b = pressure(rho,T) , Beta(rho,T)
    delr = del_rad(rho, T, r)

    dP_dr = -GM*rho/r**2 * Swz(r)**2 * (1 + 4*(4-1.5*b)*P/(rho*c**2))

    drho_dr = rho/P * dP_dr * (1 - (4-3*b)*delr)/b
    dT_dr   = T/P   * dP_dr * delr

    # s1 = '+' if drho_dr>0 else '-'
    # s2 = '+' if dT_dr>0 else '-'
    # print(s1,s2)

    if Del(rho,T,r) == del_ad: print('WARNING : CONVECTIVE')

    return [drho_dr, dT_dr]

# ---------------------------------------------- Integration -----------------------------------------------

def innerIntegration(r, rhophot, Tphot, returnResult=False):
    ''' Integrates in from the photosphere, using r as the independent variable, until rho=rhomax
        We want to match the location of p=p_inner to the NS radius '''

    if verbose:
        print('\n**** Running innerIntegration ****')

    inic = [rhophot, Tphot]
    # with stdout_redirected(): 
    #     result,info = odeint(dr, inic, r, atol=1e-10, 
    #                     rtol=1e-10,full_output=True)  # contains rho(r) and T(r)
    result,info = odeint(dr, inic, r, full_output=True)  # contains rho(r) and T(r)
        # (we can stop integrating when rho>1e10 , but need to change ode method)
    # print(info)

    flag = 0

    # Removing NaNs
    nanflag=0
    if True in np.isnan(result):
        nanflag=1
        firstnan = min([np.argwhere(np.isnan(result[:, i]) == True)[0][0]
                        for i in (0, 1) if True in np.isnan(result[:, i])])
        result = result[:firstnan]
        r = r[:firstnan]
        if verbose:
            print('Inner integration : NaNs reached after r = %.2e'%r[firstnan-1])

    # INNER B.C   :  PRESSURE
#     rho,T = result[:, 0], result[:, 1]
#     P = pressure(rho,T)

#     # Checking that we reached surface pressure
#     if P[-1]<P_inner:
#         flag = 1
#         if verbose:
#             if nanflag: print('Surface pressure never reached (NaNs before reaching p_inner)')
#             else:       print('Surface pressure never reached (max rho too small)')

#     else: # linear interpolation to find the exact radius where P=P_inner
#         x = np.argmin(np.abs(P_inner-P))
#         a,b = (x,x+1) if P_inner-P[x] > 0 else (x-1,x)
#         func = interp1d([P[a],P[b]],[r[a],r[b]])
#         RNS_calc = func(P_inner)

#         result = result[:b]

# #        print(RNS_calc)
# #        import matplotlib.pyplot as plt
# #        plt.figure()
# #        plt.plot(r,P,'k.-')
# #        plt.plot([r[0],r[-1]],[P_inner,P_inner],'k--')
# #        plt.plot([r[a],r[b]],[P[a],P[b]],'ro')
# #        plt.plot([RNS_calc],[P_inner],'bo')

    # INNER B.C   :  TEMPERATURE
    rho,T = result[:,0],result[:, 1]

    # Checking that we reached surface temperature
    if max(T)<T_inner:
        flag = 1
        if verbose:
            if nanflag: print('Surface temperature never reached (NaNs before reaching T_inner)')
            else:       print('Surface temperature never reached (min r too big)')
            print('Max temperature reached %.3e K at radius %.1f'%(max(T),r[T==max(T)]/1e5))

    else: # linear interpolation to find the exact radius where P=P_inner
        x = np.argmin(np.abs(T_inner-T))
        a,b = (x,x+1) if T_inner-T[x] > 0 else (x-1,x)
        func = interp1d([log10(T[a]),log10(T[b])],[log10(r[a]),log10(r[b])],kind='linear')
        RNS_calc = 10**func(log10(T_inner))

        result = result[:b+1]
        r = r[:b+1]
        rho,T = result[:,0],result[:, 1]

        import matplotlib.pyplot as plt
        # plt.figure()

        # plt.plot(log10(r/1e5),log10(abs(T)),'k.-')
        # plt.plot(log10([r[0]/1e5,r[-1]/1e5]),log10([T_inner,T_inner]),'k--')
        # plt.plot(log10([r[a]/1e5,r[b]/1e5]),log10([T[a],T[b]]),'ro')
        # plt.plot(log10([RNS_calc/1e5]),log10([T_inner]),'bo')
        # plt.xlim([0.8,2.4])
        # plt.ylim([6.5,9])

        # plt.loglog(rho,T)
        # plt.plot(log10(r/1e5),log10(rho),'k.-')
        # plt.ylim([-1,1])

        # plt.show()


    if returnResult:
        return result
    else:
        if flag:
            return +100
        else:
            return (RNS_calc/1e5-RNS)/RNS       # Boundary error #2




# ------------------------------------------------- Envelope ---------------------------------------------------

def MakeEnvelope(f0, Rphot_km, mode='rootsolve', Verbose=0):

    ''' Obtaining the wind solution for set of parameters Edot/LEdd and log10(Ts).
        The modes are rootsolve : not output, just obtaining the boundary errors, 
        and wind : obtain the full solutions.   '''

    global Rphot,Linf,verbose
    Rphot,verbose = Rphot_km*1e5 , Verbose

    if mode == 'rootsolve':
        
        # Star by obtaining photospheric conditions
        rho_phot,T_phot,Linf = photosphere(f0)
        # print(Linf/LEdd)
        if verbose: 
            print('Photospheric temperature : %.5f (log)'%log10(T_phot))
            print('Photospheric density : %.5f (log)'%log10(rho_phot))

        # Error is given by the outer luminosity
        # r = np.linspace(Rphot, 3*rmin, 1000)
        r = np.logspace(log10(Rphot), log10(5.1*rmin) , 1e4)
        error = innerIntegration(r,rho_phot,T_phot)

        return error 

    elif mode == 'envelope':  # Same thing but calculate variables and output all of the arrays
        pass



# Testing

# # test : In P&A, for photosphere at 200km, the solution has f0=-3.85.  That gives a photosphere temperature of 10**6.7
# Or 20 km, f0 = -3.76, T=10**7.3
# global Rphot
# Rphot = 200*1e5 
# f0 = -3.85
# Rphot = 20*1e5 
# f0 = -3.76
# rho,T,_ = photosphere(f0)
# print(log10(T),log10(rho))
# # It works!!

Rphot_km = 20
# f0 = -3.76
# err=MakeEnvelope(f0,Rphot_km,Verbose=1)
errs=[]
# for f0 in linspace(-3.9,-3.7,50):
# for f0 in linspace(-3.8401,-3.8399,100):
# #     # print('\nf0=',f0)
#     err=MakeEnvelope(f0,Rphot_km,Verbose=0)
#     print(f0,err)
#     errs.append(err)
# print(errs)

f1,out = brentq(MakeEnvelope,-3.8401,-3.8399,args=(Rphot_km,),rtol=1e-15,full_output=True)
print(f1,MakeEnvelope(f1,Rphot_km))
# print(out)



# print(MakeEnvelope(-3.8400216295448280,Rphot_km))


# We need to rootfind to a riddiculous precision. We'll break up the number f in 10 digit parts
import mpmath
mpmath.mp.dps = 100

def MakeEnvelope2(f0):      # because mpmath.findroot has no option to add function args
    return MakeEnvelope(f0, Rphot_km, mode='rootsolve', Verbose=0)

mpmath.findroot(MakeEnvelope2,f1,solver='secant',verbose=True)