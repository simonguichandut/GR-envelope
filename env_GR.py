''' Main code to calculate expanded envelopes '''

import sys
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from collections import namedtuple
import numpy as np
import IO
import physics

# --------------------------------------- Constants and parameters --------------------------------------------

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2
sigmarad = 0.25*arad*c

# Parameters
params = IO.load_params()
if params['FLD'] == True: 
    sys.exit('This script is for pure optically thick calculations')

# Generate EOS class and methods
eos = physics.EOS(params['comp'])

# Mass-dependent parameters
M,RNS,y_inner = params['M'],params['R'],params['y_inner']
GM = 6.6726e-8*2e33*M
LEdd = 4*np.pi*c*GM/eos.kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

T_inner = 10**8.5
rg = 2*GM/c**2 # gravitationnal radius

# ----------------------------------------- General Relativity ------------------------------------------------

def Swz(r):  # Schwartzchild metric term
    # return (1-2*GM/c**2/r)**(-0.5)     # NOTE : unlike wind_GR code, there is -1/2 exponent here
    return (1-2*GM/c**2/r)     

def grav(r): # local gravity
    return GM/r**2 * Swz(r)**(-1/2)

def Lcr(r,rho,T):
    return 4*np.pi*c*GM/eos.kappa(rho,T)*(1-rg/r)**(-1/2)

# ----------------------------------------- Initial conditions ------------------------------------------------

def photosphere(Rphot,f0):

    ''' Finds photospheric density and temperature (eq 9a-b) for a given luminosity-like parameter f0 
        Also sets Linf, the luminosity seen by observers at infinity '''

    def Teff_eq(T): 
        return eos.kappa(0.,T) - (GM*c/(Rphot**2*sigmarad) * Swz(Rphot)**(-1/2) * (1-10**f0))/T**4  # Assuming 0 density for opacity, which is no problem at photosphere

    Tkeep1, Tkeep2 = 0.0, 0.0
    npoints = 10
    while Tkeep1 == 0 or Tkeep2 == 0:
        logT = np.linspace(6, 8, npoints)
        for T in 10**logT:
            foo = Teff_eq(T)
            if foo < 0.0:
                Tkeep1 = T
            if foo > 0.0 and Tkeep2 == 0:
                Tkeep2 = T
        npoints += 10

    T = brentq(Teff_eq, Tkeep1, Tkeep2, xtol=1e-10, maxiter=10000)
    rho = 2/3 * eos.mu*mp/(kB*T) * grav(Rphot)/eos.kappa(0.,T) * 10**f0          # paczynski and anderson  (assuming zero density for the opacity)
    # rho = 3 * mu*mp/(kB*T) * grav(Rphot)/kappa(0.,T) * 10**f0           # if tau photosphere = 3
    Linf = 4*np.pi*Rphot**2*sigmarad*T**4* Swz(Rphot)

    return rho,T,Linf

# ------------------------------------ Paczynski and Anderson derivatives --------------------------------------------

def del_ad(rho,T):
    b = eos.Beta(rho,T)
    return (8-6*b)/(32 -24*b - 3*b**2)

def del_rad(rho, T, r, Linf):
    pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
    bi,be = eos.Beta_I(rho, T), eos.Beta_e(rho, T)
    term1 = eos.kappa(rho,T)*Linf/(16*np.pi*c*GM*(1-bi-be))*Swz(r)**(-1/2) + eos.pressure_e(rho,T)/(rho*c**2)  
    term2 = (1 + (4-1.5*bi-((3*f-4)/(f-1))*be) * eos.pressure_e(rho,T)/(rho*c**2) )**(-1)    
    return term1*term2    # (term1 is completely dominating, term2 could be removed)

def Del(rho,T, r):
    return min((del_ad(rho,T) , del_rad(rho,T, r, Linf)))



# --------------------------------------- Wind structure equations ---------------------------------------

def Y(r): 
    return np.sqrt(Swz(r)) # v=0

def Tstar(L, T, r, rho):  
    return L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/(4*r) *\
            3*rho/(arad*T**4) * Y(r)**(-1)

def A_e(rho,T):  
    pe,_,[alpha1,_,f] = eos.electrons(rho,T)
    return 1 + 1.5*eos.cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

def B_e(rho,T): 
    pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
    return eos.cs2_I(T) + pe/rho*(alpha1 + alpha2*f)

def C_e(L, T, r, rho):  
    _,_,[alpha1,_,_] = eos.electrons(rho,T)
    bi,be = eos.Beta_I(rho, T), eos.Beta_e(rho, T)

    return Tstar(L, T, r, rho) * \
            ((4-3*bi-(4-alpha1)*be)/(1-bi-be)) * arad*T**4/(3*rho)


# -------------------------------------------- Calculate derivatives ---------------------------------------

# def derivs(r, Y):
#     ''' Calculates the derivatives of rho and T with r as the independent variable '''
    
#     rho,T = Y[:2]
    
#     # rho can go <0 in the integration, we can just take abs because the its effects would be small anyway
#     if rho<0: rho=1e-10  
    
#     P,bi,be = eos.pressure_e(rho,T) , eos.Beta_I(rho,T) , eos.Beta_e(rho,T)
#     pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
#     delr = del_rad(rho, T, r)

#     dP_dr = -GM*rho/r**2 * Swz(r)**(-1) * (1 + (4-1.5*bi-((3*f-4)/(f-1))*be)*P/(rho*c**2))  # CORRECTED FROM PACZYNSKI (4(4-1.5b))


#     drho_dr = rho/P/(bi+be*(alpha1+alpha2*f)) * dP_dr * (1 - (4-3*bi-(4-alpha1)*be)*delr)
#     b=bi+be
#     drho_dr = rho/P * dP_dr * (1 - (4-3*b)*delr)/b
#     dT_dr   = T/P   * dP_dr * delr

#     # if Del(rho,T,r) == del_ad:
#     #     import sys
#     #     sys.exit('WARNING : CONVECTIVE')

#     return [drho_dr, dT_dr]

def derivs(r,Y):

    # Version with the wind equations (with A,B,C) and v=0
    rho,T = Y[:2]
    
    if rho<0:   
        rho=1e-10

    L = Linf*Swz(r)**(-1)

    # pe,Ue,[alpha1,alpha2,f] = eos.electrons(rho,T)
    # bi,be = eos.Beta_I(rho,T), eos.Beta_e(rho,T)

    # Ae = 1 + 1.5*cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)
    # Be = cs2_I(T) + pe/rho*(alpha1 + alpha2*f)
    # Ce = Swz(r)**(-1/2) * L/LEdd * kappa(rho,T)/kappa0 * GM/r * \
    #     (1 + (bi + alpha1*be)/(4*(1-bi-be)))
    # Tstar = L/LEdd * kappa(rho,T)/kappa0 * GM/(4*r) * 3*rho/(arad*T**4) 

    # dlnT_dlnr = -Tstar-GM*Swz(r)**(-1)/(c**2*r)
    # dlnrho_dlnr = (-GM*Swz(r)**(-1)/r*Ae + Ce)/Be

    # dT_dr = T/r * dlnT_dlnr
    # drho_dr = rho/r * dlnrho_dlnr

    # return [drho_dr,dT_dr]

    dlnT_dlnr = -Tstar(L, T, r, rho) - 1/Swz(r) * GM/c**2/r
    dlnrho_dlnr = (-GM/Swz(r)/r * A_e(rho,T) + C_e(L,T,r,rho))/B_e(rho,T)
    
    dT_dr = T/r * dlnT_dlnr
    drho_dr = rho/r * dlnrho_dlnr

    return [drho_dr,dT_dr]


# ---------------------------------------------- Integration -----------------------------------------------

def Shoot(rspan, rho0, T0, returnResult=False):
    ''' Integrates in from the photosphere, using r as the independent variable, until T=Tbase
        We want to match the location of p=p_inner to the NS radius '''

    inic = [rho0, T0]

#    def hit_innerTemp(r, Y): return Y[1]-T_inner
#    hit_innerTemp.terminal = True # stop integrating at this point 
    
#    def hit_zeroDensity(r, Y): return Y[0]
#    hit_zeroDensity.terminal = False # 

    def hit_innerPressure(r,Y): return eos.pressure_e(Y[0],Y[1])-P_inner
    hit_innerPressure.terminal = True # stop integrating at this point

    sol = solve_ivp(derivs, rspan, inic, method='Radau', events = hit_innerPressure, dense_output=True)

    return sol

def Error(r): # Evaluate normalized error on location of the base versus NS radius
    return (r[-1]-RNS*1e5)/(RNS*1e5)

# ------------------------------------------------- Envelope ---------------------------------------------------

Env = namedtuple('Env',
            ['rphot','Linf','r','T','rho'])

def MakeEnvelope(Rphot_km, tol=1e-4, return_stuff=False): 

    global Linf             # that way Linf does not have to always be a function input parameter
    Rphot = Rphot_km*1e5
    
    rspan = (Rphot , 1.01*rg)                       # rspan is the integration range
    Rad,Rho,Temp = [np.array([]) for i in range(3)] 

    stuff = [] # store items of the bisection method for the demo plot      
    
    # First pass to find border values of f, and their solutions sola (gives r(y8)<RNS) and solb (r(y8)>RNS)

    # For photospheres at least > RNS+1km, f is not bigger than -3.5
    if Rphot_km>=RNS+1:
        fvalues = np.linspace(-3.5,-4.5,500)

    # But apparently for ultra compact ones, say a hundred meters or less above the surface, f is much bigger
    else:
        fvalues = np.linspace(-1e-4,-4, 100)

    for i,f0 in enumerate(fvalues):
        rho_phot,T_phot,Linf = photosphere(Rphot,f0)
        solb = Shoot(rspan,rho_phot,T_phot) 
        Eb = Error(solb.t)
        print('f=',f0,'\t success: ',solb.success,'\t error:',Eb)
        if Eb<0:
            print('f=',f0,'\t suc   cess: ',solb.success,'\t error:',Eb)
            a,b = fvalues[i],fvalues[i-1]
            Ea,sola = Eb,solb
            Eb,solb = Eprev,solprev
            break
        Eprev,solprev = Eb,solb

    if return_stuff:
        stuff.append([a,sola,b,solb])
        stuff.append([]) # will store the intermediate solutions into this list

        
    def check_convergence(sola,solb,rcheck_prev):  
        ''' Checks if two solutions have similar parameters rho,T (1 part in tol^-1), some small integration distance in. 
            If converged, returns the interpolated value of rho,T at that point                            '''
        d = Rphot/100/(count+1) # 1% of photosphere at a time, reduce by count number of current iteration
        rcheck = rcheck_prev - d
        rhoa,Ta = sola.sol(rcheck)
        rhob,Tb = solb.sol(rcheck)
        
        if abs(rhoa-rhob)/rhoa < tol and abs(Ta-Tb)/Ta < tol:
            return True,rcheck,rhoa,rhob,Ta,Tb
        else:
            return False,
        

    print('Radius (km) \t Step # \t Iter count \t RNS error')    
    
    
    # Begin bisection 
    Em=100
    count_iter,count = 0,0
    rhoa,rhob,Ta,Tb = [0 for i in range(4)]

    while abs(Em)>tol:   # we can stop when the final radius is the neutron star radius close to one part in 10^5
        
        # middle point.  In the first integration, a&b are the f values.  In the rest, a&b are between 0 and 1. For interpolation
        m = (a+b)/2
    
        if count_iter == 0:  # in the first iteration, a&b represent f
            rho_phot,T_phot,Linf = photosphere(Rphot,m)
            solm = Shoot(rspan,rho_phot,T_phot)
            rcheck = Rphot
            
        else:                # in the other iterations, a&b represent the linear space in [rhoa,rhob] and [Ta,Tb]
            rhom,Tm = rhoa + m*(rhob-rhoa) , Ta + m*(Tb-Ta)
            solm = Shoot(rspan,rhom,Tm)
            
        # Bisection : check which side the new solution lands on and update either a or b
        Em=Error(solm.t)
        if Ea*Em>0:
            a,sola = m,solm
        else:
            b,solb = m,solm


        conv = check_convergence(sola,solb,rcheck)
        # When the two solutions are converging on rho and T, move the starting point inwards and reset a & b
        if conv[0] is True:
            rcheck,rhoa,rhob,Ta,Tb = conv[1:]
            rspan = (rcheck,1.01*rg)
            a,b = 0,1  
            count_iter+=1  # update step counter
            count = 0      # reset iteration counter
            
            Rad, Rho, Temp = np.append(Rad,rcheck), np.append(Rho,(rhoa+rhob)/2), np.append(Temp,(Ta+Tb)/2)
        

            if return_stuff:
                stuff[1].extend((sola,solb))


        # End of step 
        print('%.5f \t %d \t\t %d \t\t %.6e'%(rcheck/1e5,count_iter,count+1,Em))
        count+=1

        # Exit if stuck at a step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not arrive at the neutron star radius! Exiting after being stuck at the same step for %d iterations"%nitermax)


    # Reached precision criteria for error on radius
    print('Reached surface at r=%.5f km!'%(solm.t[-1]/1e5))

    # Fill out arrays    
    Rad,Rho,Temp  = np.insert(Rad,0,Rphot), np.insert(Rho,0,rho_phot), np.insert(Temp,0,T_phot)
    ind = solm.t<Rad[-1]
    Rad,Rho,Temp  = np.append(Rad,solm.t[ind]), np.append(Rho,solm.y[0][ind]), np.append(Temp,solm.y[1][ind])
    # return np.flip(Rad),np.flip(Rho),np.flip(Temp),Linf
    
    r,T,rho = np.flip(Rad),np.flip(Temp),np.flip(Rho)


    if return_stuff:
        return Env(Rphot,Linf,r,T,rho),stuff
    else:
        return Env(Rphot,Linf,r,T,rho)
            











"""
''' Copy of MakeEnvelope but with dynamic plotting to see the bisection method in action  '''

import matplotlib
import matplotlib.pyplot as plt

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def bisection_dynamic_plot_setup(p):
    plt.close('all')
    plt.ion()
    fig = plt.figure()
    plt.xlabel(r'r (km)',fontsize=14)
    plt.axvline(RNS,color='g')
    plt.show(block=False)

    if p == 1:
        plt.ylabel(r'log $\rho$',fontsize=14)
    elif p == 2:
        plt.ylabel(r'log $T$',fontsize=14)
        
    return fig


def MakeEnvelope_plot(Rphot_km, p, tol=1e-4):   

    global Linf
    Rphot = Rphot_km*1e5
    
    rspan = (Rphot , 1.01*rg)                       
    Rad,Rho,Temp = [np.array([]) for i in range(3)]       
    
    fvalues = np.linspace(-3.7,-4.5,100)
    for i,f0 in enumerate(fvalues):
        rho_phot,T_phot,Linf = photosphere(Rphot,f0)
        solb = Shoot(rspan,rho_phot,T_phot) 
        Eb = Error(solb.t)
        # print('f=',f0,'\t success: ',solb.success,'\t error:',Eb)
        if Eb<0:
            a,b = fvalues[i],fvalues[i-1]
            Ea,sola = Eb,solb
            Eb,solb = Eprev,solprev
            break
        Eprev,solprev = Eb,solb
        
    def check_convergence(sola,solb,rcheck_prev):  
        ''' Checks if two solutions have similar parameters rho,T (1 part in tol^-1), some small integration distance in. 
            If converged, returns the interpolated value of rho,T at that point                            '''
        d = Rphot/100/(count+1) 
        rcheck = rcheck_prev - d
        rhoa,Ta = sola.sol(rcheck)
        rhob,Tb = solb.sol(rcheck)
        
        if abs(rhoa-rhob)/rhoa < tol and abs(Ta-Tb)/Ta < tol:
            return True,rcheck,rhoa,rhob,Ta,Tb
        else:
            return False,
        
    fig = bisection_dynamic_plot_setup(p)
    print('Radius (km) \t Step # \t Iteration counter')    
    
    # Begin bissection 
    Em=100
    count_iter,count = 0,0
    rhoa,rhob,Ta,Tb = [0 for i in range(4)]
    while abs(Em)>1e-5:   # we can stop when the final radius is the neutron star radius close to one part in 10^5
        
        l1,=plt.semilogy(sola.t/1e5,abs(sola.y[p-1]),'r-')
        l2,=plt.semilogy(solb.t/1e5,solb.y[p-1],'b-')
        mypause(0.001)
        
        # middle point.  In the first integration, a&b are the f values.  In the rest, a&b are between 0 and 1. For interpolation
        m = (a+b)/2
    
        if count_iter == 0:  # in the first iteration, a&b represent f
            rho_phot,T_phot,Linf = photosphere(Rphot,m)
            solm = Shoot(rspan,rho_phot,T_phot)
            rcheck = Rphot
            
        else:
            rhom,Tm = rhoa + m*(rhob-rhoa) , Ta + m*(Tb-Ta)
            solm = Shoot(rspan,rhom,Tm)
            
            
        Em=Error(solm.t)
        if Ea*Em>0:
            a,sola = m,solm
        else:
            b,solb = m,solm
        conv = check_convergence(sola,solb,rcheck)
        
        
        if conv[0]:
            rcheck,rhoa,rhob,Ta,Tb = conv[1:]
            rspan = (rcheck,1.01*rg)
            a,b = 0,1 
            count_iter+=1
            count = 0
            
            Rad, Rho, Temp = np.append(Rad,rcheck), np.append(Rho,(rhoa+rhob)/2), np.append(Temp,(Ta+Tb)/2)
            
            po = rhoa if p==1 else Ta
            plt.semilogy([rcheck/1e5],[po],'k.')
            # fig.savefig('png/%06d.png'%count_iter)
            
        print('%.5f \t %d \t\t %d'%(rcheck/1e5,count_iter,count+1))
        count+=1

        nitermax=200
        if count==nitermax:
            sys.exit("Could not arrive at the neutron star radius! Exiting after being stuck at the same step for %d iterations"%nitermax)

        if abs(Em)>1e-5:
            l1.set_alpha(0.2)
            l2.set_alpha(0.2)
            l1.set_linewidth(0.5)
            l2.set_linewidth(0.5)

    print('Reached surface at r=%.5f!'%solm.t[-1]/1e5)
    l1,=plt.semilogy(solm.t/1e5,abs(solm.y[p-1]),'k-')
    # fig.savefig('png/%06d.png'%count_iter)

    # Fill out arrays    
    Rad,Rho,Temp  = np.insert(Rad,0,Rphot), np.insert(Rho,0,rho_phot), np.insert(Temp,0,T_phot)
    ind = solm.t<Rad[-1]
    Rad,Rho,Temp  = np.append(Rad,solm.t[ind]), np.append(Rho,solm.y[0][ind]), np.append(Temp,solm.y[1][ind])
    return np.flip(Rad),np.flip(Rho),np.flip(Temp),Linf

"""