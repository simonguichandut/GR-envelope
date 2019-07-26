''' Main code to calculate expanded envelopes '''

from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
from numpy import linspace, sqrt, log10, array, pi, logspace
from IO import load_params

# --------------------------------------- Constants and parameters --------------------------------------------

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2
sigmarad = 0.25*arad*c

# Parameters
M, RNS, y_inner, comp, save, img = load_params()

if comp == 'He':
    Z=2
    mu_I, mu_e, mu = 4, 2, 4/3
elif comp == 'Ni':
    Z = 28
    mu_I, mu_e = 56, 2
    mu = 1/(1/mu_I + 1/mu_e)

GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0

ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

T_inner = 10**8.5
rg = 2*GM/c**2 # gravitationnal radius


# -------------------------------------------- Microphysics ----------------------------------------------------

def kappa(rho,T):
#     return kappa0/(1.0+(T/4.5e8)**0.86)     
    if rho<0:rho=abs(rho)  
    return kappa0/(1.0+(T/4.5e8)**0.86) + 1e23*Z**2/(mu_e*mu_I)*rho*T**(-7/2)

def cs2_I(T):  # ideal gas sound speed c_s^2 IONS ONLY
    return kB*T/(mu_I*mp)

def Lcr(r,rho,T):
    return 4*pi*c*GM/kappa(rho,T)*(1-rg/r)**(-1/2)

def electrons(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 
    
    if isinstance(rho, (list, tuple, np.ndarray)): # a neat trick for this function to be able to return arrays by making it recursive
        pe,Ue,alpha1,alpha2,f = [[] for i in range(5)]
        for rhoi,ti in zip(rho,T):
            a,b,[c,d,e] = electrons(rhoi,ti)
            for x,y in zip((pe,Ue,alpha1,alpha2,f),(a,b,c,d,e)): x.append(y)
        return pe,Ue,[alpha1,alpha2,f]
    
    else:
        if rho<0:
            return 0,0,[1,0,5/3]
        else:
            rY = rho/mu_e # rho*Ye = rho/mu_e
            pednr = 9.91e12 * (rY)**(5/3)     
            pedr = 1.231e15 * (rY)**(4/3)
            ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
            pend = kB/mp*rY*T
            pe = sqrt(ped**2 + pend**2) # pressure
            
            f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
            Ue = pe/(f-1)               # energy density (erg cm-3)
        
            alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2
            
            return pe,Ue,[alpha1,alpha2,f]

def pressure(rho, T): # ideal gas + radiation pressure (eq 2c)}  PLUS(new)  electron pressure (non-degen + degen)
    pe,_,_ = electrons(rho,T)
    return rho*cs2_I(T) + arad*T**4/3.0 + pe

def Beta_I(rho, T):
    pg = rho*cs2_I(T)
    return pg/pressure(rho,T)

def Beta_e(rho, T):
    pe,_,_ = electrons(rho,T)
    return pe/pressure(rho,T)

def del_ad(rho,T):
    b = Beta_I(rho,T)+Beta_e(rho,T)
    return (8-6*b)/(32 -24*b - 3*b**2)

def del_rad(rho, T, r):
    pe,_,[alpha1,alpha2,f] = electrons(rho,T)
    bi,be = Beta_I(rho, T), Beta_e(rho, T)
    term1 = kappa(rho,T)*Linf/(16*pi*c*GM*(1-bi-be))*Swz(r) + pressure(rho,T)/(rho*c**2)  
    term2 = (1 + (4-1.5*bi-((3*f-4)/(f-1))*be) * pressure(rho,T)/(rho*c**2) )**(-1)    
    return term1*term2    # (term1 is completely dominating, term2 could be removed)

def Del(rho,T, r):
    return min((del_ad(rho,T) , del_rad(rho,T, r)))

# ----------------------------------------- General Relativity ------------------------------------------------

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)**(-0.5)     # NOTE : unlike wind_GR code, there is -1/2 exponent here

def grav(r): # local gravity
    return GM/r**2 * Swz(r)

# ----------------------------------------- Initial conditions ------------------------------------------------

def photosphere(Rphot,f0):

    ''' Finds photospheric density and temperature (eq 9a-b) for a given luminosity-like parameter f0 
        Also sets Linf, the luminosity seen by observers at infinity '''

    def Teff_eq(T): 
        return kappa(0.,T) - (GM*c/(Rphot**2*sigmarad) * Swz(Rphot) * (1-10**f0))/T**4  # Assuming 0 density for opacity, which is no problem at photosphere

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
    rho = 2/3 * mu*mp/(kB*T) * grav(Rphot)/kappa(0.,T) * 10**f0
    Linf = 4*pi*Rphot**2*sigmarad*T**4* (Swz(Rphot)**(-2))

    return rho,T,Linf

# -------------------------------------------- Calculate derivatives ---------------------------------------

def derivs(r, Y):
    ''' Calculates the derivatives of rho and T with r as the independent variable '''
    
    rho,T = Y[:2]
    
    # rho can go <0 in the integration, we can just take abs because the its effects would be small anyway
#    if rho<0:rho=abs(rho)  
    
    P,bi,be = pressure(rho,T) , Beta_I(rho,T) , Beta_e(rho,T)
    pe,_,[alpha1,alpha2,f] = electrons(rho,T)
    delr = del_rad(rho, T, r)

    dP_dr = -GM*rho/r**2 * Swz(r)**2 * (1 + (4-1.5*bi-((3*f-4)/(f-1))*be)*P/(rho*c**2))  # CORRECTED FROM PACZYNSKI (4(4-1.5b))


    drho_dr = rho/P/(bi+be*(alpha1+alpha2*f)) * dP_dr * (1 - (4-3*bi-(4-alpha1)*be)*delr)
    b=bi+be
    drho_dr = rho/P * dP_dr * (1 - (4-3*b)*delr)/b
    dT_dr   = T/P   * dP_dr * delr

    if Del(rho,T,r) == del_ad:
        import sys
        sys.exit('WARNING : CONVECTIVE')

    return [drho_dr, dT_dr]

# ---------------------------------------------- Integration -----------------------------------------------

def Shoot(rspan, rho0, T0, returnResult=False):
    ''' Integrates in from the photosphere, using r as the independent variable, until T=Tbase
        We want to match the location of p=p_inner to the NS radius '''

    inic = [rho0, T0]

#    def hit_innerTemp(r, Y): return Y[1]-T_inner
#    hit_innerTemp.terminal = True # stop integrating at this point 
    
#    def hit_zeroDensity(r, Y): return Y[0]
#    hit_zeroDensity.terminal = False # 

    def hit_innerPressure(r,Y): return pressure(Y[0],Y[1])-P_inner
    hit_innerPressure.terminal = True # stop integrating at this point

    sol = solve_ivp(derivs, rspan, inic, method='Radau', events = hit_innerPressure, dense_output=True)

    return sol

def Error(r): # Evaluate normalized error on location of the surface
    return (r[-1]-RNS*1e5)/(RNS*1e5)

# ------------------------------------------------- Envelope ---------------------------------------------------


def MakeEnvelope(Rphot_km, p=0):    # setup for relaxation method

    global Linf
    Rphot = Rphot_km*1e5
    
    rspan = (Rphot , 1.01*rg)
    Rad,Rho,Temp = [array([]) for i in range(3)]
    
    # First pass to find border values of f
    fvalues = linspace(-3.7,-4,100)
    for i,f0 in enumerate(fvalues):
        rho_phot,T_phot,Linf = photosphere(Rphot,f0)
        solb = Shoot(rspan,rho_phot,T_phot) 
#        print('f=',f0,'success: ',solb.success)
        Eb = Error(solb.t)
        if Eb<0:
            a,b = fvalues[i],fvalues[i-1]
            Ea,sola = Eb,solb
            Eb,solb = Eprev,solprev
            break
        Eprev,solprev = Eb,solb
        
#    r_grid = np.linspace((RNS*1e5)
    def check_convergence(sola,solb,rcheck_prev,tol=1e-3):  
        ''' checks if two solutions have similar parameters rho,T (1 part in 1e4), some small integration distance in 
            if is converged, returns the interpolated value of rho,T at that point                            '''
        d = Rphot/100/(count+1) # 100 meter at a time, reduce by count number of current iteration
        rcheck = rcheck_prev - d
        rhoa,Ta = sola.sol(rcheck)
        rhob,Tb = solb.sol(rcheck)
        
        if abs(rhoa-rhob)/rhoa < tol and abs(Ta-Tb)/Ta < tol:
            return True,rcheck,rhoa,rhob,Ta,Tb
        else:
            return False,
        
    if p!=0: fig = bisection_dynamic_plot_setup(p)
    print('Radius (km) \t Iteration # \t Iteration counter')    
    
    
    # Begin bissection 
    Em=100
    count_iter,count = 0,0
    while abs(Em)>1e-5:
        
        if p!=0:
#            R=linspace(*rspan,5000)
            l1,=plt.semilogy(sola.t/1e5,abs(sola.y[p-1]),'r-')
            l2,=plt.semilogy(solb.t/1e5,solb.y[p-1],'b-')
            mypause(0.001)
        
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
            a,b = 0,1  # a,b don't represent f, but the zone in between rhoa & rhob (and Ta & Tb)
            count_iter+=1
            count = 0
            
            Rad, Rho, Temp = np.append(Rad,rcheck), np.append(Rho,(rhoa+rhob)/2), np.append(Temp,(Ta+Tb)/2)
            
            if p!=0:
                po = rhoa if p==1 else Ta
                plt.semilogy([rcheck/1e5],[po],'k.')
                fig.savefig('png/%06d.png'%count_iter)
            

        print('%.5f \t %d \t\t %d'%(rcheck/1e5,count_iter+1,count+1))
        count+=1

        if count==50:
            break
        if p!=0 and abs(Em)>1e-5:
            l1.set_alpha(0.2)
            l2.set_alpha(0.2)
            l1.set_linewidth(0.5)
            l2.set_linewidth(0.5)

    print('Reached surface!')
    # Fill out arrays    
    Rad,Rho,Temp  = np.insert(Rad,0,Rphot), np.insert(Rho,0,rho_phot), np.insert(Temp,0,T_phot)
    ind = solm.t<Rad[-1]
    Rad,Rho,Temp  = np.append(Rad,solm.t[ind]), np.append(Rho,solm.y[0][ind]), np.append(Temp,solm.y[1][ind])
    return np.flip(Rad),np.flip(Rho),np.flip(Temp),Linf
            


## Extras for plotting
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




#import matplotlib.pyplot as plt
#import matplotlib
#plt.ion()
#fig = plt.figure()
#plt.xlabel(r'r (km)',fontsize=14)
#plt.axvline(RNS,color='g')
#plt.show(block=False)
#
#def mypause(interval):
#    backend = plt.rcParams['backend']
#    if backend in matplotlib.rcsetup.interactive_bk:
#        figManager = matplotlib._pylab_helpers.Gcf.get_active()
#        if figManager is not None:
#            canvas = figManager.canvas
#            if canvas.figure.stale:
#                canvas.draw()
#            canvas.start_event_loop(interval)
#            return
#
#p = 0
#if not p:
#    plt.close('all')
#elif p == 1:
#    plt.ylabel(r'log $\rho$',fontsize=14)
#elif p == 2:
#    plt.ylabel(r'log $T$',fontsize=14)

#Rphot_km = 15
#Rad,Rho,Temp,Linf=MakeEnvelope(Rphot_km,p=p)

##%% Checking errors made
#
#from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # has a derivative method
#
#frho,fT = interp1d(Rad,Rho,kind='cubic'), interp1d(Rad,Temp,kind='cubic')
#frho2,fT2 = IUS(Rad,Rho), IUS(Rad,Temp)
#
## Analytical derivatives
#drho,dT = [],[]
#for r,rho,T in zip(Rad,Rho,Temp):
#    z = derivs(r,[rho,T])
#    drho.append(z[0])
#    dT.append(z[1])
#
## Fit derivatives
#drho2,dT2 = frho2.derivative(),fT2.derivative()
## or finite difference derivative, but grid is non-uniform..
#
## Errors
#relerr_rho = (drho-drho2(Rad))/drho
#relerr_T = (dT-dT2(Rad))/dT
##R=np.linspace(min(Rad),50*1e5,5000)
#
#from matplotlib import gridspec
#fig= plt.figure(figsize=(12,8))
#plt.show()
#fig.suptitle('Solution for %d km photosphere'%Rphot_km,fontsize=15)
#
#gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 2])
#ax = []
#for i in range(6): ax.append(plt.subplot(gs[i]))
#ax1,ax2,ax3,ax4,ax5,ax6 = ax
#
#ax1.set_ylabel(r'log $\rho$ (g cm$^{-3}$)',fontsize=14)
#ax2.set_ylabel(r'log T (K)',fontsize=14)
#ax3.set_ylabel(r'log |$d\rho/dr$|',fontsize=14)
#ax4.set_ylabel(r'log |$dT/dr$|',fontsize=14)
#ax5.set_ylabel('Rel. error (%)',fontsize=14)
#ax6.set_ylabel('Rel. error (%)',fontsize=14)
#ax5.set_xlabel(r'log $r$ (km)',fontsize=14)
#ax6.set_xlabel(r'log $r$ (km)',fontsize=14)
#
#
#x=log10(Rad/1e5)
#ax1.plot(x,log10(Rho),'k-',lw=1.5)
#ax2.plot(x,log10(Temp),'k-',lw=1.5)
#ax3.plot(x,log10(np.abs(drho)),'b.',label='Analytical derivative',ms=6,alpha=0.5)
#ax3.plot(x,log10(np.abs(drho2(Rad))),'k-',lw=1.5,label='Direct derivative')
#ax4.plot(x,log10(np.abs(dT)),'b.',label='Analytical derivative',ms=6,alpha=0.5)
#ax4.plot(x,log10(np.abs(dT2(Rad))),'k-',lw=1.5,label='Direct derivative')
#ax3.legend()
#ax4.legend()
#ax5.plot(x,relerr_rho*100,'k-',lw=1.5)
#ax6.plot(x,relerr_T*100,'k-',lw=1.5)
#
#plt.tight_layout(rect=(0,0,1,0.95))
#plt.show()
#
#
#
#
##%% Many values
#
##R = (11,12,15,20,30,40,50,70,100,150,200,300)
#R = (13,15,20,30,40,50,70,100,150,200)
#N = len(R)
#
##Rads,Rhos,Temps,Linfs = [],[],[],[]
##for r in R:
##    print('\n\n\n%d km \n\n\n'%r)
##    Rad,Rho,Temp,Linf=MakeEnvelope(r,p=2)
##    Rads.append(Rad)
##    Rhos.append(Rho)
##    Temps.append(Temp)
##    Linfs.append(Linf)
##
##Rads,Rhos,Temps,Linfs = array(Rads),array(Rhos),array(Temps),array(Linfs)
#
#
#fig=plt.figure()
#plt.xlabel(r'log $R$ (km)',fontsize=14)
#plt.ylabel(r'log $T$ (K)',fontsize=14)
#plt.ylim([6.5,9])
#plt.xlim([0.8,2.4])
#for i in range(N):
#    if R[i] in (20,200):
#        plt.plot(log10(Rads[i]/1e5),log10(Temps[i]),label='%d km'%R[i])
#plt.legend()
#
#
#plt.figure()
#plt.xlabel(r'log $R$ (km)',fontsize=14)
#plt.ylabel(r'f',fontsize=14)
#farrays = []
#for i in range(N):
#    L = Linfs[i]*(1-rg/Rads[i])**(-1)
#    
#    Linf_pac = LEdd*(1-rg/R[i]/1e5)**(0.5)
#    print(Linf_pac/LEdd)
##    L = Linf_pac*(1-rg/Rads[i])**(-1)
#    
#    Lcrit = 4*pi*c*GM/kappa(Rhos[i],Temps[i])*(1-rg/Rads[i])**(-0.5)
#    f = log10(1-L/Lcrit)
#    plt.plot(log10(Rads[i]/1e5),f,label='%d km'%R[i])
#
#plt.legend()
##plt.ylim([-4.2,-3.7])
#
