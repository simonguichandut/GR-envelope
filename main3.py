''' Main code to calculate expanded envelopes '''

from scipy.optimize import brentq
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
import numpy as np
from numpy import linspace, sqrt, log10, array, pi, logspace
from IO import load_params
import os
from relax import Newton

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

def Lcr(r,T):
    return 4*pi*c*GM/kappa(T)*(1-rg/r)**(-1/2)

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

def photosphere(Rphot,f0):

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

#def derivs2(r, Y):
#    ''' Calculates the second derivatives of rho and T with r as the independent variable '''
#
#    rho,T = Y[:2]
#    P,b = pressure(rho,T) , Beta(rho,T)
#    delr = del_rad(rho, T, r)
#    dP_dr = -GM*rho/r**2 * Swz(r)**2 * (1 + 4*(4-1.5*b)*P/(rho*c**2))
#    drho_dr = rho/P * dP_dr * (1 - (4-3*b)*delr)/b
#    dT_dr   = T/P   * dP_dr * delr
#
#    dbeta_dr = Beta(rho,T) * (1/T*dT_dr + 1/rho*drho_dr - 1/P*dP_dr)
#    dkappa_dr = -kappa(T)**2/kappa0 * 0.86*(4.5e8)**(-0.86)*T**(-0.14)
#    alpha = kappa(T)*Linf/(16*pi*GM*c*(1-Beta(rho,T)))
#    ddelr_dr = delr/(alpha*Swz(r) + P/(rho*c**2))


# ---------------------------------------------- Integration -----------------------------------------------

def Shoot(rspan, rho0, T0, returnResult=False):
    ''' Integrates in from the photosphere, using r as the independent variable, until T=Tbase
        We want to match the location of p=p_inner to the NS radius '''

    inic = [rho0, T0]

    def hit_innerTemp(r, Y): return Y[1]-T_inner
    hit_innerTemp.terminal = True # stop integrating at this point 

    sol = solve_ivp(derivs, rspan, inic, method='Radau', events = hit_innerTemp, dense_output=True)

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
        
    # Begin bissection 
    Em=100
    count_iter,count = 0,0
    while abs(Em)>1e-5:
        
        if p!=0:
            R=linspace(*rspan,5000)
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
            
                
        print(count_iter,count)
        count+=1

        if count==50:
            break
        if p!=0 and abs(Em)>1e-5:
            l1.set_alpha(0.2)
            l2.set_alpha(0.2)
            l1.set_linewidth(0.5)
            l2.set_linewidth(0.5)

        
        
    # Fill out arrays    
    Rad,Rho,Temp  = np.insert(Rad,0,Rphot), np.insert(Rho,0,rho_phot), np.insert(Temp,0,T_phot)
    ind = solm.t<Rad[-1]
    Rad,Rho,Temp  = np.append(Rad,solm.t[ind]), np.append(Rho,solm.y[0][ind]), np.append(Temp,solm.y[1][ind])
    return np.flip(Rad),np.flip(Rho),np.flip(Temp),Linf
            
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
fig = plt.figure()
plt.xlabel(r'r (km)',fontsize=14)
plt.axvline(10,color='g')
plt.show(block=False)

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

p = 0
if not p:
    plt.close('all')
elif p == 1:
    plt.ylabel(r'log $\rho$',fontsize=14)
elif p == 2:
    plt.ylabel(r'log $T$',fontsize=14)

Rphot_km = 12
Rad,Rho,Temp,Linf=MakeEnvelope(Rphot_km,p=p)

#%% Checking errors made

from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # has a derivative method

frho,fT = interp1d(Rad,Rho,kind='cubic'), interp1d(Rad,Temp,kind='cubic')
frho2,fT2 = IUS(Rad,Rho), IUS(Rad,Temp)

# Analytical derivatives
drho,dT = [],[]
for r,rho,T in zip(Rad,Rho,Temp):
    z = derivs(r,[rho,T])
    drho.append(z[0])
    dT.append(z[1])

# Fit derivatives
drho2,dT2 = frho2.derivative(),fT2.derivative()
# or finite difference derivative, but grid is non-uniform..

# Errors
relerr_rho = (drho-drho2(Rad))/drho
relerr_T = (dT-dT2(Rad))/dT
#R=np.linspace(min(Rad),50*1e5,5000)

from matplotlib import gridspec
fig= plt.figure(figsize=(12,8))
plt.show()
fig.suptitle('Solution for %d km photosphere'%Rphot_km,fontsize=15)

gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 2])
ax = []
for i in range(6): ax.append(plt.subplot(gs[i]))
ax1,ax2,ax3,ax4,ax5,ax6 = ax

ax1.set_ylabel(r'log $\rho$ (g cm$^{-3}$)',fontsize=14)
ax2.set_ylabel(r'log T (K)',fontsize=14)
ax3.set_ylabel(r'log |$d\rho/dr$|',fontsize=14)
ax4.set_ylabel(r'log |$dT/dr$|',fontsize=14)
ax5.set_ylabel('Rel. error (%)',fontsize=14)
ax6.set_ylabel('Rel. error (%)',fontsize=14)
ax5.set_xlabel(r'log $r$ (km)',fontsize=14)
ax6.set_xlabel(r'log $r$ (km)',fontsize=14)


x=log10(Rad/1e5)
ax1.plot(x,log10(Rho),'k-',lw=1.5)
ax2.plot(x,log10(Temp),'k-',lw=1.5)
ax3.plot(x,log10(np.abs(drho)),'b.',label='Analytical derivative',ms=6,alpha=0.5)
ax3.plot(x,log10(np.abs(drho2(Rad))),'k-',lw=1.5,label='Direct derivative')
ax4.plot(x,log10(np.abs(dT)),'b.',label='Analytical derivative',ms=6,alpha=0.5)
ax4.plot(x,log10(np.abs(dT2(Rad))),'k-',lw=1.5,label='Direct derivative')
ax3.legend()
ax4.legend()
ax5.plot(x,relerr_rho*100,'k-',lw=1.5)
ax6.plot(x,relerr_T*100,'k-',lw=1.5)

plt.tight_layout(rect=(0,0,1,0.95))
plt.show()




#%% Many values

R = (11,12,15,20,30,40,50,70,100,150,200,300)
N = len(R)
#
#Rads,Rhos,Temps,Linfs = [],[],[],[]
#for r in R:
#    print('\n\n\n%d km \n\n\n'%r)
#    Rad,Rho,Temp,Linf=MakeEnvelope(r,p=0)
#    Rads.append(Rad)
#    Rhos.append(Rho)
#    Temps.append(Temp)
#    Linfs.append(Linf)

Rads,Rhos,Temps,Linfs = array(Rads),array(Rhos),array(Temps),array(Linfs)


fig=plt.figure()
plt.xlabel(r'log $R$ (km)',fontsize=14)
plt.ylabel(r'log $T$ (K)',fontsize=14)
plt.ylim([6.5,9])
plt.xlim([0.8,2.4])
for i in range(N):
    if R[i] in (20,200):
        plt.plot(log10(Rads[i]/1e5),log10(Temps[i]),label='%d km'%R[i])
plt.legend()


plt.figure()
plt.xlabel(r'log $R$ (km)',fontsize=14)
plt.ylabel(r'f',fontsize=14)
farrays = []
for i in range(N):
    L = Linfs[i]*(1-rg/Rads[i])**(-1)
    
    Linf_pac = LEdd*(1-rg/R[i]/1e5)**(0.5)
    print(Linf_pac/LEdd)
#    L = Linf_pac*(1-rg/Rads[i])**(-1)
    
    Lcrit = 4*pi*c*GM/kappa(Temps[i])*(1-rg/Rads[i])**(-0.5)
    f = log10(1-L/Lcrit)
    plt.plot(log10(Rads[i]/1e5),f,label='%d km'%R[i])

plt.legend()
#plt.ylim([-4.2,-3.7])





#%% Radius-f space 
plt.close('all')

n=100
R = logspace(1,2.4,n)*1e5
f = linspace(-6,-1,n)
X,Y = np.meshgrid(R,f)

Z = [[] for i in range(n)]
for i,fi in enumerate(f):
    for r in R:
        rho,T,Linf = photosphere(r,fi)
        Z[i].append(Linf/LEdd)

plt.figure()
plt.xlabel(r'log R$_{ph}$ (km)',fontsize=14)
plt.ylabel(r'log(1-L/L$_{cr}$)',fontsize=14)

plt.pcolormesh(log10(X/1e5),Y,Z,cmap='Greys',vmax=1.05)
cbar=plt.colorbar()
cbar.ax.set_title(r'L$_\infty$/LEdd')

# Draw exact line where Linf>LEdd
def foo(r,f):
    rho,T,Linf = photosphere(r,f)
    return Linf/LEdd-1

r_crit = []
for i,fi in enumerate(f):
    if max(Z[i])>1:
        r_crit.append(brentq(foo,R[0],R[-1],args=(fi,)))
x,y = log10(np.append(array(r_crit),R[-1])/1e5) , np.append(f[:len(r_crit)] , f[len(r_crit)-1])
plt.plot(x,y,'r--',label=r'L$_\infty$/LEdd=1')

# Data from Paczynski and Anderson
data = [[1.040808582062322, -3.730633802816901],
[1.0790702765817706, -3.734507042253521],
[1.3004013719623444, -3.755281690140845],
[1.1742100270013867, -3.744014084507042],
[1.4762314821571922, -3.7714788732394364],
[1.6024228271181495, -3.7827464788732392],
[1.696548201123842, -3.7911971830985913],
[1.8423994745676135, -3.8038732394366197],
[1.9985915492957749, -3.817605633802817],
[2.174436254834709, -3.8330985915492954],
[2.2985696562796476, -3.843661971830986]]

plt.plot(array(data)[:,0],array(data)[:,1],'b.',label='Paczynski & Anderson data points')

plt.legend(loc=2)


plt.text(1.3,-5.5,'static',color='w',fontsize=13)
plt.text(1.9,-5.5,'outflow',color='w',fontsize=13)

