''' Checking errors on solutions '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # has a derivative method
from scipy.interpolate import interp1d

from IO import get_phot_list,read_from_file
import env_GR
Rphotkms = get_phot_list()


def plot_stuff(radius,rho_points,T_points,rho_func,T_func,drho_points,dT_points,title):
    '''rho_points and T_points are the actual points from the solution (same for drho and dT)
       rho_func and T_func are some kind of fit of the date, like a spline, but they HAVE TO have a .derivative method()'''

    fig= plt.figure(figsize=(12,8))
    fig.suptitle(title,fontsize=15)

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
    ax5.set_ylim([-10,10])
    ax6.set_ylim([-10,10])

    x=radius/1e5
    ax1.plot(x,np.log10(rho_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax1.plot(x,np.log10(rho_func(radius)),'b-',label='Fit')
    ax2.plot(x,np.log10(T_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax2.plot(x,np.log10(T_func(radius)),'b-',label='Fit')
    ax3.plot(x,np.log10(np.abs(rho_func.derivative()(radius))),'b-',label='Fit derivative')
    ax3.plot(x,np.log10(np.abs(drho_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax4.plot(x,np.log10(np.abs(T_func.derivative()(radius))),'b-',label='Fit derivative')
    ax4.plot(x,np.log10(np.abs(dT_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Errors
    relerr_rho = (drho_points-rho_func.derivative()(radius))/drho_points
    relerr_T = (dT_points-T_func.derivative()(radius))/dT_points
    ax5.plot(x,relerr_rho*100,'k-',lw=1.5)
    ax6.plot(x,relerr_T*100,'k-',lw=1.5)

    plt.tight_layout(rect=(0,0,1,0.95))


''''''''' Raw data '''''''''

Rphot_km = 20
R,Rho,T,P,Linf = read_from_file(Rphot_km)
R *= 1e5
frho,fT = IUS(R,Rho), IUS(R,T)

# Analytical derivatives
env_GR.Linf = Linf
drho,dT = [],[]
for r,rho,t in zip(R,Rho,T):
   z = env_GR.derivs(r,[rho,t])
   drho.append(z[0])
   dT.append(z[1])

plot_stuff(R,Rho,T,frho,fT,drho,dT,'Errors on 20km solution - v1')



''''''''' Spline sampled on 1000 points '''''''''

r2 = np.linspace(R[0],Rphot_km*1e5,1000)
rho2,T2 = frho(r2) , fT(r2)
drho,dT = [],[]
for r,rho,T in zip(r2,rho2,T2):
   z = env_GR.derivs(r,[rho,T])
   drho.append(z[0])
   dT.append(z[1])

plot_stuff(r2,rho2,T2,frho,fT,drho,dT,'Errors on 20km solution - v2')




''''''''' Spline sampled on 100 points '''''''''

r3 = np.linspace(R[0],Rphot_km*1e5,100)
rho3,T3 = frho(r3) , fT(r3)
drho,dT = [],[]
for r,rho,T in zip(r3,rho3,T3):
   z = env_GR.derivs(r,[rho,T])
   drho.append(z[0])
   dT.append(z[1])

plot_stuff(r3,rho3,T3,frho,fT,drho,dT,'Errors on 20km solution - v3')




''''''''' Spline sampled on 100 points in the first km, 100 points in the rest '''''''''

r4 = np.concatenate((np.linspace(R[0],R[0]+1e5,100) , np.linspace(R[0]+1.1e5,Rphot_km*1e5,100)))
rho4,T4 = frho(r4) , fT(r4)
drho,dT = [],[]
for r,rho,T in zip(r4,rho4,T4):
   z = env_GR.derivs(r,[rho,T])
   drho.append(z[0])
   dT.append(z[1])

plot_stuff(r4,rho4,T4,frho,fT,drho,dT,'Errors on 20km solution - v4')



plt.show()