import sys
sys.path.append(".")

from env_GR import *
import IO
import os
import pickle

from scipy.optimize import fsolve

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams.update({

    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Non-italic math
    "mathtext.default": "regular",
    # Tick seetings
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top" : True,
    "ytick.right" : True
})


kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
keV = 1.602177e-9 


thesis_figsize=(5.95, 3.68) # according to document size


def Make_profiles_plot(lw = 0.8,figsize=thesis_figsize):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1, sharex=True, figsize=(figsize[0],3*figsize[1]))
    fig.subplots_adjust(hspace=0) 

    ax3.set_xlabel(r'r (cm)',labelpad=5)
    # ax3.tick_params(axis='x',pad=4)

    ax1.set_ylabel(r'$T$ (K)')
    ax2.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    ax3.set_ylabel(r'$q$')

    for ax in (ax1,ax2,ax3):
        ax.grid(alpha=0.5)
    
    colors = ['r','b','g', 'm']

    for i,rphotkm in enumerate((13, 20, 50, 70, 100, 200, 500, 1000)):
    
        env = IO.read_from_file(rphotkm)

        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]

        Lcrit = Lcr(env.r,env.rho,env.T)
        L = env.Linf*Swz(env.r)**(-1)
        q = 1-L/Lcrit

        ax1.loglog(env.r,env.T,lw=lw,ls=ls,color=col,label=str(rphotkm))
        ax2.loglog(env.r,env.rho,lw=lw,ls=ls,color=col)
        ax3.loglog(env.r,q,lw=lw,ls=ls,color=col)
    
    ax1.legend(frameon=False, ncol=4,  bbox_to_anchor=(0.92,0.94), bbox_transform=fig.transFigure)
    ax1.text(0.13,0.91,(r'$r_\mathrm{ph}$ (km)'),fontsize=9,transform=fig.transFigure,
            ha='left',va='center')


    fig.savefig('analysis/thesis_plots/env_profiles.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_profiles.pdf')


def Make_density_temperature_plot(figsize=thesis_figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylabel(r'$T$ (K)')
    
    ax.set_xlim([1e-8,1e6])
    ax.set_ylim([1e6,5e9])
    
    colors = ['r','b','g', 'm']
    for i,rphotkm in enumerate((13, 20, 50, 70, 100, 200, 500, 1000)):
    
        env = IO.read_from_file(rphotkm)

        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]
        ax.loglog(env.rho,env.T,ls=ls,color=col,lw=0.8)

        # Pressure lines
        Rho = np.logspace(-6,10,100)
        Knr,Kr = 9.91e12/eos.mu_e**(5/3), 1.231e15/eos.mu_e**(4/3)

        # Prad = Pg (non-degen)
        T1 = (3*kB*Rho/(arad*eos.mu*mp))**(1/3)

        # Pednr = Pend (non-degen) : Knr rho**(5/3) = kTrho/mu_e*mp
        T2 = Knr*eos.mu_e*mp/kB * Rho**(2/3)

        # Pedr = Pednr
        rho_rel = (Kr/Knr)**3

        ax.loglog(Rho,T1,'k-',lw=0.3)
        ax.loglog(Rho,T2,'k-',lw=0.3)
        ax.axvline(rho_rel,color='k',lw=0.7)

        # ax.loglog(Rho,T1b,'b-',lw=0.5)

        ax.text(Rho[np.argmin(np.abs(T1-2e6))]*2,2e6,(r'$P_r=P_g$'),
            transform=ax.transData,ha='left',va='center',fontsize=matplotlib.rcParams['legend.fontsize'])

        ax.text(Rho[np.argmin(np.abs(T2-2e6))]*2,2e6,(r'$P_\mathrm{end}=P_\mathrm{ed}$'),
            transform=ax.transData,ha='left',va='center',fontsize=matplotlib.rcParams['legend.fontsize'])


#    plt.show()
    fig.savefig('analysis/thesis_plots/env_rho_T.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_rho_T.pdf')


def Make_bisection_method_plot(figsize=thesis_figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$r$ (cm)')
    ax.set_ylabel(r'$T$ (K)')

    if os.path.exists('analysis/bisection_stuff_100km.p'):
        env,((a,sola,b,solb),sols) = pickle.load(open('analysis/bisection_stuff_100km.p','rb'))
    else:
        env,stuff = MakeEnvelope(100,return_stuff=True)
        pickle.dump((env,stuff),open('analysis/bisection_stuff_100km.p','wb'))
        a,sola,b,solb = stuff[0]
        sols = stuff[1]

    ax.loglog(sola.t,sola.y[1],'r-',lw=0.8,label=('%.3f'%a))
    ax.loglog(solb.t,solb.y[1],'b-',lw=0.8,label=('%.3f'%b))
    ax.loglog(env.r,env.T,'k-',lw=1)
    ax.axvline(12e5,color='k',ls=':',lw=0.5)

    r0prev = 100e5
    for sol in sols:
        if sol.t[0]!=r0prev:
            r0prev=sol.t[0]
            # ax.plot([sol.t[0]],[sol.y[1][0]],'k.',ms=3)
            ax.loglog(sol.t,sol.y[1],'k-',lw=0.1,alpha=0.2)
            

    ax.legend(frameon=False,title=r'$\log_{10}q_\mathrm{ph}$',loc=2)
    
    # plt.show()

    fig.savefig('analysis/thesis_plots/env_bisection_demo.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_bisection_demo.pdf')

    

def Make_del_plot(figsize=thesis_figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$r$ (cm)')
    ax.set_ylabel(r'$1-\nabla_\mathrm{rad}/\nabla_\mathrm{ad}$')

    colors = ['r','b','g','m','k']
    for i,rphotkm in enumerate((13, 20, 50, 70, 100, 200, 500, 1000)):

        env = IO.read_from_file(rphotkm)

        delad = del_ad(env.rho,env.T)
        delrad = del_rad(env.rho,env.T,env.r,env.Linf)

        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]

        ax.loglog(env.r,1-delrad/delad,color=col,ls=ls,lw=0.8,label=str(rphotkm))

    # ax.legend(frameon=False,title=(r'$r_{ph}$ (km)'))

    # plt.show()

    fig.savefig('analysis/thesis_plots/env_del.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_del.pdf')


def Make_rphot_plot(figsize=thesis_figsize):

    # plot log10qph and Linf/LEdd as function of rph

    fig,ax = plt.subplots(1,1,figsize=figsize)
    # ax.grid(alpha=0.5)

    ax.set_xlabel(r'$\log_{10}r_\mathrm{ph}$ (cm)')
    ax.set_ylabel(r'$L^\infty/L_\mathrm{Edd}$')

    axb = ax.twinx()
    axb.set_ylabel(r'$\log_{10}q_\mathrm{ph}$',color='b')
    axb.set_yticks([-3.74,-3.78,-3.82,-3.86,-3.9])

    Linfs,qph = [],[]

    Rphotskm = IO.get_phot_list()
    Rphotskm = Rphotskm[Rphotskm>=13]

    for rphotkm in Rphotskm:
        env = IO.read_from_file(rphotkm)
        Linfs.append(env.Linf)

        Lcritph = Lcr(env.r[-1],env.rho[-1],env.T[-1])
        Lph = env.Linf*Swz(env.r[-1])**(-1)
        qph.append(1-Lph/Lcritph)

    Linfs,qph = np.array(Linfs),np.array(qph)

    ax.plot(np.log10(np.array(Rphotskm)*1e5), Linfs/LEdd,'k-')
    axb.plot(np.log10(np.array(Rphotskm)*1e5), np.log10(qph),'b-')

    # plt.show()

    fig.savefig('analysis/thesis_plots/env_rphot.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_rphot.pdf')



def Make_taustar_plot(figsize=thesis_figsize):

    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax.grid(alpha=0.5)
    ax.set_xlabel(r'r (cm)')
    ax.set_ylabel(r'$\tau^*$')

    ax.axhline(3,color='k',linestyle=':')
    ax.set_ylim([0.5,1e3])

    colors = ['r','b','g', 'm']

    for i,rphotkm in enumerate((13, 20, 50, 70, 100, 200, 500, 1000)):
    
        env = IO.read_from_file(rphotkm)

        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]

        taus = eos.kappa(env.rho,env.T) * env.rho * env.r

        ax.loglog(env.r,taus,lw=0.8,ls=ls,color=col,label=str(rphotkm))
    
    ax.legend(frameon=False, ncol=4,  bbox_to_anchor=(0.92,1.05), bbox_transform=fig.transFigure)
    ax.text(0.14,0.96,(r'$r_\mathrm{ph}$ (km)'),fontsize=9,transform=fig.transFigure,
            ha='left',va='center')

    # plt.show()

    fig.savefig('analysis/thesis_plots/env_taustar.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_taustar.pdf')



def Make_touchdown_error_plot(figsize=thesis_figsize):

    # It's only here that we'll have the very compact envelopes

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(figsize[0],1.5*figsize[1]),sharex=True)
    fig.subplots_adjust(hspace=0) 


    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    ax2.set_xlabel(r'$r_\mathrm{ph}-R$ (cm)')

    # ax1.set_ylabel(r'$1-L^\infty/L_\mathrm{Edd}$')
    ax1.set_ylabel(r'$L^\infty/L_\mathrm{Edd}$')
    ax2.set_ylabel(r'$kT^\infty_\mathrm{eff}$ (keV)')

    rphots = IO.get_phot_list()
    rphots = rphots[rphots<500]
    finf,Teffinf = [],[]

    for rphotkm in rphots:
    
        env = IO.read_from_file(rphotkm)
        # finf.append(1-env.Linf/LEdd)
        finf.append(env.Linf/LEdd)
        Teffinf.append(env.T[-1]*Swz(rphotkm*1e5)**(+0.5))

    rphots,Teffinf = np.array(rphots), np.array(Teffinf)
    
    ax1.semilogx((rphots-RNS)*1e5, finf, 'k-', lw=0.8)
    ax2.semilogx((rphots-RNS)*1e5, kB*Teffinf/keV, 'k-', lw=0.8)


    # Add the newtonian envelopes
    from newtonian import find_rph, find_rph_x1
    x = np.logspace(-2,0,500)  # L/LEdd
    rphot_newt = []
    for xi in x[:-1]: # remove the exactly 1 value
        rphot_newt.append(find_rph(xi))

    rphot_newt.append(find_rph_x1()) # add the x=1 value

    rphot_newt = np.array(rphot_newt)

    ax1.semilogx(rphot_newt - RNS*1e5, x, 'k:', lw=0.8, label='Newtonian models')
    ax1.legend(loc='lower center',frameon=False)

    # plt.show()

    fig.savefig('analysis/thesis_plots/env_touchdown.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_touchdown.pdf')

def Make_L_Lcrit_plot(figsize=thesis_figsize):


    fig,ax = plt.subplots(1,1,figsize=figsize)
    # ax.grid(alpha=0.5)

    ax.set_xlabel(r'$r_\mathrm{ph}-R$ (cm)')
    ax.set_ylabel(r'$q_\mathrm{ph}$')

    Linfs,qph = [],[]

    Rphotskm = IO.get_phot_list()

    for rphotkm in Rphotskm:
        env = IO.read_from_file(rphotkm)
        Linfs.append(env.Linf)

        Lcritph = Lcr(env.r[-1],env.rho[-1],env.T[-1])
        Lph = env.Linf*Swz(env.r[-1])**(-1)
        qph.append(1-Lph/Lcritph)

    ax.loglog((np.array(Rphotskm)-RNS)*1e5, np.array(qph),'k-')

    # plt.show()

    fig.savefig('analysis/thesis_plots/env_L_Lcrit.pdf',bbox_inches='tight',format='pdf')
    print('Saved figure to analysis/thesis_plots/env_L_Lcrit.pdf')



# Make_profiles_plot()
# Make_density_temperature_plot()
# Make_density_luminosity_plot()


# rho-L and rho-T plots not full textwidth? Maybe 670%?
# frac = 0.7
# Make_density_temperature_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_del_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_profiles_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_bisection_method_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_rphot_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_taustar_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_touchdown_error_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))
# Make_L_Lcrit_plot(figsize=(frac*thesis_figsize[0],frac*thesis_figsize[1]))