''' Plotting various quantities from wind solutions ''' 

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import log10, pi, array, linspace, logspace, floor

rc('text', usetex = True)
# rc('font', family = 'serif')
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
mpl.rcParams.update({'font.size': 15})

from env_GR import *
from IO import *

# Parameters
M, RNS, y_inner, comp, save, img = load_params()

# Available solutions
Rphotkms = get_phot_list()

# constants
c = 2.99792458e10
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
g = GM/(RNS*1e5)**2

# save=0

########## PLOTS ###########
def set_style():
    plt.style.use(['seaborn-talk'])
def beautify(fig,ax):
    set_style()
    ax.tick_params(which='both',direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    fig.tight_layout()

# Radius-Temperature 
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xlabel(r'$r$ (km)')
ax1.set_ylabel(r'$T$ (K)')
ax1.set_xlim([9,300])

# Radius-Density
fig2, ax2 = plt.subplots(1, 1)
ax2.set_xlabel(r'$r$ (km)')
ax2.set_ylabel(r'$\rho$ (K)')
ax1.set_xlim([9,300])

# Density-Temperature 
fig3, ax3 = plt.subplots(1, 1)
ax3.set_xlabel(r'$\rho$ (g km$^{-3}$)')
ax3.set_ylabel(r'$T$ (K)')

# Radius-Pressure
fig4, ax4 = plt.subplots(1, 1)
ax4.set_xlabel(r'$r$ (km)')
ax4.set_ylabel(r'$P$ (g cm$^{-1}$ s$^{-2}$)')
ax1.set_xlim([9,300])

# Density-Opacity
fig5, ax5 = plt.subplots(1, 1)
ax5.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax5.set_ylabel(r'$\kappa$')

# Density-Optical depth
fig6, ax6 = plt.subplots(1, 1)
ax6.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax6.set_ylabel(r'$\tau^*=\kappa\rho r$')
ax6.axhline(3,color='k')
ax6.set_ylim([0.5,2e10])

# Radius-Optical depth
fig7, ax7 = plt.subplots(1, 1)
ax7.set_xlabel(r'$r$ (km)')
ax7.set_ylabel(r'$\tau^*=\kappa\rho r$')
ax7.axhline(3,color='k')
ax7.set_ylim([0.5,2e10])

# colors = ['r', 'b', 'g', 'k', 'm']
Rplot = (15,20,50,100,200)

for i,R in enumerate(Rphotkms):

    r, rho, T, P, Linf = read_from_file(R)
    Kap = kappa(rho,T)
    taustar = Kap*rho*r*1e5

    def myloglogplot(ax,x,y):
        ax.loglog(x,y,lw=1,label=('%d km'%R))
    
    if R in Rplot:
        for fig,ax,x,y in zip((fig1,fig2,fig3,fig4,fig5,fig6,fig7),(ax1,ax2,ax3,ax4,ax5,ax6,ax7),(r,r,rho,r,rho,rho,r),(T,rho,T,P,Kap,taustar,taustar)):
            myloglogplot(ax,x,y)
            beautify(fig,ax)

for ax in (ax1,ax2,ax3,ax4,ax5,ax6,ax7):
        ax.legend(title=r'R$_\text{ph}$ (km)', loc='best')



if save: 
    save_plots([fig1,fig2,fig3,fig4,fig5,fig6,fig7],['Temperature','Density','Density-Temperature','Pressure','Opacity','Optical_depth_rho','Optical_depth_r'],img)
else:
    plt.show()