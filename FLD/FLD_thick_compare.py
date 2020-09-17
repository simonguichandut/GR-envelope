import sys
sys.path.append(".")

# Comparing envelope solutions with FLD to pure optically thick
# Analyzing the behaviour of the FLD solutions (lambda)

import matplotlib.pyplot as plt 
import IO
from env_GR_FLD import *
from scipy.integrate import solve_ivp


fig,(ax1a,ax1b) = plt.subplots(2,1,figsize=(6,8),sharex=True)
fig.subplots_adjust(hspace=0) 

ax1a.set_ylabel(r'$T$')
ax1b.set_ylabel(r'$\rho$')
ax1b.set_xlabel(r'r (cm)')


fig2,ax2 = plt.subplots(1,1)
ax2.set_xlabel(r'r (cm)')
ax2.set_ylabel(r'$1/3-\lambda$')


Rphotlist = (15,30,50,100,200,1000)
colors = ('k','m','g','b','r','c')

path_thick = 'results/He_IGDE_M1.4_R12_y8/data/'
path_FLD = 'results/He_IGDE_M1.4_R12_y8_FLD/data/'

for Rphot,col in zip(Rphotlist,colors):

    env_thick = IO.read_from_file(Rphot, specific_file = path_thick+str(Rphot) + '.txt')
    env_FLD = IO.read_from_file(Rphot, specific_file = path_FLD+str(Rphot) + '.txt')

    sol_FLD_thin = solve_ivp(derivs,(env_FLD.r[-1],1e9), (env_FLD.rho[-1],env_FLD.T[-1]), 
                    args=(env_FLD.Linf,), method='Radau', dense_output=True, 
                    atol=1e-6, rtol=1e-6, max_step=1e6)

    if col=='k':
        labels = ('thick',(r'FLD, $\tau\leq2/3$'),(r'FLD, $\tau>2/3$'))
    else:
        labels = (None,None,None)

    ax1a.loglog(env_thick.r,env_thick.T,color=col,ls='--',lw=0.8,label=labels[0])
    ax1a.loglog(env_FLD.r,env_FLD.T,color=col,ls='-',lw=0.8,label=labels[1])
    ax1a.loglog(sol_FLD_thin.t,sol_FLD_thin.y[1],color=col,ls=':',lw=0.8,label=labels[2])

    ax1b.loglog(env_thick.r,env_thick.rho,color=col,ls='--',lw=0.8)
    ax1b.loglog(env_FLD.r,env_FLD.rho,color=col,ls='-',lw=0.8)
    ax1b.loglog(sol_FLD_thin.t,sol_FLD_thin.y[0],color=col,ls=':',lw=0.8)


    lam1 = FLD_Lam(env_FLD.Linf*Swz(env_FLD.r)**(-1), env_FLD.r, env_FLD.T)
    lam2 = FLD_Lam(env_FLD.Linf*Swz(sol_FLD_thin.t)**(-1), sol_FLD_thin.t, sol_FLD_thin.y[1])

    ax2.loglog(env_FLD.r, 1/3-lam1, color=col,ls='-',lw=0.8)
    ax2.loglog(sol_FLD_thin.t, 1/3-lam2, color=col,ls=':',lw=0.8)

    # In optically thin, lambda should go to tau*/2Y
    # lam_thin = eos.kappa(sol_FLD_thin.y[0],sol_FLD_thin.y[1]) * sol_FLD_thin.y[0] * sol_FLD_thin.t / 2 / Y(sol_FLD_thin.t)
    # ax2.loglog(sol_FLD_thin.t, 1/3-lam_thin, color=col, ls='-.', lw=0.8)

    ax2.axhline(1/3,color='k',ls='-',lw=0.5)


ax1a.legend(frameon=False)

plt.show()






