''' Optical depth parameter kappa*rho*r ''' 
import sys
sys.path.append('.')

from env_GR import kappa
import IO

# Parameters
M, RNS, y_inner, comp, save, img = IO.load_params()

# Available solutions
Rphotkms = IO.get_phot_list()


# Plotting 

import matplotlib.pyplot as plt
from my_plot import set_size, plot_settings

plot_settings()
fig,ax = plt.subplots(1,1,figsize=set_size('mnras2col'))

save = 1

for i,R in enumerate(Rphotkms):

    r, rho, T, P, Linf = IO.read_from_file(R)
    Kap = kappa(rho,T)
    taustar = Kap*rho*r*1e5

    ax.loglog(r,taustar,'r',linewidth=0.7)

ax.set_xlabel('r (km)')
ax.set_ylabel(r'$\tau^* = \kappa\rho r$')
ax.axhline(3,color='k',linewidth=0.7,linestyle='--')
ax.set_ylim([0.6,1e3])


if save: fig.savefig('analysis/plots/taustar.pdf', format='pdf', bbox_inches='tight')
else: plt.show()