import sys
sys.path.append('.')  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 

mpl.rcParams.update({

    # Use LaTeX to write all text
    # "text.usetex": True,
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
    # Tick settings
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top" : True,
    "ytick.right" : True,
    # Short dash sign
    "axes.unicode_minus" : True
})


from IO import read_from_file, get_phot_list, load_params
assert load_params()['FLD'] == True

model_name = 'He_IG_M1.4_R12_y8_FLD'
LEdd = 4*np.pi*3e10*6.6726e-8*2e33*1.4/0.2


def rho_plot():

    fig,ax = plt.subplots(1,1, figsize=(6, 4.))
    ax.set_xlabel('r (cm)')
    ax.set_xlim([1e6,1e7])
    ax.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylim([1e-9,1e-1])


    for Rphotkm in (13,20,50):

        # Simple Prad
        env = read_from_file(Rphotkm, specific_file=('results/'+model_name+'/data/%d.txt'%Rphotkm))
        
        label = (r'$P_R = aT^4/3$') if Rphotkm==13 else None
        ax.loglog(env.r, env.rho, 'k-', lw=0.5, label=label)
        ax.loglog([Rphotkm*1e5], env.rho[list(env.r).index(Rphotkm*1e5)], 'k.', ms=3)

        # # Exact Prad
        # env = read_from_file(Rphotkm, specific_file=('results/'+model_name+'_exact/data/%d.txt'%Rphotkm))

        label = (r'$P_R = (\lambda+\lambda^2R^2)aT^4$') if Rphotkm==13 else None
        ax.loglog(env.r, env.rho, 'b-', lw=0.5, label=label)
        ax.loglog([Rphotkm*1e5], env.rho[list(env.r).index(Rphotkm*1e5)], 'b.', ms=3)


    ax.legend(frameon=False)
    fig.savefig('analysis/Prad_exact_or_simple_profiles.png', bbox_inches='tight', dpi=300)
    # fig.savefig('analysis/Prad_exact_or_simple.pdf', bbox_inches='tight')
    # plt.show()

rho_plot()



def Lb_rph_plot():

    fig,ax = plt.subplots(1,1,figsize=(6,4))
    ax.set_xlabel(r'$L_b^\infty/L_{E}$')
    ax.set_ylabel(r'r (km)')

    for Prad,col in zip(('simple','exact'),('k','b')):

        name = model_name
        if Prad == 'exact': 
            name+='_exact'
            label = (r'$P_R = (\lambda+\lambda^2R^2)aT^4$') 
        else:
            label = (r'$P_R = aT^4/3$') 

        logMdots,_ = load_roots(specific_file='roots/roots_'+name+'.txt')

        Lbs,rs,rph = [],[],[]

        for logMdot in logMdots:

            try:
                w = read_from_file(logMdot,specific_file='results/'+name+'/data/%.2f.txt'%logMdot)

                Lbs.append(w.Lstar[0])
                rs.append(w.rs)
                rph.append(Rphot_Teff(logMdot, wind=w))
            except:
                pass

        ax.semilogy(np.array(Lbs)/LEdd,np.array(rs)/1e5,color=col,ls='--',lw=0.8)
        ax.semilogy(np.array(Lbs)/LEdd,np.array(rph)/1e5,color=col,ls='-',lw=0.8,label=label)

    ax.text(2,370,r'$r_{ph}$',ha='left',va='center')
    ax.text(2,105,r'$r_{s}$',ha='left',va='center')
    ax.legend(frameon=False)
    fig.savefig('analysis/Prad_exact_or_simple_Lbs.png',bbox_inches='tight', dpi=300)



# Lb_rph_plot()