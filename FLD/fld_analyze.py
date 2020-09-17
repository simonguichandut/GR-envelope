import sys
sys.path.append(".")

from env_GR_FLD import *

env,stuff = MakeEnvelope(13, Verbose=True, return_stuff=True)

# Inwards integration
a,sola,b,solb = stuff[0]
sols = stuff[1]

# Outwards integrations
a2,sola2,b2,solb2 = stuff[2]
sols2 = stuff[3]

if len(stuff)==5:
    r3,rho3,T3 = stuff[4]



import matplotlib.pyplot as plt 

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(6,8),sharex=True)
fig.subplots_adjust(hspace=0) 

ax1.set_ylabel('T')
ax2.set_ylabel('rho')
ax2.set_xlabel('r (km)')
ax2.set_ylim([1e-9,10*np.round(np.log10(max(env.rho)))])


ax1.semilogy(sola.t/1e5,sola.y[1],'r-',lw=0.8,label=('%.3f'%a))
ax1.semilogy(solb.t/1e5,solb.y[1],'b-',lw=0.8,label=('%.3f'%b))
ax2.semilogy(sola.t/1e5,sola.y[0],'r-',lw=0.8,label=('%.3f'%a))
ax2.semilogy(solb.t/1e5,solb.y[0],'b-',lw=0.8,label=('%.3f'%b))

ax1.semilogy(env.r/1e5,env.T,'g-',lw=2,label='final')
ax2.semilogy(env.r/1e5,env.rho,'g-',lw=2,label='final')

ax1.axvline(env.rphot/1e5,color='k')
ax2.axvline(env.rphot/1e5,color='k')


# Optically thin limit
rvec = np.logspace(6,9,1000)
T_thin = ( env.Linf*Swz(rvec)**(-1) / (4*np.pi*rvec**2*arad*c) )**0.25
ax1.loglog(rvec/1e5,T_thin,'b--',lw=0.7)

for sol in sols:
    ax1.semilogy(sol.t/1e5,sol.y[1],'k-',lw=0.5,alpha=0.5)
    ax2.semilogy(sol.t/1e5,sol.y[0],'k-',lw=0.5,alpha=0.5)


fig,ax3 = plt.subplots(1,1)
ax3.set_ylabel('lambda')
ax3.set_xlabel('r (km)')
ax3.set_ylim([0,0.35])
lama = FLD_Lam(env.Linf*Swz(sola.t)**(-1), sola.t, sola.y[1])
lamb = FLD_Lam(env.Linf*Swz(solb.t)**(-1), solb.t, solb.y[1])
lamsol = FLD_Lam(env.Linf*Swz(env.r)**(-1), env.r, env.T)
ax3.plot(sola.t/1e5,lama,'r-',lw=0.8)
ax3.plot(solb.t/1e5,lamb,'b-',lw=0.8)
ax3.plot(env.r/1e5,lamsol,'g-',lw=2)

lamthin = eos.kappa(env.rho,env.T)*env.rho*env.r/2/Y(env.r)
ax3.plot(env.r/1e5,lamthin,'b--',lw=0.7)

for sol in sols:
    lam = FLD_Lam(env.Linf*Swz(sol.t)**(-1), sol.t, sol.y[1])
    ax3.plot(sol.t/1e5, lam, 'k-', lw=0.5, alpha=0.5)

ax3.axhline(1/3,color='k',ls='--')



ax1.semilogy(sola2.t/1e5,sola2.y[1],'m-',lw=0.8)
ax1.semilogy(solb2.t/1e5,solb2.y[1],'g-',lw=0.8)
ax2.semilogy(sola2.t/1e5,sola2.y[0],'m-',lw=0.8)
ax2.semilogy(solb.t/1e5,solb.y[0],'g-',lw=0.8)

for sol in sols2:
    ax1.semilogy(sol.t/1e5,sol.y[1],'k-',lw=0.5,alpha=0.5)
    ax2.semilogy(sol.t/1e5,sol.y[0],'k-',lw=0.5,alpha=0.5)

lama = FLD_Lam(env.Linf*Swz(sola2.t)**(-1), sola2.t, sola2.y[1])
lamb = FLD_Lam(env.Linf*Swz(solb2.t)**(-1), solb2.t, solb2.y[1])
ax3.plot(sola2.t/1e5,lama,'m-',lw=0.8)
ax3.plot(solb2.t/1e5,lamb,'g-',lw=0.8)

for sol in sols2:
    lam = FLD_Lam(env.Linf*Swz(sol.t)**(-1), sol.t, sol.y[1])
    ax3.plot(sol.t/1e5, lam, 'k-', lw=0.2, alpha=0.5)



plt.show()