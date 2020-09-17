import sys
sys.path.append(".")

# Figuring out the behavior of the envelopes into the optically thin regions

import matplotlib.pyplot as plt 
import IO
from env_GR_FLD import *

Rphotkm = 50
env=IO.read_from_file(50)

fig,(ax1a,ax1b,ax1c) = plt.subplots(3,1,figsize=(6,8),sharex=True)
fig.subplots_adjust(hspace=0) 

ax1a.set_ylabel(r'$T$')
ax1b.set_ylabel(r'$\rho$')
ax1c.set_ylabel(r'$F/cE$')
ax1c.set_xlabel(r'r (cm)')

ax1a.loglog(env.r,env.T,'k-')
ax1b.loglog(env.r,env.rho,'k-')

L1 = env.Linf*Swz(env.r)**(-1)
F1 = L1/(4*np.pi*env.r**2)
x1 = F1/(c*arad*env.T**4)
ax1c.semilogx(env.r,x1,'k-')
ax1c.set_ylim([0,1])

# Optically thin limit
rvec = np.logspace(6.5,9,50)
T_thin = ( env.Linf*Swz(rvec)**(-1) / (4*np.pi*rvec**2*arad*c) )**0.25
ax1a.loglog(rvec,T_thin,'k--')

# Optically thin integration
def hit_zero_density(r,y,*args):
    return y[0]
hit_zero_density.terminal = True

# sol_thin_FLD = solve_ivp(derivs, (env.r[-1],1e9), (env.rho[-1],env.T[-1]), args=(env.Linf,), events=(hit_zero_density), method='RK45', dense_output=True, atol=1e-6, rtol=1e-10, max_step=1e5)
# env_thin = Env(Rphotkm,env.Linf,sol_thin_FLD.t,sol_thin_FLD.y[1],sol_thin_FLD.y[0])
# print(env_thin.r[-1])

# ax1a.loglog(env_thin.r,env_thin.T,'b-')
# ax1b.loglog(env_thin.r,env_thin.rho,'b-')

# L2 = env.Linf*Swz(env_thin.r)**(-1)
# F2 = L2/(4*np.pi*env_thin.r**2)
# x2 = F2/(c*arad*env_thin.T**4)
# ax1c.semilogx(env_thin.r,x2,'b-')


# # Iterate on rho_phot
# rho_phot_values = np.logspace(np.log10(env.rho[-1])+0.051, np.log10(env.rho[-1])+0.0513, 8)
# import seaborn as sns
# colors = sns.color_palette('Blues', 10)
# for rho_phot,color in zip(rho_phot_values,colors):

#     sol_thin_FLD = solve_ivp(derivs, (env.r[-1],1e9), (rho_phot,env.T[-1]), args=(env.Linf,), 
#                     events=(hit_zero_density), method='Radau', dense_output=True, rtol=1e-6)
#     env_thin = Env(Rphotkm,env.Linf,sol_thin_FLD.t,sol_thin_FLD.y[1],sol_thin_FLD.y[0])
#     print(sol_thin_FLD.message, sol_thin_FLD.t_events, env_thin.r[-1])

#     ax1a.loglog(env_thin.r,env_thin.T,'-',lw=0.7,color=color,label=('%.5f'%np.log10(rho_phot)))
#     ax1b.loglog(env_thin.r,env_thin.rho,'-',lw=0.7,color=color)

#     L2 = env.Linf*Swz(env_thin.r)**(-1)
#     F2 = L2/(4*np.pi*env_thin.r**2)
#     x2 = F2/(c*arad*env_thin.T**4)
#     ax1c.semilogx(env_thin.r,x2,'-',lw=0.7,color=color)

# ax1a.legend(title=r'log$_{10}\rho_\mathrm{ph}$',fontsize=10)

# plt.show()


# # def derivs_quinn(r,y,Linf):
# #     rho,T = y[:2]
# #     L = Linf*Swz(Rphotkm*1e5)**(-1)
# #     taus = eos.kappa(rho,T)*rho*r
# #     dT_dr = -3*taus*L/(16*np.pi*r**3*c*arad*T**3) * (1 + 2/(3*taus))
# #     drho_dr = eos.cs2(T)**(-1) * (-GM*rho/r**2 - rho*eos.cs2(T)/T*dT_dr + taus*L/(4*np.pi*r**3*c))

# #     return [drho_dr,dT_dr]

# # sol_thin_Quinn = solve_ivp(derivs_quinn, (env.r[-1],1e9), (env.rho[-1],env.T[-1]), args=(env.Linf,), method='RK45', dense_output=True, rtol=1e-6, max_step=1e5)
# # env_thin_Quinn = Env(Rphotkm,env.Linf,sol_thin_Quinn.t,sol_thin_Quinn.y[1],sol_thin_Quinn.y[0])

# # ax1a.loglog(env_thin_Quinn.r,env_thin_Quinn.T,'r-')
# # ax1b.loglog(env_thin_Quinn.r,env_thin_Quinn.rho,'r-')








###### We're going to have to do a bisection method for the optically thin regions of the envelopes, like
# we had to do for the winds. 

# First we have to get a relation for rho_ph as a function of f0=log10(1-Lph/Lcr) (similar to EdotTsrel for winds)

def get_rhophf0rel(Rphotkm,tol=1e-6,Verbose=0,f0min=-4.5,f0max=-3.7,npts=40):

    # find the value of rhoph that allow a solution to go to inf (to tol precision), for each value of f0

    if Verbose: print('\nRphot = %.2f km\n'%Rphotkm)

    f0vals = np.linspace(f0min,f0max,npts)
    f0vals = np.round(f0vals,8) # 8 decimals

    for f0 in f0vals:
        if Verbose: print('\nFinding rhoph for f0 = %.8f'%f0)

        # Start at the initial value given by the approximation for tau=2/3
        rhoph,Tph,Linf = photosphere(Rphotkm*1e5, f0)

        a = rhoph
        sola = solve_ivp(derivs, (Rphotkm*1e5,1e9), (a,Tph), args=(Linf,), 
                            events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)

        if sola.status == 1: # hit zero density (intial rhoph is in the bottom branch)
            direction = +1
        else:
            direction = -1

        # Step either up or down in rhoph until we find other branch
        step = 0.5 # 50% update
        b = a
        while True:
            b *= 1 + direction*step  # either *1.1 or *0.9
            # print('%.6e'%b)
            solb = solve_ivp(derivs, (Rphotkm*1e5,1e9), (b,Tph), args=(Linf,), 
                            events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)
            if solb.status != sola.status:
                break

        # Bissect to get two values of rhoph close to relative tolerance tol
        # print('\nBeginning Bissection')
        while abs(b-a)/a>tol:
            m = (a+b)/2
            # print('%.6e'%m)
            solm = solve_ivp(derivs, (Rphotkm*1e5,1e9), (m,Tph), args=(Linf,), 
                            events=(hit_zero_density), method='Radau', dense_output=True, rtol=tol)
            if solm.status == sola.status:
                a,sola = m,solm
            else:
                b,solb = m,solm

        # a the smaller one just to not get confused
        if a>b: (a,b) = (b,a) 

        if Verbose:
            print('\nInitial rhoph based on PA86 formula : \n%.6e'%rhoph)
            print('Final bounding values:\n%.6e\n%.6e'%(a,b))

        # Save one at a time
        IO.save_rhophf0rel(Rphotkm,[f0],[a],[b])


# get_rhophf0rel(50)
        




        # while abs(b-a)>tol and cont:
        #     print('%.8f    %.8f'%(a,b))

        #     for logTs in logTsvals[1:]:

        #         print('Current: %.8f'%logTs, end="\r")

        #         try:
        #             res = run_outer(logMdot,Edot_LEdd,logTs,Verbose)
        #         except Exception as E:
        #             print(E)
        #             print('Exiting...')
        #             cont = False
        #             break

        #         else:
        #             if res.status==1:
        #                 a = logTs
        #             elif res.status==0:
        #                 raise Exception('Reached end of integration interval (r=%.3e) without diverging!'%res.t[-1])
        #             else:
        #                 b = logTs
        #                 break

        #     logTsvals = np.linspace(a,b,6)      

        # # Take final sonic point temperature to be bottom value (the one that leads to Mach 1.  We know the real value is in between a and a+tol)
        # # Tsvals.append(a)

        # if cont==False:
        #     break

        # if a==b:
        #     print('border values equal (did not hit rs<RNS, maybe allow higher Ts). Exiting')
        #     break

        # # Save one at a time
        # IO.save_EdotTsrel(logMdot,[Edot_LEdd],[a],[b])

        # a,b = a,8  # next Edot, Ts will certainly be higher than this one
        # print('ok'.ljust(20))

    # IO.clean_EdotTsrelfile(logMdot,warning=0)




# def OuterBisection(rend=1e9,tol=1e-5):

#     """ Makes a full outer solution for the wind by integrating until 
#     divergence, interpolating values by bissection and restarting prior to 
#     divergence point, over and over until reaching rend."""

#     # get the solution from the root's Ts (rs and Ts already set as global)
#     if verbose: print('Calculating solution from Ts root')
#     rsa,Tsa = rs,Ts
#     sola = outerIntegration(r0=rsa,T0=Tsa,phi0=2.0)

#     # find other solution that diverges in different direction
#     if sola.status == 0: 
#         sys.exit('reached end of integration interval with root!')

#     elif sola.status == +1:
#         direction = +1  # reached dv/dr=0,other solution needs to have higher Ts
#     elif sola.status == -1:
#         direction = -1 # diverged, other solution needs to have smaller Ts

#     if verbose: print('Finding second solution')
#     step = 1e-6
#     Tsb,rsb,solb = Tsa,rsa,sola
#     i=0
#     while solb.status == sola.status:

#         if i>0: 
#             Tsa,rsa,sola = Tsb,rsb,solb  
#             # might as well update solution a 
#             # (since this process gets closer to the right Ts)

#         logTsb = np.log10(Tsb) + direction*step
#         Tsb = 10**logTsb
#         rsb = rSonic(Tsb)
#         solb = outerIntegration(r0=rsb,T0=Tsb,phi0=2.0)
#         i+=1
#         if i==200:
#             print('Not able to find a solution that diverges in opposite \
#                     direction after changing Ts by 200 tolerances.  \
#                     Problem in the TsEdot interpolation')

#     # if sola was the high Ts one, switch sola and solb
#     if direction == -1:
#         (rsa,Tsa,sola),(rsb,Tsb,solb) = (rsb,Tsb,solb),(rsa,Tsa,sola)
            
#     if verbose:
#         print('Two initial solutions. sonic point values:')
#         print('logTs - sola:%.6f \t solb:%.6f'%(np.log10(Tsa),np.log10(Tsb)))
#         print('logrs - sola:%.6f \t solb:%.6f'%(np.log10(rsa),np.log10(rsb)))


#     def check_convergence(sola,solb,rcheck):
#         """ checks if two solutions are converged (similar T, phi) at some r """
#         Ta,phia = sola.sol(rcheck)
#         Tb,phib = solb.sol(rcheck)
#         if abs(Ta-Tb)/Ta < tol and abs(phia-phib)/phia < tol:
#             return True,Ta,Tb,phia,phib
#         else:
#             return False,Ta,Tb,phia,phib


#     # Start by finding the first point of divergence
#     Npts = 1000
#     R = np.logspace(np.log10(rsa),np.log10(rend),Npts)   
#     # choose colder (larger) rs (rsa) as starting point because 
#     # sola(rsb) doesnt exist

#     for i,ri in enumerate(R):
#         conv = check_convergence(sola,solb,ri)
#         if conv[0] is False:
#             i0=i            # i0 is index of first point of divergence
#             break
#         else:
#             Ta,Tb,phia,phib = conv[1:]

#     if i0==0:
#         print('Diverging at rs!')
#         print(conv)
#         print('rs=%.5e \t rsa=%.5e \t rsb=%.5e'%(rs,rsa,rsb))
        

#     # Construct initial arrays
#     T,Phi = sola.sol(R[:i0])
#     def update_arrays(T,Phi,sol,R,j0,jf):
#         # Add new values from T and Phi using ODE solution object. 
#         # Radius points to add are R[j0] and R[jf]
#         Tnew,Phinew = sol(R[j0:jf+1])  # +1 so R[jf] is included
#         T,Phi = np.concatenate((T,Tnew)), np.concatenate((Phi,Phinew))
#         return T,Phi

#     # Begin bisection
#     if verbose:
#         print('\nBeginning bisection')
#         print('rconv (km) \t Step # \t Iter \t m')  
#     a,b = 0,1
#     step,count = 0,0
#     i = i0
#     rconv = R[i-1]  # converged at that radius
#     rcheck = R[i]   # checking if converged at that radius
#     do_bisect = True
#     while rconv<rend:  
#         # probably can be replaced by while True if the break conditions are ok

#         if do_bisect: # Calculate a new solution from interpolated values
            
#             m = (a+b)/2
#             Tm,phim = Ta + m*(Tb-Ta) , phia + m*(phib-phia)
#             solm = outerIntegration(r0=rconv,T0=Tm,phi0=phim) 
#             # go further than rmax to give it the chance to diverge either way

#             if solm.status == 0: # Reached rend - done
#                 T,Phi = update_arrays(T,Phi,solm.sol,R,i0,Npts)  
#                 #jf=Npts so that it includes the last point of R
#                 print('reached end of integration interval  without \
#                     necessarily converging.. perhaps wrong')
#                 return R,T,Phi

#             elif solm.status == 1:
#                 a,sola = m,solm
#             elif solm.status == -1:
#                 b,solb = m,solm

#         else:
#             i += 1
#             rconv = R[i-1]
#             rcheck = R[i] 

#         conv = check_convergence(sola,solb,rcheck)
#         if conv[0] is True:

#             # Exit here if reached rend
#             if rcheck==rend or i==Npts:  # both should be the same
#                 T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i)
#                 return R,T,Phi

#             Ta,Tb,phia,phib = conv[1:]
#             a,b = 0,1 # reset bissection parameters
#             step += 1 # update step counter
#             count = 0 # reset iteration counter

#             # Converged, so on next iteration just look further
#             do_bisect = False 
        
#         else:
#             count+=1
#             do_bisect = True

#             # Before computing new solution, add converged results to array 
#             # (but only if we made at least 1 step progress)
#             if i-1>i0:
#                 T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i-1)  # i-1 is where we converged last
#                 i0=i # next time we append

#         # Exit if stuck at one step
#         nitermax=1000
#         if count==nitermax:
#             sys.exit("Could not integrate out to rend! Exiting after being \
#                         stuck at the same step for %d iterations"%nitermax)

#         # End of step
#         if verbose: print('%.4e \t %d \t\t %d \t\t %f'%(rconv,step,count,m))

#     return R,T,Phi