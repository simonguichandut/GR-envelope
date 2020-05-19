import IO 
import matplotlib.pyplot as plt

if IO.load_params()['FLD'] == True:
    from env_GR_FLD import *
else:
    from env_GR import *

fig,(ax1,ax2)=plt.subplots(1,2) 
plt.pause(0.01)
Rphot_km=100
Rphot = Rphot_km*1e5
rspan = (Rphot , 1.01*rg)
fvalues = np.linspace(-3.7,-4.5,200)

# fvalues = np.linspace(-3.84,-3.844,100)

# for i,f0 in enumerate(fvalues):
#     print('f=',f0)
#     rho_phot,T_phot,Linf = photosphere(Rphot,f0)
#     # try:
#     sol = Shoot(rspan,rho_phot,T_phot,Linf)
#     ax1.loglog(sol.t,sol.y[0],'k-',lw=0.5)
#     ax2.loglog(sol.t,sol.y[1],'k-',lw=0.5)
#     plt.pause(0.01)

#     if len(sol.t_events[1]) == 1: 
#         break

    # except Exception as E:
    #     print(E)
    #     print('problem')
    #     break

for i,f0 in enumerate(fvalues):
    rho_phot,T_phot,Linf = photosphere(Rphot,f0)
    solb = Shoot(rspan,rho_phot,T_phot,Linf) 
    if len(solb.t_events[1]) == 1: 
        # density went down and we stopped integration, so we didn't end up 
        # landing on rb<RNS. We will still keep this solution
        a,b = fvalues[i],fvalues[i-1]
        Ea,sola = -1,solb # -1 to have a negative value
        Eb,solb = Eprev,solprev
        break

    else:
        Eb = Error(solb.t)
        print('f=',f0,'\t success: ',solb.success,'\t error:',Eb)

        if Eb<0:
            a,b = fvalues[i],fvalues[i-1]
            Ea,sola = Eb,solb
            Eb,solb = Eprev,solprev
            break
            
        Eprev,solprev = Eb,solb