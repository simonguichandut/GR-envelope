import sys
sys.path.append(".")
sys.path.append("./analysis")

import numpy as np
import IO


def export(target = "."):

    # Export useful values for analysis for each Rphot to a text file at target directory
    # Current values are : Linf,Rb,Tb,Rhob,Pb,Tphot,Rhophot
    # tsound(sound crossing time) 
    # Min&Mout (masses below & above sonic point)

    if target[-1]!='/': target += '/'
    Rphotkms = IO.get_phot_list()
    params  = IO.load_params()
    filename = target+'env_values_'+IO.get_name()+'.txt'

    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    import physics
    eos = physics.EOS(params['comp'])

    with open(filename,'w') as f:

        f.write(('{:<11s} \t '*9 +'{:<11s}\n').format(
            'Rph (km)','Linf (erg/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Tph (K)','rhoph (g/cm3)','tsound (s)','Min (g)'))


        for R in Rphotkms:

            if R>=params['R']+0.5: # not the ultra compact ones

                print(R)
                env = IO.read_from_file(R)
                
                cs = np.sqrt(eos.cs2(env.T))
                func_inverse_cs = interp1d(env.r,1/cs,kind='cubic')
                tsound,err = quad(func_inverse_cs,env.r[0],env.r[-1],epsrel=1e-5)#limit=100)
        
                # Mass contained in envelope
                rhofunc = interp1d(env.r,env.rho,kind='cubic')

                def mass_in_shell(r): 
                    return 4*np.pi*rhofunc(r)*r**2

                r0 = params['R']*1e5 + 2e2 # start integrating 2m above surface to make uniform
                Min,err = quad(mass_in_shell, r0, env.r[-1], epsrel=1e-5, limit=500)

                # Write base values
                f.write(('%0.1f \t\t' + '%0.6e \t'*5)%
                    (R,env.Linf,env.r[0],env.T[0],env.rho[0],eos.pressure_e(env.rho[0],env.T[0])))
                
                # Write photoshere values 
                f.write(('%0.6e \t'*2)%
                    (env.T[-1],env.rho[-1]))

                # Timescales
                f.write(('%0.6e \t')%
                    (tsound))

                # Mass contained
                f.write(('%0.6e \t')%
                    (Min))

                f.write('\n')

    print('Saved values to : "%s"'%filename)

# Command line call
if len(sys.argv)>=2:
    export(target = sys.argv[-1])
else:
    export()