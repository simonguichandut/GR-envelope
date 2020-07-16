''' Input and Output '''

import os
import numpy as np

def load_params(as_dict=True):
    with open('./params.txt','r') as f:
        next(f)
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        next(f)
        next(f)
        y_inner = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        EOS_type = f.readline().split()[1]
        FLD = eval(f.readline().split()[1]) # boolean
        next(f)
        next(f)
        save = f.readline().split()[1]
        img = f.readline().split()[1]

    if as_dict is True:
        return {'M':M,'R':R,'y_inner':y_inner,
                'comp':comp,'EOS_type':EOS_type,'FLD':FLD,
                'save':save,'img':img}
        
    return M,R,y_inner,comp,EOS_type,FLD,save,img


def get_name():  # We give various files and directories the same name corresponding to the setup given in the parameter file

    params = load_params()
    name = '_'.join([ 
        params['comp'], params['EOS_type'], ('M%.1f'%params['M']), 
        ('R%2d'%params['R']) , 
        ('y%1d'%np.log10(params['y_inner'])) ])
    if params['FLD'] == True: name += '_FLD'
    return name


def make_directories():

    dirname = get_name()
    path = 'results/' + dirname
    if not os.path.exists(path):   # Assuming code is being run from main directory
        os.mkdir(path)
        os.mkdir(path+'/data')
        os.mkdir(path+'/plots')


def write_to_file(Rphotkm,env):
    # Expecting env type namedtuple object

    dirname = get_name()
    path = 'results/' + dirname + '/data/'

    if Rphotkm >= load_params()['R']+1:
                filename = path + str(Rphotkm) + '.txt'
    else:
        filename = path + str(Rphotkm).replace('.','_') + '.txt'

    with open(filename,'w') as f:

        # Write header
        f.write('{:<13s} \t {:<11s} \t {:<11s} \t\t Linf = {:<6e}\n'.format(
            'r (cm)','rho (g/cm3)','T (K)',env.Linf))

        for i in range(len(env.r)):
            f.write('%0.8e \t %0.6e \t %0.6e \t'%
                (env.r[i], env.rho[i], env.T[i]))    

            if load_params()['FLD'] == True:
                pass

            f.write('\n')

def read_from_file(Rphotkm, specific_file=None):

    # output is Envelope namedtuple object       

    # Sometimes because of numpy coversions 13 gets converted to 13.0 for example.
    # We have to remove these zeros else the file isn't found
    s = str(Rphotkm)
    if '.' in s:
        if len(s[s.find('.')+1:]) == 1: # this counts the number of char after '.' (#decimals)
            if s[-1]=='0':
                Rphotkm = round(eval(s))

    if specific_file != None:
        filename = specific_file
    else:
        dirname = get_name()
        path = 'results/' + dirname + '/data/'

        if Rphotkm >= load_params()['R']+1:
                    filename = path + str(Rphotkm) + '.txt'
        else:
            filename = path + str(Rphotkm).replace('.','_') + '.txt'

    def append_vars(line,varz): 
        l=line.split()
        for col,var in enumerate(varz):
            var.append(float(l[col]))

    r, rho, T = [[] for i in range (3)]
    with open(filename,'r') as f:
        for i,line in enumerate(f): 
            if i==0: 
                Linf = float(line.split()[-1])
            else:
                append_vars(line,[r, rho, T])

    r,rho,T = [np.array(var) for var in (r,rho,T)]

    # Return as env tuple object
    # if load_params()['FLD'] == True:
    #     from env_GR import Env
    #     return Env(Rphotkm*1e5,Linf,r,T,rho)
    # else:
    #     from env_GR import Env
    #     return Env(Rphotkm*1e5,Linf,r,T,rho)

    if load_params()['FLD'] == False:
        from env_GR import Env
    else:
        from env_GR_FLD import Env
    return Env(Rphotkm*1e5,Linf,r,T,rho)



def get_phot_list():

    # Returns list of photospheric radius that have solutions
    path = 'results/' + get_name() + '/data/'

    Rphotkms = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            Rphotkms.append(eval(filename[:-4].replace('_','.')))

    return np.sort(Rphotkms)


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)


# def export_values(target='./'):

#     # Export useful values for analysis for each Rphot to a text file at target directory
#     # Current values are : Linf,Rb,Tb,Rhob,Pb,Tphot,Rhophot
#     # tsound(sound crossing time) 

#     if target[-1]!='/': target += '/'
#     Rphotkms = get_phot_list()

#     import physics
#     eos = physics.EOS(load_params()['comp'])

#     from scipy.interpolate import interp1d
#     from scipy.integrate import quad
    
#     with open(target+'envelope_values.txt','w') as f:

#         f.write('{:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \n'.format(
#             'Rph (km)','Linf (erg/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Tph (K)','rhoph (g/cm3)','tsound (s)'))

#         for R in Rphotkms:

#             if R>=load_params()['R']+0.5: # not the ultra compact ones

#                 print(R)

#                 env = read_from_file(R)
                
#                 cs = np.sqrt(eos.cs2(env.T))
#                 func_inverse_cs = interp1d(env.r,1/cs,kind='cubic')
#                 tsound,err = quad(func_inverse_cs,env.r[0],env.r[-1],epsrel=1e-5)#limit=100)
#                 # print(tsound,err)
        
#                 f.write('%0.1f \t\t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e\n'%
#                     (R,env.Linf,env.r[0]*1e5,env.T[0],env.rho[0],eos.pressure_e(env.rho[0],env.T[0]),env.T[-1],env.rho[-1],tsound))

#     print('Exported to '+target+'envelope_values.txt')

# export_values('../compare')

# def pickle_save(name):
    
#     # Save all arrays into pickle file

#     # Import Winds
#     clean_rootfile()
#     logMDOTS,roots = load_roots()

#     if not os.path.exists('pickle/'):
#         os.mkdir('pickle/')

    