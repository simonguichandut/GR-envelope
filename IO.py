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
        Prad = f.readline().split()[1]
        next(f)
        next(f)
        save = f.readline().split()[1]
        img = f.readline().split()[1]

    if as_dict is True:
        return {'M':M,'R':R,'y_inner':y_inner,
                'comp':comp,'EOS_type':EOS_type,'FLD':FLD,'Prad':Prad,
                'save':save,'img':img}
        
    return M,R,y_inner,comp,EOS_type,FLD,save,img


def get_name():  # We give various files and directories the same name corresponding to the setup given in the parameter file

    params = load_params()
    name = '_'.join([ 
        params['comp'], params['EOS_type'], ('M%.1f'%params['M']), 
        ('R%2d'%params['R']) , 
        ('y%1d'%np.log10(params['y_inner'])) ])

    if params['FLD'] == True: 
        name += '_FLD'
        if params['Prad'] == 'exact':
            name += '_exact'
 
    return name


def make_directories():

    dirname = get_name()
    path = 'results/' + dirname
    if not os.path.exists(path):   # Assuming code is being run from main directory
        os.mkdir(path)
        os.mkdir(path+'/data')
        # os.mkdir(path+'/plots')


def write_to_file(Rphotkm,env):
    # Expecting env type namedtuple object

    assert(env.rphot/1e5==Rphotkm)

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

    sorted_list = list(np.sort(Rphotkms))
    sorted_list_clean = [int(x) if str(x)[-1]=='0' else x for x in sorted_list] # changes 15.0 to 15 for filename cleanliness
    return sorted_list_clean


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)



######## FLD stuff

def save_rhophf0rel(Rphotkm, f0vals, rhophvalsA, rhophvalsB):

    path = 'FLD/' + get_name()

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + '/rhophf0rel_' + str(Rphotkm) + '.txt'
    if not os.path.exists(filepath):
        f = open(filepath, 'w+')
        f.write('{:<12s} \t {:<12s} \t {:<12s}\n'.format(
                'f0', 'log10(rhophA)', 'log10(rhophB)'))
    else:
        f = open(filepath, 'a')

    for f0, rhopha, rhophb in zip(f0vals, rhophvalsA, rhophvalsB):
        f.write('{:<11.8f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                f0, np.log10(rhopha), np.log10(rhophb)))

def load_rhophf0rel(Rphotkm):

    s = str(Rphotkm)
    if s[-2:]=='.0': s=s[:-2]

    filepath = 'FLD/' + get_name() + '/rhophf0rel_' + s + '.txt'
    if not os.path.exists(filepath):
        return False,

    else:
        f0, rhophA, rhophB = [],[],[]
        with open(filepath,'r') as f:
            next(f)
            for line in f:
                f0.append(eval(line.split()[0]))
                rhophA.append(10**eval(line.split()[1]))
                rhophB.append(10**eval(line.split()[2])) # saved as log values in the file
         
        return True,f0,rhophA,rhophB


def clean_rhophf0relfile(Rphotkm,warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest f0

    _,f0vals,rhophvalsA,rhophvalsB = load_rhophf0rel(Rphotkm)
    new_f0vals = np.sort(np.unique(f0vals))[::-1] # largest f0 value first (to work correctly in the initial search in MakeEnvelope)

    if list(new_f0vals) != list(f0vals):

        v = []
        for x in new_f0vals:
            duplicates = np.argwhere(f0vals==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_rhophvalsA, new_rhophvalsB = [],[]
        for i in v:
            new_rhophvalsA.append(rhophvalsA[i])
            new_rhophvalsB.append(rhophvalsB[i])

        if warning:
            o = input('EdotTsrel file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:
            filepath = 'FLD/'+get_name()+'/rhophf0rel_'+str(Rphotkm)+'.txt'
            os.remove(filepath)

            save_rhophf0rel(Rphotkm,new_f0vals,new_rhophvalsA,new_rhophvalsB)
    