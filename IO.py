''' Input and Output '''

import os
from numpy import log10,array,sort,sqrt

def load_params():
    with open('params.txt','r') as f:
        next(f)
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        next(f)
        next(f)
        y_inner = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        next(f)
        next(f)
        save = f.readline().split()[1]
        img = f.readline().split()[1]
        
    return M,R,y_inner,comp,save,img


def get_name():  # We give various files and directories the same name corresponding to the setup given in the parameter file

    M,R,y_inner,comp,_,_ = load_params()
    name = '_'.join( [ comp , str(M) , ('%2d'%R) , ('%1d'%log10(y_inner)) ] )
    return name


def make_directories():

    dirname = get_name()
    path = 'results/' + dirname
    if not os.path.exists(path):   # Assuming code is being run from main directory
        os.mkdir(path)
        os.mkdir(path+'/data')
        os.mkdir(path+'/plots')


def write_to_file(Rphotkm,data):

    # data is expected to be list of the following arrays : R, Rho, T, P, Linf (Linf is just a number)

    dirname = get_name()
    path = 'results/' + dirname + '/data/'
    filename = path + str(Rphotkm) + '.txt'

    with open(filename,'w') as f:
        R, Rho, T, P, Linf = data

        f.write('{:<13s} \t {:<11s} \t {:<11s} \t {:<11s} \t\t Linf = {:<6e}\n'.format(
            'r (km)','rho (g/cm3)','T (K)','P (dyne/cm2)',Linf))

        for i in range(len(R)):
            f.write('%0.8e \t %0.6e \t %0.6e \t %0.6e\n'%
                (R[i]/1e5 , Rho[i] , T[i] , P[i]))
    

def read_from_file(Rphotkm):

    # output is arrays : R, rho, T, P and Linf

    dirname = get_name()
    path = 'results/' + dirname + '/data/'
    filename = path + str(Rphotkm) + '.txt'

    def append_vars(line,varz,cols): # take line of file and append its values to variable lists 
        l=line.split()
        for var,col in zip(varz,cols):
            var.append(float(l[col]))

    R, rho, T, P = [[] for i in range (4)]
    with open(filename,'r') as f:
        for i,line in enumerate(f): 
            if i==0: 
                Linf = float(line.split()[-1])
            else:
                append_vars(line,[R, rho, T, P],[i for i in range(4)])

    return array(R),array(rho),array(T),array(P),Linf


def get_phot_list():

    # Returns list of photospheric radius that have solutions
    path = 'results/' + get_name() + '/data/'

    Rphotkms = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            Rphotkms.append(eval(filename[:-4]))

    return sort(Rphotkms)


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)


def export_values(target='./'):

    # Export useful values for analysis for each Rphot to a text file at target directory
    # Current values are : Linf,Rb,Tb,Rhob,Pb,Tphot,Rhophot
    # tsound(sound crossing time) 

    if target[-1]!='/': target += '/'
    Rphotkms = get_phot_list()

    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    # Parameters
    M, RNS, y_inner, comp, save, img = load_params()

    if comp == 'He':
        Z=2
        mu_I, mu_e, mu = 4, 2, 4/3

    kB,mp = 1.380658e-16, 1.67e-24
    def cs2(T):  # ideal gas sound speed  c_s^2  
        return kB*T/(mu*mp)
    
    with open(target+'envelope_values.txt','w') as f:

        f.write('{:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \n'.format(
            'Rph (km)','Linf (erg/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Tph (K)','rhoph (g/cm3)','tsound (s)'))

        for R in Rphotkms:
            r, rho, T, P, Linf = read_from_file(R)
            
            cs = sqrt(cs2(T))/1e5  # r is in km
            func_inverse_cs = interp1d(r,1/cs,kind='cubic')
            tsound,err = quad(func_inverse_cs,r[0],r[-1],epsrel=1e-5)#limit=100)
            print(tsound,err)
    
            f.write('%d \t\t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e\n'%
                (R,Linf,r[0]*1e5,T[0],rho[0],P[0],T[-1],rho[-1],tsound))


# export_values('../../compare')


# def pickle_save(name):
    
#     # Save all arrays into pickle file

#     # Import Winds
#     clean_rootfile()
#     logMDOTS,roots = load_roots()

#     if not os.path.exists('pickle/'):
#         os.mkdir('pickle/')

    