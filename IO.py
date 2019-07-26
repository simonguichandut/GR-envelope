''' Input and Output '''

import os
from numpy import log10,array

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

    # output is arrays : R, rho, T, P, L and Linf

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
            if i==0: Linf = float(line.split()[-1])
            append_vars(line,[R, rho, T, P],[i for i in range(4)])

    return array(R),array(rho),array(T),array(P),Linf


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)



def pickle_save(name):
    
    # Save all arrays into pickle file

    # Import Winds
    clean_rootfile()
    logMDOTS,roots = load_roots()

    if not os.path.exists('pickle/'):
        os.mkdir('pickle/')

    