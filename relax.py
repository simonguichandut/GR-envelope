'''
Relaxation method for a two-point boundary problem with two functions
* Form dY/dx = G(x,Y),  A<x<B
* Y = (y1,y2) dependent variables
* G = (g1,g2) functions of x,y1,y2
* x is continuous  and discretizable (non-regular grid spacing is fine)
* BCs : one at each boundary for one of the two functions, e.g f2(A)=alpha , f1(B)=beta

* Needs one (reasonable) trial solution of the derivatives of G can be expressed analytically, and two trial solutions
* if the derivatives (jacobian) have to be computed numerically
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import array, linspace, ones


def Newton(X, Yinit, derivs, BCs,      Jacobian='analytical', derivs2=None, Yinit2=None  ,epsilon=1e-10, nmax=100):

    ''' Arguments
    X : grid (list or numpy array)
    Yinit : list of two initial arrays [y1,y2]
    derivs : function which returns [g1,g2] - must have input arguments (x,Y(x))
    BCs : A list of the couples [which function (1 or 2) , boundary value]. For example in top description [[2,alpha] , [1,Beta]]
    Jacobian : analytical (requires derivs2) or approximate (requires yinit2) 
    derivs2 : second derivative functions
    Yinit2 : second initial solutions
    '''

    global x,m
    x,m = X,len(X)

    # Checking input
    if Jacobian == 'analytical':
        print('Running Newton algorithm using analytical jacobian')
        if derivs2 is None:
            raise Exception('Analytical jacobian method requires functions for second derivatives')

    elif Jacobian == 'approximate':
        print('Running Newton algorithm using approximate jacobian')
        if Yinit2 is None:
            raise Exception('Approximate jacobian requires two initial solutions')


    ya = stack_y(Yinit[0] , Yinit[1])
    sols = [[Yinit[0]],[Yinit[1]]]  # keep track of what happens to our solution over time

    if Jacobian == 'approximate':
        yb = stack_y(Yinit2[0] , Yinit2[1])
        sols[0].append(Yinit2[0])
        sols[1].append(Yinit2[1])

    n, go = 0, 1
    while go:

        print(n)
        J = Jac_approx(Err, ya, yb, derivs, BCs)
        
        pillow = 1
        ynew = yb - pillow*np.matmul(np.linalg.inv(J), Err(yb, derivs, BCs))
        
        while True in (ynew<0):
            pillow/=3
            print('pillow update : ',pillow)
            ynew = yb - pillow*np.matmul(np.linalg.inv(J), Err(yb, derivs, BCs))

        ya = yb[:]
        yb = ynew[:]
        
        Y = unstack_y(yb)
        sols[0].append(Y[0])
        sols[1].append(Y[1])
        

        err = np.linalg.norm(yb-ya)
#         print(err)
        if err < 1e-10:
            go = 0
            print('Converged after %d iterations\n' % n)

        n += 1
        if n == nmax+1:
            print('Maximum number of iterations (%d) reached\n'%nmax)
            go = 0

                
    return Y,sols



## Shape of function vector y

def stack_y(y1, y2): # gives y = [ y1_0,y2_0 , y1_1,y2_1, ... , y1_m,y2_m] (what is used in solving the matrix system)
    y = []
    for i in range(m):
        y.append(y1[i])
        y.append(y2[i])
    return array(y)

def unstack_y(y):   # gives y1 and y2, the m-points arrays
    y1, y2 = [], []
    for i in range(m):
        y1.append(y[2*i])
        y2.append(y[2*i+1])
    return np.array(y1), np.array(y2)

def get_yk(y, k):     # gives Y_k = (y1_k,y2_k), which is the input needed for G(x,y)
    y1, y2 = unstack_y(y)
    return (y1[k], y2[k])




# Writing problem as F=0

# def G(xk, y):  # RHS of dY/dx=G(x).  Expecting y to be [y1(x),y2(x)]
#     return [y[1], 1+2*xk]


def Err(y,derivs,BCs):  # error vector. Expecting y to be the m-stacked y1,y2
    y1, y2 = unstack_y(y)

    ## Boundary conditions
    fbc1,fbc2 =  BCs[0][0] , BCs[1][0]
    alpha,beta = BCs[0][1] , BCs[1][1]
   
    F0   = y1[0]-alpha  if fbc1==0 else y2[0]-alpha
    Fend = y1[-1]-beta if fbc2==0 else y2[-1]-beta

    F = [F0]  # first element is BC1

    ## Grid : k=1,2...m is evaluating derivatives
    for k in range(1, m):

        yk, ykm1 = get_yk(y, k), get_yk(y, k-1)
        Gk   = derivs(x[k],    yk)      
        Gkm1 = derivs(x[k-1], ykm1)

        for i in range(2):
            Fk = (yk[i] - ykm1[i])/(x[k]-x[k-1]) - 0.5*(Gk[i] + Gkm1[i])   # trapezoidal derivative
            F.append(Fk)

    # last element is BC2
    F.append(Fend)

    return array(F)


## Jacobians
def Jac_approx(func, ya, yb, *args):  # jacobian from finite difference derivatives (needs two points)

    def Jmunu(mu, nu):  # returns approximate of dF_mu/dy_nu
        num = func([ya[i] if i != nu else yb[i]
                    for i in range(2*m)],*args)[mu] - func(ya,*args)[mu]
        den = yb[nu]-ya[nu]
        return num/den

    J = np.zeros((2*m, 2*m))  # initialize matrix
    for mu in range(2*m):
        for nu in range(2*m):
            J[mu][nu] = Jmunu(mu, nu)

    return J


def Jac_exact(y):  # exact jacobian (problem specificc)
    pass
