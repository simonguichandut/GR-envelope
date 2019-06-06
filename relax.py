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
from numpy import array,linspace,ones



def Driver(X, Yinit, derivs ,BCs ,      Jacobian='analytical',derivs2=None,yinit2=None):

    ''' Arguments
    X : grid (list or numpy array)
    Yinit : list of two initial arrays [y1,y2]
    derivs : list of two functions g1,g2  (must have input arguments (x,y1,y2) )
    BCs : A list of the couples [which function (1 or 2) , boundary value]. For example in top description [[2,alpha] , [1,Beta]]
    Jacobian : analytical (requires derivs2) or approximate (requires yinit2)

    global m
    m = len(x)
    '''
    pass





# Shape of function vector y  

def stack_y(y1,y2): # gives y = [ y1_0,y2_0 , y1_1,y2_1, ... , y1_m,y2_m] (what is used in solving the matrix system)
    y = []
    for i in range(m):
        y.append(y1[i])
        y.append(y2[i])
    return array(y)

def unstack_y(y):   # gives y1 and y2, the m-points arrays
    y1,y2= [],[]
    for i in range(m):
        y1.append(y[2*i])
        y2.append(y[2*i+1])
    return np.array(y1),np.array(y2)

def get_yk(y,k):     # gives Y_k = (y1_k,y2_k), which is the input needed for G(x,y)
    y1,y2 = unstack_y(y)
    return [y1[k],y2[k]]


# Writing problem as F=0

def G(xk,y): # RHS of dY/dx=G(x).  Expecting y to be [y1(x),y2(x)] 
    return [y[1] , 1+2*xk]

def Err(y): # error vector. Expecting y to be the m-stacked y1,y2 
    y1,y2 = unstack_y(y)
    F0,Fend = y1[0]-alpha , y1[-1]-beta
#     F = [F0,0]
    F=[F0]
    for k in range(1,m):
        
        yk,ykm1 = get_yk(y,k) , get_yk(y,k-1)
        Gk   = G(x[k] , yk)
        Gkm1 = G(x[k-1] , ykm1)
        
        for i in range(2):
            Fk = (yk[i] - ykm1[i])/(x[k]-x[k-1]) - 0.5*(Gk[i] + Gkm1[i])
            F.append(Fk)

#     F.extend([0,Fend])
#     F.extend([Fend,0])
    F.append(Fend)
        
    return array(F)


# Jacobians

def Jac_approx(func,ya,yb): # jacobian from finite difference derivatives (needs two points)
    
    def Jmunu(mu,nu): # returns approximate of dF_mu/dy_nu
        num   = func([ya[i] if i!=nu else yb[i] for i in range(2*m)])[mu] - func(ya)[mu]
        den   = yb[nu]-ya[nu]
#         prit(num)
#         print(array([ya[i] if i!=nu else yb[i] for i in range(2*m)]) - array(ya))
#         print(func([ya[i] if i!=nu else yb[i] for i in range(2*m)]) - func(ya))
        return num/den
    
    J = np.zeros((2*m,2*m)) # initialize matrix
    for mu in range(2*m):
        for nu in range(2*m):
            J[mu][nu] = Jmunu(mu,nu)
            
    return J

def Jac_exact(y): # exact jacobian (problem specificc)
    pass
    
            

# Iterator
            
def Newton(func,ya,yb,nmax=100,**kwargs):
    
    n,go = 1,1
    sols = [ya]
    while go:
        
        sols.append(yb)
        print(n)
        J = Jac_approx(func,ya,yb)
        print(np.linalg.det(J))
        ynew = yb - np.matmul(np.linalg.inv(J) , func(yb))
        
        ya = yb[:]
        yb = ynew[:]
   
        err = np.linalg.norm(yb-ya)
#         print(err)
        if err<1e-10:
            go=0
            print('Converged after %d iterations'%n)
            
        n+=1
        if n==nmax:
            go=0
                
    return sols,n





