import math
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def ProjGD(X,M,O,Oc,t,flag) :
    if flag==1 :
       step=1/(t+1)
    else :
       step=1/math.sqrt(t+1)
    U,s,V = np.linalg.svd(X,full_matrices=False)
    Z=np.dot(U,V)
    X=X-step*Z
    X=X*Oc+M*O
    return X

#def descent(update, A, b, reg, T=int(1e4)):
def descent(update, X, M, O, Oc, T, flag):
    err = []
    for t in xrange(T):
        # update A (either subgradient or frank-wolfe)
        X = update(X, M, O, Oc, t, flag)
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            tmp = la.norm(M-X,'fro')
            err.append(tmp*tmp/10000)
    return X, err


def main(T=int(1e3)):
    from numpy import genfromtxt
    M = genfromtxt('M.csv', delimiter=',')
    O = genfromtxt('O.csv', delimiter=',')
    Oc= np.ones((O.shape[0],O.shape[1]))    
    Oc= -1*O+Oc
    X = M*O
    T = 1000
    X1, err1 = descent(ProjGD, X, M, O, Oc, T, 1) 
    X2, err2 = descent(ProjGD, X, M, O, Oc, T, 0) 
    
    plt.clf()
    plt.plot(err1, label='1/t')
    plt.plot(err2, label='1/sqrt(t)')
    plt.title("Error")
    plt.legend()
    plt.savefig('hw4_4.eps')
    
    #f=open('hw4_3.txt','w')
    #print >> f, "Projected GD \n", x_pgd
    #print >> f, "Mirror GD \n", x_mgd
    #f.close()

if __name__ == "__main__":
    main()
