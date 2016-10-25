import math
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def MirrorGD(X,b,y,t,c=1e-2):
    tmp=np.zeros(len(b))
    step=c/math.sqrt(t+1)
    #step=10
    v=SignVec(X,b,y)
    g=np.zeros(len(b))
    for j in xrange(0,len(b)):
	for i in xrange(0,len(b)):
	    g[j]=g[j]+v[i]*X[i,j]
    for i in xrange(0,len(b)):
        tmp[i]=b[i]*math.exp(-step*g[i])
    b=tmp/np.sum(tmp)
    return b

def ProjGD(X,b,y,t,c=1e-5):
    step=c/math.sqrt(t+1)
    v=SignVec(X,b,y)
    g=np.zeros(len(b))
    for j in xrange(0,len(b)):
	for i in xrange(0,len(b)):
	    g[j]=g[j]+v[i]*X[i,j]
    b=projsplx(b-step*g)
    return b

def projsplx(y):
    x=-1*np.ones(len(y))
    tmp=list(y)
    tmp.sort()
    i=len(y)-2
    t=np.sum(tmp[i+1:len(y)])-1
    while t<=tmp[i] :
	if i>=0 :
            i=i-1
            t=(np.sum(tmp[i+1:len(y)])-1)/(len(y)-i-1)
        else :
            t=(np.sum(tmp)-1)/len(y)
	    break
    for i in xrange(0,len(y)):
        x[i]=max(0,y[i]-t)
    return x

def SignVec(X,b,y):
    v=np.zeros(len(b))
    for i in xrange(0,len(b)):
	v[i]=np.sign(np.dot(X[i,:],b)-y[i])
    return v

#def descent(update, A, b, reg, T=int(1e4)):
def descent(update, X, y, T):
    b = np.ones(X.shape[1])
    b=b/len(b);
    l1 = []
    for t in xrange(T):
        # update A (either subgradient or frank-wolfe)
        b = update(X, b, y, t)
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            l1.append(np.sum(abs(np.dot(X,b)-y)))
    return b, l1


def main(T=int(1e3)):
    X = np.load("X.npy")
    y = np.load("y.npy")
    T=5000

    x_pgd, l1_pgd = descent(ProjGD, X, y, T) 
    x_mgd, l1_mgd = descent(MirrorGD, X, y, T) 
    # add plots for BTLS

    plt.clf()
    plt.plot(l1_pgd, label='Projected GD')
    plt.plot(l1_mgd, label='Mirror GD')
    plt.title("L1 Norm")
    plt.legend()
    plt.savefig('l1_mirror.eps')
    
    f=open('hw4_3.txt','w')
    print >> f, "Projected GD \n", x_pgd
    print >> f, "Mirror GD \n", x_mgd
    f.close()

if __name__ == "__main__":
    main()
