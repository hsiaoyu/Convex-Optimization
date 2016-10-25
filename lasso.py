import math
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt


def frank_wolfe(x, A, b, t, gam, flag):
    R=b-np.dot(A,x)
    tmp=0
    vmax=0
    alpha=0.4
    beta=0.8
    for i in xrange(0,len(x)):
	tmp=-np.dot(A[:,i].T,R)
        if abs(tmp) > vmax :
	   vmax=abs(tmp)
	   vindex=i
           vsign=np.sign(tmp)
    dd=-gam*vsign*np.eye(1,len(x),vindex).flatten()-x
    if flag==0 :
        step=2.0/(t+2.0)
    else :
        step=1.0
        while 0.5*pow(la.norm(np.dot(A,x+step*dd)-b),2)>0.5*pow(la.norm(np.dot(A,x)-b),2)+alpha*step*np.dot(np.dot(A.T,np.dot(A,x)-b).T,dd) :
		step=beta*step
    x=x+step*dd
    return x


def subgradient(x, A, b, t, lam, flag, c=1e-5):
    # update x (your code here), set c above
    #nt=c/math.sqrt(t+1)
    #x=softTH(x+nt*np.dot(A.T,b-dot(A,x)))
    v=np.zeros(len(x))
    step=c/math.sqrt(t+1)
    for i in xrange(0,len(x)):
	if x[i]>0 :
	   v[i]=lam
	elif x[i]<0 :
	   v[i]=-lam
	else :
	   v[i]=0
    sg=np.dot(A.T,np.dot(A,x)-b)+v
    x=x-step*sg
    return x

def softTH(x,lam):
    xnew=list(x)  # to avoid changing x while calling the function
    for i in xrange(0,len(x)) :
	if (xnew[i]>lam) :
	    xnew[i]=xnew[i]-lam
	elif (xnew[i]<-lam) :
	    xnew[i]=xnew[i]+lam
    return xnew

# add BTLS variants and include them in main/descent below

#def descent(update, A, b, reg, T=int(1e4)):
def descent(update, A, b, reg, T, flag):
    x = np.zeros(A.shape[1])
    error = []
    l1 = []
    for t in xrange(T):
        # update A (either subgradient or frank-wolfe)
        x = update(x, A, b, t, reg, flag)
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
#	    print t,"\n", la.norm(np.dot(A,x)-b)
            l1.append(np.sum(abs(x)))
 #           print np.sum(abs(x))
            assert not np.isnan(error[-1])
    return x, error, l1


def main(T=int(1e3)):
    A = np.load("A.npy")
    b = np.load("b.npy")

    # modify regularization parameters below
#    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=0., T=T)
#    x_fw, error_fw, l1_fw = descent(frank_wolfe, A, b, reg=0., T=T)
    # add BTLS experiments
#    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=0., T=T)
    print 0.5*pow(la.norm(b),2)
    T=10000
    reg=0.3
    gam=1.5
    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg, T, 0)
    x_fw, error_fw, l1_fw = descent(frank_wolfe, A, b, gam, T, 0)
    x_fwb, error_fwb, l1_fwb = descent(frank_wolfe, A, b, gam, T, 1)
   
    value1=0.5*pow(la.norm(np.dot(A,x_sg)-b),2)+reg*(np.sum(abs(x_sg)))
    value2=0.5*pow(la.norm(np.dot(A,x_fw)-b),2)+reg*(np.sum(abs(x_fw)))
    value3=0.5*pow(la.norm(np.dot(A,x_fwb)-b),2)+reg*(np.sum(abs(x_fwb)))
    print value1,"\n",value2,"\n",value3
    # add plots for BTLS
    plt.clf()
    plt.plot(error_sg, label='Subgradient')
    plt.plot(error_fw, label='Frank-Wolfe')
    plt.plot(error_fwb, label='FW_BTLS')
    plt.title('Error  lamda=%f'%reg)
    plt.legend()
    plt.savefig('error.eps')

    plt.clf()
    plt.plot(l1_sg, label='Subgradient')
    plt.plot(l1_fw, label='Frank-Wolfe')
    plt.plot(l1_fwb, label='FW_BTLS')
    plt.title("L1 Norm  lamda=%f"%reg)
    plt.legend()
    plt.savefig('l1.eps')


if __name__ == "__main__":
    main()
