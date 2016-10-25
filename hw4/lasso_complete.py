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

def ProxGD(x, A, b, t, lam, flag):
    step=0.001
    sg=np.dot(A.T,np.dot(A,x)-b)
    #x=x-step*sg
    x=softTH(x-step*sg,lam)
    return x
def FISTA(x, A, b, y0, lam, alpha):
    beta=100000
    sg=np.dot(A.T,np.dot(A,x)-b)
    alphaNew=(1+math.sqrt(1+4*alpha*alpha))/2
    r=(1-alpha)/alphaNew
    y=softTH(x-sg/beta,lam/beta)
    x=(1-r)*y+r*y0
    return x, alphaNew, y

def softTH(xnew, lam):
    for i in xrange(0,len(xnew)) :
	if (xnew[i]>lam) :
	    xnew[i]=xnew[i]-lam
	elif (xnew[i]<-lam) :
	    xnew[i]=xnew[i]+lam
	else :
	    xnew[i]=0
    return xnew

# add BTLS variants and include them in main/descent below

#def descent(update, A, b, reg, T=int(1e4)):
def descent(update, A, A_test, b, b_test, reg, T, flag):
    x = np.zeros(A.shape[1])
    error = []
    error_t=[]
    for t in xrange(T):
        x = update(x, A, b, t, reg, flag)
        if (t % 1 == 0) or (t == T - 1):
           error.append(la.norm(np.dot(A, x) - b))
           error_t.append(la.norm(np.dot(A_test, x) - b_test))
    return x, error, error_t


def main(T=int(1e3)):
    A = np.load("A_train.npy")
    b = np.load("b_train.npy")
    A_test = np.load("A_test.npy")
    b_test = np.load("b_test.npy")
    # modify regularization parameters below
#    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=0., T=T)
#    x_fw, error_fw, l1_fw = descent(frank_wolfe, A, b, reg=0., T=T)
    # add BTLS experiments
#    x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=0., T=T)
    T=10000
    reg=0.2
    x_sg, error_sg, err_sg_test = descent(subgradient, A, A_test, b, b_test, reg, T, 0)
    x_fw, error_fw, err_fw_test = descent(frank_wolfe, A, A_test, b, b_test, reg, T, 0)
    x_px, error_px, err_px_test = descent(ProxGD, A, A_test, b, b_test, reg, T, 0)
    
    x = np.zeros(A.shape[1])
    y0 = x
    alpha =0
    err_FISTA = []
    err_FISTA_test = []
    for t in xrange(T):
        x,alpha,y0 = FISTA(x, A, b, y0, reg, alpha)
        if (t % 1 == 0) or (t == T - 1):
            err_FISTA.append(la.norm(np.dot(A, x) - b))
            err_FISTA_test.append(la.norm(np.dot(A_test, x) - b_test))
   
    # add plots for BTLS
    plt.clf()
    plt.plot(error_sg, label='Subgradient')
    plt.plot(error_fw, label='Frank-Wolfe')
    plt.plot(error_px, label='Proximal GD')
    plt.plot(err_FISTA, label='FISTA')
    plt.title('Train Error  lamda=%f'%reg)
    plt.legend()
    plt.savefig('hw4_2_train.eps')

    plt.clf()
    plt.plot(err_sg_test, label='Subgradient')
    plt.plot(err_fw_test, label='Frank-Wolfe')
    plt.plot(err_px_test, label='Proximal GD')
    plt.plot(err_FISTA_test, label='FISTA')
    plt.title('Test Error  lamda=%f'%reg)
    plt.legend()
    plt.savefig('hw4_2_test.eps')

if __name__ == "__main__":
    main()
