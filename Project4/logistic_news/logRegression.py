import math
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt


def GD( B, X, y, mu, t, c=1e-3):
    step=c
    N=X.shape[0]
    numC=B.shape[1]
    for n in xrange(0,numC):
	tmp=np.zeros(X.shape[1])
	for i in xrange(0,N):
            z=0
            for j in xrange(0,numC):
	         z=z+math.exp(-1*np.dot(B[:,j],X[i,:]))
     #       print "z   ",z
	    val=-math.exp(-1*np.dot(B[:,n],X[i,:]))/z
            if y[i]==n :
               val=1+val
     #       print "i  ",i, "val   ",val
            val=val/N
            tmp=tmp+val*X.T[:,i]
        B[:,n]=B[:,n]-step*(tmp+2*mu*B[:,n])
    return B

def accGD (B, B0, X, y, mu, alpha) :
    Bnew=np.zeros((B.shape[0],B.shape[1]))
    beta=1000
    alphaNew=(1+math.sqrt(1+4*alpha*alpha))/2
    r=(1-alpha)/alphaNew
    N=X.shape[0]
    numC=B.shape[1]
    for n in xrange(0,numC):
	tmp=np.zeros(X.shape[1])
	for i in xrange(0,N):
            z=0
            for j in xrange(0,numC):
	         z=z+math.exp(-1*np.dot(B[:,j],X[i,:]))
	    val=-math.exp(-1*np.dot(B[:,n],X[i,:]))/z
            if y[i]==n :
               val=1+val
            val=val/N
            tmp=tmp+val*X.T[:,i]
        Bnew[:,n]=B[:,n]-(tmp+2*mu*B[:,n])/beta
    B=Bnew*(1-r)+B0*r
    return B, alphaNew, Bnew

def TestVal(B, X, y, mu):
    N=X.shape[0]
    numC=B.shape[1]
    val1 = 0
    val2 = 0
    for i in xrange(0,N):
	val1=val1+np.dot(B[:,y[i]],X[i,:])
        z=0
        for j in xrange(0,numC):
	    z=z+math.exp(-1*np.dot(B[:,j],X[i,:]))
        val2=val2+math.log10(z)/N
    val3=val2+val1/N
    val2=val2+val1/N+mu*np.sum(B**2)
    return val3

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
    from numpy import genfromtxt
    #x_test = np.random.rand(2,2)
    #y_test = [0,1]
    x_train = genfromtxt('X_train.csv', delimiter=',')
    y_train = genfromtxt('y_train.csv', delimiter=',')
    x_test = genfromtxt('X_test.csv', delimiter=',')
    y_test = genfromtxt('y_test.csv', delimiter=',')
    mu = 0.2
    T = 100
    nrow = x_train.shape[1]
    B=np.zeros((nrow,20))
    B1=np.zeros((nrow,20))
    B0 =np.empty_like(B1)
    B0[:] = B1;
    alpha =0
    err_train = []
    err_test = []
    err_acctrain = []
    err_acctest = []
    #f=open('hw4_1.txt','a')
    #(val1,val2)=TestVal(B,x_train,y_train,mu)
    #print >> f, "mu=",mu,"  T=",T, "  c=0.1\n", "  Initial val=", val1, "  ",val2
    #(val1,val2)=TestVal(B,x_test,y_test,mu)
    #print >> f, "Test set  ", "  Initial val=", val1, "  ",val2
    for t in xrange(T) :
	B=GD(B, x_train, y_train, mu, t)
        err_train.append(TestVal(B,x_train,y_train,mu))
        err_test.append(TestVal(B,x_test,y_test,mu))
	B1, alpha, B0 = accGD(B1, B0, x_train, y_train, mu, alpha)
        err_acctrain.append(TestVal(B1,x_train,y_train,mu))
        err_acctest.append(TestVal(B1,x_test,y_test,mu))

    plt.clf()
    plt.plot(err_train, label='GD')
    plt.plot(err_acctrain, label='Acc. GD')
    plt.title('Train Loss')
    plt.legend()
    plt.savefig('train_loss.eps')

    plt.clf()
    plt.plot(err_test, label='GD')
    plt.plot(err_acctest, label='accGD')
    plt.title('Test Los')
    plt.legend()
    plt.savefig('test_loss.eps')
if __name__ == "__main__":
    main()
