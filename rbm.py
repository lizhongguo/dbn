import numpy as np

class rbm:
    def __init__(self,sizes=[],learning_rate=0.01,numepochs = 1):
        print 'rbm init ,sizes:',sizes,', numepochs:',numepochs
        self.W=np.matrix(np.zeros(sizes,'float32'))
        self.vW=np.matrix(np.zeros(sizes,'float32'))

        self.b=np.matrix(np.zeros(sizes[0],'float32'))
        self.vb=np.matrix(np.zeros(sizes[0],'float32'))

        self.c=np.matrix(np.zeros(sizes[1],'float32'))
        self.vc=np.matrix(np.zeros(sizes[1],'float32'))

        self.learning_rate = learning_rate
        self.numepochs = numepochs

        self.v1=self.b
        self.v2=self.b
        self.h1=self.c
        self.h2=self.c
        self.c1=self.W
        self.c2=self.W

    def sigmrnd(self,X):
        return np.float32((1./(1.+np.exp(-X))) > np.random.random(X.shape))

    def sigm(self,X):
        return np.float32(1./(1.+np.exp(-X)))

    def train(self,X):
        print 'begin train , X.shape',X.shape,'numepochs:',self.numepochs
        for i in range(self.numepochs):
            err = 0
            for v in X:
                self.v1 = np.matrix(v)

                self.h1 = self.sigmrnd(self.c+np.dot(self.v1,self.W))
                self.v2 = self.sigmrnd(self.b+np.dot(self.h1,self.W.T))
                self.h2 = self.sigmrnd(self.c+np.dot(self.v2,self.W))


                self.c1 = np.dot(self.h1.T,self.v1).T
                self.c2 = np.dot(self.h2.T,self.v2).T

                self.vW = self.learning_rate*(self.c1-self.c2)
                self.vb = self.learning_rate*(self.v1-self.v2)
                self.vc = self.learning_rate*(self.h1-self.h2)

                self.W = self.W + self.vW
                self.b = self.b + self.vb
                self.c = self.c + self.vc

                err = err + np.linalg.norm(self.v1-self.v2)
            print "err :",err
        return self.W,self.b,self.c

    def v2h(self,v,k = True):
        if len(v.shape) == 2:
             ret = np.dot(np.ones([v.shape[0],1],'float32') ,self.c) + np.dot(v,self.W)
        else:
            ret = self.c + np.dot(v, self.W)

        if(k):
            return self.sigmrnd(ret)
        else:
            return self.sigm(ret)

    def h2v(self,h,k = True):
        if len(h.shape) == 2:
            ret = np.dot(np.ones([h.shape[0],1],'float32'), self.b) + np.dot(h, self.W.T)
        else:
            ret = self.b+np.dot(h, self.W.T)

        if(k):
            return self.sigmrnd(ret)
        else:
            return self.sigm(ret)

