import numpy as np
import idx2numpy
import Image
import pickle
import os

if not os.path.exists('X.bin'):
    images=idx2numpy.convert_from_file("train-images-idx3-ubyte")
    Y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

    data=[]
    temp=[]
    index = 0;
    for image , label in zip(images, Y):
        index = index + 1
        for i in image:
            for j in i:
                temp.append(j/255.0)

        for i in range(10):
            if(i != label):
                temp.append(0.0)
            elif(index > 50000):
                temp.append(0.0)
            else:
                temp.append(1.0)
        data.append(temp)
        temp=[]

    X = np.asarray(data , 'float32')
    f_x = open('X.bin','wb')
    pickle.dump(X,f_x)
else:
    f_x = open('X.bin','rb')
    X = pickle.load(f_x)

#X = np.matrix(X)
X_test = X[50000:]
X = X[:50000]

#make pair with X and Y

print X.shape

sizes=[X.shape[1],100]

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
        return np.float32(1./(1.+np.exp(-X)) > np.random.random(X.shape))


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
            return ret

    def h2v(self,h,k = True):
        if len(h.shape) == 2:
            ret = np.dot(np.ones([h.shape[0],1],'float32'), self.b) + np.dot(h, self.W.T)
        else:
            ret = self.b+np.dot(h, self.W.T)

        if(k):
            return self.sigmrnd(ret)
        else:
            return ret


class dbn:
    def __init__(self,sizes = [],learning_rate = 0.01,numepochs = 1):
        print 'dbn init ,sizes:',sizes,', numepochs:',numepochs
        self.sizes = sizes
        self.rbms = []
        self.learning_rate = learning_rate
        self.numepochs = numepochs

        for i in range(len(self.sizes)-1):
            self.rbms.append(rbm(sizes[i:i+2],self.learning_rate,self.numepochs))

    def train(self,X):
        #for i in range(self.numepochs):
      for j in range(len(self.sizes)-1):
            self.rbms[j].train(X)
            X = self.rbms[j].v2h(X)

    def v2h(self,X):
        for j in range(len(self.sizes)-1):
            X = self.rbms[j].v2h(X)
        return X

    def h2v(self,X):
        for j in range(len(self.sizes)-1):
            X = self.rbms[len(self.sizes)-j-1].h2v(X);
        return X

    def predict(self, X):
        return self.h2v(self.v2h(X))


size_test = [X.shape[1], 100]
rbm1=rbm(size_test, 0.01, 10)
rbm1.train(X)

im=Image.open('2.bmp')
t=np.array(im)
tt=[]
for i in t:
    for j in i:
        tt.append(j/255.0)
for k in range(10):
        tt.append(0);

print rbm1.h2v(rbm1.v2h(np.asarray(tt)),False)[-10:]
