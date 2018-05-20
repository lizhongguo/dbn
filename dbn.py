import numpy as np
from rbm import *
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
            if j == len(self.sizes)-2:
                X = self.rbms[len(self.sizes)-j-2].h2v(X,False);
            else:
                X = self.rbms[len(self.sizes)-j-2].h2v(X)
        return X

    def predict(self, X):
        return self.h2v(self.v2h(X))

