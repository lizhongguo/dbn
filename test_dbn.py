import numpy as np
import idx2numpy
import Image
import pickle
import os

from dbn import *

need_retrain = True

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

#using 0-50000 to train dbn and the other for test
X_test = X[50000:]
X = X[:50000]



size_test = [X.shape[1], 100, 10]


if os.path.exists('dbn.bin') and (not need_retrain):
    f_dbn = open('dbn.bin','rb')
    dbn1 = pickle.load(f_dbn)
else:
    f_dbn = open('dbn.bin','wb')
    dbn1=dbn(size_test, 0.01, 30)
    dbn1.train(X)
    pickle.dump(dbn1,f_dbn)


def predict(X):
    l = dbn1.predict(X)[:,-10:]
    return l.argmax(1)


Y = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
Y_test = Y[5000:]

result = np.reshape(predict(np.asarray(X_test)),-1).tolist()[0]
print len(result)

result_right = [0,0,0,0,0,0,0,0,0,0]
result_sum = [0,0,0,0,0,0,0,0,0,0]

for r,y in zip(result,Y_test):
    #print r,y
    if r == y:
        result_right[r] = result_sum[r]+1
    result_sum[r] = result_sum[r] + 1

for i in range(10):
    print str(i) +  ' accuraccy:  ' + str(float(result_right[i])/float(result_sum[i]))
