import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class Learner:

    def __init__(self, batch_size, learning_rate, n_hidden):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
    
    def softmax(self, X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    
    def relu(self, X):
        return T.maximum(X, 0)
                    
    def init_model(self, n_features, n_classes):
        self.W1 = theano.shared(0.01 * np.random.rand(n_features, self.n_hidden).astype(theano.config.floatX) - 0.005, borrow=True)
        self.B1 = theano.shared(np.zeros(self.n_hidden, dtype=theano.config.floatX))
        self.W2 = theano.shared(0.01 * np.random.rand(self.n_hidden, n_classes).astype(theano.config.floatX) - 0.005, borrow=True)
        self.B2 = theano.shared(np.zeros(n_classes, dtype=theano.config.floatX))
        self.params = [self.W1, self.B1, self.W2, self.B2]

    def model(self, X):
        h = self.relu(T.dot(X, self.W1) + self.B1)
        y = self.softmax(T.dot(h, self.W2) + self.B2)
        return y
    
    def cross_entropy(self, Yhat, Y):
        return T.mean(-T.log(T.diag(Yhat.T[Y])))
        
    def update_model(self, cost):
        updates = OrderedDict()
        for V in self.params:
            G = T.grad(cost, wrt=V)
            A = theano.shared(V.get_value()*0., borrow=False)
            A2 = A + G**2
            V2 = V - self.learning_rate * G / T.sqrt(A2+1e-6)
            updates[A] = A2
            updates[V] = V2
        return updates
    
    def fit(self, data, n_steps=100000, print_n=1000, check_n=-1, test=None):
        self.init_model(data.shape[1]-1, len(np.unique(data[:,0].reshape(-1))))
        X = T.matrix()
        Y = T.ivector()
        Yhat = self.model(X)
        cost = self.cross_entropy(Yhat, Y)
        updates = self.update_model(cost)
        train_function = theano.function([X, Y], cost, updates=updates)
        c = []
        for i in range(n_steps):
            idxs = np.random.choice(len(data), size=self.batch_size)
            x = data[idxs]
            y = x[:,0].reshape(-1).astype('int32')
            x = x[:,1:].astype(theano.config.floatX)
            c.append(train_function(x, y))
            if ((i+1) % print_n) == 0:
                print(i+1, np.mean(c))
                c = []
            if check_n > 0 and ((i+1) % check_n) == 0:
                print('Test classification accuracy after {} steps: {}'.format(i+1, self.classification_accuracy(test)))

    def predict(self, data):
        if not hasattr(self, 'predict_function'):
            X = T.matrix()
            Yhat = self.model(X)
            self.predict_function = theano.function([X], Yhat)
        yhat = self.predict_function(data)
        return yhat
        
    def classification_accuracy(self, test):
        labels = test[:,0]
        test = test[:,1:]
        yhat = np.argmax(self.predict(test), axis=1)
        return (labels==yhat).sum()/len(labels)
