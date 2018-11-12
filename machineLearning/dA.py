# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T #theano中一些常见的符号操作在子库tensor中
from theano.tensor.shared_randomstreams import RandomStreams

class dA(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=28*28, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True) #W,bvis,bhid都为共享变量
        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), borrow=True)
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.n_rng = numpy_rng
        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input #保存输入数据
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input #binomial()函数为产生0，1的分布，这里是设置产生1的概率为p

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate): #每调用该函数一次，就算出了前向传播的误差cost，网络参数及其导数

        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        y = self.get_hidden_values(tilde_x)

        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam)) #append列表中存的是参数和其导数构成的元组
        return (cost, updates, y)