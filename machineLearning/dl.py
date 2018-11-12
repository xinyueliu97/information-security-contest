# -*- coding: utf-8 -*-

import sys
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from dA import *


def test_dA(train_set_x, train_set_x_feature_num,learning_rate=0.1, training_epochs=8,
            batch_size=5, da_object = None):
			#改batch_size也要改ml.py里面的b_s

    # print train_set_x
    # accessing the third minibatch of the training set
    train_set_x = theano.shared(numpy.array(train_set_x))

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size #求出batch的个数
    #print n_train_batches				#这里是50000 / 20 = 2500
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    # 使用denoise时
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    # da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
    #         n_visible=28 * 28, n_hidden=500) # 创建dA对象时，并不需要数据x，只是给对象da中的一些网络结构参数赋值

    if da_object is None:
        da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible= train_set_x_feature_num, n_hidden= train_set_x_feature_num*3/4) # 创建dA对象时，并不需要数据x，只是给对象da中的一些网络结构参数赋值
    else:
        #da = da_object	
        da = dA(numpy_rng=da_object.n_rng, theano_rng=da_object.theano_rng, input=x,
                n_visible= train_set_x_feature_num, n_hidden= train_set_x_feature_num*3/4,
                W = da_object.W, bvis = da_object.b_prime, bhid = da_object.b) # 创建dA对象时，并不需要数据x，只是给对象da中的一些网络结构参数赋值


    cost, updates, y = da.get_cost_updates(corruption_level=0.3,
                                        learning_rate=learning_rate)

    train_da = theano.function([index], y, updates=updates, #theano.function()为定义一个符号函数，这里的自变量为indexy，指定input 参数是index（加了括号），指定output是cost，指定函数名称是上面的get_cost_updates
         givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]} ) #输出变量为cost


    for epoch in xrange(training_epochs):
        hiddeny = numpy.array([])
        for batch_index in xrange(n_train_batches):
            yy = train_da(batch_index)

            sys.stdout.write("step:%d" % (batch_index + 1) + ' / ' + "%d" % n_train_batches + "         %d%%" % ((batch_index + 1) * 100 / n_train_batches)  + '\r')
            sys.stdout.flush()
            if hiddeny.size == 0:
                hiddeny = yy
            else:
                hiddeny = numpy.concatenate((hiddeny, yy))

    print '\n'
    return hiddeny, da
