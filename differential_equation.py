# Universal approximation
# 07 / 18 / 2021
# Edgar A. M. O.


import matplotlib.pyplot as plt
from d2l import mxnet as d2l
from mxnet import gluon, autograd, np, npx
from mxnet.gluon import nn
import math
import random
npx.set_np()


def load_array(data_arrays, batch_size, is_train=True):
    """ construct a Gluon data iterator """
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, train_iter, updater, loss): 
    """ Train a model within one epoch """
    # Sum of training loss, sum of training accuracy, no. of examples
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            g0 = f0 + (X * net(X))
            g0.detach()
            g1 = f0 + ((X + inf_s) * net(X + inf_s))
            g1.detach()
            dg = (g1 - g0) / inf_s
            l = loss(dg, y)
        l.backward()
        updater(X.shape[0])


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Create synthetic features 
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

f0 = 1                                      # initial condition
inf_s = random.random() / 200               # infinitesimal step
a = -1                                      # function lower bound
b = 1                                       # function upper bound
size = 1000                                 # size of sample

set_X = (b - a) * np.random.rand(size) + a  # X sample
np.random.shuffle(set_X)
set_X = set_X.reshape(-1, 1)

set_Y = 2 * set_X                           # Y sample (u' = 2x)


def fit(epochs=200):
    # loss function
    loss = gluon.loss.L2Loss()
    # net type
    net = nn.Sequential()
    # layers
    net.add(nn.Dense(16, activation='sigmoid'))
    net.add(nn.Dense(16, activation='sigmoid'))
    net.add(nn.Dense(1))
    # initialize weights
    net.initialize()

    batch_size = 20

    train_iter = load_array((set_X, set_Y), batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    for epoch in range(epochs):
        train(net, train_iter, trainer, loss)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print('epoch ' + str(epoch + 1) + ' completed...')

    x = np.linspace(a, b, 20).reshape(-1, 1)
    f = x ** 2 + f0
    g = f0 + (x * net(x))

    plt.plot(x, f)
    plt.plot(x, g)
    plt.legend(('Analytical solution', 'Prediction'))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
 




