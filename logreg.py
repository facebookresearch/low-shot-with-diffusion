#! /usr/bin/env python2
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import faiss
import sys
import time


def train(model, X, labels, opts):
    optimizer = optim.SGD(model.parameters(),
                          lr = opts.lr,
                          momentum = opts.mom,
                          weight_decay=opts.wd)
    N = X.shape[0]
    er = 0
    for it in range(0, opts.maxiter):
        dt = opts.lr
        model.train()
        optimizer.zero_grad()

        idx = np.random.randint(0,N,opts.batchsize)
        x = Variable(torch.Tensor(X[idx]))
        y = Variable(torch.from_numpy(labels[idx]).long())
        yhat = model(x)
        loss = F.nll_loss(yhat, y)
        er = er + loss.data[0]
        loss.backward()
        optimizer.step()

        if it % opts.verbose == 1:
            print(er/opts.verbose)
            er = 0
    return er/opts.verbose



def train_balanced(model, X, labels, opts, (freq, Xte, Yte)):
    optimizer = optim.SGD(model.parameters(),
                          lr = opts.lr,
                          momentum = opts.mom,
                          weight_decay=opts.wd)
    unq, inv, cnt = np.unique(labels,
                              return_inverse=True,
                              return_counts=True)
    lid = np.split(np.argsort(inv), np.cumsum(cnt[:-1]))
    N = X.shape[0]
    er = 0
    nlabels = len(lid)
    llid = np.zeros(nlabels).astype('int')
    for i in range(nlabels):
        llid[i] = len(lid[i])
    t0 = time.time()
    for it in range(opts.maxiter):
        dt = opts.lr
        model.train()
        optimizer.zero_grad()
        idx = np.random.randint(0,nlabels,opts.batchsize)
        for t in range(opts.batchsize):
            i = idx[t]
            idx[t] = lid[i][np.random.randint(0,llid[i])]
        x = Variable(torch.Tensor(X[idx]))
        y = Variable(torch.from_numpy(labels[idx]).long())
        yhat = model(x)
        loss = F.nll_loss(yhat, y)
        er = er + loss.data[0]
        loss.backward()
        optimizer.step()

        if it % opts.verbose == 1:
            print(er/opts.verbose)
            er = er/opts.verbose
        if (it+1) % freq == 0:
            print('[%.3fs] iteration %d' % (time.time() - t0, it + 1), end=' ')
            validate(Xte, Yte, model)
    return er


def validate(tX, tlabels, model):
        """ compute top-1 and top-5 error of the model """
        model.eval()
        N = tX.shape[0]
        batch = tX
        batchlabels = tlabels
        x = Variable(torch.Tensor(batch), volatile=True)
        yhat = model(x)
        yhat = yhat.data.numpy()
        print(tlabels.do_eval(yhat))
        return yhat

class Net(nn.Module):
    def __init__(self, d, nclasses):
        super(Net, self).__init__()
        self.l1 = nn.Linear(d,nclasses)
    def forward(self, x):
        return F.log_softmax(self.l1(x))




##########################################################3
#

import diffusion_dataset


def clone_rebias(net, bidx):
    net2 = lu.Net(net.l1.weight.size(1),net.l1.weight.size(0))
    net2.l1.bias.data[:] = net.l1.bias.data
    net2.l1.weight.data[:] = net.l1.weight.data
    net2.l1.bias.data[bidx]= -20
    return net2


parser = argparse.ArgumentParser()
parser.add_argument('--nlabeled', type=int, default = 2)
parser.add_argument('--maxiter', type=int, default = 1500)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--verbose', type=int, default=500)
parser.add_argument('--pcadim', type=int, default=2048)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--mom', type=float, default=0.0)
parser.add_argument('--mode', default='val')
parser.add_argument('--storemodel', type=str, default='')
parser.add_argument('--storeL', type=str, default='')


opts = parser.parse_args()
mode = opts.mode

Xtr, Ytr, Xte, Yte = diffusion_dataset.load_traintest(
    opts.nlabeled,
    class_set=1 if mode=='val' else 2,
    seed=opts.seed,
    pca256=False)


nclasses = max(int(Ytr.max()), int(Yte.Yte.max())) + 1
Xtr_orig = Xtr
Xte_orig = Xte

print("dataset sizes: Xtr %s (%d labeled), Xte %s, %d classes (eval on %s)" % (
    Xtr.shape, (Ytr >= 0).sum(),
    Xte.shape, nclasses, Yte))


net= Net(opts.pcadim, nclasses)


print('============== start logreg')


if opts.mode == 'val':
    eval_freq = 500
else:
    eval_freq = opts.maxiter

train_balanced(net, Xtr, Ytr, opts, (eval_freq, Xte, Yte))

if opts.storeL:
    L = validate(Xte, Yte, net)
    print('writing', opts.storeL)
    np.save(opts.storeL, L)

if opts.storemodel:
    print('writing', opts.storemodel)
    torch.save(net.state_dict(), opts.storemodel)

