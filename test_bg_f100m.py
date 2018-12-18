#! /usr/bin/env python2
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import numpy as np
import sys
import time
import pdb
import cPickle
import argparse

import faiss

import diffusion
import diffusion_dataset





mode, nlabeled, seed, k, ndis = sys.argv[1:6]

niter = 60
lslice = 0
storeLname = None
storeLiters = None

if len(sys.argv) > 6:
    niter = int(sys.argv[6])

if len(sys.argv) > 7:
    lslice = int(sys.argv[7])

if len(sys.argv) > 8:
    storeLname = sys.argv[8]

if len(sys.argv) > 9:
    storeLiters = sys.argv[9]

nlabeled = int(nlabeled)
seed = int(seed)
k = int(k)
ndis = int(ndis)


featureversion = 2

if mode.endswith('_fv3'):
    featureversion = 3
    mode = mode[:-4]

if mode == 'val':
    print("========================== run on Validation")
else:
    print("========================== run on Test")


class Opt:
    pass

opt = Opt()


# graph params
opt.k = k  # we compute the graph for that

if True:

    Xtr, Ytr, Xte, Yte = diffusion_dataset.load_bharath_traintest(
        nlabeled, class_set=1 if mode=='val' else 2, seed=seed,
        nnonlabeled=0, pca256=True, include_base_class=True,
        featureversion=featureversion)

    nclasses = Ytr.max() + 1
    print("dataset sizes: Xtr %s (%d labeled), Xte %s, %d classes (eval on %s)" % (
            Xtr.shape, (Ytr >= 0).sum(),
            Xte.shape, nclasses, Yte))


    if ndis > 0:
        # add distractors
        Xdis = diffusion_dataset.load_bharath_distractors(
            ndis, pca256=True, featureversion=featureversion)

        print 'distractors shape', Xdis.shape

        opt.dis_key = 'bharath256rr_ndis%d' % ndis
        opt.indextype = 'GPUIVFFlatShards'
        opt.indexnprobe = 256
        opt.cachedir = '/mnt/vol/gfsai-east/ai-group/users/matthijs/finetune_750_250/diffusion_results/LRD_cache'
        if featureversion == 3:
            opt.cachedir += '_fv3'
        nclasses = Ytr.max() + 1

        I, D, (I2, D2) = diffusion.make_graph_with_precomputed_distractors(
            Xtr, Xdis, opt, Xte)

        if lslice > 0:
            # let's face it: we are not really using D and it consumes RAM
            D = None

        Ytr = np.hstack((Ytr, -np.ones(Xdis.shape[0], dtype=int)))

    else:
        opt.externalTest = True
        opt.pca = 0
        opt.indextype = 'IVFFlat'
        opt.indexnprobe = 256
        opt.cacheDistractorGraph = False

        I, D, (I2, D2) = diffusion.make_graph_from_points(Xtr, opt, Xtest=Xte)

    print "storing cache"

    # cPickle.dump((Ytr, I, D, I2, D2, Yte), open('/data/local/users/matthijs/tmp/cache.pickle', 'w'), -1)
    if False:
        outdir = '/mnt/vol/gfsai-east/ai-group/users/matthijs/finetune_750_250/diffusion_results/ITERSTATS//ndis100M_nl2_k30_seed1/'
        pdb.set_trace()

        np.save(outdir + 'I.npy', I)
        np.save(outdir + 'I2.npy', I2)
        cPickle.dump((Ytr, I, D, I2, D2, Yte), open(outdir + 'cache.pickle', 'w'), -1)
else:
    print "loading cache"
    (Ytr, I, D, I2, D2, Yte) = cPickle.load(open('/data/local/users/matthijs/tmp/cache.pickle', 'r'))


print "Performing diffusion"

# diffusion options
opt.weights = '1'
opt.weightsigma = 0
opt.reciprocal = False
opt.diffiters = niter
opt.tau = 1.0
opt.sinkhorn = 4
opt.reset1hot = False


if lslice > 0:
    val_L_hist = diffusion.do_diffusion_lslice(opt, Ytr, I, D, (I2, D2, Yte), lslice,
                                               storeLiters=storeLiters)
else:
    val_L_hist = diffusion.do_diffusion(opt, Ytr, I, D, (I2, D2, Yte),
                                        storeLiters=storeLiters)

if storeLname is not None:
    print 'storing L history in', storeLname
    cPickle.dump(val_L_hist, open(storeLname, 'w'), -1)
