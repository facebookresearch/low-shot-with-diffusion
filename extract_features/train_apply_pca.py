# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import faiss
import h5py
import sys
import os
from multiprocessing.dummy import Pool as ThreadPool

todo = sys.argv[1:]



def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


fv3_dir = os.getenv('DDIR') + '/features/'

if 'train' in todo:

    f = h5py.File(fv3_dir + 'f100m/block0.hdf5', 'r')

    count= f['count'][0]
    labels = f['all_labels'][:count]
    features = f['all_feats'][:count]

    pca = faiss.PCAMatrix(2048, 256, 0, True)

    pca.train(features)
    faiss.write_VectorTransform(pca, fv3_dir + 'PCAR256.vt')


if 'apply' in todo:
    pca = faiss.read_VectorTransform(fv3_dir + 'PCAR256.vt')

    def load_block(i):
        f = h5py.File(fv3_dir + 'f100m/block%d.hdf5' % i, 'r')
        count= f['count'][0]
        # labels = f['all_labels'][:count]
        features = f['all_feats'][:count]
        return features

    # one read thread, one PCA computation thread, and main thread writes result.
    src = rate_limited_imap(load_block, range(100))
    src2 = rate_limited_imap(pca.apply_py, src)
    f = open(fv3_dir + '/concatenated_PCAR256.raw', 'w')

    i = 0
    for x in src2:
        x.tofile(f)
        f.flush()
        i += x.shape[0]
        print "wrote %d\r" % i, 
        sys.stdout.flush()
