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

from scipy import sparse as SM
import faiss

import build_graph
import diffusion_dataset



def do_diffusion(Ytr, I, (indexes2, Yte), clstep,
                 niter, storeLiters=None):

    print('[%.2f GiB] begin diffusion ' % (
        faiss.get_mem_usage_kb() / float(1<<20)))

    t0 = time.time()

    nclasses = Ytr.max() + 1
    if clstep == 0:
        clstep = nclasses
    N, k = I.shape
    assert N == Ytr.size
    nl1 = (Ytr >= 0).sum()

    W = build_graph.knngraph_to_CSRMatrix(I)

    # build the graph that links test points to the diffusion graph
    N2 = indexes2.shape[0]
    indptr = np.arange(N2 + 1) * k
    vals = np.ones(N2 * k, dtype='float32')
    W2 = SM.csr_matrix((vals, indexes2[:, :k].ravel(), indptr), shape=(N2, N))
    # no more normalization because if we symmetrize the matrix it
    # will mean there is propagation between the test images

    L_train = np.zeros((nl1, nclasses), dtype='float32')
    L_train[np.arange(nl1), Ytr[:nl1]] = 1

    val_L_tab = []
    print('[%.3f s, %.2f GiB] preproc done' % (
        time.time() - t0, faiss.get_mem_usage_kb() / float(1<<20)))

    for cl0 in range(0, nclasses, clstep):
        cl1 = min(cl0 + clstep, nclasses)
        ts = [time.time()]

        def pt():
            ts.append(time.time())
            return "  [%.3f s, %.2f GiB]" % (
                ts[-1] - ts[0], faiss.get_mem_usage_kb() / float(1<<20))

        print pt(), 'Classes %d:%d' % (cl0, cl1)

        L = np.zeros((N, cl1 - cl0), dtype='float32')

        L[:nl1, :] = L_train[:, cl0:cl1]

        # normalizations
        build_graph.normalize_columns(L)

        # diffusion to external points
        val_L = build_graph.sparse_dense_mul(W2, L)

        val_L_list = [val_L]
        val_L_tab.append(val_L_list)
        print pt(), 'start iter'

        for s in range(niter):

            # one diffusion step
            L = build_graph.CSRMatrix_mul_dense(W, L)

            # normalizations
            build_graph.normalize_columns(L)

            if storeLiters:
                fname = storeLiters % (s, cl0, cl1)
                print("storing L matrix in %s" % fname)
                np.save(fname, L)

            # diffusion to external points
            val_L = build_graph.sparse_dense_mul(W2, L)

            val_L_list.append(val_L)

            print pt(), 'iter %d val nnz %.3f' % (
                s, (val_L > 0).sum() / float(val_L.size))

        del L
        print

    # we can evaluate only at the end
    val_L_hist = []
    for it in range(niter + 1):

        val_L = np.hstack([val_L_list[it] for val_L_list in val_L_tab])
        val_L_hist.append(val_L)

        acc_full, acc_nc = Yte.compute_accuracies(val_L)

        if it == 0:
            itername = 'intial state'
        else:
            itername = 'iter %d/%d' % (it - 1, niter)

        print '   %s nnz %.2f top-5 accuracy: %.3f (%.3f on novel)' % (
            itername,
            (val_L != 0).sum() / float(val_L.shape[0]),
            acc_full, acc_nc)


    return val_L_hist




def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=['val', 'test'], default='val',
        help='validation or test'
    )
    parser.add_argument(
        '--nlabeled', type=int, default=2,
        help='nb of labeled images per class'
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='random seed to select labelled images'
    ) 
    parser.add_argument(
        '--k', type=int, default=30,
        help='number of neighbors in knn-graph' 
    )
    parser.add_argument(
        '--nbg', type=int, default=10**6,
        help='nb of background images'
    )    
   
    parser.add_argument(
        '--niter', type=int, default=60,
        help='number of diffusion iterations'
    )
    parser.add_argument(
        '--lslice', type=int, default=0,
        help='size of column slices in L matrix'
    )
    parser.add_argument(
        '--storeLname', type=str, default='',
        help='store L matrix under this name'
    )
    parser.add_argument(
        '--storeLiters', type=str, default='',
        help='store L iterates'
    )

    args = parser.parse_args()
    mode = args.mode
    if mode == 'val':
        print("========================== run on Validation")
    else:
        print("========================== run on Test")
        
    print "load train + test set"
    
    Xtr, Ytr, Xte, Yte = diffusion_dataset.load_traintest(
        args.nlabeled, class_set=1 if mode=='val' else 2, seed=args.seed,
        include_base_class=True, pca256=True)

    nclasses = Ytr.max() + 1
    print("dataset sizes: Xtr %s (%d labeled), " % (
        Xtr.shape, (Ytr >= 0).sum()) +           
          "Xte %s, %d classes (eval on %s)" % (
              Xte.shape, nclasses, Yte))

    Xdis = diffusion_dataset.load_bharath_distractors(args.nbg)

    I, D, (I2, D2) = build_graph.make_graph_with_precomputed_distractors(
        Xtr, Xdis, args.k, Xte)

    if args.lslice > 0:
        # let's face it: we are not really using D and it consumes RAM
        D = None

    # add -1 labels for bg images
    Ytr = np.hstack((Ytr, -np.ones(Xdis.shape[0], dtype=int)))

    val_L_hist = do_diffusion(
        Ytr, I, (I2, Yte), args.lslice, niter=args.niter,
        storeLiters=args.storeLiters)
    
    if args.storeLname:
        print 'storing L history in', args.storeLname
        cPickle.dump(val_L_hist, open(args.storeLname, 'w'), -1)

    
if __name__ == '__main__':

    main()
    
