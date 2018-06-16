# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function

import cPickle as pickle

# import c_graph_utils
import sys
import os

import time
import copy
import pdb
import argparse
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from scipy import sparse as SM

import faiss
import graph_utils
import diffusion_dataset

#############################################################
# Elementary functions
#############################################################



def normalize_columns(a):
    sx = a.sum(axis=0)
    sx[sx == 0] = 1
    a /= sx


#############################################################
# Graph matrix
#############################################################


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


def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    def prepare_block((i0, i1)):
        xb = x[i0:i1]
        return i0, i1, preproc(xb)

    return rate_limited_imap(prepare_block, block_ranges)


def ident(x):
    return x


def search_by_blocks(index, xq, k, preproc=ident, output=None,
                     output_files=None):
    n = xq.shape[0]
    bs = 16384
    if output_files is not None:
        file_D, file_I = output_files
    elif output is None:
        similarities = np.ones((n, k), dtype='float32') * np.nan
        indexes = -np.ones((n, k), dtype='int32')
    else:
        similarities, indexes = output
        assert similarities.shape == (n, k)
        assert indexes.shape == (n, k)

    for i0, i1, block in dataset_iterator(xq, preproc, bs):
        print("   search %d:%d / %d\r" % (i0, i1, n), end=' ')
        sys.stdout.flush()
        si, ii = index.search(block, k)
        if output_files is None:
            similarities[i0:i1] = si
            indexes[i0:i1] = ii
        else:
            ii.astype('int32').tofile(file_I)
            si.tofile(file_D)
    print()
    if output_files is None:
        return similarities, indexes

    
def move_index_to_gpu(index, shard=False):
    ngpu = faiss.get_num_gpus()
    gpu_resources = [faiss.StandardGpuResources() for i in
                     range(ngpu)]

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = shard
    co.shard_type = 1

    print("   moving to %d GPUs" % ngpu)
    t0 = time.time()
    index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)
    index.dont_dealloc_me = gpu_resources
    print("      done in %.3f s" % (time.time() - t0))
    return index


def make_index(sx, preproc=ident):
    N, p = sx.shape
    ngpu = faiss.get_num_gpus()

    if N < 1000:
        indextype = 'Flat'
    elif N < 10**6:
        indextype = 'GPUFlat'        
    elif N < 100000:
        indextype = 'GPUIVFFlat'
    else:
        indextype = 'GPUIVFFlatShards'
    
    if (indextype == 'IVFFlat' or indextype == 'GPUIVFFlat' or
        indextype == 'GPUIVFFlatShards'):
        ncentroids = int(4 * np.floor(np.sqrt(N)))
        nprobe = 256
        print("using IndexIVFFlat with %d/%d centroids" % (
            nprobe, ncentroids))
        q = faiss.IndexFlatL2(p)
        index = faiss.IndexIVFFlat(q, p, ncentroids, faiss.METRIC_INNER_PRODUCT)
        if nprobe >= ncentroids * 3 / 4:
            nprobe = int(ncentroids * 3 / 4)
            print("  forcing nprobe to %d" % nprobe)
        index.nprobe = nprobe
        index.quantizer_no_dealloc = q
        if indextype.startswith('GPU') and ngpu > 0:
            index = move_index_to_gpu(index, indextype == 'GPUIVFFlatShards')
        ntrain = min(ncentroids * 100, N)
        print("prepare train set, size=%d" % ntrain)
        trainset = sx[:ntrain]
        trainset.max()   # force move to RAM
        print("train")
        index.train(trainset)

    elif indextype == 'GPUFlat' or indextype == 'Flat':
        index = faiss.IndexFlatIP(p)
        if indextype.startswith('GPU') and ngpu > 0:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co)
    else:
        assert False

    bs = 16384
    for i0, i1, block in dataset_iterator(sx, preproc, bs):
        print("   add %d:%d / %d\r" % (i0, i1, N), end=' ')
        sys.stdout.flush()
        index.add(block)

    return index

def norm_L2(x):
    x = np.array(x) # copy and remove mmap
    faiss.normalize_L2(x)
    return x


def get_distractor_graph(sx_1, k):

    ts = [time.time()]
    def pt():
        ts.append(time.time())
        return "  [%.3f s, %.2f GiB]" % (
            ts[-1] - ts[-2],
            faiss.get_mem_usage_kb() / float(1<<20))

    ndis, d = sx_1.shape

    print(pt(), "make distractor graph for ndis=%d k=%d" % (ndis, k))
    
    fname_base = '%s/knngraph/ndis%d' % (os.getenv('DDIR'), ndis)
    
    print(pt(), "fname_base=", fname_base)

    fname_index = fname_base + ".index"

    if not os.path.exists(fname_index):

        index = make_index(sx_1, preproc=norm_L2)

        print(pt(), "move to CPU")
        index_cpu = faiss.index_gpu_to_cpu(index)

        print(pt(), "store", fname_index)
        faiss.write_index(index_cpu, fname_index)
        del index_cpu
        is_new_index = True
    else:

        print(pt(), "load", fname_index)
        index_cpu = faiss.read_index(fname_index)

        if faiss.get_num_gpus() > 0:
            index = move_index_to_gpu(index_cpu, True)
        else:
            # for run on cluster
            index = index_cpu
        del index_cpu
        is_new_index = False

    D_11, I_11 = None, None

    if not is_new_index:
        # otherwise presumaby we should recompute
        for log_ki in range(11):
            ki = 1 << log_ki
            if ki < k: continue

            fname_I = fname_base + '_k%d_I_11' % ki
            fname_D = fname_base + '_k%d_D_11' % ki

            if os.path.exists(fname_I + '.npy'):
                fname_D += '.npy'
                fname_I += '.npy'
                print(pt(), 'mmap', fname_D, fname_I)
                D_11 = np.load(fname_D, mmap_mode='r')
                I_11 = np.load(fname_I, mmap_mode='r')
                break

            if os.path.exists(fname_I + '.int%d' % ki):
                fname_I += '.int%d' % ki
                fname_D += '.float%d' % ki
                print(pt(), 'mmap', fname_D, fname_I)
                D_11 = np.memmap(fname_D, mode='r', dtype='float32').reshape(-1, ki)
                I_11 = np.memmap(fname_I, mode='r', dtype='int32').reshape(-1, ki)
                break

    if I_11 is None:
        # it was not computed for this value of ki
        for log_ki in range(11):
            ki = 1 << log_ki
            if ki >= k: break

        fname_D = fname_base + '_k%d_D_11.float%d' % (ki, ki)
        fname_I = fname_base + '_k%d_I_11.int%d' % (ki, ki)

        print(pt(), 'writing on-the-fly to ', fname_D, fname_I)
        file_D = open(fname_D, 'w')
        file_I = open(fname_I, 'w')

        search_by_blocks(index, sx_1, ki, preproc=norm_L2,
                         output_files=(file_D, file_I))

        del file_D
        del file_I

        print(pt(), 'mmap', fname_D, fname_I)
        D_11 = np.memmap(fname_D, mode='r', dtype='float32').reshape(-1, ki)
        I_11 = np.memmap(fname_I, mode='r', dtype='int32').reshape(-1, ki)

    assert D_11.shape == I_11.shape
    assert D_11.shape[0] == ndis
    assert D_11.shape[1] >= k

    print(pt(), 'distractor graph ready')
    return index, D_11, I_11



def make_graph_with_precomputed_distractors(sx_0, sx_1, k, sXtest):
    """ make a knn graph from sx_0 (labelled vectors) 
    sx_1 (big background)
    sXtest (test images)
    """
    ts = [time.time()]
    def pt():
        ts.append(time.time())
        return "[%.3f s, %.2f GiB]" % (
            ts[-1] - ts[-2],
            faiss.get_mem_usage_kb() / float(1<<20))

    ndis, d = sx_1.shape
    nl1, _ = sx_0.shape
    print(pt(), "make knn graph for %d+%d k=%d d=%d" % (
        nl1, ndis, k, d))

    index, D_11, I_11 = get_distractor_graph(sx_1, k)

    print(pt(), 'spherifying all descriptors')

    faiss.normalize_L2(sx_0)
    # faiss.normalize_L2(sx_1)
    faiss.normalize_L2(sXtest)

    print(pt(), "search labelled", sx_0.shape)
    D_10, I_10 = search_by_blocks(index, sx_0, k)

    print(pt(), "search test", sXtest.shape)
    D_12, I_12 = index.search(sXtest, k)
    I_12 = I_12.astype('int32')

    del index

    N0 = sx_0.shape[0]

    if True:  # N0 < 100000:
        print(pt(), "make Flat index for labelled")
        index = faiss.IndexFlatIP(d)
        index.add(sx_0)
        if faiss.get_num_gpus() > 0:
            index = move_index_to_gpu(index, False)
    else:
        print(pt(), "make IVFFlat index for labelled")
        index = make_index(opt, sx_0)

    print(pt(), "alloc output")

    D_03 = np.empty((nl1 + ndis, k), dtype='float32')
    I_03 = np.empty((nl1 + ndis, k), dtype='int32')

    print(pt(), "search labelled")
    D_00, I_00 = index.search(sx_0, k)
    D_03[:nl1] = D_00
    I_03[:nl1] = I_00

    print(pt(), "search distractors")
    search_by_blocks(index, sx_1, k, preproc=norm_L2,
                     output=(D_03[nl1:], I_03[nl1:]))

    print(pt(), "search test")
    D_02, I_02 = index.search(sXtest, k)
    I_02 = I_02.astype('int32')

    del index

    print(pt(), "merge results")

    graph_utils.merge_int_result_table_with(
        nl1, k,
        faiss.swig_ptr(I_03[:nl1]), faiss.swig_ptr(D_03[:nl1]),
        faiss.swig_ptr(I_10), faiss.swig_ptr(D_10),
        False, nl1)

    del I_10, D_10

    if I_11.dtype != 'int32':
        I_11 = I_11.astype('int32')

    # here we need to set the strides because {I,D}_11 may be longer
    graph_utils.merge_int_result_table_with(
        ndis, k,
        faiss.swig_ptr(I_03[nl1:]), faiss.swig_ptr(D_03[nl1:]),
        faiss.swig_ptr(I_11), faiss.swig_ptr(D_11),
        False, nl1, k, I_11.shape[1])

    del I_11, D_11

    graph_utils.merge_int_result_table_with(
        I_02.shape[0], k,
        faiss.swig_ptr(I_02), faiss.swig_ptr(D_02),
        faiss.swig_ptr(I_12), faiss.swig_ptr(D_12),
        False, nl1)

    print(pt(), "graph done")

    return I_03, D_03, (I_02, D_02)


#############################################################
# Efficient multiplication
#############################################################


def knngraph_to_CSRMatrix(I):
    """ converts the neighbor indices I to a CSRMatrix, by: 
    - compute W0 + W0.T
    - L1-normalize the W0 rows"""
    
    print('   knngraph_to_CSRMatrix in: %.2f GiB' %
          (faiss.get_mem_usage_kb() / float(1<<20)))
    N, k = I.shape
    CSRMatrix = graph_utils.CSRMatrix
    swig_ptr = graph_utils.swig_ptr

    indptr = np.arange(N + 1, dtype='uint64')
    indptr *= k
    I = np.ascontiguousarray(I, dtype='int32')
    vals = np.ones(I.size, dtype='float32')

    W1 = CSRMatrix(N, N, swig_ptr(indptr), swig_ptr(I), swig_ptr(vals))

    print('      A: %.2f GiB' %
          (faiss.get_mem_usage_kb() / float(1<<20)))

    W2 = W1.transpose()

    print('      B: %.2f GiB' %
          (faiss.get_mem_usage_kb() / float(1<<20)))

    # changes I!
    W1.sort_rows()
    W2.sort_rows()

    W = W1.point_op(W2, CSRMatrix.Pop_add)
    W.rows_normalize_L1()

    print('      C: %.2f GiB' %
          (faiss.get_mem_usage_kb() / float(1<<20)))

    del W1, W2, vals

    print('      D: %.2f GiB, matrix nnz=%d' %
          (faiss.get_mem_usage_kb() / float(1<<20),
           W.count_nz()))

    return W



def sparse_dense_mul(A, B):
    if False: # A.indices.dtype == 'int32':
        return mkl_dot(A, B)
    else:
        # parallel scipy multiplication
        nt = 20
        pool = ThreadPool(nt)
        n = A.shape[0]
        Aslices = [A[i * n / nt : (i + 1) * n / nt]
                   for i in range(nt)]
        m = pool.map(lambda Ai: Ai * B, Aslices)
        return np.vstack(m)


def CSRMatrix_mul_dense(A, B, bs=1 << 17):
    swig_ptr = graph_utils.swig_ptr
    indptr = graph_utils.ulvec_rev_swig_ptr(A.lims, A.nrow + 1)
    nnz = int(indptr[-1])
    indices = graph_utils.ivec_rev_swig_ptr(A.idx, nnz)
    vals = graph_utils.fvec_rev_swig_ptr(A.val, nnz)

    m, k = int(A.nrow), int(A.ncol)
    k2, n = B.shape
    assert k == k2
    R = np.empty((m, n), dtype='float32')

    # here we have to loop over blocks to avoid indptrs of > 31 bits
    # that are not supported by MKL
    
    for i0 in range(0, m, bs):
        i1 = min(m, i0 + bs)
        
        # restriction of matrix A to rows i0:i1
        o0, o1 = indptr[i0], indptr[i1]
        indptr_i = (indptr[i0 : i1 + 1] - o0).astype('int32')
        indices_i = indices[o0:o1]
        vals_i = vals[o0:o1]

        # convert everything to MKL calling conversions (Fortran!)
        mnk = np.array([i1 - i0, n, k], dtype='int32')
        mptr = swig_ptr(mnk[0:])
        nptr = swig_ptr(mnk[1:])
        kptr = swig_ptr(mnk[2:])
        ab = np.array([1.0, 0.0], dtype='float32')
        alpha = swig_ptr(ab[0:])
        beta = swig_ptr(ab[1:])
        graph_utils.mkl_scsrmm("N", mptr, nptr, kptr,
                               alpha, "GIIC", swig_ptr(vals_i),
                               swig_ptr(indices_i),
                               swig_ptr(indptr_i),
                               swig_ptr(indptr_i[1:]),
                               swig_ptr(B), nptr, beta,
                               swig_ptr(R[i0:i1]), nptr)
    return R
    

if __name__ == '__main__':
    ndis = int(sys.argv[1])
    k = 30
    Xdis = diffusion_dataset.load_bharath_distractors(ndis)
    get_distractor_graph(Xdis, k)

    
