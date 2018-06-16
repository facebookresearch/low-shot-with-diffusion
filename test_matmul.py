# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import build_graph
import numpy as np
from scipy import sparse as SM
import graph_utils
import build_graph
import faiss

M = 100
N = 50
K = 60

swig_ptr = graph_utils.swig_ptr

rs = np.random.RandomState(123)
A = rs.rand(M, K).astype('float32')
A.ravel()[rs.rand(A.size) < 0.7] = 0


A = SM.csr_matrix(A)
print A.shape, A.nnz
B = rs.rand(K, N).astype('float32')

ref_AB = A * B

indices = A.indices.astype('int32')
indptr = A.indptr.astype('uint64')

A_csr = graph_utils.CSRMatrix(M, K, swig_ptr(indptr),
                              swig_ptr(indices),
                              swig_ptr(A.data))

new_AB = build_graph.CSRMatrix_mul_dense(A_csr, B, bs=5)


print 'err=', np.abs(ref_AB - new_AB).sum() / np.abs(ref_AB).sum()

N = 1000
K = 40

I = np.array([rs.choice(N, K, replace=False) for i in range(N)])

W0 = SM.csr_matrix((np.ones(N * K, dtype='float32'),
                    I.ravel(),
                    np.arange(N + 1) * K), shape=(N, N))

def sparsediag(v):
    # returns a sparse matrix with the entries of the
    # vector v on the diagonal
    N = v.shape[0]
    rows = np.arange(N, dtype = 'uint64')
    cols = np.arange(N, dtype = 'uint64')
    D = SM.csr_matrix((v, (rows, cols)), shape=(N, N))
    return D


def normalize_weights(W, reciprocal=False):
    # given a sparse matrix W, outputs a sparse matrix
    # that is D^{-1} (W+W^T)
    # where D is the diagonal matrix with the row sums of
    # W+W^T on the diagonal
    if not reciprocal:
        W2 = W + W.transpose()
    else:
        W2 = W.multiply(W.transpose())
    N = W2.shape[0]
    ones = np.ndarray((N), dtype='float32')
    ones.fill(1)
    s = W2*ones
    ones /= s
    D = sparsediag(ones)
    return D * W2

ref_W = normalize_weights(W0)


new_W_c = build_graph.knngraph_to_CSRMatrix(I)

def CSRMatrix_to_csr(A):
    return SM.csr_matrix(
        (graph_utils.fvec_rev_swig_ptr(A.val, A.count_nz()),
         graph_utils.ivec_rev_swig_ptr(A.idx, A.count_nz()),
         graph_utils.ulvec_rev_swig_ptr(A.lims, A.nrow + 1)),
        shape=(A.nrow, A.ncol))

new_W = CSRMatrix_to_csr(new_W_c)

print 'err=', np.abs(ref_W - new_W).sum() / np.abs(ref_W).sum()


