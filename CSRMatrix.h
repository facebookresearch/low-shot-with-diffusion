/* Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdint.h>
#include <vector>
#include <cstdio>

typedef int32_t idx_t; // not uint because there are -1 pts
typedef uint64_t lim_t;

/**
 * Compressed sparse row matrix. Used mainly to store a graph with < 2B nodes
 *
 * The edges are oriented but may or may not have weights. The
 * objective is to have a few well optimized algorithms. However,
 * there are also standard matrix ops that need to be implemented.
 *
 * CSRMatrix does not manage its memory. Inheriting objects must
 * allocate or free the data contained in the pointers.
 */
struct CSRMatrix {

    idx_t nrow; ///< nb rows
    idx_t ncol; ///< nb columns

    lim_t *lims; ///< size nrow + 1

    /** Column index for each non-zero cell, size lims[nrow],
     * (i, idx[l]) are the non-0 cells of the matrix, with
     *
     *    lims[i] <= l < lims[i+1]
     *
     * the array is not necessarily sorted and may contain -1's for
     * entries that must be ignored.
     */
    idx_t * idx;

    /** If non-NULL, is of size lims[nrow] contains values for each
     * index, ie. entry (i, idx[l]) of the matrix contains value
     * val[l] */
    float *val;

    CSRMatrix ()
    {}

    CSRMatrix (int nrow, int ncol,
               lim_t *lims,
               idx_t * idx,
               float *val):
        nrow (nrow), ncol (ncol),
        lims(lims), idx (idx), val(val)
    {}

    virtual ~CSRMatrix() {}

    /// count number of registered entries in the matrix
    lim_t count_nz () const {return lims [nrow]; }

    /// transposes (slow)
    CSRMatrix *transpose () const;

    /// sorts rows in-place (fast)
    void sort_rows ();

    bool check_valid() const ;

    enum pointwise_operation_t {
        Pop_min_union, /// union of cells, entries = min of entries
        Pop_min,       /// intersection of cells
        Pop_max,       /// union, max
        Pop_add,       /// union, sum
    };

    /// applies a pointwise operation on two matrices (both must be
    /// sorted on input, without -1 entries, output is sorted as well).
    CSRMatrix *point_op (const CSRMatrix &other,
                         pointwise_operation_t pop) const;


    // row and column normalization
    void rows_normalize_L1 ();


};


/// malloc'ed graph.
struct MallocCSRMatrix : CSRMatrix {
    std::vector<lim_t> vlims; // these are stored explictly

    // copy the data into self
    explicit MallocCSRMatrix (const CSRMatrix &mat);

    MallocCSRMatrix (int nrow, int ncol, size_t nnz, bool have_val = true);
    MallocCSRMatrix (int nrow, int ncol,
                     const lim_t * lims, const idx_t *idx, const float *vals);
    virtual ~MallocCSRMatrix();
};



// exposing MKL function
extern "C" {

typedef int MKL_INT;

void mkl_scsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
}


size_t merge_int_result_table_with (size_t n, size_t k,
                                    int *I0, float *D0,
                                    const int *I1, const float *D1,
                                    bool keep_min = true,
                                    long translation = 0,
                                    long stride0 = -1, long stride1 = -1);


#endif
