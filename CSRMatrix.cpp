/* Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "CSRMatrix.h"

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/mman.h>

#include <omp.h>
#include <algorithm>


#include "mkl.h"


/*************************************************
 * CSRMatrix
 *************************************************/


bool CSRMatrix::check_valid () const
{

    printf("Check lims\n");
    for (idx_t i = 0; i < nrow; i++) {
        if (!(lims[i] <= lims[i+1])) {
            printf ("wrong lims order row %d: %ld %ld\n",
                    i, lims[i], lims[i+1]);
            return false;
        }
    }
    lim_t nnz = lims[nrow];
    printf("Check arrays\n");
    printf("I0 %d\n", idx[0]);
    printf("I1 %d\n", idx[nnz - 1]);
    if (val) {
        printf("V0 %g\n", val[0]);
        printf("V1 %g\n", val[nnz - 1]);
    } else {
        printf("has no vals\n");
    }
    printf("Check indices\n");

    bool has_minusones = false;
    bool is_sorted = true;

    for (idx_t i = 0; i < nrow; i++) {
        lim_t l0 = lims[i], l1 = lims[i+1];
        idx_t prev_i = -1;
        for (lim_t l = l0; l < l1; l++) {
            if (idx[l] == -1) {
                has_minusones = true;
            } else if (idx[l] >= 0 && idx[l] < ncol) {
                if (idx[l] < prev_i)
                    is_sorted = false;
                prev_i = idx[l];
            } else {
                printf("wrong index %ld (line %d) = %d\n", l, i, idx[l]);
                return false;
            }
        }
    }
    printf("Check ok! has_minusones=%d is_sorted=%d\n",
           int(has_minusones), int(is_sorted));
    return true;
}

CSRMatrix *CSRMatrix::transpose () const
{
    MallocCSRMatrix *m;
    std::vector<lim_t> nnz (ncol + 1);
    lim_t ntot = 0;
#pragma omp parallel
    {
        std::vector<lim_t> nnz_l (ncol + 1);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

#pragma omp for
        for (idx_t i = 0; i < nrow; i++) {
            lim_t l0 = lims[i], l1 = lims[i+1];
            for (lim_t j = l0; j < l1; j++) {
                if (idx[j] >= 0) {
                    nnz_l [idx[j]] ++;
                }
            }
        }

#pragma omp critical
        {
            for (idx_t i = 0; i < ncol; i++) {
                ntot += nnz_l[i];
                nnz[i] += nnz_l[i];
            }
        }

#pragma omp barrier
        if (rank == 0) {
            m = new MallocCSRMatrix (ncol, nrow, ntot, val != nullptr);
            lim_t prev = 0;
            for (idx_t i = 0; i < ncol; i++) {
                prev += nnz[i];
                m->lims[i + 1] = nnz[i] = prev;
            }
        }

        for (int sl = 0; sl < nt; sl++) {
#pragma omp barrier
            int slice = (sl + rank) % nt;
            idx_t i0 = slice * long(ncol) / nt;
            idx_t i1 = (slice + 1) * long(ncol) / nt;
            for (idx_t i = i0; i < i1; i++) {
                nnz[i] -= nnz_l[i]; // reserve this many slots
                nnz_l[i] = nnz[i]; // remember where we start
            }
        }

        // assumes openmp schedules the loop in the same way as in the
        // first loop
#pragma omp for
        for (idx_t i = 0; i < nrow; i++) {
            lim_t l0 = lims[i], l1 = lims[i+1];
            for (lim_t j = l0; j < l1; j++) {
                idx_t k = idx[j];
                if (k < 0) continue;
                lim_t l = nnz_l[k]++;
                m->idx[l] = i;
                if (val) m->val[l] = val[j];
            }
        }
    }

    return m;
}

struct Cell {
    float val;
    idx_t i;
    bool operator < (const Cell & other) const {
        return i < other.i;
    }
};

void CSRMatrix::sort_rows ()
{
    if (!val) {
#pragma omp parallel for
        for (idx_t i = 0; i < nrow; i++) {
            lim_t l0 = lims[i], l1 = lims[i+1];
            std::sort (idx + l0, idx + l1);
        }
    } else {
#pragma omp parallel
        {
            std::vector<Cell> cells (ncol);

#pragma omp for
            for (idx_t i = 0; i < nrow; i++) {
                lim_t l0 = lims[i], l1 = lims[i+1];
                for (lim_t l = l0; l < l1; l++) {
                    cells[l - l0].i = idx[l];
                    cells[l - l0].val = val[l];
                }
                std::sort (cells.begin(), cells.begin() + l1 - l0);
                for (lim_t l = l0; l < l1; l++) {
                    idx[l] = cells[l - l0].i;
                    val[l] = cells[l - l0].val;
                }
            }

        }
    }

}


static lim_t count_inter (const idx_t *ai, lim_t a0, lim_t a1,
                          const idx_t *bi, lim_t b0, lim_t b1)
{
    lim_t n_inter = 0;
    while (a0 < a1 && b0 < b1) {
        if (ai[a0] == bi[b0]) {
            n_inter++;
            a0 ++; b0++;
        } else if (ai[a0] < bi[b0]) {
            a0++;
        } else {
            b0++;
        }
    }
    return n_inter;
}

static lim_t count_union (const idx_t *ai, lim_t a0, lim_t a1,
                          const idx_t *bi, lim_t b0, lim_t b1)
{
    return a1 - a0 + b1 - b0 - count_inter (ai, a0, a1, bi, b0, b1);
}



CSRMatrix *CSRMatrix::point_op (const CSRMatrix &other,
                                pointwise_operation_t pop) const
{
    assert (nrow == other.nrow && ncol == other.ncol);

    std::vector<lim_t> nnz (nrow + 1);

    // it the result on the intersection or on the union?
    bool is_inter = pop == Pop_min;

#pragma omp parallel for
    for (idx_t i = 0; i < nrow; i++) {
        lim_t l0 = lims[i], l1 = lims[i+1];
        lim_t ol0 = other.lims[i], ol1 = other.lims[i+1];
        if (is_inter)
            nnz[i] = count_inter (idx, l0, l1,
                                  other.idx, ol0, ol1);
        else
            nnz[i] = count_union (idx, l0, l1,
                                  other.idx, ol0, ol1);

    }
    lim_t accu = 0;
    for (idx_t i = 0; i < nrow; i++){
        lim_t tmp = nnz[i];
        nnz[i] = accu;
        accu += tmp;
    }
    nnz[nrow] = accu;
    //printf ("accu = %ld / (%ld + %ld)\n", accu, count_nz(), other.count_nz());
    assert (accu <= count_nz() + other.count_nz());

    MallocCSRMatrix *m = new MallocCSRMatrix
        (nrow, ncol, accu, val != nullptr);

    // printf ("alloc ok\n");

    m->vlims = nnz;
    m->lims = m->vlims.data();
    const idx_t *ai = idx;
    const idx_t *bi = other.idx;

#pragma omp parallel for
    for (idx_t i = 0; i < nrow; i++) {
        lim_t a0 = lims[i], a1 = lims[i+1];
        lim_t b0 = other.lims[i], b1 = other.lims[i+1];
        lim_t & ofs = nnz[i];
        if (is_inter) {
            while (a0 < a1 && b0 < b1) {
                if (ai[a0] == bi[b0]) {
                    // only min implemented for now
                    m->idx[ofs] = ai[a0];
                    if (val)
                        m->val[ofs] = std::min (val[a0], other.val[b0]);
                    ofs++;
                    a0++; b0++;
                } else if (ai[a0] < bi[b0]) {
                    a0++;
                } else {
                    b0++;
                }
            }
        } else {
            while (a0 < a1 || b0 < b1) {
                float va = 0, vb = 0;
                lim_t j;
                idx_t dj = a0 < a1 ?
                    (b0 < b1 ? ai[a0] - bi[b0] : -1) :
                    1;

                if (dj <= 0) {
                    if (val) va = val[a0];
                    j = ai[a0++];
                }
                if (dj >= 0) {
                    if (other.val) vb = other.val[b0];
                    j = bi[b0++];
                }
                m->idx[ofs] = j;
                if (val) {
                    if (pop == Pop_add) m->val[ofs] = va + vb;
                    else if (pop == Pop_max) m->val[ofs] = std::max(va, vb);
                    else if (pop == Pop_min_union) m->val[ofs] = std::min(va, vb);
                }
                ofs++;
            }
        }
    }
    return m;
}


void CSRMatrix::rows_normalize_L1 () {
#pragma omp parallel for
    for (idx_t i = 0; i < nrow; i++) {
        lim_t l0 = lims[i], l1 = lims[i+1];
        double sum_v = 0;
        for (lim_t j = l0; j < l1; j++) {
            sum_v += val[j];
        }
        if (sum_v > 0) {
            float f = 1 / sum_v;
            for (lim_t j = l0; j < l1; j++) {
                val[j] *= f;
            }
        }
    }

}





/*************************************************
 * MallocCSRMatrix
 *************************************************/

MallocCSRMatrix::MallocCSRMatrix (
         int nrow, int ncol,
         const lim_t * lims, const idx_t *idx, const float *val):
    CSRMatrix (nrow, ncol, nullptr, nullptr, nullptr)
{
    vlims.resize (nrow + 1);
    this->lims = vlims.data();
    memcpy(this->lims, lims, sizeof(lims[0]) * (nrow + 1));
    size_t nnz = lims[nrow];
    this->idx = new idx_t [nnz];
    memcpy(this->idx, idx, sizeof(idx[0]) * nnz);
    if (val) {
        this->val = new float [nnz];
        memcpy(this->val, val, sizeof(val[0]) * nnz);
    }
}

MallocCSRMatrix::MallocCSRMatrix (int nrow, int ncol, size_t nnz, bool have_val):
    CSRMatrix (nrow, ncol, nullptr, nullptr, nullptr)
{
    vlims.resize (nrow + 1);
    lims = vlims.data();
    idx = new idx_t [nnz];
    val = have_val ? new float [nnz] : nullptr;
}


MallocCSRMatrix::MallocCSRMatrix (const CSRMatrix &other):
    CSRMatrix (other.nrow, other.ncol, nullptr, nullptr, nullptr)
{
    vlims.resize (nrow + 1);
    lims = vlims.data();
    memcpy (lims, other.lims, sizeof (*lims) * (nrow + 1));
    lim_t nnz = count_nz();
    idx = new idx_t [nnz];
    memcpy (idx, other.idx, sizeof (*idx) * nnz);
    if (other.val) {
        val = new float [nnz];
        memcpy (val, other.val, sizeof(*val) * nnz);
    }
}

MallocCSRMatrix::~MallocCSRMatrix()
{
    delete [] idx;
    delete [] val;
}




size_t merge_int_result_table_with (size_t n, size_t k,
                                    int *I0, float *D0,
                                    const int *I1, const float *D1,
                                    bool keep_min,
                                    long translation,
                                    long stride0, long stride1)
{
    size_t n1 = 0;


    if (stride0 == -1) stride0 = k;
    if (stride1 == -1) stride1 = k;

#pragma omp parallel reduction(+:n1)
    {
        std::vector<int> tmpI (k);
        std::vector<float> tmpD (k);

        #pragma omp for
        for (size_t i = 0; i < n; i++) {
            int *lI0 = I0 + i * stride0;
            float *lD0 = D0 + i * stride0;
            const int *lI1 = I1 + i * stride1;
            const float *lD1 = D1 + i * stride1;
            size_t r0 = 0;
            size_t r1 = 0;

            if (keep_min) {
                for (size_t j = 0; j < k; j++) {

                    if (lI0[r0] >= 0 && lD0[r0] < lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            } else {
                for (size_t j = 0; j < k; j++) {
                    if (lI0[r0] >= 0 && lD0[r0] > lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            }
            n1 += r1;
            memcpy (lD0, tmpD.data(), sizeof (lD0[0]) * k);
            memcpy (lI0, tmpI.data(), sizeof (lI0[0]) * k);
        }
    }

    return n1;
}

