#include <assert.h>
#ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#else
#    include <cblas.h>
#endif
#include <err.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "scrappie_assert.h"
#include "scrappie_matrix.h"

scrappie_matrix make_scrappie_matrix(int nr, int nc) {
    // Matrix padded so row length is multiple of 4
    int nrq = (int)ceil(nr / 4.0);
    scrappie_matrix mat = malloc(sizeof(*mat));
    mat->nr = nr;
    mat->nrq = nrq;
    mat->nc = nc;

    {
        // Check for overflow to please Coverity scanner
        size_t tmp1 = nrq * sizeof(__m128);
        size_t tmp2 = tmp1 * nc;
        if (tmp1 != 0 && tmp2 / tmp1 != nc) {
            // Have overflow in memory allocation
            free(mat);
            return NULL;
        }
    }

    int status =
        posix_memalign((void **)&(mat->data.v), 16, nrq * nc * sizeof(__m128));
    if (0 != status) {
        warnx("Error allocating memory in %s.\n", __func__);
        free(mat);
        return NULL;
    }
    memset(mat->data.v, 0, nrq * nc * sizeof(__m128));
    return mat;
}

scrappie_matrix remake_scrappie_matrix(scrappie_matrix M, int nr, int nc) {
    // Could be made more efficient when there is sufficent memory already allocated
    if ((NULL == M) || (M->nr != nr) || (M->nc != nc)) {
        M = free_scrappie_matrix(M);
        M = make_scrappie_matrix(nr, nc);
    }
    return M;
}

void zero_scrappie_matrix(scrappie_matrix M) {
    if (NULL == M) {
        return;
    }
    memset(M->data.f, 0, M->nrq * 4 * M->nc * sizeof(float));
}

scrappie_matrix mat_from_array(const float *x, int nr, int nc) {
    scrappie_matrix res = make_scrappie_matrix(nr, nc);
    for (int col = 0; col < nc; col++) {
        memcpy(res->data.f + col * res->nrq * 4, x + col * nr,
               nr * sizeof(float));
    }
    return res;
}

void fprint_scrappie_matrix(FILE * fh, const char *header,
                            const scrappie_matrix mat, int nr, int nc,
                            bool include_padding) {
    assert(NULL != fh);
    assert(NULL != mat);
    const int rlim = include_padding ? (4 * mat->nrq) : mat->nr;

    if (nr <= 0 || nr > rlim) {
        nr = rlim;
    }
    if (nc <= 0 || nc > mat->nc) {
        nc = mat->nc;
    }

    if (NULL != header) {
        int ret = fputs(header, fh);
        if (EOF == ret || ret < 0) {
            return;
        }
        fputc('\n', fh);
    }
    for (int c = 0; c < nc; c++) {
        const size_t offset = c * mat->nrq * 4;
        fprintf(fh, "%4d : % 6.4f", c, mat->data.f[offset]);
        for (int r = 1; r < nr; r++) {
            fprintf(fh, "  % 6.4f", mat->data.f[offset + r]);
        }
        fputc('\n', fh);
    }
}

scrappie_matrix free_scrappie_matrix(scrappie_matrix mat) {
    if (NULL != mat) {
        free(mat->data.v);
        free(mat);
    }
    return NULL;
}

bool validate_scrappie_matrix(scrappie_matrix mat, float lower,
                              const float upper, const float maskval,
                              const bool only_finite, const char *file,
                              const int line) {
    if (NULL == mat) {
        return false;
    }
    assert(NULL != mat->data.f);
    assert(mat->nc > 0);
    assert(mat->nr > 0);
    assert(mat->nrq > 0 && (4 * mat->nrq) >= mat->nr);

    const int nc = mat->nc;
    const int nr = mat->nr;
    const int ld = mat->nrq * 4;

    //  Masked values correct
    if (!isnan(maskval)) {
        for (int c = 0; c < nc; ++c) {
            const size_t offset = c * ld;
            for (int r = nr; r < ld; ++r) {
                if (maskval != mat->data.f[offset + r]) {
                    warnx
                        ("%s:%d  Matrix entry [%d,%d] = %f violates masking rules\n",
                         file, line, r, c, mat->data.f[offset + r]);
                    return false;
                }
            }
        }
    }
    //  Check finite
    if (only_finite) {
        for (int c = 0; c < nc; ++c) {
            const size_t offset = c * ld;
            for (int r = 0; r < nr; ++r) {
                if (!isfinite(mat->data.f[offset + r])) {
                    warnx
                        ("%s:%d  Matrix entry [%d,%d] = %f contains a non-finite value\n",
                         file, line, r, c, mat->data.f[offset + r]);
                    return false;
                }
            }
        }
    }
    //  Lower bound
    if (!isnan(lower)) {
        for (int c = 0; c < nc; ++c) {
            const size_t offset = c * ld;
            for (int r = 0; r < nr; ++r) {
                if (mat->data.f[offset + r] + FLT_EPSILON < lower) {
                    warnx
                        ("%s:%d  Matrix entry [%d,%d] = %f (%e) violates lower bound\n",
                         file, line, r, c, mat->data.f[offset + r],
                         mat->data.f[offset + r] - lower);
                    return false;
                }
            }
        }
    }
    //  Upper bound
    if (!isnan(upper)) {
        for (int c = 0; c < nc; ++c) {
            const size_t offset = c * ld;
            for (int r = 0; r < nr; ++r) {
                if (mat->data.f[offset + r] > upper + FLT_EPSILON) {
                    warnx
                        ("%s:%d  Matrix entry [%d,%d] = %f (%e) violates upper bound\n",
                         file, line, r, c, mat->data.f[offset + r],
                         mat->data.f[offset + r] - upper);
                    return false;
                }
            }
        }
    }

    return true;
}

/**  Check whether two matrices are equal within a given tolerance
 *
 *  @param mat1 A `scrappie_matrix` to compare
 *  @param mat2 A `scrappie_matrix` to compare
 *  @param tol Absolute tolerance to which elements of the matrix should agree
 *
 *  Notes:
 *    The tolerance is absolute; this may not be desirable in all circumstances.
 *    The convention used here is that of equality '=='.  The standard C
 *    sorting functions expect the convention of 0 being equal and non-equality
 *    being defined by negative (less than) and positive (greater than).
 *
 *  @return A boolean of whether the two matrices are equal.
 **/
bool equality_scrappie_matrix(const scrappie_matrix mat1,
                              const scrappie_matrix mat2, const float tol) {
    if (NULL == mat1 || NULL == mat2) {
        // One or both matrices are NULL
        if (NULL == mat1 && NULL == mat2) {
            return true;
        }
        return false;
    }
    // Given non-NULL matrices, they should always contain data
    assert(NULL != mat1->data.f);
    assert(NULL != mat2->data.f);

    if (mat1->nc != mat2->nc || mat1->nr != mat2->nr) {
        // Dimension mismatch
        return false;
    }
    //  Given equal dimensions, the following should alway hold
    assert(mat1->nrq == mat2->nrq);

    for (int c = 0; c < mat1->nc; ++c) {
        const int offset = c * 4 * mat1->nrq;
        for (int r = 0; r < mat1->nr; ++r) {
            if (fabsf(mat1->data.f[offset + r] - mat2->data.f[offset + r]) >
                tol) {
                return false;
            }
        }
    }

    return true;
}

scrappie_imatrix make_scrappie_imatrix(int nr, int nc) {
    // Matrix padded so row length is multiple of 4
    int nrq = (int)ceil(nr / 4.0);
    scrappie_imatrix mat = malloc(sizeof(*mat));
    mat->nr = nr;
    mat->nrq = nrq;
    mat->nc = nc;
    int status =
        posix_memalign((void **)&(mat->data.v), 16, nrq * nc * sizeof(__m128i));
    if (0 != status) {
        warnx("Error allocating memory in %s.\n", __func__);
        free(mat);
        return NULL;
    }
    memset(mat->data.v, 0, nrq * nc * sizeof(__m128));
    return mat;
}

scrappie_imatrix remake_scrappie_imatrix(scrappie_imatrix M, int nr, int nc) {
    // Could be made more efficient when there is sufficent memory already allocated
    if ((NULL == M) || (M->nr != nr) || (M->nc != nc)) {
        M = free_scrappie_imatrix(M);
        M = make_scrappie_imatrix(nr, nc);
    }
    return M;
}

scrappie_imatrix free_scrappie_imatrix(scrappie_imatrix mat) {
    if (NULL != mat) {
        free(mat->data.v);
        free(mat);
    }
    return NULL;
}

void zero_scrappie_imatrix(scrappie_imatrix M) {
    if (NULL == M) {
        return;
    }
    memset(M->data.f, 0, M->nrq * 4 * M->nc * sizeof(int));
}

scrappie_matrix affine_map(const scrappie_matrix X, const scrappie_matrix W,
                           const scrappie_matrix b, scrappie_matrix C) {
    /*  Affine transform C = W^t X + b
     *  X is [nr, nc]
     *  W is [nr, nk]
     *  b is [nk]
     *  C is [nk, nc] or NULL.  If NULL then C is allocated.
     */
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != W);
    assert(NULL != b);
    assert(W->nr == X->nr);

    C = remake_scrappie_matrix(C, W->nc, X->nc);
    RETURN_NULL_IF(NULL == C, NULL);

    /* Copy bias */
    for (int c = 0; c < C->nc; c++) {
        memcpy(C->data.v + c * C->nrq, b->data.v, C->nrq * sizeof(__m128));
    }

    /* Affine transform */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr,
                1.0, W->data.f, W->nrq * 4, X->data.f, X->nrq * 4, 1.0,
                C->data.f, C->nrq * 4);

    return C;
}

scrappie_matrix affine_map2(const scrappie_matrix Xf, const scrappie_matrix Xb,
                            const scrappie_matrix Wf, const scrappie_matrix Wb,
                            const scrappie_matrix b, scrappie_matrix C) {
    RETURN_NULL_IF(NULL == Xf, NULL);
    RETURN_NULL_IF(NULL == Xb, NULL);

    assert(NULL != Wf);
    assert(NULL != Wb);
    assert(NULL != b);
    assert(Wf->nr == Xf->nr);
    assert(Wb->nr == Xb->nr);
    assert(Xf->nc == Xb->nc);
    assert(Wf->nc == Wb->nc);
    C = remake_scrappie_matrix(C, Wf->nc, Xf->nc);
    RETURN_NULL_IF(NULL == C, NULL);

    /* Copy bias */
    for (int c = 0; c < C->nc; c++) {
        memcpy(C->data.v + c * C->nrq, b->data.v, C->nrq * sizeof(__m128));
    }

    /* Affine transform -- forwards */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, Wf->nc, Xf->nc, Wf->nr,
                1.0, Wf->data.f, Wf->nrq * 4, Xf->data.f, Xf->nrq * 4, 1.0,
                C->data.f, C->nrq * 4);
    /* Affine transform -- backwards */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, Wb->nc, Xb->nc, Wb->nr,
                1.0, Wb->data.f, Wb->nrq * 4, Xb->data.f, Xb->nrq * 4, 1.0,
                C->data.f, C->nrq * 4);
    return C;
}

__m128 mask(int i) {
    return (__m128) (__v4sf) {
    i >= 1, i >= 2, i >= 3, 0.0f};
}

void row_normalise_inplace(scrappie_matrix C) {
    if (NULL == C) {
        // Input NULL due to earlier failure.  Propagate
        return;
    }
    for (int col = 0; col < C->nc; col++) {
        const size_t offset = col * C->nrq;
        __m128 sum = _mm_setzero_ps();
        for (int row = 0; row < C->nrq; row++) {
            sum += C->data.v[offset + row];
        }
        sum -= C->data.v[offset + C->nrq - 1] * mask(C->nr - C->nrq * 4);
        const __m128 psum = _mm_hadd_ps(sum, sum);
        const __m128 tsum = _mm_hadd_ps(psum, psum);

        for (int row = 0; row < C->nrq; row++) {
            C->data.v[offset + row] /= tsum;
        }
    }
}

float max_scrappie_matrix(const scrappie_matrix x) {
    if (NULL == x) {
        // Input NULL due to earlier failure.  Propagate
        return NAN;
    }
    float amax = x->data.f[0];
    for (int col = 0; col < x->nc; col++) {
        const size_t offset = col * x->nrq * 4;
        for (int r = 0; r < x->nr; r++) {
            if (amax < x->data.f[offset + r]) {
                amax = x->data.f[offset + r];
            }
        }
    }
    return amax;
}

float min_scrappie_matrix(const scrappie_matrix x) {
    if (NULL == x) {
        // Input NULL due to earlier failure.  Propagate
        return NAN;
    }
    float amin = x->data.f[0];
    for (int col = 0; col < x->nc; col++) {
        const size_t offset = col * x->nrq * 4;
        for (int r = 0; r < x->nr; r++) {
            if (amin < x->data.f[offset + r]) {
                amin = x->data.f[offset + r];
            }
        }
    }
    return amin;
}

int argmax_scrappie_matrix(const scrappie_matrix x) {
    if (NULL == x) {
        // Input NULL due to earlier failure.  Propagate
        return -1;
    }
    float amax = x->data.f[0];
    int imax = 0;

    for (int col = 0; col < x->nc; col++) {
        const size_t offset = col * x->nrq * 4;
        for (int r = 0; r < x->nr; r++) {
            if (amax < x->data.f[offset + r]) {
                amax = x->data.f[offset + r];
                imax = offset + r;
            }
        }
    }
    return imax;
}

int argmin_scrappie_matrix(const scrappie_matrix x) {
    if (NULL == x) {
        // Input NULL due to earlier failure.  Propagate
        return -1;
    }
    float amin = x->data.f[0];
    int imin = 0;

    for (int col = 0; col < x->nc; col++) {
        const size_t offset = col * x->nrq * 4;
        for (int r = 0; r < x->nr; r++) {
            if (amin < x->data.f[offset + r]) {
                amin = x->data.f[offset + r];
                imin = offset + r;
            }
        }
    }
    return imin;
}

bool validate_vector(float *vec, const float n, const float lower,
                     const float upper, const char *file, const int line) {
    if (NULL == vec) {
        return false;
    }
    //  Lower bound
    if (!isnan(lower)) {
        for (int i = 0; i < n; ++i) {
            if (lower > vec[i]) {
                warnx("%s:%d  Vector entry %d = %f violates lower bound\n",
                      file, line, i, vec[i]);
                return false;
            }
        }
    }
    //  Upper bound
    if (!isnan(upper)) {
        for (int i = 0; i < n; ++i) {
            if (upper < vec[i]) {
                warnx("%s:%d  Vector entry %d = %f violates upper bound\n",
                      file, line, i, vec[i]);
                return false;
            }
        }
    }

    return true;
}

bool validate_ivector(int *vec, const int n, const int lower, const int upper,
                      const char *file, const int line) {
    if (NULL == vec) {
        return false;
    }
    //  Lower bound
    for (int i = 0; i < n; ++i) {
        if (lower > vec[i]) {
            warnx("%s:%d  Vector entry %d = %d violates lower bound\n", file,
                  line, i, vec[i]);
            return false;
        }
    }

    //  Upper bound
    for (int i = 0; i < n; ++i) {
        if (upper < vec[i]) {
            warnx("%s:%d  Vector entry %d = %d violates upper bound\n", file,
                  line, i, vec[i]);
            return false;
        }
    }

    return true;
}
