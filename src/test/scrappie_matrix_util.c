#include <assert.h>
#include <err.h>
#include <math.h>

#include "scrappie_assert.h"
#include "scrappie_matrix_util.h"

/**  Simple writer for scrappie_matrix structures
 *
 *  Writes a Scrappie matrix to a file point using hexadecimal ecoded
 *  floats.
 *
 *  @param fh A FILE pointer to write to.
 *  @param mat A matrix to write out.
 *
 *  @returns Number of elements written
 **/
int write_scrappie_matrix(FILE * fh, const scrappie_matrix mat) {
    if (NULL == fh || NULL == mat) {
        return 0;
    }

    fprintf(fh, "%d\t%d\n", mat->nr, mat->nc);

    int nelt = 0;
    for (int c = 0; c < mat->nc; ++c) {
        const int offset = c * 4 * mat->nrq;
        int ret = fprintf(fh, "%a", mat->data.f[offset + 0]);
        if (ret > 0) {
            nelt += 1;
        }
        for (int r = 1; r < mat->nr; ++r) {
            ret = fprintf(fh, "\t%a", mat->data.f[offset + r]);
            if (ret > 0) {
                nelt += 1;
            }
        }
        fputc('\n', fh);
    }

    assert(nelt == mat->nr * mat->nc);
    return nelt;
}

/**   Simple reader for scrappie_matrix structures
 *
 *  @param fh A FILE pointer to read from.
 *
 *  @returns Matrix read or NULL on failure
 **/
scrappie_matrix read_scrappie_matrix(FILE * fh) {
    RETURN_NULL_IF(NULL == fh, NULL);

    int nr, nc;
    int ret = fscanf(fh, "%d\t%d\n", &nr, &nc);
    if (2 != ret) {
        warnx("Invalid header line.\n");
        return NULL;
    }
    if (nr <= 0 || nc <= 0) {
        warnx("Number of rows or columns invalid (got %d %d).\n", nr, nc);
        return NULL;
    }

    scrappie_matrix mat = make_scrappie_matrix(nr, nc);
    if (NULL == mat) {
        warnx("Failed to allocate enough memory for matrix.\n");
        return NULL;
    }

    int nelt = 0;
    for (int c = 0; c < nc; ++c) {
        const int offset = c * 4 * mat->nrq;
        ret = fscanf(fh, "%a", &mat->data.f[offset]);
        if (ret > 0) {
            nelt += 1;
        }
        for (int r = 1; r < nr; ++r) {
            ret = fscanf(fh, "\t%a", &mat->data.f[offset + r]);
            if (ret > 0) {
                nelt += 1;
            }
        }
    }

    if (nelt != nr * nc) {
        warnx
            ("Read incorrect number of elements. Got %d but expecteding %d x %d\n",
             nelt, nr, nc);
        mat = free_scrappie_matrix(mat);
    }

    assert(validate_scrappie_matrix
           (mat, NAN, NAN, 0.0, true, __FILE__, __LINE__));
    return mat;
}

/**   Uniform number distributed [lower, upper]
 *
 *  @param lower Lower bound
 *  @param upper Upper bound
 *
 *  @returns A unformly distributed random number
 **/
float scrappie_random_uniform(float lower, float upper) {
    // rand() is awful but we don't require good random numbers
    return lower + (upper - lower) * (float)rand() / RAND_MAX;
}

/**  Matrix filed with random values
 *
 *   Creates new matrix whose elemenrs are picked from a uniform distribution
 *   on [lower, upper].
 *
 *  @param nr Number of rows of matrix
 *  @param nc Number of columns of matrix
 *  @param lower Lower bound for random values
 *  @param upper Upper bound for random values
 *
 *  @returns Matrix filled with random data, or NULL on error
 **/
scrappie_matrix random_scrappie_matrix(int nr, int nc, float lower, float upper) {
    assert(nr > 0);
    assert(nc > 0);
    assert(upper >= lower);

    scrappie_matrix mat = make_scrappie_matrix(nr, nc);
    RETURN_NULL_IF(NULL == mat, NULL);

    for (int c = 0; c < mat->nc; ++c) {
        const int offset = c * 4 * mat->nrq;
        for (int r = 0; r < mat->nr; ++r) {
            mat->data.f[offset + r] = scrappie_random_uniform(lower, upper);
        }
    }

    assert(validate_scrappie_matrix
           (mat, lower, upper, 0.0, true, __FILE__, __LINE__));
    return mat;
}
