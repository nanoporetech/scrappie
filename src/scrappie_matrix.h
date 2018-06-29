#pragma once
#ifndef SCRAPPIE_MATRIX_H
#    define SCRAPPIE_MATRIX_H

#    include <immintrin.h>
#    include <stdbool.h>
#    include <stdint.h>
#    include <stdio.h>

typedef struct {
    size_t nr, nrq, nc, stride;
    union {
        __m128 *v;
        float *f;
    } data;
} _Mat;

typedef struct {
    size_t nr, nrq, nc, stride;
    union {
        __m128i *v;
        int32_t *f;
    } data;
} _iMat;

typedef _Mat *scrappie_matrix;
typedef _iMat *scrappie_imatrix;
typedef _Mat const *const_scrappie_matrix;
typedef _iMat const *const_scrappie_imatrix;

scrappie_matrix make_scrappie_matrix(size_t nr, size_t nc);
scrappie_matrix remake_scrappie_matrix(scrappie_matrix M, size_t nr, size_t nc);
scrappie_matrix copy_scrappie_matrix(const_scrappie_matrix mat);
scrappie_matrix free_scrappie_matrix(scrappie_matrix mat);
void zero_scrappie_matrix(scrappie_matrix M);
scrappie_matrix mat_from_array(const float *x, size_t nr, size_t nc);
float * array_from_scrappie_matrix(const_scrappie_matrix mat);
void fprint_scrappie_matrix(FILE * fh, const char *header,
                            const_scrappie_matrix mat, size_t nr, size_t nc,
                            bool include_padding);
bool equality_scrappie_matrix(const_scrappie_matrix mat1,
                              const_scrappie_matrix mat2, const float tol);
bool validate_scrappie_matrix(scrappie_matrix mat, float lower,
                              const float upper, const float maskval,
                              const bool only_finite, const char *file,
                              const int line);

scrappie_imatrix make_scrappie_imatrix(size_t nr, size_t nc);
scrappie_imatrix remake_scrappie_imatrix(scrappie_imatrix M, size_t nr, size_t nc);
scrappie_imatrix copy_scrappie_imatrix(const_scrappie_imatrix mat);
scrappie_imatrix free_scrappie_imatrix(scrappie_imatrix mat);
void zero_scrappie_imatrix(scrappie_imatrix M);

scrappie_matrix affine_map(const_scrappie_matrix X, const_scrappie_matrix W,
                           const_scrappie_matrix b, scrappie_matrix C);
scrappie_matrix affine_map2(const_scrappie_matrix Xf, const_scrappie_matrix Xb,
                            const_scrappie_matrix Wf, const_scrappie_matrix Wb,
                            const_scrappie_matrix b, scrappie_matrix C);
void row_normalise_inplace(scrappie_matrix C);

float min_scrappie_matrix(const_scrappie_matrix mat);
float max_scrappie_matrix(const_scrappie_matrix mat);

bool validate_ivector(int *vec, const size_t n, const int lower,
                      const int upper, const char *file, const int line);

bool validate_vector(float *vec, const size_t n, const float lower,
                     const float upper, const char *file, const int line);

void shift_scale_matrix_inplace(scrappie_matrix sigmat, float shift, float scale);
void clip_matrix_inplace(scrappie_matrix C, float thresh);
void filter_matrix_inplace(scrappie_matrix C, float fill_val,float thresh);
void difference_matrix_inplace(scrappie_matrix C, float val);


#endif                          /* SCRAPPIE_MATRIX_H */
