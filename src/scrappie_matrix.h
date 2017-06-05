#ifndef SCRAPPIE_MATRIX_H
#define SCRAPPIE_MATRIX_H

#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
        unsigned int nr, nrq, nc;
        union {
                __m128 * v;
                float * f;
        } data;
} _Mat;

typedef struct {
        unsigned int nr, nrq, nc;
        union {
                __m128i * v;
                int32_t * f;
        } data;
} _iMat;

typedef _Mat * scrappie_matrix;
typedef _iMat * scrappie_imatrix;


scrappie_matrix make_scrappie_matrix(int nr, int nc);
scrappie_matrix remake_scrappie_matrix(scrappie_matrix M, int nr, int nc);
scrappie_matrix free_scrappie_matrix(scrappie_matrix mat);
void zero_scrappie_matrix(scrappie_matrix M);
scrappie_matrix mat_from_array(const float * x, int nr, int nc);
void fprint_scrappie_matrix(FILE * fh, const char * header, const scrappie_matrix mat, int nr, int nc);
bool equality_scrappie_matrix(const scrappie_matrix mat1, const scrappie_matrix mat2, const float tol);
scrappie_matrix _validate_scrappie_matrix(scrappie_matrix mat, const float lower,
		const float upper, const float maskval, const bool only_finite,
		const char * file, const int line);

#ifdef NDEBUG
static inline scrappie_matrix validate_scrappie_matrix(scrappie_matrix mat, const float lower,
		const float upper, const float maskval, const bool only_finite,
		const char * file, const int line){
	return mat;
}
#else
static inline scrappie_matrix validate_scrappie_matrix(scrappie_matrix mat, const float lower,
		const float upper, const float maskval, const bool only_finite,
		const char * file, const int line){
	return _validate_scrappie_matrix(mat, lower, upper, maskval, only_finite, file, line);
}
#endif

scrappie_imatrix make_scrappie_imatrix(int nr, int nc);
scrappie_imatrix remake_scrappie_imatrix(scrappie_imatrix M, int nr, int nc);
scrappie_imatrix free_scrappie_imatrix(scrappie_imatrix mat);
void zero_scrappie_imatrix(scrappie_imatrix M);

scrappie_matrix affine_map(const scrappie_matrix X, const scrappie_matrix W,
                 const scrappie_matrix b, scrappie_matrix C);
scrappie_matrix affine_map2(const scrappie_matrix Xf, const scrappie_matrix Xb,
                  const scrappie_matrix Wf, const scrappie_matrix Wb,
                  const scrappie_matrix b, scrappie_matrix C);
void row_normalise_inplace(scrappie_matrix C);

float min_scrappie_matrix(const scrappie_matrix mat);
float max_scrappie_matrix(const scrappie_matrix mat);



int * _validate_ivector(int * vec, const int n, const int lower,
		const int upper, const char * file, const int line);
#ifdef NDEBUG
static inline int * validate_vector(int * vec, const int n, const int lower,
		const int upper, const char * file, const int line){
	return vec;
}
#else
static inline int * validate_ivector(int * vec, const int n, const int lower,
		const int upper, const char * file, const int line){
	return _validate_ivector(vec, n, lower, upper, file, line);
}
#endif

float * _validate_vector(float * vec, const float n, const float lower,
		const float upper, const char * file, const int line);
#ifdef NDEBUG
static inline float * validate_vector(float * vec, const float n, const float lower,
		const float upper, const char * file, const int line){
	return vec;
}
#else
static inline float * validate_vector(float * vec, const float n, const float lower,
		const float upper, const char * file, const int line){
	return _validate_vector(vec, n, lower, upper, file, line);
}
#endif

#endif /* SCRAPPIE_MATRIX_H */
