#ifndef UTIL_H
#define UTIL_H
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include "sse_mathfun.h"

typedef union {
	__m128d v;
	double f[2];
} v2f;


typedef union {
	__m128 v;
	float f[4];
} v4f;


typedef struct {
	int nr, nrq, nc;
	union {
		__m128 * v;
		float * f;
	} data;
} _Mat;

typedef _Mat * restrict Mat_rptr;

/* Create a vector of  ones.  */
extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
        _mm_setone_ps (void){ return __extension__ (__m128){ 1.0f, 1.0f, 1.0f, 1.0f}; }

extern __inline __m128d __attribute__((__gnu_inline__, __always_inline__, __artificial__))
	_mm_setone_pd (void){ return __extension__ (__m128d){ 1.0, 1.0 }; }


static inline float logisticf(float x){
	return 1.0 / (1.0 + expf(-x));
}

static inline float fast_expf(float x){
        /* Values of c
         * Mean relative error: 8
         * Minimum root-mean square error: 7
         * Min max relative error: 5
         * Exact at x = 0.0: 0
         */
        union{ uint32_t i; float f;} value = {.i = (uint32_t)(12102203.161561485 * x + 1064872507.1541044)};
        return value.f;
}

static inline float fast_logisticf(float x){
        return 1.0 / (1.0 + fast_expf(-x));
}

static inline float fast_tanhf(float x){
	const float y = fast_logisticf(x + x);
        return y * y - 1.0;
}


#ifdef FAST_EXP
	#define EXPF fast_expfv
#else
	#define EXPF expfv
#endif

static inline __m128 fast_expfv(__m128 x){
	#define _A 12102203.161561485f
	#define _B 1064872507.1541044f
	const __m128 a = (__m128)(__v4sf){_A, _A, _A, _A};
	const __m128 b = (__m128)(__v4sf){_B, _B, _B, _B};
	__m128 y = a * x + b;
	return _mm_castsi128_ps(_mm_cvtps_epi32(y));
}

static inline __m128 expfv(__m128 x){
	__v4sf y = (__v4sf)x;
	return (__m128)exp_ps(y);
}

static inline __m128 logisticfv(__m128 x){
	return _mm_rcp_ps(_mm_add_ps(_mm_setone_ps(), EXPF(-x)));
}

static inline __m128 tanhfv(__m128 x){
	const __m128 y = logisticfv(x + x);
	return y + y - _mm_setone_ps();
}

static inline __m128 fast_logfv(__m128 x){
	#define _Alogfv 8.262958294867817e-08f
	#define _Blogfv 1064872507.1541044f
	const __m128 a = (__m128)(__v4sf){_Alogfv, _Alogfv, _Alogfv, _Alogfv};
	const __m128 b = (__m128)(__v4sf){_Blogfv, _Blogfv, _Blogfv, _Blogfv};
	x = _mm_cvtepi32_ps(_mm_castps_si128(x));
	return a * (x - b);
}

static inline __m128 logfv(__m128 x){
	__v4sf y = (__v4sf)x;
	return (__m128)log_ps(y);
}

int argmaxf(const float * x, int n);
int argminf(const float * x, int n);
float valmaxf(const float * x, int n);
float valminf(const float * x, int n);



Mat_rptr make_mat(int nr, int nc);
Mat_rptr mat_from_array(const float * x, int nr, int nc);
void free_mat(Mat_rptr mat);

Mat_rptr affine_map(const Mat_rptr X, const Mat_rptr W,
		 const Mat_rptr b, Mat_rptr C);
Mat_rptr affine_map2(const Mat_rptr Xf, const Mat_rptr Xb,
		  const Mat_rptr Wf, const Mat_rptr Wb,
		  const Mat_rptr b, Mat_rptr C);
void row_normalise_inplace(Mat_rptr C);

float min_mat(const Mat_rptr mat);
float max_mat(const Mat_rptr mat);

#endif /* UTIL_H */
