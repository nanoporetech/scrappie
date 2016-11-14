#ifndef UTIL_H
#define UTIL_H
#include <immintrin.h>
#include <math.h>
#include <stdint.h>

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
} Mat;

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
	return (__m128){expf(y[0]), expf(y[1]), expf(y[2]), expf(y[3])};

}

static inline __m128 logisticfv(__m128 x){
	return _mm_rcp_ps(_mm_add_ps(_mm_setone_ps(), EXPF(-x)));
}

static inline __m128 tanhfv(__m128 x){
	const __m128 y = logisticfv(x + x);
	return y + y - _mm_setone_ps();
}





Mat * make_mat(int nr, int nc);
Mat * mat_from_array(const float * x, int nr, int nc);
void free_mat(Mat * mat);

Mat * affine_map(const Mat * X, const Mat * W,
		 const Mat * b, Mat * C);
Mat * affine_map2(const Mat * Xf, const Mat * Xb,
		  const Mat * Wf, const Mat * Wb,
		  const Mat * b, Mat * C);
void row_normalise_inplace(Mat * C);

#endif /* UTIL_H */
