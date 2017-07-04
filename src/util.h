#pragma once
#ifndef UTIL_H
#    define UTIL_H
#    include <immintrin.h>
#    include <math.h>
#    include <stdint.h>
#    include <stdio.h>
#    include "sse_mathfun.h"

#    ifdef FAST_LOG
#        define LOGFV fast_logfv
#    else
#        define LOGFV logfv
#    endif

#    ifdef FAST_EXP
#        define EXPFV fast_expfv
#    else
#        define EXPFV expfv
#    endif

/* Create a vector of  ones.  */
extern __inline __m128 __attribute__ ((__gnu_inline__, __always_inline__))
    _mm_setone_ps(void) {
    return __extension__(__m128) {
    1.0f, 1.0f, 1.0f, 1.0f};
}

extern __inline __m128d __attribute__ ((__gnu_inline__, __always_inline__))
    _mm_setone_pd(void) {
    return __extension__(__m128d) {
    1.0, 1.0};
}

static inline float logisticf(float x) {
    return 1.0 / (1.0 + expf(-x));
}


// Constants for fast exp approximation.  See Schraudolph (1999)
#    define _A 12102203.161561485f
//  Minimum maximum relative error
//#    define _B 1064986822.5027076f
//  No bias at zero
#    define _B 1065353216.0f
//  Minimum RMS relative error
//#    define _B 1064866803.6193156f
//  Minimum mean relative error
//#    define _B 1064807268.0425934f
#    define _BOUND 88.02969193111305
static inline float fast_expf(float x) {
    x = fmaxf(-_BOUND, fminf(_BOUND, x));
    union {
        uint32_t i;
        float f;
    } value = {
    .i = (uint32_t) (_A * x + _B)};
    return value.f;
}

static inline __m128 fast_expfv(__m128 x) {
    const __m128 a = (__m128) (__v4sf) { _A, _A, _A, _A };
    const __m128 b = (__m128) (__v4sf) { _B, _B, _B, _B };
    const __m128 _bound = (__m128) (__v4sf) { _BOUND, _BOUND, _BOUND, _BOUND };
    x = _mm_max_ps(-_bound, _mm_min_ps(_bound, x));

    __m128 y = a * x + b;
    return _mm_castsi128_ps(_mm_cvtps_epi32(y));
}

static inline float fast_logisticf(float x) {
    return 1.0 / (1.0 + fast_expf(-x));
}

static inline float fast_tanhf(float x) {
    const float y = fast_logisticf(x + x);
    return y + y - 1.0;
}

static inline __m128 __attribute__ ((__always_inline__)) expfv(__m128 x) {
    __v4sf y = (__v4sf) x;
    return (__m128) exp_ps(y);
}

static inline __m128 __attribute__ ((__always_inline__)) logisticfv(__m128 x) {
    return _mm_rcp_ps(_mm_add_ps(_mm_setone_ps(), EXPFV(-x)));
}

static inline __m128 __attribute__ ((__always_inline__)) tanhfv(__m128 x) {
    const __m128 y = logisticfv(_mm_add_ps(x, x));
    return _mm_sub_ps(_mm_add_ps(y, y), _mm_setone_ps());
}

static inline __m128 fast_logfv(__m128 x) {
#    define _Alogfv 8.262958294867817e-08f
#    define _Blogfv 1064872507.1541044f
    const __m128 a = (__m128) (__v4sf) { _Alogfv, _Alogfv, _Alogfv, _Alogfv };
    const __m128 b = (__m128) (__v4sf) { _Blogfv, _Blogfv, _Blogfv, _Blogfv };
    x = _mm_cvtepi32_ps(_mm_castps_si128(x));
    return a * (x - b);
}

static inline __m128 logfv(__m128 x) {
    __v4sf y = (__v4sf) x;
    return (__m128) log_ps(y);
}

char *strip_filename_extension(char *filename);

int argmaxf(const float *x, int n);
int argminf(const float *x, int n);
float valmaxf(const float *x, int n);
float valminf(const float *x, int n);

static inline int iceil(int x, int y) {
    return (x + y - 1) / y;
}

static inline int ifloor(int x, int y) {
    return x / y;
}

void quantilef(const float *x, size_t nx, float *p, size_t np);
float medianf(const float *x, size_t n);
float madf(const float *x, size_t n, const float *med);
void medmad_normalise_array(float *x, size_t n);
void studentise_array_kahan(float *x, size_t n);

#endif                          /* UTIL_H */
