#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nnfeatures.h"
#include "scrappie_assert.h"

/** Studentise features
 *
 *  Studentise a matrix of four features. The matrix is updated inplace.
 *
 *  @param features A matrix containing features to normalise
 *  @see studentise_features_kahan
 *  @return void
 **/
void studentise_features(scrappie_matrix features) {
    assert(4 == features->nr);
    const int nevent = features->nc;

    __m128 sum, sumsq;
    sumsq = sum = _mm_setzero_ps();
    for (int ev = 0; ev < nevent; ev++) {
        sum += features->data.v[ev];
        sumsq += features->data.v[ev] * features->data.v[ev];
    }
    sum /= _mm_set1_ps((float)nevent);
    sumsq /= _mm_set1_ps((float)nevent);
    sumsq -= sum * sum;

    sumsq = _mm_rsqrt_ps(sumsq);
    sum *= sumsq;
    for (int ev = 0; ev < nevent; ev++) {
        features->data.v[ev] = sumsq * features->data.v[ev] - sum;
    }
}

/** Studentise features using Kahan summation algorithm
 *
 *  Studentise a matrix of four features using the Kahan summation
 *  algorithm for numerical stability. The matrix is updated inplace.
 *  https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *
 *  @param features A matrix containing features to normalise
 *  @see studentise_features
 *  @return void
 **/
void studentise_features_kahan(scrappie_matrix features) {
    assert(4 == features->nr);
    const int nevent = features->nc;

    __m128 sum, sumsq, comp, compsq;
    sumsq = sum = comp = compsq = _mm_setzero_ps();
    for (int ev = 0; ev < nevent; ev++) {
        __m128 d1 = features->data.v[ev] - comp;
        __m128 sum_tmp = _mm_add_ps(sum, d1);
        comp = (sum_tmp - sum) - d1;
        sum = sum_tmp;

        __m128 d2 = features->data.v[ev] * features->data.v[ev] - compsq;
        __m128 sumsq_tmp = _mm_add_ps(sumsq, d2);
        compsq = (sumsq_tmp - sumsq) - d2;
        sumsq = sumsq_tmp;
    }
    sum /= _mm_set1_ps((float)nevent);
    sumsq /= _mm_set1_ps((float)nevent);
    sumsq -= sum * sum;

    sumsq = _mm_rsqrt_ps(sumsq);
    sum *= sumsq;
    for (int ev = 0; ev < nevent; ev++) {
        features->data.v[ev] = sumsq * features->data.v[ev] - sum;
    }
}

scrappie_matrix nanonet_features_from_events(const event_table evtbl,
                                             bool normalise) {
    const size_t nevent = evtbl.end - evtbl.start;
    const size_t offset = evtbl.start;
    scrappie_matrix features = make_scrappie_matrix(4, nevent);
    for (size_t ev = 0; ev < nevent - 1; ev++) {
        features->data.v[ev] = _mm_setr_ps(evtbl.event[ev + offset].mean,
                                           evtbl.event[ev + offset].stdv,
                                           evtbl.event[ev + offset].length,
                                           fabs(evtbl.event[ev + offset].mean -
                                                evtbl.event[ev + 1 +
                                                            offset].mean));
    }
    features->data.v[nevent - 1] = _mm_setr_ps(evtbl.event[evtbl.end - 1].mean,
                                               evtbl.event[evtbl.end - 1].stdv,
                                               evtbl.event[evtbl.end -
                                                           1].length, 0.0f);

    if (normalise) {
        studentise_features_kahan(features);
    }

    return features;
}

scrappie_matrix nanonet_features_from_raw(const raw_table signal) {
    ASSERT_OR_RETURN_NULL(signal.n > 0 && NULL != signal.raw, NULL);
    const size_t nsample = signal.end - signal.start;
    scrappie_matrix sigmat = make_scrappie_matrix(1, nsample);
    ASSERT_OR_RETURN_NULL(NULL != sigmat, NULL);

    const size_t offset = signal.start;
    for (size_t i = 0; i < nsample; i++) {
        // Copy with stride 4 because of required padding for matrix
        sigmat->data.f[i * 4] = signal.raw[i + offset];
    }
    return sigmat;
}
