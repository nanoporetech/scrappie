#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nnfeatures.h"


/** Studentise features
 *
 *  Studentise a matrix of four features. The matrix is updated inplace.
 *
 *  @param features A matrix containing features to normalise
 *  @see studentise_features_kahan
 *  @return void
 **/
void studentise_features(Mat_rptr features){
	const int nevent = features->nc;

	__m128 sum, sumsq;
	sumsq = sum = _mm_setzero_ps();
	for(int ev=0 ; ev<nevent ; ev++){
		sum += features->data.v[ev];
		sumsq += features->data.v[ev] * features->data.v[ev];
	}
	sum /= _mm_set1_ps((float)nevent);
	sumsq /= _mm_set1_ps((float)nevent);
	sumsq -= sum * sum;

	sumsq = _mm_rsqrt_ps(sumsq);
	sum *= sumsq;
	for(int ev=0 ; ev<nevent ; ev++){
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
void studentise_features_kahan(Mat_rptr features){
	assert(4 == features->nr);
	const int nevent = features->nc;

	__m128 sum, sumsq, comp, compsq;
	sumsq = sum = comp = compsq = _mm_setzero_ps();
	for(int ev=0 ; ev<nevent ; ev++){
		__m128 d1 = features->data.v[ev] - comp;
		__m128 sum_tmp = sum + d1;
		comp = (sum_tmp - sum) - d1;
		sum = sum_tmp;

		__m128 d2 = features->data.v[ev] * features->data.v[ev] - compsq;
		__m128 sumsq_tmp = sumsq + d2;
		compsq = (sumsq_tmp - sumsq) - d2;
		sumsq = sumsq_tmp;
	}
	sum /= _mm_set1_ps((float)nevent);
	sumsq /= _mm_set1_ps((float)nevent);
	sumsq -= sum * sum;

	sumsq = _mm_rsqrt_ps(sumsq);
	sum *= sumsq;
	for(int ev=0 ; ev<nevent ; ev++){
		features->data.v[ev] = sumsq * features->data.v[ev] - sum;
	}
}


Mat_rptr make_features(const event_table evtbl, int trim, bool normalise){
	trim += evtbl.start;
	const int nevent = evtbl.end - trim;
	Mat_rptr features = make_mat(4, nevent);
	for(int ev=0 ; ev<nevent - 1 ; ev++){
		features->data.v[ev] = _mm_setr_ps(
			evtbl.event[ev + trim].mean,
			evtbl.event[ev + trim].stdv,
			evtbl.event[ev + trim].length,
			fabs(evtbl.event[ev + trim].mean - evtbl.event[ev + 1 + trim].mean));
	}
	features->data.v[nevent - 1] = _mm_setr_ps(
			evtbl.event[evtbl.end - 1].mean,
			evtbl.event[evtbl.end - 1].stdv,
			evtbl.event[evtbl.end - 1].length,
			0.0f);

	if(normalise){
		studentise_features_kahan(features);
	}

	return features;
}

