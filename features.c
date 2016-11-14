#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "features.h"

Mat * make_features(const event_table evtbl, bool normalise){
	const int nevent = evtbl.n;
	Mat * features = make_mat(4, nevent);
	for(int ev=0 ; ev<nevent - 1 ; ev++){
		features->data.v[ev] = _mm_setr_ps(
			evtbl.event[ev].mean,
			evtbl.event[ev].stdv,
			(float)evtbl.event[ev].length,
			fabsf(evtbl.event[ev].mean - evtbl.event[ev + 1].mean));
	}
	features->data.v[nevent - 1] = _mm_setr_ps(
			evtbl.event[nevent - 1].mean,
			evtbl.event[nevent - 1].stdv,
			(float)evtbl.event[nevent - 1].length,
			0.0f);

	if(normalise){
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

	return features;
}

