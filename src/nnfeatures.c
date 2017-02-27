#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nnfeatures.h"

Mat_rptr make_features(const event_table evtbl, int trim, bool normalise){
	trim += evtbl.start;
	const int nevent = evtbl.end - trim;
	Mat_rptr features = make_mat(4, nevent);
	for(int ev=0 ; ev<nevent - 1 ; ev++){
		features->data.v[ev] = _mm_setr_ps(
			evtbl.event[ev + trim].mean,
			evtbl.event[ev + trim].stdv,
			(float)evtbl.event[ev + trim].length,
			(float)fabs(evtbl.event[ev + trim].mean - evtbl.event[ev + 1 + trim].mean));
	}
	features->data.v[nevent - 1] = _mm_setr_ps(
			evtbl.event[evtbl.end - 1].mean,
			evtbl.event[evtbl.end - 1].stdv,
			(float)evtbl.event[evtbl.end - 1].length,
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

