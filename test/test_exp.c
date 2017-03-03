#include "util.h"
#include <math.h>
#include <stdio.h>

const __m128 c0123 = {.75f, 0.5, 0.25f, 0.0f};

void main(void){
	for( int i=0 ; i<50 ; i++){
		__m128 x = _mm_set1_ps((i-25.0) / 2.5 ) + c0123;

		__v4sf xv = (__v4sf) x;
		__m128 y = fast_expfv(x);
		__v4sf yv = (__v4sf) y;
		float y0 = expf(xv[0]);
		float y1 = expf(xv[1]);
		float y2 = expf(xv[2]);
		float y3 = expf(xv[3]);

		printf("(%f %f %f %f) -> (%f %f %f %f)\n",
			xv[0], xv[1], xv[2], xv[3],
			(yv[0] - y0) / fabsf(yv[0] + y0), (yv[1] - y1) / fabsf(yv[1] + y1), (yv[2] - y2) / fabsf(yv[2] + y2), (yv[3] - y3) / fabsf(yv[3] + y3));
	}
}
