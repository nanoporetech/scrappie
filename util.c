#include <assert.h>
#include <openblas/cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

Mat * make_mat(int nr, int nc){
	// Matrix padded so row length is multiple of 4
	int nrq = (int)ceil(nr / 4.0);
	Mat * mat = malloc(sizeof(Mat));
	mat->nr = nr;
	mat->nrq = nrq;
	mat->nc = nc;
	mat->data.v = calloc(nrq * nc, sizeof(__m128));
	return mat;
}

Mat * mat_from_array(const float * x, int nr, int nc){
	Mat * res = make_mat(nr, nc);
	for(int col=0 ; col < nc ; col++){
		memcpy(res->data.f + col * res->nrq * 4, x + col * nr, nr * sizeof(float));
	}
	return res;
}

void free_mat(Mat * mat){
	free(mat->data.v);
	free(mat);
}

Mat * affine_map(const Mat * X, const Mat * W,
                 const Mat * b, Mat * C){
        /*  Affine transform C = W^t X + b
         *  X is [nr, nc]
         *  W is [nr, nk]
         *  b is [nk]
         *  C is [nk, nc] or NULL.  If NULL then C is allocated.
         */
	assert(W->nr == X->nr);
        if(NULL == C){
        	C = make_mat(W->nc, X->nc);
        }
	assert(C->nr == W->nc);
	assert(C->nc == X->nc);

        /* Copy bias */
        for( int c = 0 ; c < C->nc; c++){
                memcpy(C->data.v + c * C->nrq, b->data.v, C->nrq * sizeof(__m128));
        }

        /* Affine transform */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr, 1.0, W->data.f, W->nrq * 4, X->data.f, X->nrq * 4, 1.0, C->data.f, C->nrq * 4);


        return C;
}

Mat * affine_map2(const Mat * Xf, const Mat * Xb,
		  const Mat * Wf, const Mat * Wb,
		  const Mat * b, Mat * C){
	assert(Wf->nr == Xf->nr);
	assert(Wb->nr == Xb->nr);
	assert(Xf->nc == Xb->nc);
	assert(Wf->nc == Wb->nc);
	if(NULL == C){
		C = make_mat(Wf->nc, Xf->nc);
	}
	assert(C->nr == Wf->nc);
	assert(C->nc == Xf->nc);

        /* Copy bias */
        for( int c = 0 ; c < C->nc; c++){
                memcpy(C->data.v + c * C->nrq, b->data.v, C->nrq * sizeof(__m128));
        }

        /* Affine transform -- forwards*/
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, Wf->nc, Xf->nc, Wf->nr, 1.0, Wf->data.f, Wf->nrq * 4, Xf->data.f, Xf->nrq * 4, 1.0, C->data.f, C->nrq * 4);
        /* Affine transform -- backwards*/
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, Wb->nc, Xb->nc, Wb->nr, 1.0, Wb->data.f, Wb->nrq * 4, Xb->data.f, Xb->nrq * 4, 1.0, C->data.f, C->nrq * 4);
	return C;
}



void row_normalise_inplace(Mat * C){
	assert(NULL != C);
	for(int col=0 ; col < C->nc ; col++){
		const int offset = col * C->nrq;
		__m128 sum = _mm_setzero_ps();
		for(int row=0 ; row < C->nrq ; row++){
			sum += C->data.v[offset + row];
		}
		const __m128 psum = _mm_hadd_ps(sum, sum);
		const __m128 tsum = _mm_hadd_ps(psum, psum);

		const __m128 isumv = _mm_rcp_ps(tsum);
		for(int row=0 ; row < C->nrq ; row++){
			C->data.v[offset + row] *= isumv;
		}
	}
}
