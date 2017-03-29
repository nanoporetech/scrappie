#include <assert.h>
#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif
#include <err.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

/**  Strips the extension from a filename
 *
 *   An extension is located and its initial period is replaced with a
 *  null byte to terminate the string at that location.  A filename beginning
 *  with a period but no other extension is not modified.
 *
 *  @param filename A string containing filename to be modified [updated]
 *  @return pointer to beginning of filename.
 **/
char * strip_filename_extension(char * filename){
	char * loc = strrchr(filename, '.');
	if(NULL != loc && loc != filename){
		// Filename contains '.' and it is not the first character
		*loc = '\0';
	}
	return filename;
}

int argmaxf(const float * x, int n){
	assert(n > 0);
	if(NULL == x){ return -1; }
	int imax = 0;
	float vmax = x[0];
	for(int i=1 ; i<n ; i++){
		if(x[i] > vmax){
			vmax = x[i];
			imax = i;
		}
	}
	return imax;
}

int argminf(const float * x, int n){
	assert(n > 0);
	if(NULL == x){ return -1; }
	int imin = 0;
	float vmin = x[0];
	for(int i=1 ; i<n ; i++){
		if(x[i] > vmin){
			vmin = x[i];
			imin = i;
		}
	}
	return imin;
}

float valmaxf(const float * x, int n){
	assert(n > 0);
	if(NULL == x){ return NAN; }
	float vmax = x[0];
	for(int i=1 ; i<n ; i++){
		if(x[i] > vmax){
			vmax = x[i];
		}
	}
	return vmax;
}

float valminf(const float * x, int n){
	assert(n > 0);
	if(NULL == x){ return NAN; }
	float vmin = x[0];
	for(int i=1 ; i<n ; i++){
		if(x[i] > vmin){
			vmin = x[i];
		}
	}
	return vmin;
}

Mat_rptr make_mat(int nr, int nc){
	// Matrix padded so row length is multiple of 4
	int nrq = (int)ceil(nr / 4.0);
	Mat_rptr mat = malloc(sizeof(*mat));
	mat->nr = nr;
	mat->nrq = nrq;
	mat->nc = nc;
	int status = posix_memalign((void **) &(mat->data.v), 16, nrq * nc * sizeof(__m128));
	if(0 != status){
		warnx("Error allocating memory in %s.\n", __func__);
		free(mat);
		return NULL;
	}
	memset(mat->data.v, 0, nrq * nc * sizeof(__m128));
	return mat;
}

Mat_rptr remake_mat(Mat_rptr M, int nr, int nc){
	// Could be made more efficient when there is sufficent memory already allocated
	if((NULL == M) || (M->nr != nr) || (M->nc != nc)){
		M = free_mat(M);
		M = make_mat(nr, nc);
	}
	return M;
}

void zero_mat(Mat_rptr M) {
	if(NULL != M){ return; }
	memset(M->data.f, 0, M->nrq * 4 * M->nc * sizeof(float));
}


Mat_rptr mat_from_array(const float * x, int nr, int nc){
	Mat_rptr res = make_mat(nr, nc);
	for(int col=0 ; col < nc ; col++){
		memcpy(res->data.f + col * res->nrq * 4, x + col * nr, nr * sizeof(float));
	}
	return res;
}


void fprint_mat(FILE * fh, const char * header, const Mat_rptr mat, int nr, int nc){
	assert(NULL != fh);
	assert(NULL != mat);
	if(nr <= 0 || nr > mat->nr){nr = mat->nr;}
	if(nc <= 0 || nc > mat->nc){nc = mat->nc;}

	if(NULL != header){
		fputs(header, fh);
		fputc('\n', fh);
	}
	for(int c=0 ; c < nc ; c++){
	        const size_t offset = c * mat->nrq * 4;
	        fprintf(fh, "%4d : % 6.4f", c, mat->data.f[offset]);
	        for(int r=1 ; r<nr ; r++){
	                fprintf(fh, "  % 6.4f", mat->data.f[offset + r]);
	        }
	        fputc('\n', fh);
	}
}


Mat_rptr free_mat(Mat_rptr mat){
	if(NULL != mat){
		free(mat->data.v);
		free(mat);
	}
	return NULL;
}

iMat_rptr make_imat(int nr, int nc){
	// Matrix padded so row length is multiple of 4
	int nrq = (int)ceil(nr / 4.0);
	iMat_rptr mat = malloc(sizeof(*mat));
	mat->nr = nr;
	mat->nrq = nrq;
	mat->nc = nc;
	int status = posix_memalign((void **) &(mat->data.v), 16, nrq * nc * sizeof(__m128i));
	if(0 != status){
		warnx("Error allocating memory in %s.\n", __func__);
		free(mat);
		return NULL;
	}
	memset(mat->data.v, 0, nrq * nc * sizeof(__m128));
	return mat;
}

iMat_rptr remake_imat(iMat_rptr M, int nr, int nc){
	// Could be made more efficient when there is sufficent memory already allocated
	if((NULL == M) || (M->nr != nr) || (M->nc != nc)){
		M = free_imat(M);
		M = make_imat(nr, nc);
	}
	return M;
}

iMat_rptr free_imat(iMat_rptr mat){
	if(NULL != mat){
		free(mat->data.v);
		free(mat);
	}
	return NULL;
}

void zero_imat(iMat_rptr M) {
	if(NULL != M){ return; }
	memset(M->data.f, 0, M->nrq * 4 * M->nc * sizeof(int));
}


Mat_rptr affine_map(const Mat_rptr X, const Mat_rptr W,
	         const Mat_rptr b, Mat_rptr C){
	/*  Affine transform C = W^t X + b
	 *  X is [nr, nc]
	 *  W is [nr, nk]
	 *  b is [nk]
	 *  C is [nk, nc] or NULL.  If NULL then C is allocated.
	 */
	if(NULL == X){
		// Input NULL due to earlier failure.  Propagate
		return NULL;
	}
	assert(NULL != W);
	assert(NULL != b);
	assert(W->nr == X->nr);
	C = remake_mat(C, W->nc, X->nc);
	if(NULL == C){
		return NULL;
	}

	/* Copy bias */
	for( int c = 0 ; c < C->nc; c++){
	        memcpy(C->data.v + c * C->nrq, b->data.v, C->nrq * sizeof(__m128));
	}

	/* Affine transform */
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, X->nc, W->nr, 1.0, W->data.f, W->nrq * 4, X->data.f, X->nrq * 4, 1.0, C->data.f, C->nrq * 4);


	return C;
}

Mat_rptr affine_map2(const Mat_rptr Xf, const Mat_rptr Xb,
		  const Mat_rptr Wf, const Mat_rptr Wb,
		  const Mat_rptr b, Mat_rptr C){
	if(NULL == Xf || NULL == Xb){
		// Input NULL due to earlier failure.  Propagate
		return NULL;
	}
	assert(NULL != Wf);
	assert(NULL != Wb);
	assert(NULL != b);
	assert(Wf->nr == Xf->nr);
	assert(Wb->nr == Xb->nr);
	assert(Xf->nc == Xb->nc);
	assert(Wf->nc == Wb->nc);
	C = remake_mat(C, Wf->nc, Xf->nc);
	if(NULL == C){
		return NULL;
	}

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


__m128 mask(int i){
	return (__m128)(__v4sf){i>=1, i>=2, i>=3, 0.0f};
}

void row_normalise_inplace(Mat_rptr C){
	if(NULL == C){
		// Input NULL due to earlier failure.  Propagate
		return;
	}
	for(int col=0 ; col < C->nc ; col++){
		const size_t offset = col * C->nrq;
		__m128 sum = _mm_setzero_ps();
		for(int row=0 ; row < C->nrq ; row++){
			sum += C->data.v[offset + row];
		}
		sum -= C->data.v[offset + C->nrq - 1] * mask(C->nr - C->nrq * 4);
		const __m128 psum = _mm_hadd_ps(sum, sum);
		const __m128 tsum = _mm_hadd_ps(psum, psum);

		const __m128 isumv = _mm_rcp_ps(tsum);
		for(int row=0 ; row < C->nrq ; row++){
			C->data.v[offset + row] *= isumv;
		}
	}
}

float max_mat(const Mat_rptr x){
	if(NULL == x){
		// Input NULL due to earlier failure.  Propagate
		return NAN;
	}
	float amax = x->data.f[0];
	for(int col=0 ; col < x->nc ; col++){
		const size_t offset = col * x->nrq * 4;
		for(int r=0 ; r < x->nr ; r++){
			if(amax < x->data.f[offset + r]){
				amax = x->data.f[offset + r];
			}
		}
	}
	return amax;
}

float min_mat(const Mat_rptr x){
	if(NULL == x){
		// Input NULL due to earlier failure.  Propagate
		return NAN;
	}
	float amin = x->data.f[0];
	for(int col=0 ; col < x->nc ; col++){
		const size_t offset = col * x->nrq * 4;
		for(int r=0 ; r < x->nr ; r++){
			if(amin < x->data.f[offset + r]){
				amin = x->data.f[offset + r];
			}
		}
	}
	return amin;
}

int argmax_mat(const Mat_rptr x){
	if(NULL == x){
		// Input NULL due to earlier failure.  Propagate
		return -1;
	}
	float amax = x->data.f[0];
	int imax = 0;

	for(int col=0 ; col < x->nc ; col++){
		const size_t offset = col * x->nrq * 4;
		for(int r=0 ; r < x->nr ; r++){
			if(amax < x->data.f[offset + r]){
				amax = x->data.f[offset + r];
				imax = offset + r;
			}
		}
	}
	return imax;
}

int argmin_mat(const Mat_rptr x){
	if(NULL == x){
		// Input NULL due to earlier failure.  Propagate
		return -1;
	}
	float amin = x->data.f[0];
	int imin = 0;

	for(int col=0 ; col < x->nc ; col++){
		const size_t offset = col * x->nrq * 4;
		for(int r=0 ; r < x->nr ; r++){
			if(amin < x->data.f[offset + r]){
				amin = x->data.f[offset + r];
				imin = offset + r;
			}
		}
	}
	return imin;
}


int floatcmp(const void * x, const void * y){
	float d = *(float *)x - *(float *)y;
	if(d > 0){
		return 1;
	}
	return -1;
}


/**  Quantiles from n array
 *
 *  Using a relatively inefficent qsort resulting in O(n log n)
 *  performance but better performance is possible for small np.
 *  The array p is modified inplace, containing which quantiles to
 *  calculation on input and the quantiles on output; on error, p
 *  is filled with the value NAN.
 *
 *  @param x An array to calculate quantiles from
 *  @param nx Length of array x
 *  @param p An array containing quantiles to calculate [in/out]
 *  @param np Length of array p
 *
 *  @return void
 **/
void quantilef(const float * x, size_t nx, float * p, size_t np){
	if(NULL == p){
		return;
	}
	for(int i=0 ; i < np ; i++){
		assert(p[i] >= 0.0f && p[i] <= 1.0f);
	}
	if(NULL == x){
		for(int i=0 ; i < np ; i++){
			p[i] = NAN;
		}
		return;
	}

	// Sort array
	float * space = malloc(nx * sizeof(float));
	if(NULL == space){
		for(int i=0 ; i < np ; i++){
			p[i] = NAN;
		}
		return;
	}
	memcpy(space, x, nx * sizeof(float));
	qsort(space, nx, sizeof(float), floatcmp);

	// Extract quantiles
	for(int i=0 ; i < np ; i++){
		const size_t idx = p[i] * (nx - 1);
		const float remf = p[i] * (nx - 1) - idx;
		if(idx < nx - 1){
			p[i] = (1.0 - remf) * space[idx] + remf * space[idx + 1];
		} else {
			// Should only occur when p is exactly 1.0
			p[i] = space[idx];
		}
	}

	free(space);
	return;
}


/** Median of an array
 *
 *  Using a relatively inefficent qsort resulting in O(n log n)
 *  performance but O(n) is possible.
 *
 *  @param x An array to calculate median of
 *  @param n Length of array
 *
 *  @return Median of array on success, NAN otherwise.
 **/
float medianf(const float * x, size_t n){
	float p = 0.5;
	quantilef(x, n, &p, 1);
	return p;
}


/** Median Absolute Deviation of an array
 *
 *  @param x An array to calculate the MAD of
 *  @param n Length of array
 *  @param med Median of the array.  If NAN then median is calculated.
 *
 *  @return MAD of array on success, NAN otherwise.
 **/
float madf(const float * x, size_t n, const float * med){
	const float mad_scaling_factor = 1.4826;
	if(NULL == x){
		return NAN;
	}
	if(1 == n){
		return 0.0f;
	}

	float * absdiff = malloc(n * sizeof(float));
	if(NULL == absdiff){
		return NAN;
	}

	const float _med = (NULL == med) ? medianf(x, n) : *med;

	for(size_t i=0 ; i < n ; i++){
		absdiff[i] = fabsf(x[i] - _med);
	}

	const float mad = medianf(absdiff, n);
	free(absdiff);
	return mad * mad_scaling_factor;
}


/** Med-MAD normalisation of an array
 *
 *  Normalise an array using the median and MAD as measures of
 *  location and scale respectively.  The array is updated inplace.
 *
 *  @param x An array containing values to normalise
 *  @param n Length of array
 *  @return void
 **/
void medmad_normalise_array(float * x, size_t n){
	if(NULL == x){
		return;
	}
	if(1 == n){
		x[0] = 0.0;
		return;
	}

	const float xmed = medianf(x, n);
	const float xmad = madf(x, n, &xmed);
	for(int i=0 ; i < n ; i++){
		x[i] = (x[i] - xmed)  / xmad;
	}
}


/** Studentise array using Kahan summation algorithm
 *
 *  Studentise an array using the Kahan summation
 *  algorithm for numerical stability. The array is updated inplace.
 *  https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *
 *  @param x An array to normalise
 *  @param n Length of array
 *  @return void
 **/
void studentise_array_kahan(float * x, size_t n){
	if(NULL == x){
		return;
	}

	double sum, sumsq, comp, compsq;
	sumsq = sum = comp = compsq = 0.0;
	for(int i=0 ; i < n ; i++){
		double d1 = x[i] - comp;
		double sum_tmp = sum + d1;
		comp = (sum_tmp - sum) - d1;
		sum = sum_tmp;

		double d2 = x[i] * x[i] - compsq;
		double sumsq_tmp = sumsq + d2;
		compsq = (sumsq_tmp - sumsq) - d2;
		sumsq = sumsq_tmp;
	}
	sum /= n;
	sumsq /= n;
	sumsq -= sum * sum;

	sumsq = sqrt(sumsq);

	const float sumf = sum;
	const float sumsqf = sumsq;
	for(int i=0 ; i < n ; i++){
		x[i] = (x[i] - sumf) / sumsqf;
	}
}
