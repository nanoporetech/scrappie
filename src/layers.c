#ifdef __APPLE__
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif
#include <math.h>
#include <string.h>
#include "layers.h"
#include "scrappie_assert.h"
#include "util.h"



Mat_rptr window(const Mat_rptr input, int w, int stride){
	ASSERT_OR_RETURN_NULL(NULL != input, NULL);
	assert(w > 0);
	const int wh = (w + 1) / 2;

	Mat_rptr output = make_mat(input->nr * w, (int)ceilf(input->nc / (float)stride));

	for(int col=0 ; col<output->nc ; col++){
		// First and last columns are special cases
		const size_t out_offset = col * output->nrq * 4;
		const int icol = col * stride;
		for (int i = 0, w1 = (icol - wh + 1); w1 <= icol + wh; w1++) {
			if(w1 < 0 || w1 >= input->nc){
				i += input->nr;
				continue;
			}
			const size_t in_offset = w1 * input->nrq * 4;
			for(int row=0 ; row < input->nr ; row++, i++){
				output->data.f[out_offset + i] = input->data.f[in_offset + row];
			}
		}
	}

	return output;
}


/**  Convolution of the input data
 *  @param X Input data matrix (features x nobs)
 *  @param W Filter matrix (winlen * features x nfilter)
 *
 *  The input is padded with zeros such that the resultant matrix has the
 *  same size as the input (under a stride of 1).
 *
 *  Note: The rows of the input matrix X are padded with zeros to make them
 *  a multiple of the SSE vector size (4).  The filter matrix must have been
 *  expanded accordingly.
 **/
Mat_rptr Convolution(const Mat_rptr X, const Mat_rptr W, const Mat_rptr b, int stride, Mat_rptr C) {
	ASSERT_OR_RETURN_NULL(NULL != X, NULL);
	assert(NULL != W);
	assert(NULL != b);
	assert(W->nc == b->nr);
	assert(stride > 0);
	// Window length of filter
	assert((W->nrq % X->nrq) == 0);
	const int winlen = W->nrq / X->nrq;
	const int nfilter = W->nc;
	// Padding -- right-hand side is longer when asymmetric padding is required
	const int padL = (winlen - 1) / 2;
	const int padR = winlen / 2;
	const int ncolC = iceil(X->nc, stride);
	C = remake_mat(C, nfilter, ncolC);

	// Matrix strides
	const int ldC = C->nrq * 4;
	const int ldW = W->nrq * 4;
	const int ldX = X->nrq * 4;
	const int ldFeature = ldX;

	// Copy bias into result matrix
	for(int i=0 ; i < C->nc ; i++){
		memcpy(C->data.v + i * C->nrq, b->data.v, C->nrq * sizeof(__m128));
	}


	// Left-hand side edge case where only part of the filter covers the input
	for (int w = 0; w < padL; w += stride) {
		const int offsetW = ldFeature * (padL - w);
		const int ncol = w / stride;
		cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc,
			1.0, W->data.f + offsetW, ldW,
			X->data.f, 1, 1.0, C->data.f + ldC * ncol,
			1);
	}

	// Number of columns of X already filled * ldC
	const int ncolsL_complete = iceil(padL, stride);
	const int offsetC_L = ldC * ncolsL_complete;
	// Because of stride, first valid filter may not start at beginning of X
	//const int shiftX_L = stride - (padL % stride);
	const int shiftX_L = ncolsL_complete * stride - padL;
	const int offsetX_L = shiftX_L * ldX;
	// Find multiple of stride greater or equal to winlen
	const int nstepC = iceil(winlen, stride);
	const int nstepX = stride * nstepC;

	for (int w = 0; w < winlen; w += stride) {
		//  Multiply reshaped X matrix by filter matrix
		//  The rows of X are padded by zeros to make a multiple of 4.
		//  Input matrix 'X'
		//   - stride is ldX * nstepX
		//   - offset by ldX * w (w cols)
		//   - Ncolumns is (X->nc - w) / nstepX + adjustment if a final window fits
		//  Filter matrix needs to be padded appropriately for the padding of X.
		//
		const int ncol_processed = ifloor(X->nc - shiftX_L - w, nstepX);
		const int initial_col = ifloor(w, stride);
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc, ncol_processed, W->nr,
			1.0, W->data.f, ldW,
			X->data.f + ldX * w + offsetX_L, ldX * nstepX,
			1.0, C->data.f + ldC * initial_col + offsetC_L,
			ldC * nstepC);
	}

	// Right-hand side edge case where only part of the filter covers the input
	/*const int maxCol_reshape = ifloor(X->nc - shiftX_L, nstepX);
	const int remainder_reshape = (X->nc - shiftX_L) % nstepX;
	const int offsetC_R = offsetC_L + ldC * nstepC * (maxCol_reshape - 1) + ldC * (remainder_reshape / stride) + ldC;
	const int offsetX_R = offsetX_L + ldX * nstepX * (maxCol_reshape - 1) + ldX * stride * (remainder_reshape / stride) + ldX;
	for (int w = 0 ; w < padR; w += stride) {
        	const int offsetW = ldFeature * (padR - w);
		cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc, 1.0,
			W->data.f, ldW,
			X->data.f + offsetX_R + ldX * w, 1, 1.0,
			C->data.f + offsetC_R + ldC * (w / stride), 1);
	}*/

	// Apply an activation function
	for(int c=0 ; c<C->nc ; c++){
		const size_t offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = tanhfv(C->data.v[offset +r]);
		}
	}
	return C;
}


Mat_rptr feedforward_linear(const Mat_rptr X, const Mat_rptr W,
		         const Mat_rptr b, Mat_rptr C){
	return affine_map(X, W, b, C);
}


Mat_rptr feedforward_tanh(const Mat_rptr X, const Mat_rptr W,
	               const Mat_rptr b, Mat_rptr C){
	C = affine_map(X, W, b, C);
	ASSERT_OR_RETURN_NULL(NULL != C, NULL);

	for(int c=0 ; c<C->nc ; c++){
		const size_t offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = tanhfv(C->data.v[offset +r]);
		}
	}
	return C;
}


Mat_rptr feedforward_exp(const Mat_rptr X, const Mat_rptr W,
	              const Mat_rptr b, Mat_rptr C){
	C = affine_map(X, W, b, C);
	ASSERT_OR_RETURN_NULL(NULL != C, NULL);

	for(int c=0 ; c<C->nc ; c++){
		const size_t offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = EXPFV(C->data.v[offset +r]);
		}
	}
	return C;
}


Mat_rptr softmax(const Mat_rptr X, const Mat_rptr W,
              const Mat_rptr b, Mat_rptr C){
	C = feedforward_exp(X, W, b, C);
	ASSERT_OR_RETURN_NULL(NULL != C, NULL);

	row_normalise_inplace(C);
	return C;
}


Mat_rptr feedforward2_tanh(const Mat_rptr Xf, const Mat_rptr Xb,
	  	       const Mat_rptr Wf, const Mat_rptr Wb,
	               const Mat_rptr b, Mat_rptr C){
	C = affine_map2(Xf, Xb, Wf, Wb, b, C);
	ASSERT_OR_RETURN_NULL(NULL != C, NULL);

	for(int c=0 ; c<C->nc ; c++){
		const size_t offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = tanhfv(C->data.v[offset +r]);
		}
	}
	return C;
}


Mat_rptr gru_forward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW, const Mat_rptr sW2, const Mat_rptr b, Mat_rptr ostate){
	ASSERT_OR_RETURN_NULL(NULL != X, NULL);

	assert(NULL != iW);
	assert(NULL != sW);
	assert(NULL != sW2);
	assert(NULL != b);

	assert(X->nr == iW->nr);
	const int bsize = X->nc;
	const int size = sW2->nc;
	assert(sW->nr == size);
	assert(sW2->nr == size);
	assert(b->nr == 3 * size);
	assert(iW->nc == 3 * size);
	assert(sW->nc == 2 * size);
	assert(sW2->nc == size);
	ostate = remake_mat(ostate, size, bsize);
	ASSERT_OR_RETURN_NULL(NULL != ostate, NULL);

	_Mat xCol, sCol1, sCol2;
	Mat_rptr tmp = make_mat(3 * size, 1);

	/* First step state is zero.  Set second column of ostate to zero and use that */
	memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
	xCol = *X; sCol1 = *ostate; sCol2 = *ostate;
	xCol.nc = sCol1.nc = sCol2.nc = 1;
	sCol1.data.v = ostate->data.v + ostate->nrq;
	sCol2.data.v = ostate->data.v;
	gru_step(&xCol, &sCol1, iW, sW, sW2, b, tmp, &sCol2);
	for(int i=1 ; i < bsize ; i++){
		xCol.data.v = X->data.v + i * X->nrq;
		sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
		sCol2.data.v = ostate->data.v + i * ostate->nrq;
		gru_step(&xCol, &sCol1, iW, sW, sW2, b, tmp, &sCol2);
	}

	tmp = free_mat(tmp);
	return ostate;
}


Mat_rptr gru_backward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW, const Mat_rptr sW2, const Mat_rptr b, Mat_rptr ostate){
	ASSERT_OR_RETURN_NULL(NULL != X, NULL);
	assert(NULL != iW);
	assert(NULL != sW);
	assert(NULL != sW2);
	assert(NULL != b);

	assert(X->nr == iW->nr);
	const int size = sW2->nc;
	const int bsize = X->nc;
	assert(sW->nr == size);
	assert(sW2->nr == size);
	assert(b->nr == 3 * size);
	assert(iW->nc == 3 * size);
	assert(sW->nc == 2 * size);
	assert(sW2->nc == size);
	ostate = remake_mat(ostate, size, bsize);
	ASSERT_OR_RETURN_NULL(NULL != ostate, NULL);

	_Mat xCol, sCol1, sCol2;
	Mat_rptr tmp = make_mat(3 * size, 1);

	/* First step state is zero.  Set first column of ostate to zero and use that */
	memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
	xCol = *X; sCol1 = *ostate; sCol2 = *ostate;
	xCol.nc = sCol1.nc = sCol2.nc = 1;
	xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
	sCol1.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
	sCol2.data.v = ostate->data.v;
	gru_step(&xCol, &sCol1, iW, sW, sW2, b, tmp, &sCol2);
        for(int i=1 ; i < bsize ; i++){
		const int index = bsize - i - 1;
		xCol.data.v = X->data.v + index * X->nrq;
		sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
		sCol2.data.v = ostate->data.v + index * ostate->nrq;
		gru_step(&xCol, &sCol1, iW, sW, sW2, b, tmp, &sCol2);
	}

	tmp = free_mat(tmp);
	return ostate;
}


void gru_step(const Mat_rptr x, const Mat_rptr istate,
	      const Mat_rptr xW, const Mat_rptr sW, const Mat_rptr sW2, const Mat_rptr bias,
	      Mat_rptr xF, Mat_rptr ostate){
	/* Perform a single GRU step
	 * x      is [isize]
	 * istate is [size]
	 * xW     is [isize, 3 * size]
	 * sW     is [size, 2 * size]
	 * sW2    is [size, size]
	 * bias   is [3 * size]
	 * xF     is [3 * size]
	 * ostate is [size]
	 */
	assert(NULL != x);
	assert(NULL != xW);
	assert(NULL != sW);
	assert(NULL != sW2);
	assert(NULL != bias);
	assert(x->nr == xW->nr);
	const int size = istate->nr;
	assert(3 * size == xW->nc);
	assert(size == sW->nr);
	assert(2 * size == sW->nc);
	assert(size == sW2->nr);
	assert(size == sW2->nc);
	assert(3 * size == bias->nr);
	assert(3 * size == xF->nr);
	assert(size == ostate->nr);

	/* Copy bias vector to output*/
	memcpy(xF->data.v, bias->data.v, bias->nrq * sizeof(__m128));
	//  Following line could be lifted out of GRU iteration at the expense of more memory use
	cblas_sgemv(CblasColMajor, CblasTrans, xW->nr, xW->nc, 1.0, xW->data.f, xW->nrq * 4,
		    x->data.f, 1, 1.0, xF->data.f, 1);
	/*  Add sW * istate to first 2 * size elts of xF
	 *  then apply gate function to get r and z
	 */
	cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f, sW->nrq * 4,
		    istate->data.f, 1, 1.0, xF->data.f, 1);
	for(int i=0 ; i < (size / 2) ; i++){
		xF->data.v[i] = logisticfv(xF->data.v[i]);
	}

	const int sizeq = size / 4;
	const __m128 * z = xF->data.v;
	__m128 * r = xF->data.v + sizeq;
	__m128 * hbar = xF->data.v + 2 * sizeq;
	for(int i=0 ; i < sizeq ; i++){
		r[i] *= istate->data.v[i];
	}
	cblas_sgemv(CblasColMajor, CblasTrans, sW2->nr, sW2->nc, 1.0, sW2->data.f, sW2->nrq * 4, (float *)r, 1, 1.0, (float *)hbar, 1);
	for(int i=0 ; i < sizeq ; i++){
		hbar[i] = tanhfv(hbar[i]);
	}


	const __m128 ones = _mm_set1_ps(1.0f);
	for(int i=0 ; i < sizeq ; i++){
		ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
	}
}


Mat_rptr lstm_forward(const Mat_rptr Xaffine, const Mat_rptr sW, const Mat_rptr p, Mat_rptr output){
	ASSERT_OR_RETURN_NULL(NULL != Xaffine, NULL);
	assert(NULL != sW);
	assert(NULL != p);

	const int size = sW->nr;
	const int bsize = Xaffine->nc;
	assert(Xaffine->nr == 4 * size);
	assert(p->nr == 3 * size);
	assert(sW->nc == 4 * size);
	output = remake_mat(output, size, bsize);
	ASSERT_OR_RETURN_NULL(NULL != output, NULL);

	Mat_rptr tmp = make_mat(4 * size, 1);
	Mat_rptr state = make_mat(size, 1);

	/* First step state & output are zero.  Set second column of output to zero and use that */
	memset(output->data.v + output->nrq, 0, output->nrq * sizeof(__m128));
	_Mat xCol, sCol1, sCol2;
	xCol = *Xaffine; sCol1 = *output; sCol2 = *output;
	xCol.nc = sCol1.nc = sCol2.nc = 1;
	sCol1.data.v = output->data.v + output->nrq;
	sCol2.data.v = output->data.v;
	lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
	for(int i=1 ; i < bsize ; i++){
		xCol.data.v = Xaffine->data.v + i * Xaffine->nrq;
		sCol1.data.v = output->data.v + (i - 1) * output->nrq;
		sCol2.data.v = output->data.v + i * output->nrq;
		lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
	}

	state = free_mat(state);
	state = free_mat(tmp);
	return output;
}


Mat_rptr lstm_backward(const Mat_rptr Xaffine, const Mat_rptr sW, const Mat_rptr p, Mat_rptr output){
	ASSERT_OR_RETURN_NULL(NULL != Xaffine, NULL);
	assert(NULL != sW);
	assert(NULL != p);

	const int size = sW->nr;
	const int bsize = Xaffine->nc;
	assert(Xaffine->nr == 4 * size);
	assert(sW->nc == 4 * size);
	assert(p->nr == 3 * size);
	output = remake_mat(output, size, bsize);
	ASSERT_OR_RETURN_NULL(NULL != output, NULL);

	Mat_rptr tmp = make_mat(4 * size, 1);
	Mat_rptr state = make_mat(size, 1);

	/* First step state is zero.  Set first column of ostate to zero and use that */
	memset(output->data.v, 0, output->nrq * sizeof(__m128));
	_Mat xCol, sCol1, sCol2;
	xCol = *Xaffine; sCol1 = *output; sCol2 = *output;
	xCol.nc = sCol1.nc = sCol2.nc = 1;
	xCol.data.v = Xaffine->data.v + (bsize - 1) * Xaffine->nrq;
	sCol1.data.v = output->data.v;
	sCol2.data.v = output->data.v + (bsize - 1) * output->nrq;
	lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
        for(int i=1 ; i < bsize ; i++){
		const int index = bsize - i - 1;
		xCol.data.v = Xaffine->data.v + index * Xaffine->nrq;
		sCol1.data.v = output->data.v + (index + 1) * output->nrq;
		sCol2.data.v = output->data.v + index * output->nrq;
		lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
	}

	state = free_mat(state);
	tmp = free_mat(tmp);
	return output;
}


void lstm_step(const Mat_rptr xAffine, const Mat_rptr out_prev,
	       const Mat_rptr sW, const Mat_rptr peep,
	       Mat_rptr xF, Mat_rptr state, Mat_rptr output){
	/* Perform a single LSTM step
	 * xAffine  is [isize] (== iW x + b, where x is the input to the LSTM layer)
	 * out_prev is [size]
	 * sW       is [size, 4 * size]
	 * peep     is [4 * size]
	 * xF       is [4 * size]
	 * state    is [size]
	 * output   is [size]
	 */
	assert(NULL != xAffine);
	assert(NULL != out_prev);
	assert(NULL != sW);
	assert(NULL != peep);
	assert(NULL != xF);
	assert(NULL != state);
	assert(NULL != output);
	const int size = state->nr;
	assert(xAffine->nr == 4 * size);
	assert(size == out_prev->nr);
	assert(size == sW->nr);
	assert(4 * size == sW->nc);
	assert(3 * size == peep->nr);
	assert(4 * size == xF->nr);
	assert(size == output->nr);

	// Copy input vector = iW x + b to temporary vector
	memcpy(xF->data.v, xAffine->data.v, xAffine->nrq * sizeof(__m128));
	//  + sW' * xprev
	cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f, sW->nrq * 4,
		    out_prev->data.f, 1, 1.0, xF->data.f, 1);

	assert((size % 4) == 0);
	const int sizeq = size / 4;
	for(int i=0 ; i < sizeq ; i++){
		// Forget gate
		__m128 forget = logisticfv(xF->data.v[2 * sizeq + i]
				          + state->data.v[i] * peep->data.v[sizeq + i])
			      * state->data.v[i];
		// Update gate
		__m128 update = logisticfv(xF->data.v[sizeq + i]
			                  + state->data.v[i] * peep->data.v[i])
			      * tanhfv(xF->data.v[i]);
		state->data.v[i] = forget + update;
		// Output gate
		output->data.v[i] = logisticfv(xF->data.v[3 * sizeq + i]
				              + state->data.v[i] * peep->data.v[2 * sizeq + i])
			          * tanhfv(state->data.v[i]);
	}
}
