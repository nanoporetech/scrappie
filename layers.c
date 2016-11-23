#include <assert.h>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include "layers.h"
#include "util.h"



Mat_rptr window(const Mat_rptr input, int w){
	assert(NULL != input);
	assert(w > 0 && (w%2) == 1);
	const int wh = w / 2;

	Mat_rptr output = make_mat(input->nr * w, input->nc);

	for(int col=0 ; col<output->nc ; col++){
		// First and last columns are special cases
		const size_t out_offset = col * output->nrq * 4;
		for(int i=0, w1=col-wh ; w1<=col+wh ; w1++){
			if(w1<0 || w1>=input->nc){
				i += input->nr;
				continue;
			}
			const size_t in_offset = w1 * input->nrq * 4;
			for(int row=0 ; row<input->nr ; row++, i++){
				output->data.f[out_offset + i] = input->data.f[in_offset + row];
			}
		}
	}

	return output;
}

Mat_rptr feedforward_linear(const Mat_rptr X, const Mat_rptr W,
		         const Mat_rptr b, Mat_rptr C){
	C = affine_map(X, W, b, C);
	return C;
}


Mat_rptr feedforward_tanh(const Mat_rptr X, const Mat_rptr W,
	               const Mat_rptr b, Mat_rptr C){
	C = affine_map(X, W, b, C);
	for(int c=0 ; c<C->nc ; c++){
		const int offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = tanhfv(C->data.v[offset +r]);
		}
	}
	return C;
}


Mat_rptr feedforward_exp(const Mat_rptr X, const Mat_rptr W,
	              const Mat_rptr b, Mat_rptr C){
	C = affine_map(X, W, b, C);
	for(int c=0 ; c<C->nc ; c++){
		const int offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = expfv(C->data.v[offset +r]);
		}
	}
	return C;
}

Mat_rptr softmax(const Mat_rptr X, const Mat_rptr W,
              const Mat_rptr b, Mat_rptr C){
	C = feedforward_exp(X, W, b, C);
	row_normalise_inplace(C);
	return C;
}


Mat_rptr feedforward2_tanh(const Mat_rptr Xf, const Mat_rptr Xb,
	  	       const Mat_rptr Wf, const Mat_rptr Wb,
	               const Mat_rptr b, Mat_rptr C){
	C = affine_map2(Xf, Xb, Wf, Wb, b, C);

	for(int c=0 ; c<C->nc ; c++){
		const int offset = c * C->nrq;
		for(int r=0 ; r<C->nrq ; r++){
			C->data.v[offset + r] = tanhfv(C->data.v[offset +r]);
		}
	}
	return C;
}

Mat_rptr gru_forward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW, const Mat_rptr sW2, const Mat_rptr b, Mat_rptr ostate){
	assert(X->nr == iW->nr);
	const int bsize = X->nc;
	const int size = sW2->nc;
	assert(sW->nr == size);
	assert(sW2->nr == size);
	assert(b->nr == 3 * size);
	assert(iW->nc == 3 * size);
	assert(sW->nc == 2 * size);
	assert(sW2->nc == size);
	if(NULL == ostate){
		ostate = make_mat(size, bsize);
	}
	assert(ostate->nr == size);
	assert(ostate->nc == X->nc);

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

	free_mat(tmp);
	return ostate;
}

Mat_rptr gru_backward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW, const Mat_rptr sW2, const Mat_rptr b, Mat_rptr ostate){
	assert(X->nr == iW->nr);
	const int size = sW2->nc;
	const int bsize = X->nc;
	assert(sW->nr == size);
	assert(sW2->nr == size);
	assert(b->nr == 3 * size);
	assert(iW->nc == 3 * size);
	assert(sW->nc == 2 * size);
	assert(sW2->nc == size);
	if(NULL == ostate){
		ostate = make_mat(size, bsize);
	}
	assert(ostate->nr == size);
	assert(ostate->nc == X->nc);

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

	free_mat(tmp);
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
	const int size = sW->nr;
	const int bsize = Xaffine->nc;
	assert(Xaffine->nr == 4 * size);
	assert(p->nr == 3 * size);
	assert(sW->nc == 4 * size);
	if(NULL == output){
		output = make_mat(size, bsize);
	}
	assert(output->nr == size);
	assert(output->nc == Xaffine->nc);

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

	free_mat(state);
	free_mat(tmp);
	return output;
}

Mat_rptr lstm_backward(const Mat_rptr Xaffine, const Mat_rptr sW, const Mat_rptr p, Mat_rptr output){
	const int size = sW->nr;
	const int bsize = Xaffine->nc;
	assert(Xaffine->nr == 4 * size);
	assert(sW->nc == 4 * size);
	assert(p->nr == 3 * size);
	if(NULL == output){
		output = make_mat(size, bsize);
	}
	assert(output->nr == size);
	assert(output->nc == Xaffine->nc);

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

	free_mat(state);
	free_mat(tmp);
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
