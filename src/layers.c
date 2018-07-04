#ifdef __APPLE__
#    include <Accelerate/Accelerate.h>
#else
#    include <cblas.h>
#endif
#include <math.h>
#include "layers.h"
#include "scrappie_stdlib.h"
#include "util.h"

/**  Apply tanh to a matrix element-wise
 *  @param C Matrix
 *
 **/
void tanh_activation_inplace(scrappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = TANHFV(C->data.v[offset + r]);
        }
    }
    (void)validate_scrappie_matrix(C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__);
}

/**  Apply exp to a matrix element-wise
 *  @param C Matrix
 *
 **/
void exp_activation_inplace(scrappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = EXPFV(C->data.v[offset + r]);
        }
    }
    (void)validate_scrappie_matrix(C, 0.0, INFINITY, 1.0, true, __FILE__,
                                   __LINE__);
}

/**  Apply log to a matrix element-wise
 *  @param C Matrix
 *
 **/
void log_activation_inplace(scrappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = LOGFV(C->data.v[offset + r]);
        }
    }
}

/**  Apply ELU activation function to a matrix element-wise
 *  @param C Matrix
 *
 **/
void elu_activation_inplace(scrappie_matrix C) {
    RETURN_NULL_IF(NULL == C, );
    for (size_t c = 0; c < C->nc; ++c) {
        const size_t offset = c * C->nrq;
        for (size_t r = 0; r < C->nrq; ++r) {
            C->data.v[offset + r] = ELUFV(C->data.v[offset + r]);
        }
    }
}

/** Apply robost log activation
 *
 *  Applies log(min_prob / nrow + (1 - min_prob) * x) elementwise to matrix
 *  where x in element and nrow is the number of rows
 *
 *  @param C Matrix
 *  @param min_prob  Minimum probability
 *
 **/
void robustlog_activation_inplace(scrappie_matrix C, float min_prob) {
    assert(min_prob >= 0.0);
    assert(min_prob <= 1.0);
    RETURN_NULL_IF(NULL == C, );

    const size_t nblock = C->nc;
    const __m128 mpv = _mm_set1_ps(min_prob);
    const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
    for (size_t i = 0; i < nblock; i++) {
        const size_t offset = i * C->nrq;
        for (size_t r = 0; r < C->nrq; r++) {
            C->data.v[offset + r] =
                LOGFV(mpv + mpvm1 * C->data.v[offset + r]);
        }
    }
}


scrappie_matrix embedding(int const * index, size_t n, const_scrappie_matrix E, scrappie_matrix C){
    RETURN_NULL_IF(NULL == index, NULL);

    const size_t nr = E->nr;
    const size_t nrq = E->nrq;
    const size_t nc = n;
    C = remake_scrappie_matrix(C, nr, nc);
    RETURN_NULL_IF(NULL == C, NULL);

    for(size_t c=0 ; c < nc ; c++){
        assert(index[c] >= 0 && index[c] < E->nc);
        const size_t offsetC = c * nrq;
        const size_t offsetE = index[c] * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            C->data.v[offsetC + r] = E->data.v[offsetE + r];
        }
    }

    return C;
}


scrappie_matrix window(const_scrappie_matrix input, size_t w, size_t stride) {
    RETURN_NULL_IF(NULL == input, NULL);
    assert(w > 0);
    const size_t wh = (w + 1) / 2;

    scrappie_matrix output = make_scrappie_matrix(input->nr * w,
                                                  (size_t)ceilf(input->nc /
                                                             (float)stride));
    RETURN_NULL_IF(NULL == output, NULL);

    for (size_t col = 0; col < output->nc; col++) {
        // First and last columns are special cases
        const size_t out_offset = col * output->stride;
        const int icol = (int)(col * stride);
        for (int i = 0, w1 = (icol - wh + 1); w1 <= icol + wh; w1++) {
            if (w1 < 0 || w1 >= input->nc) {
                i += input->nr;
                continue;
            }
            const size_t in_offset = w1 * input->stride;
            for (size_t row = 0; row < input->nr; row++, i++) {
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
scrappie_matrix convolution(const_scrappie_matrix X, const_scrappie_matrix W,
                            const_scrappie_matrix b, size_t stride,
                            scrappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != W);
    assert(NULL != b);
    assert(W->nc == b->nr);
    assert(stride > 0);
    // Window length of filter
    assert((W->nrq % X->nrq) == 0);
    const size_t winlen = W->nrq / X->nrq;
    const size_t nfilter = W->nc;
    // Padding -- right-hand side is longer when asymmetric padding is required
    const size_t padL = (winlen - 1) / 2;
    const size_t padR = winlen / 2;
    const size_t ncolC = iceil(X->nc, stride);
    C = remake_scrappie_matrix(C, nfilter, ncolC);
    RETURN_NULL_IF(NULL == C, NULL);

    // Matrix strides
    const size_t ldC = C->stride;
    const size_t ldW = W->stride;
    const size_t ldX = X->stride;
    const size_t ldFeature = ldX;

    // Copy bias into result matrix
    for (size_t i = 0; i < C->nc; i++) {
        memcpy(C->data.v + i * C->nrq, b->data.v, C->nrq * sizeof(__m128));
    }

    // Left-hand side edge case where only part of the filter covers the input
    for (size_t w = 0; w < padL; w += stride) {
        const size_t offsetW = ldFeature * (padL - w);
        const size_t ncol = w / stride;
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc,
                    1.0, W->data.f + offsetW, ldW,
                    X->data.f, 1, 1.0, C->data.f + ldC * ncol, 1);
    }

    // Number of columns of X already filled * ldC
    const size_t ncolsL_complete = iceil(padL, stride);
    const size_t offsetC_L = ldC * ncolsL_complete;
    // Because of stride, first valid filter may not start at beginning of X
    //const int shiftX_L = stride - (padL % stride);
    const size_t shiftX_L = ncolsL_complete * stride - padL;
    const size_t offsetX_L = shiftX_L * ldX;
    // Find multiple of stride greater or equal to winlen
    const size_t nstepC = iceil(winlen, stride);
    const size_t nstepX = stride * nstepC;

    for (size_t w = 0; w < winlen; w += stride) {
        //  Multiply reshaped X matrix by filter matrix
        //  The rows of X are padded by zeros to make a multiple of 4.
        //  Input matrix 'X'
        //   - stride is ldX * nstepX
        //   - offset by ldX * w (w cols)
        //   - Ncolumns is (X->nc - w) / nstepX + adjustment if a final window fits
        //  Filter matrix needs to be padded appropriately for the padding of X.
        //
        const size_t ncol_processed = ifloor(X->nc - shiftX_L - w, nstepX);
        const size_t initial_col = ifloor(w, stride);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, W->nc,
                    ncol_processed, W->nr, 1.0, W->data.f, ldW,
                    X->data.f + ldX * w + offsetX_L, ldX * nstepX, 1.0,
                    C->data.f + ldC * initial_col + offsetC_L, ldC * nstepC);
    }

    // Right-hand side edge case where only part of the filter covers the input
    const size_t maxCol_reshape = ifloor(X->nc - shiftX_L, nstepX);
    const size_t remainder_reshape = (X->nc - shiftX_L) % nstepX;
    const size_t offsetC_R =
        offsetC_L + ldC * nstepC * (maxCol_reshape - 1) +
        ldC * (remainder_reshape / stride) + ldC;
    const size_t offsetX_R = (X->nc - winlen + 1) * ldX;
    // How far into padding is first block
    const int startR = stride - (padL + X->nc - winlen) % stride - 1;
    for (size_t w = startR; w < padR; w += stride) {
        const size_t offsetW = ldFeature * (w + 1);
        cblas_sgemv(CblasColMajor, CblasTrans, W->nr - offsetW, W->nc, 1.0,
                    W->data.f, ldW,
                    X->data.f + offsetX_R + ldX * w, 1, 1.0,
                    C->data.f + offsetC_R + ldC * (w / stride), 1);
    }

    assert(validate_scrappie_matrix
           (C, NAN, NAN, 0.0, true, __FILE__, __LINE__));
    return C;
}

scrappie_matrix feedforward_linear(const_scrappie_matrix X,
                                   const_scrappie_matrix W,
                                   const_scrappie_matrix b, scrappie_matrix C) {
    return affine_map(X, W, b, C);
}

scrappie_matrix feedforward_tanh(const_scrappie_matrix X,
                                 const_scrappie_matrix W,
                                 const_scrappie_matrix b, scrappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    tanh_activation_inplace(C);

    assert(validate_scrappie_matrix
           (C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return C;
}

scrappie_matrix feedforward_exp(const_scrappie_matrix X,
                                const_scrappie_matrix W,
                                const_scrappie_matrix b, scrappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    exp_activation_inplace(C);

    assert(validate_scrappie_matrix
           (C, 0.0, NAN, 1.0, true, __FILE__, __LINE__));
    return C;
}

scrappie_matrix residual(const_scrappie_matrix X, const_scrappie_matrix fX, scrappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);
    RETURN_NULL_IF(NULL == fX, NULL);
    const size_t nr = X->nr;
    const size_t nrq = X->nrq;
    const size_t nc = X->nc;
    assert(nr == fX->nr);
    assert(nrq == fX->nrq);
    assert(nc == fX->nc);

    C = remake_scrappie_matrix(C, nr, nc);
    RETURN_NULL_IF(NULL == C, NULL);

    for(size_t c=0 ; c < nc ; c++){
        const size_t offset = c * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            C->data.v[offset + r] = X->data.v[offset + r] + fX->data.v[offset + r];
        }
    }

    return C;
}

void residual_inplace(const_scrappie_matrix X, scrappie_matrix fX) {
    RETURN_NULL_IF(NULL == X, );
    RETURN_NULL_IF(NULL == fX, );

    const size_t nrq = X->nrq;
    const size_t nc = X->nc;
    assert(X->nr == fX->nr);
    assert(nrq == fX->nrq);
    assert(nc == fX->nc);

    for(size_t c=0 ; c < nc ; c++){
        const size_t offset = c * nrq;
        for(size_t r=0 ; r < nrq ; r++){
            fX->data.v[offset + r] += X->data.v[offset + r];
        }
    }
}

scrappie_matrix softmax(const_scrappie_matrix X, const_scrappie_matrix W,
                        const_scrappie_matrix b, scrappie_matrix C) {
    C = feedforward_exp(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    row_normalise_inplace(C);

    assert(validate_scrappie_matrix
           (C, 0.0, 1.0, NAN, true, __FILE__, __LINE__));
    return C;
}

/**   Softmax with separate temperatures on weights and bias
 *
 *    Calculates softmax( A x / tempW + b / tempb ) as
 *    softmax( (A (x * tempb / tempW ) + b) / tempb )
 *
 *    @returns matrix containing softmax
 **/
scrappie_matrix softmax_with_temperature(scrappie_matrix X, const_scrappie_matrix W,
                                         const_scrappie_matrix b, float tempW, float tempb,
                                         scrappie_matrix C) {
    RETURN_NULL_IF(NULL == X, NULL);

    shift_scale_matrix_inplace(X, 0.0f, tempW / tempb);

    C = feedforward_linear(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    shift_scale_matrix_inplace(C, 0.0f, tempb);
    exp_activation_inplace(C);
    row_normalise_inplace(C);

    assert(validate_scrappie_matrix
           (C, 0.0, 1.0, NAN, true, __FILE__, __LINE__));
    return C;
}

scrappie_matrix feedforward2_tanh(const_scrappie_matrix Xf,
                                  const_scrappie_matrix Xb,
                                  const_scrappie_matrix Wf,
                                  const_scrappie_matrix Wb,
                                  const_scrappie_matrix b, scrappie_matrix C) {
    C = affine_map2(Xf, Xb, Wf, Wb, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    tanh_activation_inplace(C);

    assert(validate_scrappie_matrix(C, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return C;
}

scrappie_matrix gru_forward(const_scrappie_matrix X, const_scrappie_matrix sW,
                            const_scrappie_matrix sW2, scrappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t bsize = X->nc;
    const size_t size = sW2->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_scrappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set second column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = ostate->data.v + ostate->nrq;
    sCol2.data.v = ostate->data.v;
    gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = X->data.v + i * X->nrq;
        sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + i * ostate->nrq;
        gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

scrappie_matrix gru_backward(const_scrappie_matrix X, const_scrappie_matrix sW,
                             const_scrappie_matrix sW2, scrappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);
    assert(NULL != sW2);

    const size_t size = sW2->nc;
    const size_t bsize = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nr == size);
    assert(sW2->nr == size);
    assert(sW->nc == 2 * size);
    assert(sW2->nc == size);

    ostate = remake_scrappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
    gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = X->data.v + index * X->nrq;
        sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + index * ostate->nrq;
        gru_step(&xCol, &sCol1, sW, sW2, tmp, &sCol2);
    }

    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

void gru_step(const_scrappie_matrix x, const_scrappie_matrix istate,
              const_scrappie_matrix sW, const_scrappie_matrix sW2,
              scrappie_matrix xF, scrappie_matrix ostate) {
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
    assert(NULL != sW);
    assert(NULL != sW2);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(2 * size == sW->nc);
    assert(size == sW2->nr);
    assert(size == sW2->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);


    // Copy input vector = iW x + b to temporary vector
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    /*  Add sW * istate to first 2 * size elts of xF
     *  then apply gate function to get r and z
     */
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);
    for (size_t i = 0; i < (sizeq +sizeq); i++) {
        xF->data.v[i] = LOGISTICFV(xF->data.v[i]);
    }

    const __m128 *z = xF->data.v;
    __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        r[i] *= istate->data.v[i];
    }
    cblas_sgemv(CblasColMajor, CblasTrans, sW2->nr, sW2->nc, 1.0, sW2->data.f,
                sW2->stride, (float *)r, 1, 1.0, (float *)hbar, 1);
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = TANHFV(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    }
}

scrappie_matrix grumod_forward(const_scrappie_matrix X, const_scrappie_matrix sW,
                               scrappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);

    assert(NULL != sW);

    const size_t bsize = X->nc;
    const size_t size = sW->nr;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    ostate = remake_scrappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set second column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v + ostate->nrq, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = ostate->data.v + ostate->nrq;
    sCol2.data.v = ostate->data.v;
    grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = X->data.v + i * X->nrq;
        sCol1.data.v = ostate->data.v + (i - 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + i * ostate->nrq;
        grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    }

    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

scrappie_matrix grumod_backward(const_scrappie_matrix X, const_scrappie_matrix sW,
                                scrappie_matrix ostate) {
    RETURN_NULL_IF(NULL == X, NULL);
    assert(NULL != sW);

    const size_t size = sW->nr;
    const size_t bsize = X->nc;
    assert(X->nr == 3 * size);
    assert(sW->nc == 3 * size);

    ostate = remake_scrappie_matrix(ostate, size, bsize);
    RETURN_NULL_IF(NULL == ostate, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(3 * size, 1);
    if(NULL == tmp){
        //  Memory allocation falled, clean-up and return
        free(ostate);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    _Mat xCol, sCol1, sCol2;
    memset(ostate->data.v, 0, ostate->nrq * sizeof(__m128));
    xCol = *X;
    sCol1 = *ostate;
    sCol2 = *ostate;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = X->data.v + (X->nc - 1) * X->nrq;
    sCol1.data.v = ostate->data.v;
    sCol2.data.v = ostate->data.v + (ostate->nc - 1) * ostate->nrq;
    grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = X->data.v + index * X->nrq;
        sCol1.data.v = ostate->data.v + (index + 1) * ostate->nrq;
        sCol2.data.v = ostate->data.v + index * ostate->nrq;
        grumod_step(&xCol, &sCol1, sW, tmp, &sCol2);
    }

    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (ostate, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return ostate;
}

void grumod_step(const_scrappie_matrix x, const_scrappie_matrix istate,
                 const_scrappie_matrix sW, scrappie_matrix xF,
                 scrappie_matrix ostate) {
    /* Perform a single modified GRU step
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
    assert(NULL != sW);
    const size_t size = istate->nr;
    assert(x->nr == 3 * size);
    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    assert(size == sW->nr);
    assert(3 * size == sW->nc);
    assert(3 * size == xF->nr);
    assert(size == ostate->nr);


    // Copy input vector = iW x + b to temporary vector and zero last chunk
    memcpy(xF->data.v, x->data.v, x->nrq * sizeof(__m128));
    memset(xF->data.v + sizeq + sizeq, 0, sizeq *sizeof(__m128));
    /*  Add sW * istate to first 3 * size elts of xF
     *  then apply gate function to get r and z
     */
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, istate->data.f, 1, 1.0, xF->data.f, 1);
    for (size_t i = 0; i < (sizeq + sizeq); i++) {
        xF->data.v[i] = LOGISTICFV(xF->data.v[i]);
    }

    const __m128 *z = xF->data.v;
    const __m128 *r = xF->data.v + sizeq;
    __m128 *hbar = xF->data.v + sizeq + sizeq;
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = r[i] * hbar[i] + x->data.v[sizeq + sizeq + i];
    }
    for (size_t i = 0; i < sizeq; i++) {
        hbar[i] = TANHFV(hbar[i]);
    }

    const __m128 ones = _mm_set1_ps(1.0f);
    for (size_t i = 0; i < sizeq ; i++) {
        ostate->data.v[i] = z[i] * istate->data.v[i] + (ones - z[i]) * hbar[i];
    }
}

scrappie_matrix lstm_forward(const_scrappie_matrix Xaffine,
                             const_scrappie_matrix sW, const_scrappie_matrix p,
                             scrappie_matrix output) {
    RETURN_NULL_IF(NULL == Xaffine, NULL);
    assert(NULL != sW);
    assert(NULL != p);

    const size_t size = sW->nr;
    const size_t bsize = Xaffine->nc;
    assert(Xaffine->nr == 4 * size);
    assert(p->nr == 3 * size);
    assert(sW->nc == 4 * size);

    output = remake_scrappie_matrix(output, size, bsize);
    RETURN_NULL_IF(NULL == output, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(4 * size, 1);
    scrappie_matrix state = make_scrappie_matrix(size, 1);
    if(NULL == tmp || NULL == state){
        //  Memory allocation falled, clean-up and return
        free(state);
        free(tmp);
        free(output);
        return NULL;
    }

    /* First step state & output are zero.  Set second column of output to zero and use that */
    memset(output->data.v + output->nrq, 0, output->nrq * sizeof(__m128));
    _Mat xCol, sCol1, sCol2;
    xCol = *Xaffine;
    sCol1 = *output;
    sCol2 = *output;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    sCol1.data.v = output->data.v + output->nrq;
    sCol2.data.v = output->data.v;
    lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        xCol.data.v = Xaffine->data.v + i * Xaffine->nrq;
        sCol1.data.v = output->data.v + (i - 1) * output->nrq;
        sCol2.data.v = output->data.v + i * output->nrq;
        lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    }

    state = free_scrappie_matrix(state);
    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (output, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return output;
}

scrappie_matrix lstm_backward(const_scrappie_matrix Xaffine,
                              const_scrappie_matrix sW, const_scrappie_matrix p,
                              scrappie_matrix output) {
    RETURN_NULL_IF(NULL == Xaffine, NULL);
    assert(NULL != sW);
    assert(NULL != p);

    const size_t size = sW->nr;
    const size_t bsize = Xaffine->nc;
    assert(Xaffine->nr == 4 * size);
    assert(sW->nc == 4 * size);
    assert(p->nr == 3 * size);

    output = remake_scrappie_matrix(output, size, bsize);
    RETURN_NULL_IF(NULL == output, NULL);

    scrappie_matrix tmp = make_scrappie_matrix(4 * size, 1);
    scrappie_matrix state = make_scrappie_matrix(size, 1);
    if(NULL == tmp || NULL == state){
        //  Memory allocation falled, clean-up and return
        free(state);
        free(tmp);
        free(output);
        return NULL;
    }

    /* First step state is zero.  Set first column of ostate to zero and use that */
    memset(output->data.v, 0, output->nrq * sizeof(__m128));
    _Mat xCol, sCol1, sCol2;
    xCol = *Xaffine;
    sCol1 = *output;
    sCol2 = *output;
    xCol.nc = sCol1.nc = sCol2.nc = 1;
    xCol.data.v = Xaffine->data.v + (bsize - 1) * Xaffine->nrq;
    sCol1.data.v = output->data.v;
    sCol2.data.v = output->data.v + (bsize - 1) * output->nrq;
    lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    for (size_t i = 1; i < bsize; i++) {
        const size_t index = bsize - i - 1;
        xCol.data.v = Xaffine->data.v + index * Xaffine->nrq;
        sCol1.data.v = output->data.v + (index + 1) * output->nrq;
        sCol2.data.v = output->data.v + index * output->nrq;
        lstm_step(&xCol, &sCol1, sW, p, tmp, state, &sCol2);
    }

    state = free_scrappie_matrix(state);
    tmp = free_scrappie_matrix(tmp);

    assert(validate_scrappie_matrix
           (output, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    return output;
}

void lstm_step(const_scrappie_matrix xAffine, const_scrappie_matrix out_prev,
               const_scrappie_matrix sW, const_scrappie_matrix peep,
               scrappie_matrix xF, scrappie_matrix state,
               scrappie_matrix output) {
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
    const size_t size = state->nr;
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
    cblas_sgemv(CblasColMajor, CblasTrans, sW->nr, sW->nc, 1.0, sW->data.f,
                sW->stride, out_prev->data.f, 1, 1.0, xF->data.f, 1);

    assert(size % 4 == 0);  // Vectorisation assumes size divisible by 4
    const size_t sizeq = size / 4;
    for (size_t i = 0; i < sizeq; i++) {
        // Forget gate
        __m128 forget = LOGISTICFV(xF->data.v[2 * sizeq + i]
                                   + state->data.v[i] * peep->data.v[sizeq + i])
            * state->data.v[i];
        // Update gate
        __m128 update = LOGISTICFV(xF->data.v[sizeq + i]
                                   + state->data.v[i] * peep->data.v[i])
            * TANHFV(xF->data.v[i]);
        state->data.v[i] = _mm_add_ps(forget, update);
        // Output gate
        output->data.v[i] = LOGISTICFV(xF->data.v[3 * sizeq + i]
                                       +
                                       state->data.v[i] * peep->data.v[2 *
                                                                       sizeq +
                                                                       i])
            * TANHFV(state->data.v[i]);
    }
}


float crf_partition_function(const_scrappie_matrix C){
    RETURN_NULL_IF(NULL == C, NAN);

    const size_t nstate = roundf(sqrtf((float)C->nr));
    assert(nstate * nstate == C->nr);
    float * mem = calloc(2 * nstate, sizeof(float));
    RETURN_NULL_IF(NULL==mem, NAN);

    float * curr = mem;
    float * prev = mem + nstate;

    for(size_t c=0 ; c < C->nc ; c++){
        const size_t offset = c * C->stride;
        //  Swap
        {
            float * tmp = curr;
            curr = prev;
            prev = tmp;
        }
        for(size_t st1=0 ; st1 < nstate ; st1++){
            const size_t offsetS = offset + st1 * nstate;
            curr[st1] = C->data.f[offsetS + 0] + prev[0];
            for(size_t st2=1 ; st2 < nstate ; st2++){
                curr[st1] = logsumexpf(curr[st1],  C->data.f[offsetS + st2] + prev[st2]);
            }
        }
    }

    float logZ = curr[0];
    for(size_t st=1 ; st < nstate ; st++){
        logZ = logsumexpf(logZ, curr[st]);
    }

    free(mem);

    return logZ;
}


scrappie_matrix globalnorm(const_scrappie_matrix X, const_scrappie_matrix W,
                           const_scrappie_matrix b, scrappie_matrix C) {
    C = affine_map(X, W, b, C);
    RETURN_NULL_IF(NULL == C, NULL);

    float logZ = crf_partition_function(C) / (float)C->nc;

    for(size_t c=0 ; c < C->nc ; c++){
        const size_t offset = c * C->stride;
        for(size_t r=0 ; r < C->nr ; r++){
            C->data.f[offset + r] -= logZ;
        }
    }

    return C;
}
