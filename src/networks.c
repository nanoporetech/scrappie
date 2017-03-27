#include <string.h>
#include "layers.h"
#include "nanonet_lstm_events.h"
#include "nanonet_raw.h"
#include "networks.h"
#include "nnfeatures.h"
#include "scrappie_assert.h"

Mat_rptr nanonet_posterior(const event_table events, float min_prob, bool return_log){
	assert(min_prob >= 0.0 && min_prob <= 1.0);
	ASSERT_OR_RETURN_NULL(events.n > 0 && NULL != events.event, NULL);

	const int WINLEN = 3;

        //  Make features
        Mat_rptr features = make_features(events, true);
        Mat_rptr feature3 = window(features, WINLEN, 1);
        features = free_mat(features);

        // Initial transformation of input for LSTM layer
        Mat_rptr lstmXf = feedforward_linear(feature3, lstmF1_iW, lstmF1_b, NULL);
        Mat_rptr lstmXb = feedforward_linear(feature3, lstmB1_iW, lstmB1_b, NULL);
        feature3 = free_mat(feature3);
        Mat_rptr lstmF = lstm_forward(lstmXf, lstmF1_sW, lstmF1_p, NULL);
        Mat_rptr lstmB = lstm_backward(lstmXb, lstmB1_sW, lstmB1_p, NULL);

        //  Combine LSTM output
        Mat_rptr lstmFF = feedforward2_tanh(lstmF, lstmB, FF1_Wf, FF1_Wb, FF1_b, NULL);

        lstmXf = feedforward_linear(lstmFF, lstmF2_iW, lstmF2_b, lstmXf);
        lstmXb = feedforward_linear(lstmFF, lstmB2_iW, lstmB2_b, lstmXb);
        lstmF = lstm_forward(lstmXf, lstmF2_sW, lstmF2_p, lstmF);
        lstmXf = free_mat(lstmXf);
        lstmB = lstm_backward(lstmXb, lstmB2_sW, lstmB2_p, lstmB);
        lstmXb = free_mat(lstmXb);

        // Combine LSTM output
        lstmFF = feedforward2_tanh(lstmF, lstmB, FF2_Wf, FF2_Wb, FF2_b, lstmFF);
        lstmF = free_mat(lstmF);
        lstmB = free_mat(lstmB);

        Mat_rptr post = softmax(lstmFF, FF3_W, FF3_b, NULL);
        lstmFF = free_mat(lstmFF);
	ASSERT_OR_RETURN_NULL(NULL != post, NULL);

	if(return_log){
		const int nev = post->nc;
		const int nstate = post->nr;
		const __m128 mpv = _mm_set1_ps(min_prob / nstate);
		const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
		for(int i=0 ; i < nev ; i++){
			const size_t offset = i * post->nrq;
			for(int r=0 ; r < post->nrq ; r++){
				post->data.v[offset + r] = LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
			}
		}
	}

	return post;
}


Mat_rptr mat_raw(const raw_table signal){
	ASSERT_OR_RETURN_NULL(signal.n > 0 && NULL != signal.raw, NULL);
	const size_t nsample = signal.end - signal.start;
	Mat_rptr sigmat = make_mat(1, nsample);
	ASSERT_OR_RETURN_NULL(NULL != sigmat, NULL);

	const size_t offset = signal.start;
	for( size_t i=0 ; i < nsample ; i++){
		// Copy with stride 4 because of required padding for matrix
		sigmat->data.f[i * 4] = signal.raw[i + offset];
	}
	return sigmat;
}



Mat_rptr nanonet_raw_posterior(const raw_table signal, float min_prob, bool return_log){
	assert(min_prob >= 0.0 && min_prob <= 1.0);
	ASSERT_OR_RETURN_NULL(signal.n > 0 && NULL != signal.raw, NULL);

	Mat_rptr raw_mat = mat_raw(signal);
	Mat_rptr conv = Convolution(raw_mat, conv_raw_W, conv_raw_b, conv_raw_stride, NULL);
	raw_mat = free_mat(raw_mat);
	//  First GRU layer
	Mat_rptr gruF = gru_forward(conv, gruF1_raw_iW, gruF1_raw_sW, gruF1_raw_sW2, gruF1_raw_b, NULL);
	Mat_rptr gruB = gru_backward(conv, gruB1_raw_iW, gruB1_raw_sW, gruB1_raw_sW2, gruB1_raw_b, NULL);
	conv = free_mat(conv);
	//  Combine with feed forward layer
	Mat_rptr gruFF = feedforward2_tanh(gruF, gruB, FF1_raw_Wf, FF1_raw_Wb, FF1_raw_b, NULL);
	//  Second GRU layer
	gruF = gru_forward(gruFF, gruF2_raw_iW, gruF2_raw_sW, gruF2_raw_sW2, gruF2_raw_b, gruF);
	gruB = gru_backward(gruFF, gruB2_raw_iW, gruB2_raw_sW, gruB2_raw_sW2, gruB2_raw_b, gruB);
	//  Combine with feed forward layer
	gruFF = feedforward2_tanh(gruF, gruB, FF2_raw_Wf, FF2_raw_Wb, FF2_raw_b, gruFF);
	gruF = free_mat(gruF);
	gruB = free_mat(gruB);


        Mat_rptr post = softmax(gruFF, FF3_raw_W, FF3_raw_b, NULL);
        gruFF = free_mat(gruFF);
	ASSERT_OR_RETURN_NULL(NULL != post, NULL);

	if(return_log){
		const int nev = post->nc;
		const int nstate = post->nr;
		const __m128 mpv = _mm_set1_ps(min_prob / nstate);
		const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
		for(int i=0 ; i < nev ; i++){
			const size_t offset = i * post->nrq;
			for(int r=0 ; r < post->nrq ; r++){
				post->data.v[offset + r] = LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
			}
		}
	}

	return post;
}
