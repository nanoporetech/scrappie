#include "layers.h"
#include "nanonet_lstm_events.h"
#include "nanonet_lstm_raw.h"
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


Mat_rptr nanonet_raw_posterior(const event_table events, float min_prob, bool return_log){
	return NULL;
}
