#include <string.h>
#include "layers.h"
#include "nanonet_events.h"
#include "nanonet_raw.h"
#include "networks.h"
#include "nnfeatures.h"
#include "scrappie_assert.h"

scrappie_matrix nanonet_posterior(const event_table events, float min_prob,
                                  bool return_log) {
    assert(min_prob >= 0.0 && min_prob <= 1.0);
    RETURN_NULL_IF(events.n == 0, NULL);
    RETURN_NULL_IF(NULL == events.event, NULL);

    const int WINLEN = 3;

    //  Make features
    scrappie_matrix features = nanonet_features_from_events(events, true);
    scrappie_matrix feature3 = window(features, WINLEN, 1);
    features = free_scrappie_matrix(features);

    // Initial transformation of input for LSTM layer
    scrappie_matrix lstmXf =
        feedforward_linear(feature3, lstmF1_iW, lstmF1_b, NULL);
    scrappie_matrix lstmXb =
        feedforward_linear(feature3, lstmB1_iW, lstmB1_b, NULL);
    feature3 = free_scrappie_matrix(feature3);
    scrappie_matrix lstmF = lstm_forward(lstmXf, lstmF1_sW, lstmF1_p, NULL);
    scrappie_matrix lstmB = lstm_backward(lstmXb, lstmB1_sW, lstmB1_p, NULL);

    //  Combine LSTM output
    scrappie_matrix lstmFF =
        feedforward2_tanh(lstmF, lstmB, FF1_Wf, FF1_Wb, FF1_b, NULL);

    lstmXf = feedforward_linear(lstmFF, lstmF2_iW, lstmF2_b, lstmXf);
    lstmXb = feedforward_linear(lstmFF, lstmB2_iW, lstmB2_b, lstmXb);
    lstmF = lstm_forward(lstmXf, lstmF2_sW, lstmF2_p, lstmF);
    lstmXf = free_scrappie_matrix(lstmXf);
    lstmB = lstm_backward(lstmXb, lstmB2_sW, lstmB2_p, lstmB);
    lstmXb = free_scrappie_matrix(lstmXb);

    // Combine LSTM output
    lstmFF = feedforward2_tanh(lstmF, lstmB, FF2_Wf, FF2_Wb, FF2_b, lstmFF);
    lstmF = free_scrappie_matrix(lstmF);
    lstmB = free_scrappie_matrix(lstmB);

    scrappie_matrix post = softmax(lstmFF, FF3_W, FF3_b, NULL);
    lstmFF = free_scrappie_matrix(lstmFF);
    RETURN_NULL_IF(NULL == post, NULL);

    if (return_log) {
        const int nev = post->nc;
        const int nstate = post->nr;
        const __m128 mpv = _mm_set1_ps(min_prob / nstate);
        const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
        for (int i = 0; i < nev; i++) {
            const size_t offset = i * post->nrq;
            for (int r = 0; r < post->nrq; r++) {
                post->data.v[offset + r] =
                    LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
            }
        }
    }

    return post;
}

scrappie_matrix nanonet_raw_posterior(const raw_table signal, float min_prob,
                                      bool return_log) {
    assert(min_prob >= 0.0 && min_prob <= 1.0);
    RETURN_NULL_IF(signal.n == 0, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv =
        Convolution(raw_mat, conv_raw_W, conv_raw_b, conv_raw_stride, NULL);
    tanh_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);
    //  First GRU layer
    scrappie_matrix gruF =
        gru_forward(conv, gruF1_raw_iW, gruF1_raw_sW, gruF1_raw_sW2,
                    gruF1_raw_b, NULL);
    scrappie_matrix gruB =
        gru_backward(conv, gruB1_raw_iW, gruB1_raw_sW, gruB1_raw_sW2,
                     gruB1_raw_b, NULL);
    conv = free_scrappie_matrix(conv);
    //  Combine with feed forward layer
    scrappie_matrix gruFF =
        feedforward2_tanh(gruF, gruB, FF1_raw_Wf, FF1_raw_Wb, FF1_raw_b, NULL);
    //  Second GRU layer
    gruF =
        gru_forward(gruFF, gruF2_raw_iW, gruF2_raw_sW, gruF2_raw_sW2,
                    gruF2_raw_b, gruF);
    gruB =
        gru_backward(gruFF, gruB2_raw_iW, gruB2_raw_sW, gruB2_raw_sW2,
                     gruB2_raw_b, gruB);
    //  Combine with feed forward layer
    gruFF =
        feedforward2_tanh(gruF, gruB, FF2_raw_Wf, FF2_raw_Wb, FF2_raw_b, gruFF);
    gruF = free_scrappie_matrix(gruF);
    gruB = free_scrappie_matrix(gruB);

    scrappie_matrix post = softmax(gruFF, FF3_raw_W, FF3_raw_b, NULL);
    gruFF = free_scrappie_matrix(gruFF);
    RETURN_NULL_IF(NULL == post, NULL);

    if (return_log) {
        const int nblock = post->nc;
        const int nstate = post->nr;
        const __m128 mpv = _mm_set1_ps(min_prob / nstate);
        const __m128 mpvm1 = _mm_set1_ps(1.0f - min_prob);
        for (int i = 0; i < nblock; i++) {
            const size_t offset = i * post->nrq;
            for (int r = 0; r < post->nrq; r++) {
                post->data.v[offset + r] =
                    LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
            }
        }
    }

    return post;
}
