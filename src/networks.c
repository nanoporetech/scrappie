#include "layers.h"
#include "models/nanonet_events.h"
#include "models/raw_r94.h"
#include "models/rgrgr_r94.h"
#include "models/rgrgr_r941.h"
#include "models/rgrgr_r10.h"
#include "models/rnnrf_r94.h"
#include "networks.h"
#include "nnfeatures.h"
#include "scrappie_stdlib.h"
#include "util.h"

#include "models/squiggle_r94.h"
#include "models/squiggle_r94_rna.h"
#include "models/squiggle_r10.h"

enum raw_model_type get_raw_model(const char * modelstr){
    if(0 == strcmp(modelstr, "raw_r94")){
        return SCRAPPIE_MODEL_RAW;
    }
    if(0 == strcmp(modelstr, "rgrgr_r94")){
        return SCRAPPIE_MODEL_RGRGR_R9_4;
    }
    if(0 == strcmp(modelstr, "rgrgr_r941")){
        return SCRAPPIE_MODEL_RGRGR_R9_4_1;
    }
    if(0 == strcmp(modelstr, "rgrgr_r10")){
        return SCRAPPIE_MODEL_RGRGR_R10;
    }
    if(0 == strcmp(modelstr, "rnnrf_r94")){
        return SCRAPPIE_MODEL_RNNRF_R9_4;
    }
    return SCRAPPIE_MODEL_INVALID;
}

enum squiggle_model_type get_squiggle_model(const char * squigmodelstr){
    if(0 == strcmp(squigmodelstr, "squiggle_r94")){
        return SCRAPPIE_SQUIGGLE_MODEL_R9_4;
    }
    if(0 == strcmp(squigmodelstr, "squiggle_r94_rna")){
        return SCRAPPIE_SQUIGGLE_MODEL_R9_4_RNA;
    }
    if(0 == strcmp(squigmodelstr, "squiggle_r10")){
        return SCRAPPIE_SQUIGGLE_MODEL_R10;
    }
    return SCRAPPIE_SQUIGGLE_MODEL_INVALID;
}

const char * raw_model_string(const enum raw_model_type model){
    switch(model){
    case SCRAPPIE_MODEL_RAW:
        return "raw_r94";
    case SCRAPPIE_MODEL_RGRGR_R9_4:
        return "rgrgr_r94";
    case SCRAPPIE_MODEL_RGRGR_R9_4_1:
        return "rgrgr_r941";
    case SCRAPPIE_MODEL_RGRGR_R10:
        return "rgrgr_r10";
    case SCRAPPIE_MODEL_RNNRF_R9_4:
        return "rnnrf_r94";
    case SCRAPPIE_MODEL_INVALID:
        errx(EXIT_FAILURE, "Invalid scrappie model %s:%d", __FILE__, __LINE__);
    default:
        errx(EXIT_FAILURE, "Scrappie enum failure -- report bug\n");
    }

    return NULL;
}

const char * squiggle_model_string(const enum squiggle_model_type squiggle_model){
    switch(squiggle_model){
    case SCRAPPIE_SQUIGGLE_MODEL_R9_4:
        return "squiggle_r94";
    case SCRAPPIE_SQUIGGLE_MODEL_R9_4_RNA:
        return "squiggle_r94_rna";
    case SCRAPPIE_SQUIGGLE_MODEL_R10:
        return "squiggle_r10";
    case SCRAPPIE_SQUIGGLE_MODEL_INVALID:
        errx(EXIT_FAILURE, "Invalid scrappie squiggle model %s:%d", __FILE__, __LINE__);
    default:
        errx(EXIT_FAILURE, "Scrappie enum failure -- report bug\n");
    }

    return NULL;
}

int get_raw_model_stride(const enum raw_model_type model){
    switch(model){
    case SCRAPPIE_MODEL_RAW:
        return conv_raw_stride;
    case SCRAPPIE_MODEL_RGRGR_R9_4:
        return conv_rgrgr_r94_stride;
    case SCRAPPIE_MODEL_RGRGR_R9_4_1:
        return conv_rgrgr_r941_stride;
    case SCRAPPIE_MODEL_RGRGR_R10:
        return conv_rgrgr_r10_stride;
    case SCRAPPIE_MODEL_RNNRF_R9_4:
        return conv_rnnrf_r94_stride;
    case SCRAPPIE_MODEL_INVALID:
        errx(EXIT_FAILURE, "Invalid scrappie model %s:%d", __FILE__, __LINE__);
    default:
        errx(EXIT_FAILURE, "Scrappie enum failure -- report bug\n");
    }

    return -1;
}

posterior_function_ptr get_posterior_function(const enum raw_model_type model){
    switch(model){
    case SCRAPPIE_MODEL_RAW:
        return nanonet_raw_posterior;
    case SCRAPPIE_MODEL_RGRGR_R9_4:
        return nanonet_rgrgr_r94_posterior;
    case SCRAPPIE_MODEL_RGRGR_R9_4_1:
        return nanonet_rgrgr_r941_posterior;
    case SCRAPPIE_MODEL_RGRGR_R10:
        return nanonet_rgrgr_r10_posterior;
    case SCRAPPIE_MODEL_RNNRF_R9_4:
        return nanonet_rnnrf_r94_transitions;
    case SCRAPPIE_MODEL_INVALID:
        errx(EXIT_FAILURE, "Invalid scrappie model %s:%d", __FILE__, __LINE__);
    default:
        errx(EXIT_FAILURE, "Scrappie enum failure -- report bug\n");
    }

    return NULL;
}

squiggle_function_ptr get_squiggle_function(const enum squiggle_model_type squiggle_model){
    switch(squiggle_model){
    case SCRAPPIE_SQUIGGLE_MODEL_R9_4:
        return squiggle_r94;
    case SCRAPPIE_SQUIGGLE_MODEL_R9_4_RNA:
        return squiggle_r94_rna;
    case SCRAPPIE_SQUIGGLE_MODEL_R10:
        return squiggle_r10;
    case SCRAPPIE_SQUIGGLE_MODEL_INVALID:
        errx(EXIT_FAILURE, "Invalid scrappie squiggle model %s:%d", __FILE__, __LINE__);
    default:
        errx(EXIT_FAILURE, "Scrappie enum failure -- report bug\n");
    }

    return NULL;
}

scrappie_matrix nanonet_posterior(const event_table events, float min_prob,
                                  float tempW, float tempb, bool return_log) {
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == events.n, NULL);
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

    scrappie_matrix post = softmax_with_temperature(lstmFF, FF3_W, FF3_b, tempW, tempb, NULL);
    lstmFF = free_scrappie_matrix(lstmFF);
    RETURN_NULL_IF(NULL == post, NULL);

    if (return_log) {
        robustlog_activation_inplace(post, min_prob);
    }

    return post;
}

scrappie_matrix nanonet_raw_posterior(const raw_table signal, float min_prob,
                                      float tempW, float tempb, bool return_log) {
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == signal.n, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv = convolution(raw_mat, conv_raw_W, conv_raw_b, conv_raw_stride, NULL);
    tanh_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);

    //  First GRU layer
    scrappie_matrix gruF1in = feedforward_linear(conv, gruF1_raw_iW, gruF1_raw_b, NULL);
    scrappie_matrix gruB1in = feedforward_linear(conv, gruB1_raw_iW, gruB1_raw_b, NULL);
    conv = free_scrappie_matrix(conv);

    scrappie_matrix gruF = gru_forward(gruF1in, gruF1_raw_sW, gruF1_raw_sW2, NULL);
    gruF1in = free_scrappie_matrix(gruF1in);
    scrappie_matrix gruB = gru_backward(gruB1in, gruB1_raw_sW, gruB1_raw_sW2, NULL);
    gruB1in = free_scrappie_matrix(gruB1in);

    //  Combine with feed forward layer
    scrappie_matrix gruFF =
        feedforward2_tanh(gruF, gruB, FF1_raw_Wf, FF1_raw_Wb, FF1_raw_b, NULL);

    //  Second GRU layer
    scrappie_matrix gruF2in = feedforward_linear(gruFF, gruF2_raw_iW, gruF2_raw_b, NULL);
    scrappie_matrix gruB2in = feedforward_linear(gruFF, gruB2_raw_iW, gruB2_raw_b, NULL);
    gruFF = free_scrappie_matrix(gruFF);
    gruF = gru_forward(gruF2in, gruF2_raw_sW, gruF2_raw_sW2, gruF);
    gruF2in = free_scrappie_matrix(gruF2in);
    gruB = gru_backward(gruB2in, gruB2_raw_sW, gruB2_raw_sW2, gruB);
    gruB2in = free_scrappie_matrix(gruB2in);


    //  Combine with feed forward layer
    gruFF =
        feedforward2_tanh(gruF, gruB, FF2_raw_Wf, FF2_raw_Wb, FF2_raw_b, gruFF);
    gruF = free_scrappie_matrix(gruF);
    gruB = free_scrappie_matrix(gruB);

    scrappie_matrix post = softmax_with_temperature(gruFF, FF3_raw_W, FF3_raw_b, tempW, tempb, NULL);
    gruFF = free_scrappie_matrix(gruFF);
    RETURN_NULL_IF(NULL == post, NULL);

    if (return_log) {
        robustlog_activation_inplace(post, min_prob);
    }

    return post;
}


scrappie_matrix nanonet_rgrgr_r94_posterior(const raw_table signal, float min_prob,
                                            float tempW, float tempb, bool return_log) {
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == signal.n, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv =
        convolution(raw_mat, conv_rgrgr_r94_W, conv_rgrgr_r94_b, conv_rgrgr_r94_stride, NULL);
    elu_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);
    //  First GRU layer
    scrappie_matrix gruB1in = feedforward_linear(conv, gruB1_rgrgr_r94_iW, gruB1_rgrgr_r94_b, NULL);
    conv = free_scrappie_matrix(conv);
    scrappie_matrix gruB1 = gru_backward(gruB1in, gruB1_rgrgr_r94_sW, gruB1_rgrgr_r94_sW2, NULL);
    gruB1in = free_scrappie_matrix(gruB1in);
    //  Second GRU layer
    scrappie_matrix gruF2in = feedforward_linear(gruB1, gruF2_rgrgr_r94_iW, gruF2_rgrgr_r94_b, NULL);
    gruB1 = free_scrappie_matrix(gruB1);
    scrappie_matrix gruF2 = gru_forward(gruF2in, gruF2_rgrgr_r94_sW, gruF2_rgrgr_r94_sW2, NULL);
    gruF2in = free_scrappie_matrix(gruF2in);
    //  Third GRU layer
    scrappie_matrix gruB3in = feedforward_linear(gruF2, gruB3_rgrgr_r94_iW, gruB3_rgrgr_r94_b, NULL);
    gruF2 = free_scrappie_matrix(gruF2);
    scrappie_matrix gruB3 = gru_backward(gruB3in, gruB3_rgrgr_r94_sW, gruB3_rgrgr_r94_sW2, NULL);
    gruB3in = free_scrappie_matrix(gruB3in);
    //  Fourth GRU layer
    scrappie_matrix gruF4in = feedforward_linear(gruB3, gruF4_rgrgr_r94_iW, gruF4_rgrgr_r94_b, NULL);
    gruB3 = free_scrappie_matrix(gruB3);
    scrappie_matrix gruF4 = gru_forward(gruF4in, gruF4_rgrgr_r94_sW, gruF4_rgrgr_r94_sW2, NULL);
    gruF4in = free_scrappie_matrix(gruF4in);
    //  Fifth GRU layer
    scrappie_matrix gruB5in = feedforward_linear(gruF4, gruB5_rgrgr_r94_iW, gruB5_rgrgr_r94_b, NULL);
    gruF4 = free_scrappie_matrix(gruF4);
    scrappie_matrix gruB5 = gru_backward(gruB5in, gruB5_rgrgr_r94_sW, gruB5_rgrgr_r94_sW2, NULL);
    gruB5in = free_scrappie_matrix(gruB5in);

    scrappie_matrix post = softmax_with_temperature(gruB5, FF_rgrgr_r94_W, FF_rgrgr_r94_b, tempW, tempb, NULL);
    gruB5 = free_scrappie_matrix(gruB5);

    if (return_log) {
        robustlog_activation_inplace(post, min_prob);
    }

    return post;
}


scrappie_matrix nanonet_rgrgr_r941_posterior(const raw_table signal, float min_prob,
                                            float tempW, float tempb, bool return_log) {
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == signal.n, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv =
        convolution(raw_mat, conv_rgrgr_r941_W, conv_rgrgr_r941_b, conv_rgrgr_r941_stride, NULL);
    elu_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);
    //  First GRU layer
    scrappie_matrix gruB1in = feedforward_linear(conv, gruB1_rgrgr_r941_iW, gruB1_rgrgr_r941_b, NULL);
    conv = free_scrappie_matrix(conv);
    scrappie_matrix gruB1 = gru_backward(gruB1in, gruB1_rgrgr_r941_sW, gruB1_rgrgr_r941_sW2, NULL);
    gruB1in = free_scrappie_matrix(gruB1in);
    //  Second GRU layer
    scrappie_matrix gruF2in = feedforward_linear(gruB1, gruF2_rgrgr_r941_iW, gruF2_rgrgr_r941_b, NULL);
    gruB1 = free_scrappie_matrix(gruB1);
    scrappie_matrix gruF2 = gru_forward(gruF2in, gruF2_rgrgr_r941_sW, gruF2_rgrgr_r941_sW2, NULL);
    gruF2in = free_scrappie_matrix(gruF2in);
    //  Third GRU layer
    scrappie_matrix gruB3in = feedforward_linear(gruF2, gruB3_rgrgr_r941_iW, gruB3_rgrgr_r941_b, NULL);
    gruF2 = free_scrappie_matrix(gruF2);
    scrappie_matrix gruB3 = gru_backward(gruB3in, gruB3_rgrgr_r941_sW, gruB3_rgrgr_r941_sW2, NULL);
    gruB3in = free_scrappie_matrix(gruB3in);
    //  Fourth GRU layer
    scrappie_matrix gruF4in = feedforward_linear(gruB3, gruF4_rgrgr_r941_iW, gruF4_rgrgr_r941_b, NULL);
    gruB3 = free_scrappie_matrix(gruB3);
    scrappie_matrix gruF4 = gru_forward(gruF4in, gruF4_rgrgr_r941_sW, gruF4_rgrgr_r941_sW2, NULL);
    gruF4in = free_scrappie_matrix(gruF4in);
    //  Fifth GRU layer
    scrappie_matrix gruB5in = feedforward_linear(gruF4, gruB5_rgrgr_r941_iW, gruB5_rgrgr_r941_b, NULL);
    gruF4 = free_scrappie_matrix(gruF4);
    scrappie_matrix gruB5 = gru_backward(gruB5in, gruB5_rgrgr_r941_sW, gruB5_rgrgr_r941_sW2, NULL);
    gruB5in = free_scrappie_matrix(gruB5in);

    scrappie_matrix post = softmax_with_temperature(gruB5, FF_rgrgr_r941_W, FF_rgrgr_r941_b, tempW, tempb, NULL);
    gruB5 = free_scrappie_matrix(gruB5);

    if (return_log) {
        robustlog_activation_inplace(post, min_prob);
    }

    return post;
}


scrappie_matrix nanonet_rgrgr_r10_posterior(const raw_table signal, float min_prob,
		                                    float tempW, float tempb, bool return_log) {
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == signal.n, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv =
        convolution(raw_mat, conv_rgrgr_r10_W, conv_rgrgr_r10_b, conv_rgrgr_r10_stride, NULL);
    tanh_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);
    //  First GRU layer
    scrappie_matrix gruB1in = feedforward_linear(conv, gruB1_rgrgr_r10_iW, gruB1_rgrgr_r10_b, NULL);
    conv = free_scrappie_matrix(conv);
    scrappie_matrix gruB1 = gru_backward(gruB1in, gruB1_rgrgr_r10_sW, gruB1_rgrgr_r10_sW2, NULL);
    gruB1in = free_scrappie_matrix(gruB1in);
    //  Second GRU layer
    scrappie_matrix gruF2in = feedforward_linear(gruB1, gruF2_rgrgr_r10_iW, gruF2_rgrgr_r10_b, NULL);
    gruB1 = free_scrappie_matrix(gruB1);
    scrappie_matrix gruF2 = gru_forward(gruF2in, gruF2_rgrgr_r10_sW, gruF2_rgrgr_r10_sW2, NULL);
    gruF2in = free_scrappie_matrix(gruF2in);
    //  Third GRU layer
    scrappie_matrix gruB3in = feedforward_linear(gruF2, gruB3_rgrgr_r10_iW, gruB3_rgrgr_r10_b, NULL);
    gruF2 = free_scrappie_matrix(gruF2);
    scrappie_matrix gruB3 = gru_backward(gruB3in, gruB3_rgrgr_r10_sW, gruB3_rgrgr_r10_sW2, NULL);
    gruB3in = free_scrappie_matrix(gruB3in);
    //  Fourth GRU layer
    scrappie_matrix gruF4in = feedforward_linear(gruB3, gruF4_rgrgr_r10_iW, gruF4_rgrgr_r10_b, NULL);
    gruB3 = free_scrappie_matrix(gruB3);
    scrappie_matrix gruF4 = gru_forward(gruF4in, gruF4_rgrgr_r10_sW, gruF4_rgrgr_r10_sW2, NULL);
    gruF4in = free_scrappie_matrix(gruF4in);
    //  Fifth GRU layer
    scrappie_matrix gruB5in = feedforward_linear(gruF4, gruB5_rgrgr_r10_iW, gruB5_rgrgr_r10_b, NULL);
    gruF4 = free_scrappie_matrix(gruF4);
    scrappie_matrix gruB5 = gru_backward(gruB5in, gruB5_rgrgr_r10_sW, gruB5_rgrgr_r10_sW2, NULL);
    gruB5in = free_scrappie_matrix(gruB5in);

    scrappie_matrix post = softmax_with_temperature(gruB5, FF_rgrgr_r10_W, FF_rgrgr_r10_b, tempW, tempb, NULL);
    gruB5 = free_scrappie_matrix(gruB5);

    if (return_log) {
        robustlog_activation_inplace(post, min_prob);
    }

    return post;
}


scrappie_matrix squiggle_r94(int const * sequence, size_t n, bool transform_units){
    RETURN_NULL_IF(NULL == sequence, NULL);

    scrappie_matrix seq_embedding = embedding(sequence, n, embed_squiggle_r94_W, NULL);
    scrappie_matrix conv1 = convolution(seq_embedding, conv1_squiggle_r94_W, conv1_squiggle_r94_b,
                                        conv1_squiggle_r94_stride, NULL);
    seq_embedding = free_scrappie_matrix(seq_embedding);
    tanh_activation_inplace(conv1);

    // Convolution 2, wrapped in residual layer
    scrappie_matrix conv2 = convolution(conv1, conv2_squiggle_r94_W, conv2_squiggle_r94_b,
                                        conv2_squiggle_r94_stride, NULL);
    tanh_activation_inplace(conv2);
    residual_inplace(conv1, conv2);
    conv1 = free_scrappie_matrix(conv1);

    // Convolution 3, wrapped in residual layer
    scrappie_matrix conv3 = convolution(conv2, conv3_squiggle_r94_W, conv3_squiggle_r94_b,
                                        conv3_squiggle_r94_stride, NULL);
    tanh_activation_inplace(conv3);
    residual_inplace(conv2, conv3);
    conv2 = free_scrappie_matrix(conv2);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv4 = convolution(conv3, conv4_squiggle_r94_W, conv4_squiggle_r94_b,
                                        conv4_squiggle_r94_stride, NULL);
    tanh_activation_inplace(conv4);
    residual_inplace(conv3, conv4);
    conv3 = free_scrappie_matrix(conv3);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv5 = convolution(conv4, conv5_squiggle_r94_W, conv5_squiggle_r94_b,
                                        conv5_squiggle_r94_stride, NULL);
    tanh_activation_inplace(conv5);
    residual_inplace(conv4, conv5);
    conv4 = free_scrappie_matrix(conv4);

    scrappie_matrix conv6 = convolution(conv5, conv6_squiggle_r94_W, conv6_squiggle_r94_b,
                                        conv6_squiggle_r94_stride, NULL);
    conv5 = free_scrappie_matrix(conv5);

    RETURN_NULL_IF(NULL == conv6, NULL);

    if(transform_units){
        for(size_t c=0 ; c < conv6->nc ; c++){
            size_t offset = c * conv6->stride;
            //  Convert logsd to sd
            conv6->data.f[offset + 1] = expf(conv6->data.f[offset + 1]);
            //  Convert transformed dwell into expected samples
            conv6->data.f[offset + 2] = expf(-conv6->data.f[offset + 2]);
        }
    }

    return conv6;
}


scrappie_matrix squiggle_r94_rna(int const * sequence, size_t n, bool transform_units){
    RETURN_NULL_IF(NULL == sequence, NULL);

    scrappie_matrix seq_embedding = embedding(sequence, n, embed_squiggle_rna_W, NULL);
    scrappie_matrix conv1 = convolution(seq_embedding, conv1_squiggle_rna_W, conv1_squiggle_rna_b,
                                        conv1_squiggle_rna_stride, NULL);
    seq_embedding = free_scrappie_matrix(seq_embedding);
    tanh_activation_inplace(conv1);

    // Convolution 2, wrapped in residual layer
    scrappie_matrix conv2 = convolution(conv1, conv2_squiggle_rna_W, conv2_squiggle_rna_b,
                                        conv2_squiggle_rna_stride, NULL);
    tanh_activation_inplace(conv2);
    residual_inplace(conv1, conv2);
    conv1 = free_scrappie_matrix(conv1);

    // Convolution 3, wrapped in residual layer
    scrappie_matrix conv3 = convolution(conv2, conv3_squiggle_rna_W, conv3_squiggle_rna_b,
                                        conv3_squiggle_rna_stride, NULL);
    tanh_activation_inplace(conv3);
    residual_inplace(conv2, conv3);
    conv2 = free_scrappie_matrix(conv2);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv4 = convolution(conv3, conv4_squiggle_rna_W, conv4_squiggle_rna_b,
                                        conv4_squiggle_rna_stride, NULL);
    tanh_activation_inplace(conv4);
    residual_inplace(conv3, conv4);
    conv3 = free_scrappie_matrix(conv3);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv5 = convolution(conv4, conv5_squiggle_rna_W, conv5_squiggle_rna_b,
                                        conv5_squiggle_rna_stride, NULL);
    tanh_activation_inplace(conv5);
    residual_inplace(conv4, conv5);
    conv4 = free_scrappie_matrix(conv4);

    scrappie_matrix conv6 = convolution(conv5, conv6_squiggle_rna_W, conv6_squiggle_rna_b,
                                        conv6_squiggle_rna_stride, NULL);
    conv5 = free_scrappie_matrix(conv5);

    RETURN_NULL_IF(NULL == conv6, NULL);

    if(transform_units){
        for(size_t c=0 ; c < conv6->nc ; c++){
            size_t offset = c * conv6->stride;
            //  Convert logsd to sd
            conv6->data.f[offset + 1] = expf(conv6->data.f[offset + 1]);
            //  Convert transformed dwell into expected samples
            conv6->data.f[offset + 2] = expf(-conv6->data.f[offset + 2]);
        }
    }

    return conv6;
}


scrappie_matrix squiggle_r10(int const * sequence, size_t n, bool transform_units){
    RETURN_NULL_IF(NULL == sequence, NULL);

    scrappie_matrix seq_embedding = embedding(sequence, n, embed_squiggle_r10_W, NULL);
    scrappie_matrix conv1 = convolution(seq_embedding, conv1_squiggle_r10_W, conv1_squiggle_r10_b,
                                        conv1_squiggle_r10_stride, NULL);
    seq_embedding = free_scrappie_matrix(seq_embedding);
    tanh_activation_inplace(conv1);

    // Convolution 2, wrapped in residual layer
    scrappie_matrix conv2 = convolution(conv1, conv2_squiggle_r10_W, conv2_squiggle_r10_b,
                                        conv2_squiggle_r10_stride, NULL);
    tanh_activation_inplace(conv2);
    residual_inplace(conv1, conv2);
    conv1 = free_scrappie_matrix(conv1);

    // Convolution 3, wrapped in residual layer
    scrappie_matrix conv3 = convolution(conv2, conv3_squiggle_r10_W, conv3_squiggle_r10_b,
                                        conv3_squiggle_r10_stride, NULL);
    tanh_activation_inplace(conv3);
    residual_inplace(conv2, conv3);
    conv2 = free_scrappie_matrix(conv2);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv4 = convolution(conv3, conv4_squiggle_r10_W, conv4_squiggle_r10_b,
                                        conv4_squiggle_r10_stride, NULL);
    tanh_activation_inplace(conv4);
    residual_inplace(conv3, conv4);
    conv3 = free_scrappie_matrix(conv3);

    // Convolution 4, wrapped in residual layer
    scrappie_matrix conv5 = convolution(conv4, conv5_squiggle_r10_W, conv5_squiggle_r10_b,
                                        conv5_squiggle_r10_stride, NULL);
    tanh_activation_inplace(conv5);
    residual_inplace(conv4, conv5);
    conv4 = free_scrappie_matrix(conv4);

    scrappie_matrix conv6 = convolution(conv5, conv6_squiggle_r10_W, conv6_squiggle_r10_b,
                                        conv6_squiggle_r10_stride, NULL);
    conv5 = free_scrappie_matrix(conv5);

    RETURN_NULL_IF(NULL == conv6, NULL);

    if(transform_units){
        for(size_t c=0 ; c < conv6->nc ; c++){
            size_t offset = c * conv6->stride;
            //  Convert logsd to sd
            conv6->data.f[offset + 1] = expf(conv6->data.f[offset + 1]);
            //  Convert transformed dwell into expected samples
            conv6->data.f[offset + 2] = expf(-conv6->data.f[offset + 2]);
        }
    }

    return conv6;
}

scrappie_matrix nanonet_rnnrf_r94_transitions(const raw_table signal, float min_prob,
                                              float tempW, float tempb,bool return_log) {
    assert(return_log);  // Returning non-log transformed not supported
    assert(min_prob >= 0.0f && min_prob <= 1.0f);
    assert(tempW > 0.0f && tempb > 0.0f);
    RETURN_NULL_IF(0 == signal.n, NULL);
    RETURN_NULL_IF(NULL == signal.raw, NULL);

    scrappie_matrix raw_mat = nanonet_features_from_raw(signal);
    scrappie_matrix conv =
        convolution(raw_mat, conv_rnnrf_r94_W, conv_rnnrf_r94_b, conv_rnnrf_r94_stride, NULL);
    elu_activation_inplace(conv);
    raw_mat = free_scrappie_matrix(raw_mat);
    //  First GRU layer
    scrappie_matrix gruB1in = feedforward_linear(conv, gruB1_rnnrf_r94_iW, gruB1_rnnrf_r94_b, NULL);
    scrappie_matrix gruB1 = gru_backward(gruB1in, gruB1_rnnrf_r94_sW, gruB1_rnnrf_r94_sW2, NULL);
    residual_inplace(conv, gruB1);
    conv = free_scrappie_matrix(conv);
    gruB1in = free_scrappie_matrix(gruB1in);
    //  Second GRU layer
    scrappie_matrix gruF2in = feedforward_linear(gruB1, gruF2_rnnrf_r94_iW, gruF2_rnnrf_r94_b, NULL);
    scrappie_matrix gruF2 = gru_forward(gruF2in, gruF2_rnnrf_r94_sW, gruF2_rnnrf_r94_sW2, NULL);
    residual_inplace(gruB1, gruF2);
    gruB1 = free_scrappie_matrix(gruB1);
    gruF2in = free_scrappie_matrix(gruF2in);
    //  Third GRU layer
    scrappie_matrix gruB3in = feedforward_linear(gruF2, gruB3_rnnrf_r94_iW, gruB3_rnnrf_r94_b, NULL);
    scrappie_matrix gruB3 = gru_backward(gruB3in, gruB3_rnnrf_r94_sW, gruB3_rnnrf_r94_sW2, NULL);
    residual_inplace(gruF2, gruB3);
    gruF2 = free_scrappie_matrix(gruF2);
    gruB3in = free_scrappie_matrix(gruB3in);
    //  Fourth GRU layer
    scrappie_matrix gruF4in = feedforward_linear(gruB3, gruF4_rnnrf_r94_iW, gruF4_rnnrf_r94_b, NULL);
    scrappie_matrix gruF4 = gru_forward(gruF4in, gruF4_rnnrf_r94_sW, gruF4_rnnrf_r94_sW2, NULL);
    residual_inplace(gruB3, gruF4);
    gruB3 = free_scrappie_matrix(gruB3);
    gruF4in = free_scrappie_matrix(gruF4in);
    //  Fifth GRU layer
    scrappie_matrix gruB5in = feedforward_linear(gruF4, gruB5_rnnrf_r94_iW, gruB5_rnnrf_r94_b, NULL);
    scrappie_matrix gruB5 = gru_backward(gruB5in, gruB5_rnnrf_r94_sW, gruB5_rnnrf_r94_sW2, NULL);
    residual_inplace(gruF4, gruB5);
    gruF4 = free_scrappie_matrix(gruF4);
    gruB5in = free_scrappie_matrix(gruB5in);

    scrappie_matrix trans = globalnorm(gruB5, FF_rnnrf_r94_W, FF_rnnrf_r94_b, NULL);
    gruB5 = free_scrappie_matrix(gruB5);

    return trans;
}
