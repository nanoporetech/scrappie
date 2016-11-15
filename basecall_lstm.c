#include <assert.h>
#include <math.h>
#include <openblas/cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include "features.h"
#include "layers.h"
#include "read_events.h"
#include "util.h"

#include "lstm_model.h"

const int NOUT = 5;


Mat * calculate_post(char * filename, int analysis){
	event_table et = read_detected_events(filename, analysis);

	//  Make features
	Mat * features = make_features(et, true);
	Mat * feature3 = window(features, 3);

	Mat * vlstmF1_iW = mat_from_array(lstmF1_iW, 12, 288);
	Mat * vlstmF1_sW = mat_from_array(lstmF1_sW, 96, 192);
	Mat * vlstmF1_sW2 = mat_from_array(lstmF1_sW2, 96, 96);
	Mat * vlstmF1_b = mat_from_array(lstmF1_b, 288, 1);
	Mat * lstmF = lstm_forward(feature3, vlstmF1_iW, vlstmF1_sW, vlstmF1_b, NULL);

	Mat * vlstmB1_iW = mat_from_array(lstmB1_iW, 12, 288);
	Mat * vlstmB1_sW = mat_from_array(lstmB1_sW, 96, 192);
	Mat * vlstmB1_sW2 = mat_from_array(lstmB1_sW2, 96, 96);
	Mat * vlstmB1_b = mat_from_array(lstmB1_b, 288, 1);
	Mat * lstmB = lstm_backward(feature3, vlstmB1_iW, vlstmB1_sW, vlstmB1_b, NULL);

	//  Combine LSTM output
	Mat * vFF1_fW = mat_from_array(FF1_Wf, 96, 128);
	Mat * vFF1_bW = mat_from_array(FF1_Wb, 96, 128);
	Mat * vFF1_b = mat_from_array(FF1_b, 128, 1);
	Mat * lstmFF = feedforward2_tanh(lstmF, lstmB, vFF1_fW, vFF1_bW, vFF1_b, NULL);


	Mat * vlstmF2_iW = mat_from_array(lstmF2_iW, 128, 288);
	Mat * vlstmF2_sW = mat_from_array(lstmF2_sW, 96, 192);
	Mat * vlstmF2_sW2 = mat_from_array(lstmF2_sW2, 96, 96);
	Mat * vlstmF2_b = mat_from_array(lstmF2_b, 288, 1);
	lstm_forward(lstmFF, vlstmF2_iW, vlstmF2_sW, vlstmF2_sW2, vlstmF2_b, lstmF);

	Mat * vlstmB2_iW = mat_from_array(lstmB2_iW, 128, 288);
	Mat * vlstmB2_sW = mat_from_array(lstmB2_sW, 96, 192);
	Mat * vlstmB2_sW2 = mat_from_array(lstmB2_sW2, 96, 96);
	Mat * vlstmB2_b = mat_from_array(lstmB2_b, 288, 1);
	lstm_backward(lstmFF, vlstmB2_iW, vlstmB2_sW, vlstmB2_sW2, vlstmB2_b, lstmB);

	// Combine LSTM output
	Mat * vFF2_fW = mat_from_array(FF2_Wf, 96, 128);
	Mat * vFF2_bW = mat_from_array(FF2_Wb, 96, 128);
	Mat * vFF2_b = mat_from_array(FF2_b, 128, 1);
	feedforward2_tanh(lstmF, lstmB, vFF2_fW, vFF2_bW, vFF2_b, lstmFF);



	Mat * vFF3_W = mat_from_array(FF3_W, 128, 1025);
	Mat * vFF3_b = mat_from_array(FF3_b, 1025, 1);
	Mat * post = softmax(lstmFF, vFF3_W, vFF3_b, NULL);


	free_mat(vFF3_b);
	free_mat(vFF3_W);
	free_mat(vFF2_b);
	free_mat(vFF2_bW);
	free_mat(vFF2_fW);
	free_mat(vlstmB2_b);
	free_mat(vlstmB2_sW2);
	free_mat(vlstmB2_sW);
	free_mat(vlstmF2_iW);
	free_mat(vlstmF2_b);
	free_mat(vlstmF2_sW2);
	free_mat(vlstmF2_sW);
	free_mat(lstmFF);
	free_mat(vFF1_b);
	free_mat(vFF1_bW);
	free_mat(vFF1_fW);
	free_mat(lstmB);
	free_mat(vlstmB1_b);
	free_mat(vlstmB1_sW2);
	free_mat(vlstmB1_sW);
	free_mat(lstmF);
	free_mat(vlstmF1_iW);
	free_mat(vlstmF1_b);
	free_mat(vlstmF1_sW2);
	free_mat(vlstmF1_sW);
	free_mat(feature3);
	free_mat(features);

	return post;
}

int main(int argc, char * argv[]){
	openblas_set_num_threads(1);
	assert(argc > 1);

	#pragma omp parallel for
	for(int fn=1 ; fn<argc ; fn++){
		Mat * post = calculate_post(argv[fn], 0);
		printf("%s -- %d events\n", argv[fn], post->nc);

		for(int i=2000 ; i<2010 ; i++){
			const int offset = i * post->nrq * 4;
			float sum = 0.0;
			for(int j=0 ; j<post->nr ; j++){
				sum += post->data.f[offset + j];
			}
			printf("%d  %f (%f %f %f %f)\n", i, sum,
				post->data.f[offset],
				post->data.f[offset + 1],
				post->data.f[offset + 2],
				post->data.f[offset + 3]);
		}
		free_mat(post);
	}

	return EXIT_SUCCESS;
}
