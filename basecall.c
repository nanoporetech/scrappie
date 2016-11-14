#include <assert.h>
#include <math.h>
#include <openblas/cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include "features.h"
#include "layers.h"
#include "read_events.h"
#include "util.h"

#include "model.h"

const int NOUT = 5;


Mat * calculate_post(char * filename, int analysis){
	event_table et = read_detected_events(filename, analysis);

	//  Make features
	Mat * features = make_features(et, true);
	Mat * feature3 = window(features, 3);

	Mat * vgruF1_iW = mat_from_array(gruF1_iW, 12, 288);
	Mat * vgruF1_sW = mat_from_array(gruF1_sW, 96, 192);
	Mat * vgruF1_sW2 = mat_from_array(gruF1_sW2, 96, 96);
	Mat * vgruF1_b = mat_from_array(gruF1_b, 288, 1);
	Mat * gruF = gru_forward(feature3, vgruF1_iW, vgruF1_sW, vgruF1_sW2, vgruF1_b, NULL);

	Mat * vgruB1_iW = mat_from_array(gruB1_iW, 12, 288);
	Mat * vgruB1_sW = mat_from_array(gruB1_sW, 96, 192);
	Mat * vgruB1_sW2 = mat_from_array(gruB1_sW2, 96, 96);
	Mat * vgruB1_b = mat_from_array(gruB1_b, 288, 1);
	Mat * gruB = gru_backward(feature3, vgruB1_iW, vgruB1_sW, vgruB1_sW2, vgruB1_b, NULL);

	//  Combine GRU output
	Mat * vFF1_fW = mat_from_array(FF1_Wf, 96, 128);
	Mat * vFF1_bW = mat_from_array(FF1_Wb, 96, 128);
	Mat * vFF1_b = mat_from_array(FF1_b, 128, 1);
	Mat * gruFF = feedforward2_tanh(gruF, gruB, vFF1_fW, vFF1_bW, vFF1_b, NULL);


	Mat * vgruF2_iW = mat_from_array(gruF2_iW, 128, 288);
	Mat * vgruF2_sW = mat_from_array(gruF2_sW, 96, 192);
	Mat * vgruF2_sW2 = mat_from_array(gruF2_sW2, 96, 96);
	Mat * vgruF2_b = mat_from_array(gruF2_b, 288, 1);
	gru_forward(gruFF, vgruF2_iW, vgruF2_sW, vgruF2_sW2, vgruF2_b, gruF);

	Mat * vgruB2_iW = mat_from_array(gruB2_iW, 128, 288);
	Mat * vgruB2_sW = mat_from_array(gruB2_sW, 96, 192);
	Mat * vgruB2_sW2 = mat_from_array(gruB2_sW2, 96, 96);
	Mat * vgruB2_b = mat_from_array(gruB2_b, 288, 1);
	gru_backward(gruFF, vgruB2_iW, vgruB2_sW, vgruB2_sW2, vgruB2_b, gruB);

	// Combine GRU output
	Mat * vFF2_fW = mat_from_array(FF2_Wf, 96, 128);
	Mat * vFF2_bW = mat_from_array(FF2_Wb, 96, 128);
	Mat * vFF2_b = mat_from_array(FF2_b, 128, 1);
	feedforward2_tanh(gruF, gruB, vFF2_fW, vFF2_bW, vFF2_b, gruFF);



	Mat * vFF3_W = mat_from_array(FF3_W, 128, 1025);
	Mat * vFF3_b = mat_from_array(FF3_b, 1025, 1);
	Mat * post = softmax(gruFF, vFF3_W, vFF3_b, NULL);

	free_mat(vFF3_b);
	free_mat(vFF3_W);
	free_mat(vFF2_b);
	free_mat(vFF2_bW);
	free_mat(vFF2_fW);
	free_mat(vgruB2_b);
	free_mat(vgruB2_sW2);
	free_mat(vgruB2_sW);
	free_mat(vgruF2_iW);
	free_mat(vgruF2_b);
	free_mat(vgruF2_sW2);
	free_mat(vgruF2_sW);
	free_mat(gruFF);
	free_mat(vFF1_b);
	free_mat(vFF1_bW);
	free_mat(vFF1_fW);
	free_mat(gruB);
	free_mat(vgruB1_b);
	free_mat(vgruB1_sW2);
	free_mat(vgruB1_sW);
	free_mat(gruF);
	free_mat(vgruF1_iW);
	free_mat(vgruF1_b);
	free_mat(vgruF1_sW2);
	free_mat(vgruF1_sW);
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
