#include <assert.h>
#include <math.h>
#include <openblas/cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include "features.h"
#include "layers.h"
#include "read_events.h"
#include "util.h"

#include "gru_model.h"

const int NOUT = 5;


Mat * calculate_post(char * filename, int analysis){
	event_table et = read_detected_events(filename, analysis);

	//  Make features
	Mat * features = make_features(et, true);
	Mat * feature3 = window(features, 3);

	Mat * gruF = gru_forward(feature3, gruF1_iW, gruF1_sW, gruF1_sW2, gruF1_b, NULL);
	Mat * gruB = gru_backward(feature3, gruB1_iW, gruB1_sW, gruB1_sW2, gruB1_b, NULL);

	//  Combine GRU output
	Mat * gruFF = feedforward2_tanh(gruF, gruB, FF1_Wf, FF1_Wb, FF1_b, NULL);

	gru_forward(gruFF, gruF2_iW, gruF2_sW, gruF2_sW2, gruF2_b, gruF);
	gru_backward(gruFF, gruB2_iW, gruB2_sW, gruB2_sW2, gruB2_b, gruB);

	// Combine GRU output
	feedforward2_tanh(gruF, gruB, FF2_Wf, FF2_Wb, FF2_b, gruFF);

	Mat * post = softmax(gruFF, FF3_W, FF3_b, NULL);


	free_mat(gruFF);
	free_mat(gruB);
	free_mat(gruF);
	free_mat(feature3);
	free_mat(features);

	return post;
}

int main(int argc, char * argv[]){
	openblas_set_num_threads(1);
	assert(argc > 1);
	setup();

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
