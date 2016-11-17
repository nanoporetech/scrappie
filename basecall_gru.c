#include <assert.h>
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decode.h"
#include "features.h"
#include "layers.h"
#include "read_events.h"
#include "util.h"

#include "gru_model.h"

const int NOUT = 5;
const float SKIP_PEN = 0.0;
const float MIN_PROB = 1e-5;
const float MIN_PROB1M = 1.0 - 1e-5;

struct _bs {
	float score;
	int nev;
	char * bases;
};


char bases[4] = {'A','C','G','T'};
char * kmer_from_state(int state, int klen, char * kmer){
	assert(NULL!=kmer);
	for(int i=0 ; i<klen ; i++){
		int b = state &3;
		kmer[klen - i - 1] = bases[b];
		state >>= 2;
	}
	return kmer;
}

	


struct _bs calculate_post(char * filename, int analysis){
	event_table et = read_detected_events(filename, analysis);

	//  Make features
	Mat_rptr features = make_features(et, true);
	Mat_rptr feature3 = window(features, 3);

	Mat_rptr gruF = gru_forward(feature3, gruF1_iW, gruF1_sW, gruF1_sW2, gruF1_b, NULL);
	Mat_rptr gruB = gru_backward(feature3, gruB1_iW, gruB1_sW, gruB1_sW2, gruB1_b, NULL);

	//  Combine GRU output
	Mat_rptr gruFF = feedforward2_tanh(gruF, gruB, FF1_Wf, FF1_Wb, FF1_b, NULL);

	gru_forward(gruFF, gruF2_iW, gruF2_sW, gruF2_sW2, gruF2_b, gruF);
	gru_backward(gruFF, gruB2_iW, gruB2_sW, gruB2_sW2, gruB2_b, gruB);

	// Combine GRU output
	feedforward2_tanh(gruF, gruB, FF2_Wf, FF2_Wb, FF2_b, gruFF);

	Mat_rptr post = softmax(gruFF, FF3_W, FF3_b, NULL);

        for(int i=0 ; i < post->nc ; i++){
		const int offset = i * post->nrq;
		for(int r=0 ; r < post->nrq ; r++){
			post->data.v[offset + r] = fast_logfv(MIN_PROB + MIN_PROB1M * post->data.v[offset + r]);
		}
	}


	int nev = post->nc;
	int * seq = calloc(post->nc, sizeof(int));
	float score = decode_transducer(post, SKIP_PEN, seq);
	char * bases = overlapper(seq, post->nc, post->nr - 1);


/*
	for(int i=0 ; i<50 ; i++){
		const int offset = i * post->nrq * 4;
                char kmer[6] = {0, 0, 0, 0, 0, 0};
                char blank[] = "-----";
		printf("%d (%s): stay=%f ", i, (seq[i]==-1)?blank:kmer_from_state(seq[i],5,kmer), 
                                           expf(post->data.f[offset]));
		for(int j=1 ; j < post->nr ; j++){
			float ep = expf(post->data.f[offset+j]);
			if(ep > 0.05){
                                kmer_from_state(j - 1, 5, kmer);
				printf("%s (%f)  ", kmer,ep);
			}
		}
		fputc('\n', stdout);
	}
*/



	free(seq);
	free_mat(post);
	free_mat(gruFF);
	free_mat(gruB);
	free_mat(gruF);
	free_mat(feature3);
	free_mat(features);

	return (struct _bs){score, nev, bases};
}

int main(int argc, char * argv[]){
	assert(argc > 1);
	setup();

	#pragma omp parallel for schedule(dynamic)
	for(int fn=1 ; fn<argc ; fn++){
		struct _bs res = calculate_post(argv[fn], 0);
		printf(">%s   %f (%d ev -> %lu bases)\n%s\n", basename(argv[fn]), res.score, res.nev, strlen(res.bases), res.bases);
		free(res.bases);
	}

	return EXIT_SUCCESS;
}
