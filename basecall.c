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

#include "lstm_model.h"

const int NOUT = 5;
const int analysis = 0;
const int TRIM = 50;
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

void fprint_mat(FILE * fh, char * header, Mat_rptr mat, int nr, int nc){
        fputs(header, fh);
        fputc('\n', fh);
        for(int c=0 ; c < nc ; c++){
                const int offset = c * mat->nrq * 4;
                fprintf(fh, "%4d : %6.4e", c, mat->data.f[offset]);
                for(int r=1 ; r<nr ; r++){
                        fprintf(fh, "  %6.4e", mat->data.f[offset + r]);
                }
                fputc('\n', fh);
        }
}



struct _bs calculate_post(char * filename, int analysis){
	event_table et = read_detected_events(filename, analysis);
	if(NULL == et.event){
		return (struct _bs){0, 0, NULL};
	}
	/*
        fputs("* Data\n", stdout);
        for(int i=0 ; i< 100 ; i++){
                fprintf(stdout, "%d : %6d  %6d  %6.4e %6.4e\n", i, et.event[i].start, et.event[i].length, et.event[i].mean, et.event[i].stdv);
        }*/

	//  Make features
	Mat_rptr features = make_features(et, TRIM, true);
	//fprint_mat(stdout, "* Features", features, 4, 10);
	Mat_rptr feature3 = window(features, 3);
	//fprint_mat(stdout, "* Window", feature3, 12, 10);

	// Initial transformation of input for LSTM layer
	Mat_rptr lstmXf = feedforward_linear(feature3, lstmF1_iW, lstmF1_b, NULL);
	Mat_rptr lstmXb = feedforward_linear(feature3, lstmB1_iW, lstmB1_b, NULL);
	Mat_rptr lstmF = lstm_forward(lstmXf, lstmF1_sW, lstmF1_p, NULL);
	//fprint_mat(stdout, "* lstmForward", lstmF, 8, 10);
	Mat_rptr lstmB = lstm_backward(lstmXb, lstmB1_sW, lstmB1_p, NULL);
	//fprint_mat(stdout, "* lstmBackward", lstmB, 8, 10);

	//  Combine LSTM output
	Mat_rptr lstmFF = feedforward2_tanh(lstmF, lstmB, FF1_Wf, FF1_Wb, FF1_b, NULL);
	//fprint_mat(stdout, "* feedforward", lstmFF, 8, 10);

	feedforward_linear(lstmFF, lstmF2_iW, lstmF2_b, lstmXf);
	feedforward_linear(lstmFF, lstmB2_iW, lstmB2_b, lstmXb);
	lstm_forward(lstmXf, lstmF2_sW, lstmF2_p, lstmF);
	lstm_backward(lstmXb, lstmB2_sW, lstmB2_p, lstmB);

	// Combine LSTM output
	feedforward2_tanh(lstmF, lstmB, FF2_Wf, FF2_Wb, FF2_b, lstmFF);

	Mat_rptr post = softmax(lstmFF, FF3_W, FF3_b, NULL);

        for(int i=0 ; i < post->nc ; i++){
		const int offset = i * post->nrq;
		for(int r=0 ; r < post->nrq ; r++){
			post->data.v[offset + r] = logfv(MIN_PROB + MIN_PROB1M * post->data.v[offset + r]);
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
	} */




	free(seq);
	free_mat(post);
	free_mat(lstmFF);
	free_mat(lstmB);
	free_mat(lstmF);
	free_mat(lstmXb);
	free_mat(lstmXf);
	free_mat(feature3);
	free_mat(features);
	free(et.event);

	return (struct _bs){score, nev, bases};
}

int main(int argc, char * argv[]){
	assert(argc > 1);
	setup();

	#pragma omp parallel for schedule(dynamic)
	for(int fn=1 ; fn<argc ; fn++){
		struct _bs res = calculate_post(argv[fn], analysis);
		if(NULL == res.bases){
			continue;
		}
		//printf(">%s   %f (%d ev -> %lu bases)\n", basename(argv[fn]), res.score, res.nev, strlen(res.bases));
		printf(">%s   %f (%d ev -> %lu bases)\n%s\n", basename(argv[fn]), res.score, res.nev, strlen(res.bases), res.bases);
		free(res.bases);
	}

	return EXIT_SUCCESS;
}
