#include <assert.h>
#include <libgen.h>
#include <math.h>
#if defined(_OPENMP)
	#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decode.h"
#include "nnfeatures.h"
#include "layers.h"
#include <version.h>

#include "lstm_model.h"

// Doesn't play nice with other headers, include last
#include <argp.h>

const float MIN_PROB1M = 1.0 - 1e-5;

#if !defined(SCRAPPIE_VERSION)
#define SCRAPPIE_VERSION "unknown"
#endif

struct _bs {
	float score;
	int nev;
	char * bases;
	event_table et;
};

const char * argp_program_version = "scrappie " SCRAPPIE_VERSION;
const char * argp_program_bug_address = "<tim.massingham@nanoporetech.com>";
static char doc[] = "Scrappie basecaller -- scrappie attempts to call homopolymers";
static char args_doc[] = "fast5 [fast5 ...]";
static struct argp_option options[] = {
	{"analysis", 'a', "number", 0, "Analysis to read events from"},
	{"limit", 'l', "nreads", 0, "Maximum number of reads to call (0 is unlimited)"},
	{"min_prob", 'm', "probability", 0, "Minimum bound on probability of match"},
	{"skip", 's', "penalty", 0, "Penalty for skipping a base"},
	{"trim", 't', "nevents", 0, "Number of events to trim"},
	{"slip", 1, 0, 0, "Use slipping"},
	{"no-slip", 2, 0, 0, "Disable slipping"},
        {"segmentation", 3, "group", 0, "Fast5 group from which to reads segmentation"},
	{"dump", 4, "filename", 0, "Dump annotated events to HDF5 file"},
#if defined(_OPENMP)
	{"threads", '#', "nreads", 0, "Number of reads to call in parallel"},
#endif
	{0}
};

struct arguments {
	int analysis;
	int limit;
	float min_prob;
	float skip_pen;
	bool use_slip;
	int trim;
	char * segmentation;
	char * dump;
	char ** files;
};
static struct arguments args = {0, 0, 1e-5, 0.0, false, 50, "Segment_Linear", NULL};

static error_t parse_arg(int key, char * arg, struct  argp_state * state){
	switch(key){
	case 'a':
		args.analysis = atoi(arg);
		assert(args.analysis > 0 && args.analysis < 1000);
		break;
	case 'l':
		args.limit = atoi(arg);
		assert(args.limit > 0);
		break;
	case 'm':
		args.min_prob = atof(arg);
		assert(isfinite(args.min_prob) && args.min_prob >= 0.0);
		break;
	case 's':
		args.skip_pen = atof(arg);
		assert(isfinite(args.skip_pen) && args.skip_pen >= 0.0);
		break;
	case 't':
		args.trim = atoi(arg);
		assert(args.trim >= 0);
		break;
	case 1:
		args.use_slip = true;
		break;
	case 2:
		args.use_slip = false;
		break;
	case 3:
		args.segmentation = arg;
		break;
	case 4:
		args.dump = arg;
		break;

	#if defined(_OPENMP)
	case '#':
		{
			int nthread = atoi(arg);
			const int maxthread = omp_get_max_threads();
			if(nthread < 1){nthread = 1;}
			if(nthread > maxthread){nthread = maxthread;}
			omp_set_num_threads(nthread);
		}
		break;
	#endif

	case ARGP_KEY_NO_ARGS:
		argp_usage (state);

	case ARGP_KEY_ARG:
		args.files = &state->argv[state->next - 1];
		state->next = state->argc;
		break;

	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}
static struct argp argp = {options, parse_arg, args_doc, doc};



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


struct _bs calculate_post(char * filename){
	const int WINLEN = 3;
	event_table et = read_detected_events(filename, args.analysis, args.segmentation);
	if(NULL == et.event){
		return (struct _bs){0, 0, NULL};
	}
	if(et.n <= args.trim){
		free(et.event);
		return (struct _bs){0, 0, NULL};
	}
	/*
        fputs("* Data\n", stdout);
        for(int i=0 ; i< 100 ; i++){
                fprintf(stdout, "%d : %6d  %6d  %6.4e %6.4e\n", i, et.event[i].start, et.event[i].length, et.event[i].mean, et.event[i].stdv);
        }*/

	//  Make features
	Mat_rptr features = make_features(et, args.trim, true);
	//fprint_mat(stdout, "* Features", features, 4, 10);
	Mat_rptr feature3 = window(features, WINLEN);
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

	lstmXf = feedforward_linear(lstmFF, lstmF2_iW, lstmF2_b, lstmXf);
	lstmXb = feedforward_linear(lstmFF, lstmB2_iW, lstmB2_b, lstmXb);
	lstmF = lstm_forward(lstmXf, lstmF2_sW, lstmF2_p, lstmF);
	lstmB = lstm_backward(lstmXb, lstmB2_sW, lstmB2_p, lstmB);

	// Combine LSTM output
	lstmFF = feedforward2_tanh(lstmF, lstmB, FF2_Wf, FF2_Wb, FF2_b, lstmFF);

	Mat_rptr post = softmax(lstmFF, FF3_W, FF3_b, NULL);

	const int nstate = FF3_b->nr;
	const __m128 mpv = _mm_set1_ps(args.min_prob / nstate);
	const __m128 mpvm1 = _mm_set1_ps(1.0f - args.min_prob);
        for(int i=0 ; i < post->nc ; i++){
		const int offset = i * post->nrq;
		for(int r=0 ; r < post->nrq ; r++){
			post->data.v[offset + r] = LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
		}
	}


	int nev = post->nc;
	int * seq = calloc(post->nc, sizeof(int));
	float score = decode_transducer(post, args.skip_pen, seq, args.use_slip);
	int * pos = calloc(post->nc, sizeof(int));
	char * bases = overlapper(seq, post->nc, nstate - 1, pos);

	const int woffset = (WINLEN - 1) / 2;  // Offset due to windowing.  Right padded
	const int evoffset = et.start + args.trim + woffset;
	for(int ev=0 ; ev < nev ; ev++){
		et.event[ev + evoffset].state = 1 + seq[ev];
		et.event[ev + evoffset].pos = pos[ev];
	}

	free(pos);
	free(seq);
	free_mat(&post);
	free_mat(&lstmFF);
	free_mat(&lstmB);
	free_mat(&lstmF);
	free_mat(&lstmXb);
	free_mat(&lstmXf);
	free_mat(&feature3);
	free_mat(&features);

	return (struct _bs){score, nev, bases, et};
}

int main(int argc, char * argv[]){
	argp_parse(&argp, argc, argv, 0, 0, NULL);
	setup();

	int nfile = 0;
	for( ; args.files[nfile] ; nfile++);
	if(args.limit != 0 && nfile > args.limit){
		nfile = args.limit;
	}

	hid_t hdf5out = -1;
	if(NULL != args.dump){
		hdf5out = H5Fopen(args.dump, H5F_ACC_RDWR, H5P_DEFAULT);
		if(hdf5out < 0){
			hdf5out = H5Fcreate(args.dump, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		}
	}

	#pragma omp parallel for schedule(dynamic)
	for(int fn=0 ; fn < nfile ; fn++){
		struct _bs res = calculate_post(args.files[fn]);
		if(NULL == res.bases){
			continue;
		}
		const int nbase = strlen(res.bases);
		#pragma omp critical
		{
			printf(">%s  { \"normalised_score\" : %f,  \"nevent\" : %d,  \"sequence_length\" : %d,  \"events_per_base\" : %f }\n%s\n", basename(args.files[fn]), -res.score / res.nev, res.nev, nbase, (float)res.nev / (float) nbase, res.bases);
			if(hdf5out >= 0){
				write_annotated_events(hdf5out, basename(args.files[fn]), res.et);
			}
		}
		free(res.et.event);
		free(res.bases);
	}

	if(hdf5out >= 0){
		H5Fclose(hdf5out);
	}
	return EXIT_SUCCESS;
}
