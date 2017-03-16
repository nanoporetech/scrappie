#include <assert.h>
#include <err.h>
#include <libgen.h>
#include <math.h>

#if defined(_OPENMP)
	#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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
	{"dwell", 5, 0, 0, "Perform dwell correction of homopolymer lengths"},
	{"no-dwell", 6, 0, OPTION_ALIAS, "Don't perform dwell correction of homopolymer lengths"},
	{"limit", 'l', "nreads", 0, "Maximum number of reads to call (0 is unlimited)"},
	{"min_prob", 'm', "probability", 0, "Minimum bound on probability of match"},
	{"outformat", 'o', "format", 0, "Format to output reads (FASTA or SAM)"},
	{"skip", 's', "penalty", 0, "Penalty for skipping a base"},
	{"trim", 't', "nevents", 0, "Number of events to trim"},
	{"slip", 1, 0, 0, "Use slipping"},
	{"no-slip", 2, 0, OPTION_ALIAS, "Disable slipping"},
        {"segmentation", 3, "group", 0, "Fast5 group from which to reads segmentation"},
	{"segmentation-analysis", 7, "number", 0, "Analysis number to read seqmentation from"},
	{"dump", 4, "filename", 0, "Dump annotated events to HDF5 file"},
	{"albacore", 8, 0, 0, "Assume fast5 have been called using Albacore"},
	{"no-albacore", 9, 0, OPTION_ALIAS, "Assume fast5 have been called using Albacore"},
#if defined(_OPENMP)
	{"threads", '#', "nreads", 0, "Number of reads to call in parallel"},
#endif
	{0}
};

enum format { FORMAT_FASTA, FORMAT_SAM};

struct arguments {
	int analysis;
	int seganalysis;
	bool dwell_correction;
	int limit;
	float min_prob;
	enum format outformat;
	float skip_pen;
	bool use_slip;
	int trim;
	char * segmentation;
	char * dump;
	char ** files;
	bool albacore;
};
static struct arguments args = {-1, -1, true, 0, 1e-5, FORMAT_FASTA, 0.0, false, 50, "Segment_Linear", NULL, false};


static error_t parse_arg(int key, char * arg, struct  argp_state * state){
	switch(key){
	case 'a':
		args.analysis = atoi(arg);
		assert(args.analysis >= -1 && args.analysis < 1000);
		break;
	case 'l':
		args.limit = atoi(arg);
		assert(args.limit > 0);
		break;
	case 'm':
		args.min_prob = atof(arg);
		assert(isfinite(args.min_prob) && args.min_prob >= 0.0);
		break;
	case 'o':
		if(0 == strcasecmp("FASTA", arg)){
			args.outformat = FORMAT_FASTA;
		} else if(0 == strcasecmp("SAM", arg)){
			args.outformat = FORMAT_SAM;
		} else {
			errx(EXIT_FAILURE, "Unrecognised format");
		}
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
	case 5:
		args.dwell_correction = true;
		break;
	case 6:
		args.dwell_correction = false;
		break;
	case 7:
		args.seganalysis = atoi(arg);
		assert(args.seganalysis >= -1 && args.seganalysis < 1000);
		break;
	case 8:
		args.albacore = true;
		break;
	case 9:
		args.albacore = false;
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

	event_table et = args.albacore ?
		read_albacore_events(filename, args.analysis, "template") :
		read_detected_events(filename, args.analysis, args.segmentation, args.seganalysis);

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

	const int nev = post->nc;
	const int nstate = FF3_b->nr;
	const __m128 mpv = _mm_set1_ps(args.min_prob / nstate);
	const __m128 mpvm1 = _mm_set1_ps(1.0f - args.min_prob);
        for(int i=0 ; i < nev ; i++){
		const int offset = i * post->nrq;
		for(int r=0 ; r < post->nrq ; r++){
			post->data.v[offset + r] = LOGFV(mpv + mpvm1 * post->data.v[offset + r]);
		}
	}


	int * seq = calloc(nev, sizeof(int));
	float score = decode_transducer(post, args.skip_pen, seq, args.use_slip);
	int * pos = calloc(nev, sizeof(int));
	char * bases = overlapper(seq, nev, nstate - 1, pos);

	const int evoffset = et.start + args.trim;
	for(int ev=0 ; ev < nev ; ev++){
		et.event[ev + evoffset].state = 1 + seq[ev];
		et.event[ev + evoffset].pos = pos[ev];
	}

	if(args.dwell_correction){
		const float prior_scale = (et.event[nev + evoffset - 1].length + et.event[nev + evoffset - 1].start - et.event[evoffset].start)
					/ (float)strlen(bases);
		int * dwell = calloc(nev, sizeof(int));
		for(int ev=0 ; ev < nev ; ev ++){
			dwell[ev] = et.event[ev + evoffset].length;
		}

		/*   Calibrate scaling factor for homopolymer estimation.
		 *   Simple mean of the dwells of all 'step' movements in
		 * the basecall.  Steps within homopolymers are ignored.
		 *   A more complex calibration could be used.
		 */
		int tot_step_dwell = 0;
		int nstep = 0;
		for(int ev=0, ppos=-2, evdwell=0, pstate=-1 ; ev < nev ; ev++){
			// Sum over dwell of all steps excluding those within homopolymers
			if(et.event[ev + evoffset].pos == ppos){
				// Stay. Accumulate dwell
				evdwell += dwell[ev];
				continue;
			}

			if(et.event[ev + evoffset].pos == ppos + 1 && et.event[ev + evoffset].state != pstate){
				// Have a step that is not within a homopolymer
				tot_step_dwell += evdwell;
				nstep += 1;
			}

			evdwell = dwell[ev];
			ppos = et.event[ev + evoffset].pos;
			pstate = et.event[ev + evoffset].state;
		}
                // Estimate of scale with a prior with weight equal to a single observation.
		const float homo_scale = (prior_scale + tot_step_dwell) / (1.0 + nstep);
		const dwell_model dm = {homo_scale, {0.0f, 0.0f, 0.0f, 0.0f}};

		free(bases);
		bases = dwell_corrected_overlapper(seq, dwell, nev, nstate - 1, dm);

		free(dwell);
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

int fprintf_fasta(FILE * fp, const char * readname, const struct _bs res){
	const int nbase = strlen(res.bases);
	return fprintf(fp, ">%s  { \"normalised_score\" : %f,  \"nevent\" : %d,  \"sequence_length\" : %d,  \"events_per_base\" : %f }\n%s\n", readname, -res.score / res.nev, res.nev, nbase, (float)res.nev / (float) nbase, res.bases);
}

int fprintf_sam(FILE * fp, const char * readname, const struct _bs res){
	return fprintf(fp, "%s\t4\t*\t0\t0\t*\t*\t0\t0\t%s\t*\n", readname, res.bases);
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

		#pragma omp critical
		{
			switch(args.outformat){
			case FORMAT_FASTA:
				fprintf_fasta(stdout, basename(args.files[fn]), res);
				break;
			case FORMAT_SAM:
				fprintf_sam(stdout, basename(args.files[fn]), res);
				break;
			default:
				errx(EXIT_FAILURE, "Unrecognised output format");
			}

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
