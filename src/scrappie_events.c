#include <assert.h>
#include <dirent.h>
#include <err.h>
#include <glob.h>
#include <libgen.h>
#include <math.h>

#if defined(_OPENMP)
	#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include "decode.h"
#include "networks.h"
#include "scrappie_licence.h"
#include "util.h"

void scrappie_network_setup(void);


// Doesn't play nice with other headers, include last
#include <argp.h>

struct _bs {
	float score;
	int nev;
	char * bases;
	event_table et;
};

extern const char * argp_program_version;
extern const char * argp_program_bug_address;
static char doc[] = "Scrappie basecaller -- basecall from events";
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
        {"segmentation", 3, "group:summary", 0, "Fast5 group from which to read segmentation"},
	{"segmentation-analysis", 7, "number", 0, "Analysis number to read segmentation from"},
	{"dump", 4, "filename", 0, "Dump annotated events to HDF5 file"},
	{"albacore", 8, 0, 0, "Assume fast5 have been called using Albacore"},
	{"no-albacore", 9, 0, OPTION_ALIAS, "Assume fast5 have been called using Albacore"},
	{"licence", 10, 0, 0, "Print licensing information"},
	{"license", 11, 0, OPTION_ALIAS, "Print licensing information"},
	{"hdf5-compression", 12, "level", 0, "Gzip compression level for HDF5 output (0:off, 1: quickest, 9: best)"},
	{"hdf5-chunk", 13, "size", 0, "Chunk size for HDF5 output"},
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
	char * segloc1;
	char * segloc2;
	char * dump;
	bool albacore;
	int compression_level;
	int compression_chunk_size;
	char ** files;
};

static struct arguments args = {
	.analysis = -1,
	.seganalysis = -1,
	.dwell_correction = true,
	.limit = 0,
	.min_prob = 1e-5,
	.outformat = FORMAT_FASTA,
	.skip_pen = 0.0,
	.use_slip = false,
	.trim = 50,
	.segloc1 = "Segmentation",
	.segloc2 = "segmentation",
	.dump = NULL,
	.albacore = false,
	.compression_level = 1,
	.compression_chunk_size = 200,
	.files = NULL
};


static error_t parse_arg(int key, char * arg, struct  argp_state * state){
	switch(key){
		int ret = 0;
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
		args.segloc1 = strtok(arg, ":");
		char * next_token = strtok(NULL, ":");
		if(NULL == next_token){
			warnx("Segmentation should be of form 'loc1:loc2' but loc2 not found.  Going to use default of %s.", args.segloc2);
		} else {
			args.segloc2 = next_token;
		}
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
	case 10:
	case 11:
		ret = fputs(scrappie_licence_text, stdout);
		exit((EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE);
		break;
	case 12:
		args.compression_level = atoi(arg);
		assert(args.compression_level >= 0 && args.compression_level <= 9);
		break;
	case 13:
		args.compression_chunk_size = atoi(arg);
		assert(args.compression_chunk_size > 0);
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

static struct _bs calculate_post(char * filename){

	event_table et = args.albacore ?
		read_albacore_events(filename, args.analysis, "template") :
		read_detected_events(filename, args.analysis, args.segloc1, args.segloc2, args.seganalysis);

	if(NULL == et.event){
		return (struct _bs){0, 0, NULL};
	}
	const int nevent = et.end - et.start;
	if(nevent <= 2 * args.trim){
		warnx("Too few events in %s to call (%d after segmentation, originally %lu).", filename, nevent, et.n);
		free(et.event);
		return (struct _bs){0, 0, NULL};
	}
	et.start += args.trim;
	et.end -= args.trim;

	scrappie_matrix post = nanonet_posterior(et, args.min_prob, true);
	if(NULL == post){
		free(et.event);
		return (struct _bs){0, 0, NULL};
	}
	const int nev = post->nc;
	const int nstate = post->nr;


	int * seq = calloc(nev, sizeof(int));
	float score = decode_transducer(post, args.skip_pen, seq, args.use_slip);
	post = free_scrappie_matrix(post);
	int * pos = calloc(nev, sizeof(int));
	char * basecall = overlapper(seq, nev, nstate - 1, pos);
	const size_t basecall_len = strlen(basecall);

	const int evoffset = et.start;
	for(int ev=0 ; ev < nev ; ev++){
		et.event[ev + evoffset].state = 1 + seq[ev];
		et.event[ev + evoffset].pos = pos[ev];
	}

	if(args.dwell_correction){
		char * newbasecall = homopolymer_dwell_correction(et, seq, nstate, basecall_len);
		if(NULL != newbasecall){
			free(basecall);
			basecall = newbasecall;
		}
	}

	free(pos);
	free(seq);

	return (struct _bs){score, nev, basecall, et};
}

static int fprintf_fasta(FILE * fp, const char * readname, const struct _bs res){
	const int nbase = strlen(res.bases);
	return fprintf(fp, ">%s  { \"normalised_score\" : %f,  \"nevent\" : %d,  \"sequence_length\" : %d,  \"events_per_base\" : %f }\n%s\n", readname, -res.score / res.nev, res.nev, nbase, (float)res.nev / (float) nbase, res.bases);
}

static int fprintf_sam(FILE * fp, const char * readname, const struct _bs res){
	return fprintf(fp, "%s\t4\t*\t0\t0\t*\t*\t0\t0\t%s\t*\n", readname, res.bases);
}


int main_events(int argc, char * argv[]){
	#if defined(_OPENMP)
		omp_set_nested(1);
	#endif
	argp_parse(&argp, argc, argv, 0, 0, NULL);

	hid_t hdf5out = -1;
	if(NULL != args.dump){
		hdf5out = H5Fopen(args.dump, H5F_ACC_RDWR, H5P_DEFAULT);
		if(hdf5out < 0){
			hdf5out = H5Fcreate(args.dump, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
		}
	}

	int nfile = 0;
	for( ; args.files[nfile] ; nfile++);

	int reads_started = 0;
	const int reads_limit = args.limit;
	#pragma omp parallel for schedule(dynamic)
	for(int fn=0 ; fn < nfile ; fn++){
		if(reads_limit > 0 && reads_started >= reads_limit){
			continue;
		}
		//  Iterate through all files and directories on command line.
		//  Nested parallelism for OpenMP is enabled so worker threads are used for all open
		// directories but this less than optimal since many directories may be open at once.
		glob_t globbuf;
		{
			// Find all files matching commandline argument using system glob
			const size_t rootlen = strlen(args.files[fn]);
			char * globpath = calloc(rootlen + 9, sizeof(char));
			memcpy(globpath, args.files[fn], rootlen * sizeof(char));
			if(NULL != opendir(args.files[fn])){
				// If filename is a directory, add wildcard to find all fast5 files within it
				memcpy(globpath + rootlen, "/*.fast5", 8 * sizeof(char));
			}
			int globret = glob(globpath, GLOB_NOSORT, NULL, &globbuf);
			free(globpath);
			if(0 != globret){
				globfree(&globbuf);
				continue;
			}
		}
		#pragma omp parallel for schedule(dynamic)
		for(int fn2=0 ; fn2 < globbuf.gl_pathc ; fn2++){
			if(reads_limit > 0 && reads_started >= reads_limit){
				continue;
			}
			#pragma omp atomic
			reads_started += 1;

			char * filename = globbuf.gl_pathv[fn2];
			struct _bs res = calculate_post(filename);
			if(NULL == res.bases){
				warnx("No basecall returned for %s", filename);
				continue;
			}

			#pragma omp critical(sequence_output)
			{
				switch(args.outformat){
				case FORMAT_FASTA:
					fprintf_fasta(stdout, strip_filename_extension(basename(filename)), res);
					break;
				case FORMAT_SAM:
					fprintf_sam(stdout, strip_filename_extension(basename(filename)), res);
					break;
				default:
					errx(EXIT_FAILURE, "Unrecognised output format");
				}

				if(hdf5out >= 0){
					write_annotated_events(hdf5out, basename(filename), res.et,
						args.compression_chunk_size, args.compression_level);
				}
			}
			free(res.et.event);
			free(res.bases);
		}
		globfree(&globbuf);
	}

	if(hdf5out >= 0){
		H5Fclose(hdf5out);
	}
	return EXIT_SUCCESS;
}
