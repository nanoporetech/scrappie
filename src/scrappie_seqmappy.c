#include <math.h>
#include <stdio.h>
//#include <string.h>
#include <strings.h>
#include <sys/types.h>

#include "decode.h"
#include "fast5_interface.h"
#include "networks.h"
#include "scrappie_common.h"
#include "scrappie_licence.h"
#include "scrappie_seq_helpers.h"
#include "scrappie_stdlib.h"
#include "util.h"


// Doesn't play nice with other headers, include last
#include <argp.h>


extern const char *argp_program_version;
extern const char *argp_program_bug_address;
static char doc[] = "Scrappie seqmappy (local-global)";
static char args_doc[] = "fasta fast5";
static struct argp_option options[] = {
    {"localpen", 'l', "float", 0, "Penalty for local matching"},
    {"min_prob", 'm', "probability", 0, "Minimum bound on probability of match"},
    {"output", 'o', "filename", 0, "Write to file rather than stdout"},
    {"prefix", 'p', "string", 0, "Prefix to append to name of read"},
    {"segmentation", 3, "chunk:percentile", 0, "Chunk size and percentile for variance based segmentation"},
    {"skip", 's', "penalty", 0, "Penalty for skipping a base"},
    {"stay", 'y', "penalty", 0, "Penalty for staying"},
    {"trim", 't', "start:end", 0, "Number of samples to trim, as start:end"},
    {"temperature1", 7, "factor", 0, "Temperature for softmax weights"},
    {"temperature2", 8, "factor", 0, "Temperature for softmax bias"},
    {"licence", 10, 0, 0, "Print licensing information"},
    {"license", 11, 0, OPTION_ALIAS, "Print licensing information"},
    {0}
};


struct arguments {
    float local_pen;
    float min_prob;
    float skip_pen;
    float stay_pen;
    float temperature1;
    float temperature2;
    FILE * output;
    char * prefix;
    int trim_start;
    int trim_end;
    int varseg_chunk;
    float varseg_thresh;

    char * fasta_file;
    char * fast5_file;
};

static struct arguments args = {
    .local_pen = 4.0f,
    .min_prob = 1e-5f,
    .stay_pen = 0.0f,
    .skip_pen = 0.0f,
    .temperature1 = 1.0f,
    .temperature2 = 1.0f,
    .output = NULL,
    .prefix = "",
    .trim_start = 200,
    .trim_end = 10,
    .varseg_chunk = 100,
    .varseg_thresh = 0.0f,

    .fasta_file = NULL,
    .fast5_file = NULL
};

static error_t parse_arg(int key, char *arg, struct argp_state *state) {
    int ret = 0;
    char * next_tok = NULL;

    switch (key) {
    case 'l':
        args.local_pen = atof(arg);
        break;
    case 'm':
        args.min_prob = atof(arg);
        break;
    case 'o':
        args.output = fopen(arg, "w");
        if(NULL == args.output){
            errx(EXIT_FAILURE, "Failed to open \"%s\" for output.", arg);
        }
        break;
    case 'p':
        args.prefix = arg;
        break;
    case 's':
        args.skip_pen = atof(arg);
        assert(isfinite(args.skip_pen));
        break;
    case 't':
        args.trim_start = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if(NULL != next_tok){
            args.trim_end = atoi(next_tok);
        } else {
            args.trim_end = args.trim_start;
        }
        assert(args.trim_start >= 0);
        assert(args.trim_end >= 0);
        break;
    case 'y':
        args.stay_pen = atof(arg);
        assert(isfinite(args.stay_pen));
        break;
    case 3:
        args.varseg_chunk = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if(NULL == next_tok){
            errx(EXIT_FAILURE, "--segmentation should be of form chunk:percentile");
        }
        args.varseg_thresh = atof(next_tok) / 100.0;
        assert(args.varseg_chunk >= 0);
        assert(args.varseg_thresh > 0.0 && args.varseg_thresh < 1.0);
        break;
    case 7:
        args.temperature1 = atof(arg);
        assert(isfinite(args.temperature1) && args.temperature1 > 0.0f);
        break;
    case 8:
        args.temperature2 = atof(arg);
        assert(isfinite(args.temperature2) && args.temperature2 > 0.0f);
        break;
    case 10:
    case 11:
        ret = fputs(scrappie_licence_text, stdout);
        exit((EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE);
        break;

    case ARGP_KEY_NO_ARGS:
        argp_usage(state);
        break;

    case ARGP_KEY_ARG:
        args.fasta_file = state->argv[state->next - 1];
        if(NULL == state->argv[state->next]){
            errx(EXIT_FAILURE, "fast5 file is a required argument");
        }
        args.fast5_file = state->argv[state->next];
        state->next = state->argc;
        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_arg, args_doc, doc };



int main_seqmappy(int argc, char *argv[]) {
    argp_parse(&argp, argc, argv, 0, 0, NULL);
    if(NULL == args.output){
        args.output = stdout;
    }


    //  Open sequence file
    scrappie_seq_t seq = read_sequence_from_fasta(args.fasta_file);
    if(NULL == seq.seq){
        warnx("Failed to open \"%s\" for input.\n", args.fasta_file);
        return EXIT_FAILURE;
    }

    const size_t state_len = 5;
    const size_t nstate = seq.n - state_len + 1;
    int * states = encode_bases_to_integers(seq.seq, seq.n, state_len);
    if(NULL == states){
        warnx("Memory allocation failure");
        return EXIT_FAILURE;
    }

    //  Read raw signal and normalise
    raw_table rt = read_raw(args.fast5_file, true);
    rt = trim_and_segment_raw(rt, args.trim_start, args.trim_end, args.varseg_chunk, args.varseg_thresh);
    if(NULL == rt.raw){
        warnx("Failed to open \"%s\" for input and trim signal.\n", args.fasta_file);
        free(states);
        return EXIT_FAILURE;
    }
    medmad_normalise_array(rt.raw + rt.start, rt.end - rt.start);


    scrappie_matrix logpost = nanonet_rgrgr_r94_posterior(rt, args.min_prob, args.temperature1, args.temperature2, true);
    if(NULL == logpost){
        warnx("Failed to calculate posterior for \"%s\" ", args.fasta_file);
        free(states);
        return EXIT_FAILURE;
    }

    const size_t nblock = logpost->nc;
    int * path = calloc(nblock, sizeof(int));
    if(NULL != path){
        float score = map_to_sequence_viterbi(logpost, args.stay_pen, args.skip_pen, args.local_pen, states, nstate, path);

        fprintf(args.output, "# %s to %s -- score %f over %zu blocks (%f per block)\n", args.fast5_file, args.fasta_file, -score, nblock, -score / nblock);
        fprintf(args.output, "block\tpos\n");
        for(size_t i=0 ; i < nblock ; i++){
            const int32_t pos = path[i];
            fprintf(args.output, "%zu\t%d\n", i, pos);
        }

        free(path);
    }


    free(states);
    logpost = free_scrappie_matrix(logpost);
    free(seq.seq);
    free(seq.name);

    return EXIT_SUCCESS;
}
