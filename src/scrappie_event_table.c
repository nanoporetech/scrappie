#include <dirent.h>
#include <glob.h>
#include <libgen.h>
#include <math.h>
#if defined(_OPENMP)
#    include <omp.h>
#endif
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>

#include "decode.h"
#include "event_detection.h"
#include "fast5_interface.h"
#include "networks.h"
#include "scrappie_common.h"
#include "scrappie_licence.h"
#include "scrappie_stdlib.h"
#include "util.h"

// Doesn't play nice with other headers, include last
#include <argp.h>

static const event_table _et_null = {
    .n = 0,
    .start = 0,
    .end = 0,
    .event = NULL
};

extern const char *argp_program_version;
extern const char *argp_program_bug_address;
static char doc[] = "Scrappie basecaller -- basecall via events";
static char args_doc[] = "fast5 [fast5 ...]";
static struct argp_option options[] = {
    {"output", 'o', "filename", 0, "Write to file rather than stdout"},
    {"trim", 't', "start:end", 0, "Number of events to trim, as start:end"},
    {"licence", 10, 0, 0, "Print licensing information"},
    {"license", 11, 0, OPTION_ALIAS, "Print licensing information"},
    {"segmentation", 14, "chunk:percentile", 0,
     "Chunk size and percentile for variance based segmentation"},
    {0}
};


struct arguments {
    FILE * output;
    int trim_start;
    int trim_end;
    int varseg_chunk;
    float varseg_thresh;
    char **files;
};

static struct arguments args = {
    .output = NULL,
    .trim_start = 200,
    .trim_end = 10,
    .varseg_chunk = 100,
    .varseg_thresh = 0.0f,
    .files = NULL
};

static error_t parse_arg(int key, char *arg, struct argp_state *state) {
    int ret = 0;
    char *next_tok = NULL;
    switch (key) {
    case 'o':
        args.output = fopen(arg, "w");
        if(NULL == args.output){
            errx(EXIT_FAILURE, "Failed to open \"%s\" for output.", arg);
        }
        break;
    case 't':
        args.trim_start = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if (NULL != next_tok) {
            args.trim_end = atoi(next_tok);
        } else {
            args.trim_end = args.trim_start;
        }
        assert(args.trim_start >= 0);
        assert(args.trim_end >= 0);
        break;
    case 10:
    case 11:
        ret = fputs(scrappie_licence_text, stdout);
        exit((EOF != ret) ? EXIT_SUCCESS : EXIT_FAILURE);
        break;
    case 14:
        args.varseg_chunk = atoi(strtok(arg, ":"));
        next_tok = strtok(NULL, ":");
        if (NULL == next_tok) {
            errx(EXIT_FAILURE,
                 "--segmentation should be of form chunk:percentile");
        }
        args.varseg_thresh = atof(next_tok) / 100.0;
        assert(args.varseg_chunk >= 0);
        assert(args.varseg_thresh > 0.0 && args.varseg_thresh < 1.0);
        break;

    case ARGP_KEY_NO_ARGS:
        argp_usage(state);
        break;

    case ARGP_KEY_ARG:
        args.files = &state->argv[state->next - 1];
        state->next = state->argc;
        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_arg, args_doc, doc };

static event_table create_events(char *filename) {
    RETURN_NULL_IF(NULL == filename, _et_null);
    raw_table rt = read_raw(filename, true);
    RETURN_NULL_IF(NULL == rt.raw, _et_null);
    rt = trim_and_segment_raw(rt, args.trim_start, args.trim_end, args.varseg_chunk, args.varseg_thresh);
    RETURN_NULL_IF(NULL == rt.raw, _et_null);

    event_table et = detect_events(rt, event_detection_defaults);
    free(rt.raw);
    return et;

}



int main_event_table(int argc, char *argv[]) {
    argp_parse(&argp, argc, argv, 0, 0, NULL);
    if(NULL == args.output){
        args.output = stdout;
    }


    int nfile = 0;
    for (; args.files[nfile]; nfile++) ;


    for (int fn = 0; fn < nfile; fn++) {
        event_table et = create_events(args.files[fn]);
        if (NULL == et.event) {
            warnx("No events returned for %s", args.files[fn]);
            continue;
        }
        fprintf(args.output, "# %s\n", args.files[fn]);
        fprintf(args.output, "#event\tstart\tmean\tstdv\tdwell\n");
        for(size_t i=0 ; i < et.n ; i++){
            event_t ev = et.event[i];
            fprintf(args.output, "%zu\t%zu\t%f\t%f\t%d\n", i, ev.start, ev.mean, ev.stdv, (int)ev.length);
        }
        free(et.event);
    }

    if(stdout != args.output){
        fclose(args.output);
    }

    return EXIT_SUCCESS;
}
