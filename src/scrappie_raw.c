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
#include "fast5_interface.h"
#include "networks.h"
#include "scrappie_common.h"
#include "scrappie_licence.h"
#include "scrappie_stdlib.h"
#include "util.h"
#include "homopolymer.h"

// Doesn't play nice with other headers, include last
#include <argp.h>

struct _raw_basecall_info {
    float score;
    raw_table rt;

    char *basecall;
    size_t basecall_length;

    int *pos;
    size_t nblock;
};

extern const char *argp_program_version;
extern const char *argp_program_bug_address;
static char doc[] = "Scrappie basecaller -- basecall from raw signal";
static char args_doc[] = "fast5 [fast5 ...]";
static struct argp_option options[] = {
    {"format", 'f', "format", 0, "Format to output reads (FASTA or SAM)"},
    {"limit", 'l', "nreads", 0, "Maximum number of reads to call (0 is unlimited)"},
    {"min_prob", 'm', "probability", 0, "Minimum bound on probability of match"},
    {"output", 'o', "filename", 0, "Write to file rather than stdout"},
    {"prefix", 'p', "string", 0, "Prefix to append to name of each read"},
    {"skip", 's', "penalty", 0, "Penalty for skipping a base"},
    {"stay", 'y', "penalty", 0, "Penalty for staying"},
    {"local", 6, "penalty", 0, "Penalty for local basecalling"},
    {"temperature1", 7, "factor", 0, "Temperature for softmax weights"},
    {"temperature2", 8, "factor", 0, "Temperature for softmax bias"},
    {"trim", 't', "start:end", 0, "Number of samples to trim, as start:end"},
    {"slip", 1, 0, 0, "Use slipping"},
    {"no-slip", 2, 0, OPTION_ALIAS, "Disable slipping"},
    {"model", 5, "name", 0, "Raw model to use: \"raw_r94\", \"rgrgr_r94\", \"rgrgr_r941\", \"rgrgr_r10\", \"rnnrf_r94\""},
    // Currently disabled
    //{"dump", 4, "filename", 0, "Dump annotated blocks to HDF5 file"},
    {"licence", 10, 0, 0, "Print licensing information"},
    {"license", 11, 0, OPTION_ALIAS, "Print licensing information"},
    {"hdf5-compression", 12, "level", 0, "Gzip compression level for HDF5 output (0:off, 1: quickest, 9: best)"},
    {"hdf5-chunk", 13, "size", 0, "Chunk size for HDF5 output"},
    {"segmentation", 3, "chunk:percentile", 0, "Chunk size and percentile for variance based segmentation"},
    {"homopolymer", 'H',"homopolymer", 0, "Homopolymer run calc. to use: choose from \"nochange\" or \"mean\" (default). Not implemented for CRF."},
    {"uuid", 14, 0, 0, "Output UUID"},
    {"no-uuid", 15, 0, OPTION_ALIAS, "Output read file"},
#if defined(_OPENMP)
    {"threads", '#', "nparallel", 0, "Number of reads to call in parallel"},
#endif
    {0}
};

enum format { FORMAT_FASTA, FORMAT_SAM };

struct arguments {
    enum format outformat;
    int limit;
    float min_prob;
    FILE * output;
    char * prefix;
    float skip_pen;
    float stay_pen;
    float local_pen;
    float temperature1;
    float temperature2;
    bool use_slip;
    int trim_start;
    int trim_end;
    int varseg_chunk;
    float varseg_thresh;
    char * dump;
    int compression_level;
    int compression_chunk_size;
    enum raw_model_type model_type;
    char ** files;
    enum homopolymer_calculation homopolymer;
    bool uuid;
};

static struct arguments args = {
    .limit = 0,
    .min_prob = 1e-5f,
    .output = NULL,
    .outformat = FORMAT_FASTA,
    .prefix = "",
    .skip_pen = 0.0f,
    .stay_pen = 0.0f,
    .local_pen = 2.0f,
    .temperature1 = 1.0f,
    .temperature2 = 1.0f,
    .use_slip = false,
    .trim_start = 200,
    .trim_end = 10,
    .varseg_chunk = 100,
    .varseg_thresh = 0.0f,
    .dump = NULL,
    .compression_level = 1,
    .compression_chunk_size = 200,
    .model_type = SCRAPPIE_MODEL_RGRGR_R9_4,
    .files = NULL,
    .homopolymer = HOMOPOLYMER_MEAN,
    .uuid = false
};

static error_t parse_arg(int key, char * arg, struct  argp_state * state){
    int ret = 0;
    char * next_tok = NULL;

    switch(key){
    case 'f':
        if(0 == strcasecmp("FASTA", arg)){
            args.outformat = FORMAT_FASTA;
        } else if(0 == strcasecmp("SAM", arg)){
            args.outformat = FORMAT_SAM;
        } else {
            errx(EXIT_FAILURE, "Unrecognised format");
        }
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
    case 'H':
        args.homopolymer = get_homopolymer_calculation(arg);
        if(HOMOPOLYMER_INVALID == args.homopolymer){
            errx(EXIT_FAILURE, "Invalid homopolymer calculation \"%s\"", arg);
        }
        break;
    case 1:
        args.use_slip = true;
        break;
    case 2:
        args.use_slip = false;
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
    case 4:
        args.dump = arg;
        break;
    case 5:
        args.model_type = get_raw_model(arg);
        if(SCRAPPIE_MODEL_INVALID == args.model_type){
            errx(EXIT_FAILURE, "Invalid model name \"%s\"", arg);
        }
        break;
    case 6:
        args.local_pen = atof(arg);
        assert(isfinite(args.local_pen));
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
    case 12:
        args.compression_level = atoi(arg);
        assert(args.compression_level >= 0 && args.compression_level <= 9);
        break;
    case 13:
        args.compression_chunk_size = atoi(arg);
        assert(args.compression_chunk_size > 0);
        break;
    case 14:
        args.uuid = true;
        break;
    case 15:
        args.uuid = false;
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


static struct argp argp = {options, parse_arg, args_doc, doc};

static struct _raw_basecall_info calculate_post(char * filename, enum raw_model_type model){
    RETURN_NULL_IF(NULL == filename, (struct _raw_basecall_info){0});
    RETURN_NULL_IF(SCRAPPIE_MODEL_INVALID == model, (struct _raw_basecall_info){0});
    posterior_function_ptr calcpost = get_posterior_function(model);

    raw_table rt = read_raw(filename, true);
    RETURN_NULL_IF(NULL == rt.raw, (struct _raw_basecall_info){0});

    rt = trim_and_segment_raw(rt, args.trim_start, args.trim_end, args.varseg_chunk, args.varseg_thresh);
    RETURN_NULL_IF(NULL == rt.raw, (struct _raw_basecall_info){0});

    medmad_normalise_array(rt.raw + rt.start, rt.end - rt.start);
    scrappie_matrix post = calcpost(rt, args.min_prob, args.temperature1, args.temperature2, true);

    if (NULL == post) {
        free(rt.raw);
        free(rt.uuid);
        return (struct _raw_basecall_info){0};
    }
    const int nblock = post->nc;
    int * path = calloc(nblock + 1, sizeof(int));
    int * pos = calloc(nblock + 1, sizeof(int));

    float score = NAN;
    char * basecall = NULL;
    if(SCRAPPIE_MODEL_RNNRF_R9_4 != model){
        const int nstate = post->nr;
        score = decode_transducer(post, args.stay_pen, args.skip_pen, args.local_pen, path, args.use_slip);
        int runcount = homopolymer_path(post, path, args.homopolymer);
        if(runcount < 0){
            // On error, clean up and return
            free(pos);
            free(path);
            post = free_scrappie_matrix(post);
            free(rt.raw);
            free(rt.uuid);
            return (struct _raw_basecall_info){0};
        }
        basecall = overlapper(path, nblock + 1, nstate - 1, pos);
    } else{
        score = decode_crf(post, path);
        basecall = crfpath_to_basecall(path, nblock, pos);
    }

    free(path);
    post = free_scrappie_matrix(post);
    const size_t basecall_len = strlen(basecall);

    return (struct _raw_basecall_info) {
    score, rt, basecall, basecall_len, pos, nblock};
}

static int fprintf_fasta(FILE * fp, const char * uuid, const char *readname, bool uuid_primary, const char * prefix,
                         const struct _raw_basecall_info res) {
    return fprintf(fp,
                   ">%s%s  { \"filename\" : \"%s\", \"uuid\" : \"%s\", \"normalised_score\" : %f,  \"nblock\" : %zu,  \"sequence_length\" : %zu,  \"blocks_per_base\" : %f, \"nsample\" : %zu, \"trim\" : [ %zu, %zu ] }\n%s\n",
                   prefix, uuid_primary ? uuid : readname, readname, uuid, -res.score / res.nblock, res.nblock,
                   res.basecall_length,
                   (float)res.nblock / (float)res.basecall_length,
                   res.rt.n, res.rt.start, res.rt.end, res.basecall);
}

static int fprintf_sam(FILE * fp,  const char * uuid, const char *readname, bool uuid_primary, const char * prefix,
                       const struct _raw_basecall_info res) {
    return fprintf(fp, "%s%s\t4\t*\t0\t0\t*\t*\t0\t0\t%s\t*\n", prefix,
                   uuid_primary ? uuid : readname, res.basecall);
}

int main_raw(int argc, char * argv[]){
    #if defined(_OPENMP)
        omp_set_nested(1);
    #endif
    argp_parse(&argp, argc, argv, 0, 0, NULL);
    if(NULL == args.output){
        args.output = stdout;
    }

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
            {
                DIR * dirp = opendir(args.files[fn]);
                if(NULL != dirp){
                    // If filename is a directory, add wildcard to find all fast5 files within it
                    memcpy(globpath + rootlen, "/*.fast5", 8 * sizeof(char));
                    closedir(dirp);
                }
            }
            int globret = glob(globpath, GLOB_NOSORT, NULL, &globbuf);
            free(globpath);
            if(0 != globret){
                if(GLOB_NOMATCH == globret){
                    warnx("File or directory \"%s\" does not exist or no fast5 files found.", args.files[fn]);
                }
                globfree(&globbuf);
                continue;
            }
        }
        #pragma omp parallel for schedule(dynamic)
        for(size_t fn2=0 ; fn2 < globbuf.gl_pathc ; fn2++){
            if(reads_limit > 0 && reads_started >= reads_limit){
                continue;
            }
            #pragma omp atomic
            reads_started += 1;

            char * filename = globbuf.gl_pathv[fn2];
            struct _raw_basecall_info res = calculate_post(filename, args.model_type);
            if(NULL == res.basecall){
                warnx("No basecall returned for %s", filename);
                continue;
            }

            #pragma omp critical(sequence_output)
            {
                switch(args.outformat){
                case FORMAT_FASTA:
                    fprintf_fasta(args.output, res.rt.uuid, basename(filename), args.uuid, args.prefix, res);
                    break;
                case FORMAT_SAM:
                    fprintf_sam(args.output, res.rt.uuid, basename(filename), args.uuid, args.prefix, res);
                    break;
                default:
                    errx(EXIT_FAILURE, "Unrecognised output format");
                }

                if(hdf5out >= 0){
                    write_annotated_raw(hdf5out, basename(filename), res.rt,
                        args.compression_chunk_size, args.compression_level);
                }
            }
            free(res.rt.raw);
            free(res.rt.uuid);
            free(res.basecall);
            free(res.pos);
        }
        globfree(&globbuf);
    }

    if(hdf5out >= 0){
        H5Fclose(hdf5out);
    }

    if(stdout != args.output){
        fclose(args.output);
    }

    return EXIT_SUCCESS;
}
