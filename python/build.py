import os

from cffi import FFI

if 'MANYLINUX' in os.environ:
    src_dir = os.path.join('/io', 'src')
    # build-wheels.sh determines these
    libraries=['openblas']
    library_dirs=['/usr/local/lib/']
else:
    if os.path.isfile(os.path.join('..', 'src','decode.h')):
        # assume the git repo
        src_dir = os.path.join('..', 'src')
    elif os.path.isfile(os.path.join('src','decode.h')):
        # else we're from an sdist
        src_dir = 'src'
    else:
        raise IOError('Cannot find scrappie C sources.')
    # this might want to be cblas on some systems
    libraries=['blas']
    library_dirs=[]

ffibuilder = FFI()
ffibuilder.set_source("libscrappy",
    r"""
      #include "decode.h"
      #include "networks.h"
      #include "scrappie_common.h"
      #include "util.h"
      #include "scrappie_seq_helpers.h"

      int get_raw_model_stride_from_string(const char * modelstr){
        // Obtain the model stride from its str name
        // avoid the intermediate errx from C signalling bad model
        // name with a return of -1.
        const enum raw_model_type modeltype = get_raw_model(modelstr);
        if(modeltype == SCRAPPIE_MODEL_INVALID){
          return -1;
        } else {
          return get_raw_model_stride(modeltype);
        }
      }

    """,
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=[src_dir],
    sources=[
        os.path.join(src_dir, '{}.c'.format(x)) for x in
        r'''decode
            event_detection
            layers
            networks
            nnfeatures
            scrappie_common
            scrappie_matrix
            scrappie_seq_helpers
            util'''.split()
    ],
    extra_compile_args=['-std=c99', '-msse3', '-O3']
)

ffibuilder.cdef("""
  typedef struct {
    size_t n;
    size_t start;
    size_t end;
    float *raw;
  } raw_table;


  void medmad_normalise_array(float *x, size_t n);
  raw_table trim_and_segment_raw(raw_table rt,
    int trim_start, int trim_end, int varseg_chunk, float varseg_thresh
  );
  raw_table trim_raw_by_mad(raw_table rt, int chunk_size, float perc);

  typedef struct {
    unsigned int nr, nrq, nc, stride;
    union {
      //__m128 *v; // we don't need this
      float *f;
    } data;
  } _Mat;
  typedef _Mat *scrappie_matrix;
  typedef _Mat const *const_scrappie_matrix;

  scrappie_matrix free_scrappie_matrix(scrappie_matrix mat);

  // Transducer basecalling
  scrappie_matrix nanonet_rgrgr_r94_posterior(const raw_table signal, float min_prob, bool return_log);
  scrappie_matrix nanonet_rgrgr_r95_posterior(const raw_table signal, float min_prob, bool return_log);
  float decode_transducer(
    const_scrappie_matrix logpost, float stay_pen, float skip_pen, float local_pen, int *seq, bool allow_slip
  );
  char *overlapper(const int *seq, int n, int nkmer, int *pos);

  // RNN-CRF
  scrappie_matrix nanonet_rnnrf_r94_transitions(const raw_table signal, float min_prob, bool return_log);
  float decode_crf(const_scrappie_matrix trans, int * path);
  char * crfpath_to_basecall(int const * path, size_t npos, int * pos);
  scrappie_matrix posterior_crf(const_scrappie_matrix trans);

  // Squiggle generation
  scrappie_matrix dna_squiggle(int const * sequence, size_t n, bool transform_units);

  // Scrappy Mappy
  float squiggle_match_viterbi(const raw_table signal, const_scrappie_matrix params,
                               float prob_back, float localpen, float minscore,
                               int32_t * path_padded);

  // Block-based mapping
  bool are_bounds_sane(int const * low, int const * high,
                       size_t nblock, size_t seqlen);
  float map_to_sequence_forward(const_scrappie_matrix logpost,
                                float stay_pen, float skip_pen, float local_pen,
                                int const *seq, size_t seqlen);
  float map_to_sequence_forward_banded(const_scrappie_matrix logpost,
                                       float stay_pen, float skip_pen, float local_pen,
                                       int const *seq, size_t seqlen,
                                       int const * poslow, int const * poshigh);

  float map_to_sequence_viterbi(const_scrappie_matrix logpost,
                                float stay_pen, float skip_pen, float local_pen,
                                int const *seq, size_t seqlen,
                                int *path);
  float map_to_sequence_viterbi_banded(const_scrappie_matrix logpost,
                                       float stay_pen, float skip_pen, float local_pen,
                                       int const *seq, size_t seqlen,
                                       int const * poslow, int const * poshigh);

  // Misc
  int * encode_bases_to_integers(char const * seq, size_t n, size_t state_len);
  int get_raw_model_stride_from_string(const char * modelstr);

""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
