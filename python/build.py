import os

from cffi import FFI

if 'MANYLINUX' in os.environ:
    # build-wheels.sh determines these
    libraries=['openblas']
    library_dirs=['/usr/local/lib/']
else:
    # this might want to be cblas on some systems
    libraries=['blas']
    library_dirs=[]

if 'SCRAPPIESRC' in os.environ:
    src_dir = os.environ['SCRAPPIESRC']
    if not os.path.isfile(os.path.join(src_dir, 'decode.h')):
        raise IOError(
            'Scrappie sources not found at supplied location: {}'.format(src_dir)
        )
else:
    if os.path.isfile(os.path.join('..', 'src', 'decode.h')):
        # assume the git repo
        src_dir = os.path.join('..', 'src')
    elif os.path.isfile(os.path.join('src', 'decode.h')):
        # else we're from an sdist
        src_dir = 'src'
    else:
        raise IOError('Cannot find scrappie C sources.')

ffibuilder = FFI()
ffibuilder.set_source("libscrappy",
    r"""
      #include "decode.h"
      #include "networks.h"
      #include "scrappie_common.h"
      #include "util.h"
      #include "scrappie_seq_helpers.h"
      #include "scrappie_matrix.h"

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

with open('pyscrap.h', 'r') as fh:
    pyscrap_function_prototypes = fh.read()

ffibuilder.cdef("""
typedef struct {
  char * uuid;
  size_t n;
  size_t start;
  size_t end;
  float *raw;
} raw_table;

typedef struct {
  size_t nr, nrq, nc, stride;
  union {
    //__m128 *v; // we don't need this
    float *f;
  } data;
} _Mat;
typedef _Mat *scrappie_matrix;
typedef _Mat const *const_scrappie_matrix;
""" + pyscrap_function_prototypes)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
