# Scrappie basecaller

Scrappie attempts to call homopolymers.  It makes some mistakes.
```
Ref   : GACACAGTGAGGCTGCGTCTC-AAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGTTTGCATTTAGATCTCTT
Query : GACACAG-GAGGCTGCGTCTCAAAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGCTT-CA--CAGA-CT-TT
```

## Dependencies
* A good BLAS library + development headers including cblas.h.
* The HDF5 library and development headers.

On Debian based systems, the following packages are sufficient (tested Ubuntu 14.04 and 16.04)
* libopenblas-base
* libopenblas-dev
* libhdf5-10
* libhdf5-dev

## Compiling
```bash
make
```

## Running
```bash
#  Set some enviromental variables.  
# Allow scrappie to use as many threads as the system will support
export OMP_NUM_THREADS=`nproc`
# Use openblas in single-threaded mode
export OPENBLAS_NUM_THREADS=1
# Reads are assumed to be in the reads/ folder.
find reads -name \*.fast5 | xargs basecall > basecalls.fa
```

## Commandline options
```
basecall --help
Usage: basecall [OPTION...] fast5 [fast5 ...]
Scrappie basecaller -- scrappie attempts to call homopolymers

  -#, --threads=nreads       Number of reads to call in parallel
  -a, --analysis=number      Analysis to read events from
  -k, --skip=penalty         Penalty for skipping a base
  -l, --slip                 Enable slipping
  -m, --min_prob=probability Minimum bound on probability of match
  -n, --no-slip              Disable slipping
  -t, --trim=nevents         Number of events to trim
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

## Gotya's
* Scrappie does not call events are relies on information already being present in the fast5 files.  In particular:
  * Event calls are taken from /Analyses/EventDetection\_XXX/Reads/Read\_???/Events
  * Segmentation are taken from /Analyses/Segment\_Linear\_XXX/Summary/split\_hairpin
* Model is hard-coded.  Generate new header files using `parse_lstm.py model.pkl > lstm_model.h`
* The output is in Fasta format and no per-base quality scores are provided.  The order of the fasta header is:
  * filename
  * normalised score
  * number of events
  * bases called
* The normalised score (- total score / number of events) correlates well with read accuracy.
* Events with unusual rate metrics (number of event / bases called) may be unreliable.
