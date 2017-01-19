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
* libhdf5
* libhdf5-dev

## Compiling
```bash
mkdir build && cd build && cmake .. && make
```

## Running
```bash
#  Set some enviromental variables.  
# Allow scrappie to use as many threads as the system will support
export OMP_NUM_THREADS=`nproc`
# Use openblas in single-threaded mode
export OPENBLAS_NUM_THREADS=1
# Reads are assumed to be in the reads/ folder.
find reads -name \*.fast5 | xargs scrappie > basecalls.fa
# Or using a strand list (skipping first line)
tail -n +2 strand_list.txt | sed 's:^:/path/to/reads/:' | xargs scrappie > basecalls.fa
```

## Commandline options
```
scrappie --help
Usage: scrappie [OPTION...] fast5 [fast5 ...]
Scrappie basecaller -- scrappie attempts to call homopolymers

  -#, --threads=nreads       Number of reads to call in parallel
  -a, --analysis=number      Analysis to read events from
  -d, --dwell, --no-dwell    Perform dwell correction of homopolymer lengths
      --dump=filename        Dump annotated events to HDF5 file
  -l, --limit=nreads         Maximum number of reads to call (0 is unlimited)
  -m, --min_prob=probability Minimum bound on probability of match
  -s, --skip=penalty         Penalty for skipping a base
      --segmentation=group   Fast5 group from which to reads segmentation
      --slip, --no-slip      Use slipping
  -t, --trim=nevents         Number of events to trim
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version
```

## Gotya's
* Scrappie does not call events and relies on this information already being present in the fast5 files.  In particular:
  * Event calls are taken from /Analyses/EventDetection\_XXX/Reads/Read\_???/Events, where XXX is the number set by the `--analysis` flag.
  * Segmentation is taken (by default) from /Analyses/Segment\_Linear\_XXX/Summary/split\_hairpin.  The group name for the segmentation data, here Segment\_Linear, can be set using the `--segmentation` flag.
* Model is hard-coded.  Generate new header files using `parse_lstm.py model.pkl > lstm_model.h`
* The output is in Fasta format and no per-base quality scores are provided.
  * The sequence ID is the name of the file that was basecalled.
  * The *description* element of the Fasta header is a JSON strong containing the following elements:
    * `normalised_score` Normalised score (total score / number of events).
    * `nevents` Number of events
    * `sequence_length` Length of sequence called
    * `events_per_base` Number of events per base called
* The normalised score (- total score / number of events) correlates well with read accuracy.
* Events with unusual rate metrics (number of event / bases called) may be unreliable.
