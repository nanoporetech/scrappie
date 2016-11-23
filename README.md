# Scrappie basecaller

Scrappie attempts to call homopolymers.
```
Ref   : GACACAGTGAGGCTGCGTCTC-AAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGTTTGCATTTAGATCTCTT
Query : GACACAG-GAGGCTGCGTCTCAAAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGCTT-CA--CAGA-CT-TT
```

## Dependencies
* A good BLAS library + development headers.
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
find reads -name \*.fast5 | xargs scrappie/basecall > basecalls.fa
```

## Gotya's
* Scrappie does not call events are relies on information already being present in the file.  In particular:
  * Event calls are taken from /Analyses/EventDetection\_000/Reads/Read\_???/Events
  * Segmentation are taken from /Analyses/Segment\_Linear\_000/Summary/split\_hairpin
* Analysis number is hard-coded to zero, see top of basecall\_\*.c
* Basecall parameters (min\_prob and skip\_pen) are hard-coded. See top of basecall\_\*.c
* Model is hard-coded.  Generate new header files using parse\_\*.py model.pkl
* The output is in Fasta format and no per-base quality scores are provided.  The normalised score (-score / nevents) correlates well with read accuracy.
