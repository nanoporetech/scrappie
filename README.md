# Scrappie basecaller

Scrappie attempts to call homopolymers.
```
Ref   : GACACAGTGAGGCTGCGTCTC-AAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGTTTGCATTTAGATCTCTT
Query : GACACAG-GAGGCTGCGTCTCAAAAAAAAAAAAAAAAAAAAAAAAAATTGCCCCTTCTTAAGCTT-CA--CAGA-CT-TT
```

## Compiling
```bash
make
```

## Running
```bash
export OMP_NUM_THREADS=ncore
export OPENBLAS_NUM_THREADS=1
find reads -name \*.fast5 | xargs crappie/basecall > basecalls.fa
```

## Gotya's
* Analysis number is hard-coded to zero, see top of basecall\_\*.c
* Basecall parameters (min\_prob and skip\_pen) are hard-coded. See top of basecall\_\*.c
* Model is hard-coded.  Generate new header files using parse\_\*.py model.pkl
