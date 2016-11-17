# Crappie
An experiment in nominative determinism.


## Running
```bash
export OMP_NUM_THREADS=ncore
export OPENBLAS_NUM_THREADS=1
find reads -name \*.fast5 | xargs crappie/basecall > basecalls.fa
```

## Gotya's
* Analysis number is hard-coded to zero, see top of basecall\*.c
* Basecall parameters (min\_prob and skip\_pen) are hard-coded. See top
* Model is hard-coded.  Generate new header files using parse\_\*.py model.pkl


