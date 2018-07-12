Scrappy
=======

Scrappy provides a minimal python binding to some of the core functionality of
the `scrappie raw` basecaller. The performance of scrappy should be comparable
(if not essentially identical) to scrappie.

Installation
------------

Scrappy is available on pypi. For linux it is available as a pre-compiled
wheel, for MacOS the distribution will be built from source. Scrappy can be
installed simply as:

    pip install scrappie

Note the spelling! The above may fail on MacOS, due to various compiler and
library versions which might be present. In this case please report an issue
on github.

Building from the source is also supported, providing one has met the
requirements for building the scrappie itself. It is recommended to first
ensure scrappie builds cleanly before attempting to build it's python
counterpart. To build from the source, run:

    pip install -r requirements
    python setup.py install

The second command may be modified as appropriate (perhaps by adding `--user`
for non-root installations).

API
---

Scrappy directly exposes the key parts of scrappie's "from raw" basecalling
pipeline. The steps are:

    1) `scrappy.trim_raw`: attempt to remove data points corresponding to adapters,
    2) `scrappy.scale_raw`: perform a robust normalization of the data,
    3) `scrappy.calc_post`: calculate time-resolved label probabilties ("posterior probabilities"),
    4) `scrappy.decode_post`: a Viterbi decoding to obtain optimal base sequence.

The above is illustrated in `scrappy.basecall_raw`.

Additionally the squiggle simulation and alignment functionality is exposed via:

    1) `scrappy.sequence_to_squiggle`: simulate a squiggle from a base sequence,
    2) `scrappy.map_signal_to_squiggle`: align raw data to a simulated sequence.

Demo
----

After installation the scrappy program can be run to demonstrate the API's
function. The program takes one or more filepaths to `.fast5` files and outputs
FASTA formatted basecalls to stdout, e.g.:

    scrappy file1.fast5 <file2.fast5> <file3.fast5> ...

As with scrappie to achieve best performance it is recommended to disable
threading in your BLAS library (this is true also if building an application
using the API).
