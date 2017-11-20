# Scrappie
Scrappie is a technology demonstrator for the Oxford Nanopore Research Algorithms group

This software is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

(c) 2017 Oxford Nanopore Technologies Ltd.

The vectorised math functions `src/sse_mathfun.h` are from
http://gruntthepeon.free.fr/ssemath/ and the original version of this file is
under the 'zlib' licence.  See the top of `src/sse_mathfun.h` for details.

The Fasta / Fastq input library 'src/kseq.h' is from
https://github.com/attractivechaos/klib/blob/master/kseq.h and the original
version of the file is under the 'MIT' licence.  See the top of `src/kseq.h`
for details.



This project began life as a proof (bet) that a base caller could be written from scratch in a low level language in under 8 hours.  Some of the poor and just plain odd design decisions, along with the lack of documentation, are a result of its inception. In keeping with ONT's fish naming policy, the project was originally called Crappie (genus *Pomoxis*).


Scrappie's purpose is to demonstrate the next generation of base calling and, as such, may change drastically between releases and breaks backwards compatibility.  A new version may not support older features.



# Release history
The intention is that behaviour will be stable within a series, with only bug fixes or minor improvements being applied.  An improvement or change in behaviour that is not a major shift in the algorithm will be a new series with a bump of the minor version number.  Any major changes in the algorithm will be a new series with the major number bumped.
* 1.2 series: Sequence-to-squiggle
    * *release-1.2.0*
        * Ability to predict expected squiggle from sequence
* 1.1 series: Improved basecalling from raw signal
    * *release-1.1.1*
        * Improved rgrgr models
        * Ability to write output to named file
        * Rename argument '--outformat' to '--format'
    * *release-1.1.0*
        * Implementation of raw data 'pirate' networks (rGr and rgrgr).
        * Stand-alone event detection.
        * Stay penalty to allow ratio of insertions to deletions to be changed.
        * Local (Smith-Waterman like) penalty on edges of basecall to remove noisy bases.
        * Improvements to error reporting.
* 1.0 series: Public release
    * *release-1.0.1*
        * Integration with Travis-ci
        * Rudimentaty unittests
    * *release-1.0.0*
        * Change licence to MPL.
	* Increased number for assertions.
	* A few stability and bug fixes.
* 0.3 series: Basecalling from raw signal
    * *release-0.3.2* Minor fixes to support use of OpenBLAS and HDF5 libraries in non-standard locations 
    * *release-0.3.1* Expose options for segmenting raw signal
    * *release-0.3.0* Initial release implementing calling from raw signal.
* 0.2 series: Post-hoc correction of homopolymer lengths from dwells.
    * *release-0.2.8*
        * Updated model.
	* Use Kahan summation during normalisation.
	* Build scripts don't require presence of C++ compiler.
    * *release-0.2.7*
        * Allow segmentation to be taken from different analysis to event detection.
        * Basic support for recalling files produced by Albacore.
    * *release-0.2.6* Support for building on OSX
    * *release-0.2.5* Allow output to be in simple SAM format
    * *release-0.2.4* Add document describing release history
    * *release-0.2.3* Bugfix: Segfault when there is insufficent steps to calibrate dwell scaling factor.
    * *release-0.2.2* Remove redundant Python scripts for dwell correct.
    * *release-0.2.1* Make dwell correction the default
    * *release-0.2.0* Initial release
* 0.1 series: Refactoring
    * *release-0.1.1* Allow location of segmentation to be specified on commandline.
    * *release-0.1.0* Reorganisation + move to CMake.
* 0.0 series: Transducers
    * *release-0.0.3* Output more information about each read + internal changes to memory management.
    * *release-0.0.2* Initial release of Scrappie.
