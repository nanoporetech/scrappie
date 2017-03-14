# Scrappie
Scrappie is a technology demonstrator for the Oxford Nanopore Research Algorithms group


This project began life as a proof (bet) that a base caller could be written in a low level language in under 8 hours.  Some of the poor and just plain odd design decisions, along with the lack of documentation, are a result of its inception. In keeping with ONT's fish naming policy, the project was originally called Crappie.


Scrappie's purpose is to demonstrate the next generation of base calling and, as such, may change drastically between releases and breaks backwards compatibility.  A new version may not support older features.



# Release history
The intention is that behaviour will be stable within a series, with only bug fixes or minor improvements being applied.  An improvement or change in behaviour that is not a major shift in the algorithm will be a new series with a bump of the minor version number.  Any major changes in the algorithm will be a new series with the major number bumped.
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
