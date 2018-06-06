#pragma once
#ifndef HOMOPOLYMER_H
#    define HOMOPOLYMER_H
#    include <stdbool.h>
#    include "scrappie_matrix.h"
#    include "scrappie_structures.h"
//Modes of operation (to be sent to the input pathCalculationFlag)
#    define HOMOPOLYMER_NOCHANGE 0
#    define HOMOPOLYMER_MEAN 1
int homopolymer_path(scrappie_matrix post, int *viterbipath, int pathCalculationFlag);
int change_temperature(double temperature,scrappie_matrix post);
#endif