#pragma once
#ifndef HOMOPOLYMER_H
#    define HOMOPOLYMER_H
#    include <stdbool.h>
#    include "scrappie_matrix.h"
#    include "scrappie_structures.h"
//Modes of operation (to be sent to the input pathCalculationFlag)
enum homopolymer_calculation{
    HOMOPOLYMER_NOCHANGE,
    HOMOPOLYMER_MEAN};
    
int homopolymer_path(const scrappie_matrix post, int *viterbipath, enum homopolymer_calculation pathCalculationFlag);
#endif