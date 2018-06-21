#pragma once
#ifndef HOMOPOLYMER_H
#    define HOMOPOLYMER_H
#    include <stdbool.h>
#    include "scrappie_matrix.h"
#    include "scrappie_structures.h"
//Modes of operation (to be sent to the input pathCalculationFlag)
enum homopolymer_calculation{
    HOMOPOLYMER_NOCHANGE,
    HOMOPOLYMER_MEAN,
    HOMOPOLYMER_INVALID};

enum homopolymer_calculation get_homopolymer_calculation(const char * calcstring);    
int homopolymer_path(const_scrappie_matrix post, int *viterbipath, enum homopolymer_calculation pathCalculationFlag);
#endif