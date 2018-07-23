#pragma once
#ifndef DECODE_H
#    define DECODE_H
#    include <stdbool.h>
#    include "scrappie_matrix.h"
#    include "scrappie_structures.h"

typedef struct {
    float scale;
    float base_adj[4];
} dwell_model;

float decode_transducer(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                        float local_pen, int *seq, bool allow_slip);
char *overlapper(const int *seq, size_t n, int nkmer, int *pos);
char *homopolymer_dwell_correction(const event_table et, const int *seq,
                                   size_t nstate, size_t basecall_len);
char *dwell_corrected_overlapper(const int *seq, const int *dwell, int n,
                                 int nkmer, const dwell_model dm);

float sloika_viterbi(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                     float local_pen, int *seq);

float argmax_decoder(const_scrappie_matrix logpost, int *seq);
char *ctc_remove_stays_and_repeats(const int *seq, size_t n, int *pos);

float decode_crf(const_scrappie_matrix trans, int * path);
scrappie_matrix posterior_crf(const_scrappie_matrix trans);
char * crfpath_to_basecall(int const * path, size_t npos, int * pos);

float squiggle_match_viterbi(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen, float minscore,
                             int32_t * path_padded);

float squiggle_match_forward(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen, float minscore);


bool are_bounds_sane(size_t const * low, size_t const * high, size_t nblock, size_t seqlen);
float map_to_sequence_viterbi(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                              float local_pen, int const *seq, size_t seqlen, int * path);
float map_to_sequence_forward(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                              float local_pen, int const *seq, size_t seqlen);
float map_to_sequence_viterbi_banded(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                                     float local_pen, int const *seq, size_t seqlen,
                                     size_t const * poslow, size_t const * poshigh);
float map_to_sequence_forward_banded(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                                     float local_pen, int const *seq, size_t seqlen,
                                     size_t const * poslow, size_t const * poshigh);


#endif                          /* DECODE_H */
