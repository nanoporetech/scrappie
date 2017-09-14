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
char *overlapper(const int *seq, int n, int nkmer, int *pos);
char *homopolymer_dwell_correction(const event_table et, const int *seq,
                                   size_t nstate, size_t basecall_len);
char *dwell_corrected_overlapper(const int *seq, const int *dwell, int n,
                                 int nkmer, const dwell_model dm);

float sloika_viterbi(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                     float local_pen, int *seq);

float argmax_decoder(const_scrappie_matrix logpost, int *seq);
char *ctc_remove_stays_and_repeats(const int *seq, int n, int *pos);

#endif                          /* DECODE_H */
