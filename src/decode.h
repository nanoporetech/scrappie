#pragma once
#ifndef DECODE_H
#    define DECODE_H
#    include <stdbool.h>
#    include "fast5_interface.h"
#    include "scrappie_matrix.h"

typedef struct {
    float scale;
    float base_adj[4];
} dwell_model;

float decode_transducer(const scrappie_matrix logpost, float skip_pen, int *seq,
                        bool use_slip);
char *overlapper(const int *seq, int n, int nkmer, int *pos);
char *homopolymer_dwell_correction(const event_table et, const int *seq,
                                   size_t nstate, size_t basecall_len);
char *dwell_corrected_overlapper(const int *seq, const int *dwell, int n,
                                 int nkmer, const dwell_model dm);

#endif                          /* DECODE_H */
