#ifndef DECODE_H
#define DECODE_H
#include <stdbool.h>
#include "util.h"

float decode_transducer(const Mat_rptr logpost, float skip_pen, int * seq, bool use_slip);
char * overlapper(const int * seq, int n, int nkmer, int *pos);
char * dwell_corrected_overlapper(const int * seq, const int * dwell, int n, int nkmer);

#endif  /* DECODE_H */
