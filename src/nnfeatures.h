#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "fast5_interface.h"
#include "util.h"

Mat_rptr make_features(const event_table evtbl, bool normalise);
Mat_rptr mat_raw(const raw_table signal);

#endif /* FEATURES_H */
