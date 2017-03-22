#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "fast5_interface.h"
#include "util.h"

Mat_rptr make_features(const event_table evtbl, bool normalise);

#endif /* FEATURES_H */
