#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "fast5_interface.h"
#include "util.h"

scrappie_matrix make_features(const event_table evtbl, bool normalise);
scrappie_matrix mat_raw(const raw_table signal);

#endif /* FEATURES_H */
