#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "fast5_interface.h"
#include "scrappie_matrix.h"

scrappie_matrix nanonet_features_from_events(const event_table evtbl, bool normalise);
scrappie_matrix nanonet_features_from_raw(const raw_table signal);

#endif /* FEATURES_H */
