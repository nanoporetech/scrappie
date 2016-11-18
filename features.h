#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "read_events.h"
#include "util.h"

Mat_rptr make_features(const event_table evtbl, int trim, bool normalise);

#endif /* FEATURES_H */
