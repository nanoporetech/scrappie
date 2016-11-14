#ifndef FEATURES_H
#define FEATURES_H
#include <stdbool.h>
#include <stdint.h>
#include "read_events.h"
#include "util.h"

Mat * make_features(const event_table evtbl, bool normalise);

#endif /* FEATURES_H */
