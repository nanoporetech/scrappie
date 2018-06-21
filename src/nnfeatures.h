#pragma once
#ifndef FEATURES_H
#    define FEATURES_H
#    include <stdbool.h>
#    include "scrappie_structures.h"
#    include "scrappie_matrix.h"

scrappie_matrix nanonet_features_from_events(const event_table evtbl,
                                             bool normalise);
scrappie_matrix nanonet_features_from_raw(const raw_table signal);
scrappie_matrix deltasample_features_from_raw(const raw_table signal, float shift,
                                              float scale, float sdthresh);

#endif /* FEATURES_H */
