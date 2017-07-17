#pragma once
#ifndef NETWORKS_H
#    define NETWORKS_H
#    include <stdbool.h>
#    include "scrappie_structures.h"

scrappie_matrix nanonet_posterior(const event_table events, float min_prob,
                                  bool return_log);
scrappie_matrix nanonet_raw_posterior(const raw_table signal, float min_prob,
                                      bool return_log);
scrappie_matrix nanonet_rgr_posterior(const raw_table signal, float min_prob,
                                      bool return_log);

#endif                          /* NETWORKS_H */
