#pragma once
#ifndef NETWORKS_H
#    define NETWORKS_H
#    include <stdbool.h>
#    include "scrappie_structures.h"

enum raw_model_type {
    SCRAPPIE_MODEL_RAW=0,
    SCRAPPIE_MODEL_RGR,
    SCRAPPIE_MODEL_RGRGR_R94,
    SCRAPPIE_MODEL_RGRGR_R95,
    SCRAPPIE_MODEL_INVALID};

typedef scrappie_matrix (*posterior_function_ptr)(const raw_table, float, bool);

enum raw_model_type get_raw_model(const char * modelstr);
const char * raw_model_string(const enum raw_model_type model);
posterior_function_ptr get_posterior_function(const enum raw_model_type model);


//  Events posterior.  Other models via factory function
scrappie_matrix nanonet_posterior(const event_table events, float min_prob,
                                  bool return_log);

#endif    /* NETWORKS_H */
