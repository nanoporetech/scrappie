#pragma once
#ifndef NETWORKS_H
#    define NETWORKS_H
#    include <stdbool.h>
#    include "scrappie_matrix.h"
#    include "scrappie_structures.h"

enum raw_model_type {
    SCRAPPIE_MODEL_RAW=0,
    SCRAPPIE_MODEL_RGR,
    SCRAPPIE_MODEL_RGRGR_R94,
    SCRAPPIE_MODEL_RGRGR_R95,
    SCRAPPIE_MODEL_RNNRF_R94,
    SCRAPPIE_MODEL_INVALID};

typedef scrappie_matrix (*posterior_function_ptr)(const raw_table, float, bool);

enum raw_model_type get_raw_model(const char * modelstr);
const char * raw_model_string(const enum raw_model_type model);
int get_raw_model_stride(const enum raw_model_type model);
posterior_function_ptr get_posterior_function(const enum raw_model_type model);


//  Events posterior.  Other models via factory function
scrappie_matrix nanonet_posterior(const event_table events, float min_prob,
                                  bool return_log);

//  Raw posterior -- for preference, use get_posterior_function
scrappie_matrix nanonet_raw_posterior(const raw_table signal, float min_prob, bool return_log);
scrappie_matrix nanonet_rgr_posterior(const raw_table signal, float min_prob, bool return_log);
scrappie_matrix nanonet_rgrgr_r94_posterior(const raw_table signal, float min_prob, bool return_log);
scrappie_matrix nanonet_rgrgr_r95_posterior(const raw_table signal, float min_prob, bool return_log);
scrappie_matrix nanonet_rnnrf_r94_transitions(const raw_table signal, float min_prob, bool return_log);

//  Squiggle functions
scrappie_matrix dna_squiggle(int const * sequence, size_t n, bool transform_units);

#endif    /* NETWORKS_H */
