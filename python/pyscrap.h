
void medmad_normalise_array(float *x, size_t n);
raw_table trim_and_segment_raw(raw_table rt, size_t trim_start, size_t trim_end,
                               size_t varseg_chunk, float varseg_thresh);
raw_table trim_raw_by_mad(raw_table rt, size_t chunk_size, float perc);

scrappie_matrix mat_from_array(const float * x, size_t nr, size_t nc);
scrappie_matrix free_scrappie_matrix(scrappie_matrix mat);

// Transducer basecalling
scrappie_matrix nanonet_rgrgr_r94_posterior(const raw_table signal, float min_prob,
                                            float tempW, float tempb, bool return_log);
scrappie_matrix nanonet_rgrgr_r941_posterior(const raw_table signal, float min_prob,
                                             float tempW, float tempb, bool return_log);
scrappie_matrix nanonet_rgrgr_r10_posterior(const raw_table signal, float min_prob,
                                            float tempW, float tempb, bool return_log);
float decode_transducer(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                        float local_pen, int *seq, bool allow_slip);
char *overlapper(const int *seq, size_t n, int nkmer, int *pos);

// RNN-CRF
scrappie_matrix nanonet_rnnrf_r94_transitions(const raw_table signal, float min_prob,
                                              float tempW, float tempb, bool return_log);
float decode_crf(const_scrappie_matrix trans, int * path);
char * crfpath_to_basecall(int const * path, size_t npos, int * pos);
scrappie_matrix posterior_crf(const_scrappie_matrix trans);

// Squiggle generation
scrappie_matrix squiggle_r94(int const * sequence, size_t n, bool transform_units);
scrappie_matrix squiggle_r94_rna(int const * sequence, size_t n, bool transform_units);
scrappie_matrix squiggle_r10(int const * sequence, size_t n, bool transform_units);

// Scrappy Mappy
float squiggle_match_viterbi(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen,
                             float minscore, int32_t * path_padded);
float squiggle_match_forward(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen,
                             float minscore);

// Block-based mapping
bool are_bounds_sane(size_t const * low, size_t const * high,
                     size_t nblock, size_t seqlen);
float map_to_sequence_forward(const_scrappie_matrix logpost,
                              float stay_pen, float skip_pen, float local_pen,
                              int const *seq, size_t seqlen);
float map_to_sequence_forward_banded(const_scrappie_matrix logpost,
                                     float stay_pen, float skip_pen, float local_pen,
                                     int const *seq, size_t seqlen,
                                     size_t const * poslow, size_t const * poshigh);

float map_to_sequence_viterbi(const_scrappie_matrix logpost,
                              float stay_pen, float skip_pen, float local_pen,
                              int const *seq, size_t seqlen, int *path);
float map_to_sequence_viterbi_banded(const_scrappie_matrix logpost,
                                     float stay_pen, float skip_pen, float local_pen,
                                     int const *seq, size_t seqlen,
                                     size_t const * poslow, size_t const * poshigh);

// Misc
int * encode_bases_to_integers(char const * seq, size_t n, size_t state_len);
int get_raw_model_stride_from_string(const char * modelstr);
