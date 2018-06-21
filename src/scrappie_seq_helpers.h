#pragma once
#ifndef SCRAPPIE_SEQ_HELPERS
#define SCRAPPIE_SEQ_HELPERS

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    size_t n;
    char * seq;
    char * name;
} scrappie_seq_t;


int base_to_int(char c, bool allow_lower);
int * encode_bases_to_integers(char const * seq, size_t n, size_t state_len);
scrappie_seq_t read_sequence_from_fasta(char const * filename);
int repeatblock(int b, int nrep);
int kmerlength_fromnblocks(int n);

#endif /* SCRAPPIE_SEQ_HELPERS */
