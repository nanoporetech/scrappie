#pragma once
#ifndef SCRAPPIE_SEQ_HELPERS
#define SCRAPPIE_SEQ_HELPERS

#include <stdbool.h>

int base_to_int(char c, bool allow_lower);
int * encode_bases_to_integers(char const * seq, size_t n);

#endif /* SCRAPPIE_SEQ_HELPERS */
