#include <ctype.h>
#include <err.h>
#include <stdlib.h>

#include "scrappie_seq_helpers.h"

static int nbase = 4;

/**  Converts a nucleotide base into integer
 *
 *   Input may be uppercase or (optionally) lowercase.  Ambiguous
 *   bases are treated as errors.
 *
 *   a, A -> 0
 *   c, C -> 1
 *   g, G -> 2
 *   t, T -> 3
 *
 *   @param base  Nucleotide to convert
 *   @param allow_lower  Whether to treat lowercase bases as valid input
 *
 *   @returns integer representing base. -1 if base not recognised
 **/
int base_to_int(char base, bool allow_lower){
    base = allow_lower ? toupper(base) : base;
    switch(base){
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:
            warnx("Unrecognised base %d in read", base);
    }
    return -1;
}


/**  Encode an array of nucleotides into integers
 *
 *   @param seq An array of characters containing ASCII encoded
 *   nucleotides (does not explicitly need to be null-terminated).
 *   @param n Length of array `seq`
 *
 *   @returns Array [n] containing encoding of sequence or NULL if an
 *   invalid base was encountered
 **/
int * encode_bases_to_integers(char const * seq, size_t n, size_t state_len){
    const size_t nstate = n - state_len + 1;

    int * iseq = calloc(nstate, sizeof(int));
    for(size_t i=0 ; i < nstate ; i++){
        int ib = 0;
        for(size_t j=0 ; j < state_len ; j++){
            int newbase = base_to_int(seq[i + j], true);
            if(-1 == newbase){
                free(iseq);
                iseq = NULL;
                break;
            }

            ib *= nbase;
            ib += newbase;
        }
        iseq[i] = ib;
    }

    return iseq;
}
