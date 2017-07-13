#ifndef SCRAPPIE_MATRIX_UTIL
#    define SCRAPPIE_MATRIX_UTIL

#    include <scrappie_matrix.h>
#    include <stdio.h>

int write_scrappie_matrix(FILE * fh, const scrappie_matrix mat);
scrappie_matrix read_scrappie_matrix(FILE * fh);

scrappie_matrix random_scrappie_matrix(int nr, int nc, float lower,
                                       float upper);

#endif                          /* SCRAPPIE_MATRIX_UTIL */
