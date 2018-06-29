#ifndef SCRAPPIE_MATRIX_UTIL
#    define SCRAPPIE_MATRIX_UTIL

#    include <scrappie_matrix.h>
#    include <stdio.h>

size_t write_scrappie_matrix(const char * fn, const_scrappie_matrix mat);
size_t write_scrappie_matrix_to_handle(FILE * fh, const_scrappie_matrix mat);
scrappie_matrix read_scrappie_matrix_from_handle(FILE * fh);
scrappie_matrix read_scrappie_matrix(char const * fn);

scrappie_matrix random_scrappie_matrix(size_t nr, size_t nc, float lower,
                                       float upper);

#endif                          /* SCRAPPIE_MATRIX_UTIL */
