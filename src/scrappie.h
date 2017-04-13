#ifndef SCRAPPIE_H
#define SCRAPPIE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <stdbool.h>
#include <stdlib.h>



/*  Structure definitions from fast5_interface.h  */
typedef struct {
	double start;
	float length;
	float mean, stdv;
	int pos, state;
} event_t;

typedef struct {
	size_t n, start, end;
	event_t * event;
} event_table;

typedef struct {
	size_t n, start, end;
	float * raw;
} raw_table;


/*  Matrix definitions from util.h  */
typedef struct {
        int nr, nrq, nc;
        union {
                __m128 * v;
                float * f;
        } data;
} _Mat;
typedef _Mat * scrappie_matrix;


scrappie_matrix nanonet_posterior(const event_table events, float min_prob, bool return_log);
scrappie_matrix nanonet_raw_posterior(const raw_table signal, float min_prob, bool return_log);

scrappie_matrix free_scrappie_matrix(scrappie_matrix mat);


#ifdef __cplusplus
}
#endif

#endif /* SCRAPPIE_H */
