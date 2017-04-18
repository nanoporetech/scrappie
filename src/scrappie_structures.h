#ifndef SCRAPPIE_STRUCTURES_H
#define SCRAPPIE_STRUCTURES_H

typedef struct {
        double start;
        float length;
        float mean, stdv;
        int pos, state;
} event_t;

typedef struct {
        unsigned int n, start, end;
        event_t * event;
} event_table;

typedef struct {
        unsigned int n, start, end;
        float * raw;
} raw_table;


#endif /* SCRAPPIE_DATA_H */
