#ifndef READ_EVENTS_H
#define READ_EVENTS_H

#include <stdint.h>

typedef struct {
	int start, length;
	double mean, stdv;
} event_t;

typedef struct {
	size_t n, start, end;
	event_t * event;
} event_table;

struct _pi { int x1, x2;};

event_table read_events(const char * filename, const char * tablepath, struct _pi index);
event_table read_detected_events(const char * filename, int analysis_no);

#endif /* READ_EVENTS_H */
