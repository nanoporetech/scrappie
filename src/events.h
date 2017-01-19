#ifndef EVENTS_H
#define EVENTS_H

#include <hdf5.h>
#include <stdint.h>

typedef struct {
	int start, length;
	double mean, stdv;
	int pos, state;
} event_t;

typedef struct {
	size_t n, start, end;
	event_t * event;
} event_table;

struct _pi { int x1, x2;};

event_table read_events(const char * filename, const char * tablepath, struct _pi index);
event_table read_detected_events(const char * filename, int analysis_no, const char * segmentation);

void write_annotated_events(hid_t hdf5file, const char * readname, const event_table ev);

#endif /* EVENTS_H */
