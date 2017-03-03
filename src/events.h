#ifndef EVENTS_H
#define EVENTS_H

#include <hdf5.h>
#include <stdint.h>

typedef struct {
	double start, length;
	double mean, stdv;
	int pos, state;
} event_t;

typedef struct {
	size_t n, start, end;
	event_t * event;
} event_table;


event_table read_events(const char * filename, const char * tablepath);
event_table read_detected_events(const char * filename, int analysis_no, const char * segmentation, int seganalysis_no);
event_table read_albacore_events(const char * filename, int analysis_no, const char * section);

void write_annotated_events(hid_t hdf5file, const char * readname, const event_table ev);

#endif /* EVENTS_H */
