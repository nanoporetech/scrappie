#pragma once
#ifndef EVENTS_H
#    define EVENTS_H

#    include <hdf5.h>
#    include <stdbool.h>
#    include "scrappie_structures.h"

typedef struct {
    int start, end;
} range_t;

raw_table read_raw(const char *filename, bool scale_to_pA);

range_t trim_raw_by_mad(const raw_table rt, int chunk_size, float proportion);

void write_annotated_events(hid_t hdf5file, const char *readname,
                            const event_table ev, hsize_t chunk_size,
                            int compression_level);
void write_annotated_raw(hid_t hdf5file, const char *readname,
                         const raw_table rt, hsize_t chunk_size,
                         int compression_level);

#endif                          /* EVENTS_H */
