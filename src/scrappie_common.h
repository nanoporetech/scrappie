#pragma once
#ifndef SCRAPPIE_COMMON_H
#define SCRAPPIE_COMMON_H

#include "scrappie_structures.h"

raw_table read_trim_and_segment_raw(char *filename, int trim_start, int trim_end, int varseg_chunk, float varseg_thresh);

#endif /* SCRAPPIE_COMMON_H */
