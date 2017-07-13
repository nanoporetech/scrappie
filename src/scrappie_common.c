#include "scrappie_assert.h"
#include "fast5_interface.h"
#include "scrappie_common.h"

raw_table read_trim_and_segment_raw(char *filename, int trim_start, int trim_end, int varseg_chunk, float varseg_thresh) {
    const raw_table zeroRawTable = { 0 };

    RETURN_NULL_IF(NULL == filename, zeroRawTable);

    raw_table rt = read_raw(filename, true);
    RETURN_NULL_IF(NULL == rt.raw, zeroRawTable);

    const size_t nsample = rt.end - rt.start;
    if (nsample <= trim_end + trim_start) {
        warnx("Too few samples in %s to call (%zu, originally %lu).", filename,
              nsample, rt.n);
        free(rt.raw);
        return zeroRawTable;
    }

    rt.start += trim_start;
    rt.end -= trim_end;

    range_t segmentation =
        trim_raw_by_mad(rt, varseg_chunk, varseg_thresh);
    rt.start = segmentation.start;
    rt.end = segmentation.end;

    return rt;
}
