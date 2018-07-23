#include <stdio.h>

#include "decode.h"
#include "scrappie_stdlib.h"
#include "util.h"

#define NBASE 4
#define BIG_FLOAT 1.e30f

#ifndef __SSE2__
#    error "Compilation of function decode_transducer requires a processor that supports at least SSE2"
#endif

#ifndef __SSE4_1__
/**  Multiply two vectors of 32 bit integers together
 *
 *   Emulates the SSE4.1 instruction _mm_mullo_epi32 on hardware that only supports
 *   SSE2.  See https://software.intel.com/en-us/forums/intel-c-compiler/topic/288768
 *
 *   @param x first vector to multiply
 *   @param y second vector to multiply
 *
 *   @returns vector of integers containing the lower 32 bits of x * y
 **/
static inline __m128i __attribute__((__gnu_inline__, __always_inline__)) _mm_mullo_epi32(const __m128i x, const __m128i y) {
    __m128i tmp1 = _mm_mul_epu32(x, y);
    __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(x, 4), _mm_srli_si128(y, 4));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
                              _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));

}
#endif

float viterbi_backtrace(float const *score, size_t n, const_scrappie_imatrix traceback, int * seq){
    RETURN_NULL_IF(NULL == score, NAN);
    RETURN_NULL_IF(NULL == seq, NAN);

    const size_t nblock = traceback->nc;
    for(size_t i=0 ; i < nblock ; i++){
        // Initialise entries to stay
        seq[i] = -1;
    }

    int last_state = argmaxf(score, n);
    float logscore = score[last_state];
    for(size_t i=0 ; i < nblock ; i++){
        const size_t ri = nblock - i - 1;
        const int state = traceback->data.f[ri * traceback->stride + last_state];
        if(state >= 0){
            seq[ri] = last_state;
            last_state = state;
        }
    }

    return logscore;
}

float viterbi_local_backtrace(float const *score, size_t n, const_scrappie_imatrix traceback, int * seq){
    RETURN_NULL_IF(NULL == score, NAN);
    RETURN_NULL_IF(NULL == seq, NAN);

    const size_t nblock = traceback->nc;
    for(size_t i=0 ; i <= nblock ; i++){
        // Initialise entries to stay
        seq[i] = -1;
    }

    int last_state = argmaxf(score, n + 2);
    float logscore = score[last_state];
    for(size_t i=0 ; i < nblock ; i++){
        const size_t ri = nblock - i - 1;
        const int state = traceback->data.f[ri * traceback->stride + last_state];
        if(state >= 0){
            seq[ri + 1] = last_state;
            last_state = state;
        }
    }
    seq[0] = last_state;

    //  Transcode start to stay
    for(size_t i=0 ; i < nblock ; i++){
        if(seq[i] == n){
            seq[i] = -1;
        } else {
            break;
        }
    }
    //  Transcode end to stay
    for(int i=nblock ; i >= 0 ; i--){
        if(seq[i] == n + 1){
            seq[i] = -1;
        } else {
            break;
        }
    }

    return logscore;
}

float argmax_decoder(const_scrappie_matrix logpost, int *seq) {
    RETURN_NULL_IF(NULL == logpost, NAN);
    RETURN_NULL_IF(NULL == seq, NAN);

    const int nblock = logpost->nc;
    const int nstate = logpost->nr;
    assert(nstate > 0);
    const int stride = logpost->stride;
    assert(stride > 0);
    int offset;

    float logscore = 0;
    int imax;
    for (int blk = 0; blk < nblock; blk++) {
        offset = blk * stride;
        imax = argmaxf(logpost->data.f + offset, nstate);
        logscore += logpost->data.f[offset + imax];
        seq[blk] = (imax == nstate - 1) ? -1 : imax;
    }

    return logscore;
}

float decode_transducer(const_scrappie_matrix logpost, float stay_pen, float skip_pen, float local_pen, int *seq,
                        bool allow_slip) {
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);

    const int nblock = logpost->nc;
    const int nstate = logpost->nr;
    const int nhistory = nstate - 1;
    assert((nhistory % 4) == 0);
    const int32_t nhistoryq = nhistory / 4;
    const __m128i nhistoryqv = _mm_set1_epi32(nhistoryq);
    assert((nhistoryq % 4) == 0);
    const int32_t nhistoryqq = nhistoryq / 4;
    const __m128i nhistoryqqv = _mm_set1_epi32(nhistoryqq);
    assert((nhistoryqq % 4) == 0);
    const int32_t nhistoryqqq = nhistoryqq / 4;
    const __m128i nhistoryqqqv = _mm_set1_epi32(nhistoryqqq);
    if (allow_slip) {
        assert((nhistoryqqq % 4) == 0);
    }
    //  Forwards memory + traceback
    scrappie_matrix score = make_scrappie_matrix(nhistory + 2, 1);
    scrappie_matrix prev_score = make_scrappie_matrix(nhistory + 2, 1);
    scrappie_matrix tmp = make_scrappie_matrix(nhistory, 1);
    scrappie_imatrix itmp = make_scrappie_imatrix(nhistory, 1);
    scrappie_imatrix traceback = make_scrappie_imatrix(nhistory + 2, nblock);
    if(NULL == score || NULL == prev_score || NULL == tmp || NULL == itmp || NULL == traceback){
        goto cleanup;
    }

    //  Initialise
    for (int i = 0; i < nhistoryq; i++) {
        score->data.v[i] = _mm_set1_ps(-BIG_FLOAT);
    }
    score->data.f[nhistory] = 0.0f;
    score->data.f[nhistory + 1] = -BIG_FLOAT;

    //  Forwards Viterbi iteration
    for (int blk = 0; blk < nblock; blk++) {
        const size_t offsetTq = blk * traceback->nrq;
        const size_t offsetT = offsetTq * 4;
        const size_t offsetPq = blk * logpost->nrq;
        const size_t offsetP = offsetPq * 4;
        // Swap score and previous score
        {
            scrappie_matrix tmptr = score;
            score = prev_score;
            prev_score = tmptr;
        }

        // Stay
        const __m128 stay_m128 =
            _mm_set1_ps(logpost->data.f[offsetP + nhistory] - stay_pen);
        const __m128i negone_m128i = _mm_set1_epi32(-1);
        for (int i = 0; i < nhistoryq; i++) {
            // Traceback for stay is negative
            score->data.v[i] = prev_score->data.v[i] + stay_m128;
            traceback->data.v[offsetTq + i] = negone_m128i;
        }

        // Step
        // Following three loops find maximum over suffix and record index
        for (int i = 0; i < nhistoryqq; i++) {
            tmp->data.v[i] = prev_score->data.v[i];
            itmp->data.v[i] = _mm_setzero_si128();
        }
        for (int r = 1; r < NBASE; r++) {
            const size_t offset = r * nhistoryqq;
            const __m128i itmp_fill = _mm_set1_epi32(r);
            for (int i = 0; i < nhistoryqq; i++) {
                __m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i],
                                                             prev_score->data.
                                                             v[offset + i]));
                tmp->data.v[i] =
                    _mm_max_ps(tmp->data.v[i], prev_score->data.v[offset + i]);
                itmp->data.v[i] =
                    _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
                                 _mm_and_si128(mask, itmp_fill));
            }
        }
        const __m128i c0123_m128i = _mm_setr_epi32(0, 1, 2, 3);
        for (int i = 0; i < nhistoryqq; i++) {
            itmp->data.v[i] =
                _mm_add_epi32(_mm_mullo_epi32(itmp->data.v[i], nhistoryqv),
                              _mm_add_epi32(c0123_m128i,
                                            _mm_set1_epi32(i * 4)));
        }

        for (int pref = 0; pref < nhistoryq; pref++) {
            const size_t i = pref;
            const __m128 step_score =
                logpost->data.v[offsetPq + i] + _mm_set1_ps(tmp->data.f[pref]);
            __m128i mask =
                _mm_castps_si128(_mm_cmplt_ps(score->data.v[i], step_score));
            score->data.v[i] = _mm_max_ps(score->data.v[i], step_score);
            traceback->data.v[offsetTq + i] =
                _mm_or_si128(_mm_andnot_si128
                             (mask, traceback->data.v[offsetTq + i]),
                             _mm_and_si128(mask,
                                           _mm_set1_epi32(itmp->data.f[pref])));
        }

        // Skip
        const __m128 skip_penv = _mm_set1_ps(skip_pen);
        for (int i = 0; i < nhistoryqqq; i++) {
            tmp->data.v[i] = prev_score->data.v[i];
            itmp->data.v[i] = _mm_setzero_si128();
        }
        for (int r = 1; r < NBASE * NBASE; r++) {
            const size_t offset = r * nhistoryqqq;
            const __m128i itmp_fill = _mm_set1_epi32(r);
            for (int i = 0; i < nhistoryqqq; i++) {
                __m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i],
                                                             prev_score->data.
                                                             v[offset + i]));
                tmp->data.v[i] =
                    _mm_max_ps(tmp->data.v[i], prev_score->data.v[offset + i]);
                itmp->data.v[i] =
                    _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
                                 _mm_and_si128(mask, itmp_fill));
            }
        }
        for (int i = 0; i < nhistoryqqq; i++) {
            itmp->data.v[i] =
                _mm_add_epi32(_mm_mullo_epi32(itmp->data.v[i], nhistoryqqv),
                              _mm_add_epi32(c0123_m128i,
                                            _mm_set1_epi32(i * 4)));
        }
        for (int pref = 0; pref < nhistoryqq; pref++) {
            for (int i = 0; i < NBASE; i++) {
                const size_t oi = pref * NBASE + i;
                // This cycling through prefixes
                const __m128 skip_score = logpost->data.v[offsetPq + oi]
                    + _mm_set1_ps(tmp->data.f[pref])
                    - skip_penv;
                __m128i mask =
                    _mm_castps_si128(_mm_cmplt_ps
                                     (score->data.v[oi], skip_score));
                score->data.v[oi] = _mm_max_ps(score->data.v[oi], skip_score);
                traceback->data.v[offsetTq + oi] =
                    _mm_or_si128(_mm_andnot_si128
                                 (mask, traceback->data.v[offsetTq + oi]),
                                 _mm_and_si128(mask,
                                               _mm_set1_epi32(itmp->
                                                              data.f[pref])));
            }
        }

        // Slip
        if (allow_slip) {
            const int32_t nhistoryqqqq = nhistoryqqq / 4;
            const __m128 slip_penv = _mm_set1_ps(2.0 * skip_pen);
            for (int i = 0; i < nhistoryqqqq; i++) {
                tmp->data.v[i] = prev_score->data.v[i];
                itmp->data.v[i] = _mm_setzero_si128();
            }
            for (int r = 1; r < NBASE * NBASE * NBASE; r++) {
                const size_t offset = r * nhistoryqqqq;
                const __m128i itmp_fill = _mm_set1_epi32(r);
                for (int i = 0; i < nhistoryqqqq; i++) {
                    __m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i],
                                                                 prev_score->
                                                                 data.v[offset +
                                                                        i]));
                    tmp->data.v[i] =
                        _mm_max_ps(tmp->data.v[i],
                                   prev_score->data.v[offset + i]);
                    itmp->data.v[i] =
                        _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
                                     _mm_and_si128(mask, itmp_fill));
                }
            }
            for (int i = 0; i < nhistoryqqqq; i++) {
                itmp->data.v[i] =
                    _mm_add_epi32(_mm_mullo_epi32
                                  (itmp->data.v[i], nhistoryqqqv),
                                  _mm_add_epi32(c0123_m128i,
                                                _mm_set1_epi32(i * 4)));
            }
            for (int pref = 0; pref < nhistoryqqq; pref++) {
                for (int i = 0; i < NBASE * NBASE; i++) {
                    const size_t oi = pref * NBASE * NBASE + i;
                    // This cycling through prefixes
                    const __m128 skip_score = logpost->data.v[offsetPq + oi]
                        + _mm_set1_ps(tmp->data.f[pref])
                        - slip_penv;
                    __m128i mask =
                        _mm_castps_si128(_mm_cmplt_ps
                                         (score->data.v[oi], skip_score));
                    score->data.v[oi] =
                        _mm_max_ps(score->data.v[oi], skip_score);
                    traceback->data.v[offsetTq + oi] =
                        _mm_or_si128(_mm_andnot_si128
                                     (mask, traceback->data.v[offsetTq + oi]),
                                     _mm_and_si128(mask,
                                                   _mm_set1_epi32(itmp->data.f
                                                                  [pref])));
                }
            }
        }

        // Remain in start state (stay or local penalty)
        score->data.f[nhistory] = prev_score->data.f[nhistory]
                                + fmaxf(-local_pen, logpost->data.f[offsetP + nhistory] - stay_pen);
        traceback->data.f[offsetT + nhistory] = nhistory;
        // Exit start state
        for(int hst=0 ; hst < nhistory ; hst++){
            const float scoref = prev_score->data.f[nhistory] + logpost->data.f[offsetP + hst];
            if(scoref > score->data.f[hst]){
                score->data.f[hst] = scoref;
                traceback->data.f[offsetT + hst] = nhistory;
            }
        }

        // Remain in end state (stay or local penalty)
        score->data.f[nhistory + 1] = prev_score->data.f[nhistory + 1]
                                    + fmax(-local_pen, logpost->data.f[offsetP + nhistory] - stay_pen);
        traceback->data.f[offsetT + nhistory + 1] = nhistory + 1;
        // Enter end state
        for(int hst=0 ; hst < nhistory ; hst++){
            const float scoref = prev_score->data.f[hst] - local_pen;
            if(scoref > score->data.f[nhistory + 1]){
                score->data.f[nhistory + 1] = scoref;
                traceback->data.f[offsetT + nhistory + 1] = hst;
            }
        }
    }

    //  Viterbi traceback
    logscore = viterbi_local_backtrace(score->data.f, nhistory, traceback, seq);

    assert(validate_ivector(seq, nblock, -1, nhistory - 1, __FILE__, __LINE__));

cleanup:
    traceback = free_scrappie_imatrix(traceback);
    itmp = free_scrappie_imatrix(itmp);
    tmp = free_scrappie_matrix(tmp);
    prev_score = free_scrappie_matrix(prev_score);
    score = free_scrappie_matrix(score);

    return logscore;
}

int overlap(int k1, int k2, int nkmer) {
    // Neither k1 nor k2 can be stays
    assert(k1 >= 0);
    assert(k2 >= 0);

    int kmer_mask = nkmer - 1;
    int overlap = 0;
    do {
        kmer_mask >>= 2;
        k1 &= kmer_mask;
        k2 >>= 2;
        overlap += 1;
    } while (k1 != k2);

    return overlap;
}

size_t position_highest_bit(size_t x) {
    size_t i = 0;
    for (; x != 0; i++, x >>= 1) ;
    return i;
}

size_t first_nonnegative(const int *seq, size_t n) {
    RETURN_NULL_IF(NULL == seq, n);
    size_t st;
    for (st = 0; st < n && seq[st] < 0; st++) ;
    return st;
}

bool iskmerhomopolymer(int kmer, int klen) {
    const int b = kmer & 3;

    for (int k = 1; k < klen; k++) {
        kmer >>= 2;
        if (b != (kmer & 3)) {
            return false;
        }
    }

    return true;
}

const char base_lookup[4] = { 'A', 'C', 'G', 'T' };


// This method assumes a model which outputs single bases
char *ctc_remove_stays_and_repeats(const int *seq, size_t n, int *pos) {
    RETURN_NULL_IF(NULL == seq, NULL);
    RETURN_NULL_IF(NULL == pos, NULL);

    //  Determine length of final sequence
    int length = 0;
    if (seq[0] >= 0) { length += 1; }
    for (size_t blk = 1; blk < n; blk++) {
        if (seq[blk] >= 0 && seq[blk - 1] != seq[blk]) {
            length += 1;
        }
    }

    // Initialise basespace sequence with terminating null
    char *bases = calloc(length + 1, sizeof(char));
    RETURN_NULL_IF(NULL == bases, NULL);

    int loc = -1;
    int prev = -2;

    for (size_t blk = 0; blk < n; blk++) {
        int this = seq[blk];
        if (this >= 0 && this != prev) {
            bases[loc] = base_lookup[this];
            prev = this;
            loc += 1;
        }
        if (NULL != pos) {
            pos[blk] = loc;
        }
     }

    return bases;
}

char *overlapper(const int *seq, size_t n, int nkmer, int *pos) {
    RETURN_NULL_IF(NULL == seq, NULL);
    const size_t kmer_len = position_highest_bit(nkmer) / 2;

    //  Determine length of final sequence
    size_t length = kmer_len;
    // Find first non-stay
    const size_t st = first_nonnegative(seq, n);
    RETURN_NULL_IF(st == n, NULL);

    int kprev = seq[st];
    for (size_t k = st + 1; k < n; k++) {
        if (seq[k] < 0) {
            // Short-cut stays
            continue;
        }
        assert(seq[k] >= 0);
        length += overlap(kprev, seq[k], nkmer);
        kprev = seq[k];
        assert(kprev >= 0);
    }

    // Initialise basespace sequence with terminating null
    char *bases = calloc(length + 1, sizeof(char));
    RETURN_NULL_IF(NULL == bases, NULL);

    // Fill with first kmer
    for (size_t kmer = seq[st], k = 1; k <= kmer_len; k++) {
        size_t b = kmer & 3;
        kmer >>= 2;
        bases[kmer_len - k] = base_lookup[b];
    }

    if(NULL != pos){
        // Initial pos array if required -- start at beginning
        pos[0] = 0;
    }
    for (size_t last_idx = kmer_len - 1, kprev = seq[st], k = st + 1; k < n; k++) {
        if (seq[k] < 0) {
            // Short-cut stays
            if (NULL != pos) {
                pos[k] = pos[k - 1];
            }
            continue;
        }
        int ol = overlap(kprev, seq[k], nkmer);
        if (NULL != pos) {
            pos[k] = pos[k - 1] + ol;
        }
        kprev = seq[k];

        for (size_t kmer = seq[k], i = 0; i < ol; i++) {
            size_t b = kmer & 3;
            kmer >>= 2;
            bases[last_idx + ol - i] = base_lookup[b];
        }
        last_idx += ol;
    }

    return bases;
}

int calibrated_dwell(int hdwell, int inhomo, const dwell_model dm) {
    const int b = inhomo & 3;
    return (int)roundf(((float)hdwell - dm.base_adj[b]) / dm.scale);
}

char *dwell_corrected_overlapper(const int *seq, const int *dwell, int n,
                                 int nkmer, const dwell_model dm) {
    RETURN_NULL_IF(NULL == seq, NULL);
    RETURN_NULL_IF(NULL == dwell, NULL);
    const int kmer_len = position_highest_bit(nkmer) / 2;

    //  Determine length of final sequence
    int length = kmer_len;
    //  Find first non-stay
    const int st = first_nonnegative(seq, n);
    assert(st != n);
    int kprev = seq[st];
    int inhomo = -1;
    int hdwell = 0;
    for (int k = st + 1; k < n; k++) {
        /* Simple state machine tagged by inhomo
         * inhomo < 0  --  not in a homopolymer
         * inhomo >= 0  -- in homopolymer and value is homopolymer state
         */
        if (seq[k] < 0) {
            // State is stay.  Short circuit rest of logic
            if (inhomo >= 0) {
                // Accumate dwell if in homopolymer
                hdwell += dwell[k];
            }
            continue;
        }

        if (seq[k] == inhomo) {
            // Not stay but still in same homopolymer
            hdwell += dwell[k];
            continue;
        }

        if (inhomo >= 0) {
            // Changed state.  Leave homopolymer
            length += calibrated_dwell(hdwell, inhomo, dm);
            inhomo = -1;
            hdwell = 0;
        }

        assert(seq[k] >= 0);
        length += overlap(kprev, seq[k], nkmer);
        kprev = seq[k];
        assert(kprev >= 0);

        if (iskmerhomopolymer(kprev, kmer_len)) {
            // Entered a new homopolymer
            inhomo = kprev;
            hdwell = dwell[k];
        }
    }

    if (inhomo >= 0) {
        //  Correction for final homopolymer
        length += calibrated_dwell(hdwell, inhomo, dm);
    }
    // Initialise basespace sequence with terminating null
    char *bases = calloc(length + 1, sizeof(char));
    // Fill with first kmer
    for (int kmer = seq[st], k = 1; k <= kmer_len; k++) {
        int b = kmer & 3;
        kmer >>= 2;
        bases[kmer_len - k] = base_lookup[b];
    }

    int last_idx = kmer_len - 1;
    inhomo = -1;
    hdwell = 0;
    for (int kprev = seq[st], k = st + 1; k < n; k++) {
        if (seq[k] < 0) {
            // State is stay.
            if (inhomo >= 0) {
                // Accumate dwell if in homopolymer
                hdwell += dwell[k];
            }
            continue;
        }

        if (seq[k] == inhomo) {
            // Not stay but still in same homopolymer
            hdwell += dwell[k];
            continue;
        }

        if (inhomo >= 0) {
            // Changed state.  Leave homopolymer
            int hlen = calibrated_dwell(hdwell, inhomo, dm);
            char hbase = base_lookup[inhomo & 3];
            for (int i = 0; i < hlen; i++, last_idx++) {
                bases[last_idx + 1] = hbase;
            }
            inhomo = -1;
            hdwell = 0;
        }

        int ol = overlap(kprev, seq[k], nkmer);
        kprev = seq[k];

        for (int kmer = seq[k], i = 0; i < ol; i++) {
            int b = kmer & 3;
            kmer >>= 2;
            bases[last_idx + ol - i] = base_lookup[b];
        }
        last_idx += ol;

        if (iskmerhomopolymer(kprev, kmer_len)) {
            // Entered a new homopolymer
            inhomo = kprev;
            hdwell += dwell[k];
        }
    }

    if (inhomo >= 0) {
        //  Correction for final homopolymer
        int hlen = calibrated_dwell(hdwell, inhomo, dm);
        char hbase = base_lookup[inhomo & 3];
        for (int i = 0; i < hlen; i++, last_idx++) {
            bases[last_idx] = hbase;
        }
    }
    if (last_idx + 1 != length) {
        printf("last_idx %d length %d\n\n", last_idx, length);
        assert(last_idx + 1 == length);
    }

    return bases;
}

char *homopolymer_dwell_correction(const event_table et, const int *seq,
                                   size_t nstate, size_t basecall_len) {
    RETURN_NULL_IF(NULL == et.event, NULL);
    RETURN_NULL_IF(NULL == seq, NULL);

    const int nev = et.end - et.start;
    const int evoffset = et.start;
    assert(et.event[nev + evoffset - 1].start >= et.event[evoffset].start);

    int *dwell = calloc(nev, sizeof(int));
    RETURN_NULL_IF(NULL == dwell, NULL);

    for (int ev = 0; ev < nev; ev++) {
        dwell[ev] = et.event[ev + evoffset].length;
    }

    /* Calibrate scaling factor for homopolymer estimation.
     * Simple mean of the dwells of all 'step' movements in
     * the basecall.  Steps within homopolymers are ignored.
     * A more complex calibration could be used.
     */
    int tot_step_dwell = 0;
    int nstep = 0;
    for (int ev = 0, ppos = -2, evdwell = 0, pstate = -1; ev < nev; ev++) {
        // Sum over dwell of all steps excluding those within homopolymers
        if (et.event[ev + evoffset].pos == ppos) {
            // Stay. Accumulate dwell
            evdwell += dwell[ev];
            continue;
        }

        if (et.event[ev + evoffset].pos == ppos + 1
            && et.event[ev + evoffset].state != pstate) {
            // Have a step that is not within a homopolymer
            tot_step_dwell += evdwell;
            nstep += 1;
        }

        evdwell = dwell[ev];
        ppos = et.event[ev + evoffset].pos;
        pstate = et.event[ev + evoffset].state;
    }

    // Estimate of scale with a prior with weight equal to a single observation.
    const float start_delta = (float)(et.event[nev + evoffset - 1].start
                                    - et.event[evoffset          ].start);
    const float prior_scale =
        (et.event[nev + evoffset - 1].length + start_delta) / (float)basecall_len;
    const float homo_scale = (prior_scale + tot_step_dwell) / (1.0 + nstep);
    const dwell_model dm = { homo_scale, {0.0f, 0.0f, 0.0f, 0.0f} };

    char *newbases =
        dwell_corrected_overlapper(seq, dwell, nev, nstate - 1, dm);

    free(dwell);

    return newbases;
}

void colmaxf(float * x, int nr, int nc, int * idx){
    assert(nr > 0);
    assert(nc > 0);
    RETURN_NULL_IF(NULL == x,);
    RETURN_NULL_IF(NULL == idx,);

    for(int r=0 ; r < nr ; r++){
        // Initialise
        idx[r] = 0;
    }

    for(int c=1 ; c < nc ; c++){
        const size_t offset2 = c * nr;
        for(int r=0 ; r<nr ; r++){
            if(x[offset2 + r] > x[idx[r] * nr + r]){
                idx[r] = c;
            }
        }
    }
}

float sloika_viterbi(const_scrappie_matrix logpost, float stay_pen, float skip_pen, float local_pen, int *seq){
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);

    const int nbase = 4;
    const size_t nblock = logpost->nc;
    const size_t nst = logpost->nr;
    const size_t nhst = nst - 1;
    const size_t nstep = nbase;
    const size_t nskip = nbase * nbase;
    assert(nhst % nstep == 0);
    assert(nhst % nskip == 0);
    const size_t step_rem = nhst / nstep;
    const size_t skip_rem = nhst / nskip;

    float * cscore = calloc(nhst + 2, sizeof(float));
    float * pscore = calloc(nhst + 2, sizeof(float));
    int * step_idx = calloc(step_rem, sizeof(int));
    int * skip_idx = calloc(skip_rem, sizeof(int));
    scrappie_imatrix traceback = make_scrappie_imatrix(nhst + 2, nblock);
    if(NULL != cscore && NULL != pscore && NULL != step_idx && NULL != skip_idx && NULL != traceback){
        // Initialise -- must begin in start state
        for(size_t i=0 ; i < (nhst + 2) ; i++){
            cscore[i] = -BIG_FLOAT;
        }
        cscore[nhst] = 0.0f;

        //  Forwards Viterbi
        for(size_t i=0 ; i < nblock ; i++){
            const size_t lpoffset = i * logpost->stride;
            const size_t toffset = i * traceback->stride;
            {  // Swap vectors
                float * tmp = pscore;
                pscore = cscore;
                cscore = tmp;
            }

            //  Step indices
            colmaxf(pscore, step_rem, nstep, step_idx);
            //  Skip indices
            colmaxf(pscore, skip_rem, nskip, skip_idx);

            // Update score for step and skip
            for(size_t hst=0 ; hst < nhst ; hst++){
                size_t step_prefix = hst / nstep;
                size_t skip_prefix = hst / nskip;
                size_t step_hst = step_prefix + step_idx[step_prefix] * step_rem;
                size_t skip_hst = skip_prefix + skip_idx[skip_prefix] * skip_rem;

                float step_score = pscore[step_hst];
                float skip_score = pscore[skip_hst] - skip_pen;
                if(step_score > skip_score){
                    // Arbitrary assumption here!  Should be >= ?
                    cscore[hst] = step_score;
                    traceback->data.f[toffset + hst] = step_hst;
                } else {
                    cscore[hst] = skip_score;
                    traceback->data.f[toffset + hst] = skip_hst;
                }
                cscore[hst] += logpost->data.f[lpoffset + hst];
            }

            // Stay
            for(size_t hst=0 ; hst < nhst ; hst++){
                const float score = pscore[hst] + logpost->data.f[lpoffset + nhst] - stay_pen;
                if(score > cscore[hst]){
                    // Arbitrary assumption here!  Should be >= ?
                    cscore[hst] = score;
                    traceback->data.f[toffset + hst] = -1;
                }
            }

            // Remain in start state -- local penalty or stay
            cscore[nhst] = pscore[nhst] + fmaxf(-local_pen, logpost->data.f[lpoffset + nhst] - stay_pen);
            traceback->data.f[toffset + nhst] = nhst;
            // Exit start state
            for(size_t hst=0 ; hst < nhst ; hst++){
                const float score = pscore[nhst] + logpost->data.f[lpoffset + hst];
                if(score > cscore[hst]){
                    cscore[hst] = score;
                    traceback->data.f[toffset + hst] = nhst;
                }
            }

            // Remain in end state -- local penalty or stay
            cscore[nhst + 1] = pscore[nhst + 1] + fmaxf(-local_pen, logpost->data.f[lpoffset + nhst] - stay_pen);
            traceback->data.f[toffset + nhst + 1] = nhst + 1;
            // Enter end state
            for(size_t hst=0 ; hst < nhst ; hst++){
                const float score = pscore[hst] - local_pen;
                if(score > cscore[nhst + 1]){
                    cscore[nhst + 1] = score;
                    traceback->data.f[toffset + nhst + 1] = hst;
                }
            }

        }

        logscore = viterbi_local_backtrace(cscore, nhst, traceback, seq);
    }

    traceback = free_scrappie_imatrix(traceback);
    free(skip_idx);
    free(step_idx);
    free(pscore);
    free(cscore);

    return logscore;
}

float decode_crf(const_scrappie_matrix trans, int * path){
    RETURN_NULL_IF(NULL == trans, NAN);
    RETURN_NULL_IF(NULL == path, NAN);

    const size_t nblk = trans->nc;
    const size_t nstate = roundf(sqrtf((float)trans->nr));
    assert(nstate * nstate == trans->nr);
    float * mem = calloc(2 * nstate, sizeof(float));
    scrappie_imatrix tb = make_scrappie_imatrix(nstate, nblk);
    if(NULL == mem || NULL == tb){
        tb = free_scrappie_imatrix(tb);
        free(mem);
        return NAN;
    }

    float * curr = mem;
    float * prev = mem + nstate;


    //  Forwards Viterbi pass
    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t offset = blk * trans->stride;
        const size_t tboffset = blk * tb->stride;
        {   // Swap
            float * tmp = curr;
            curr = prev;
            prev = tmp;
        }

        for(size_t st1=0 ; st1 < nstate ; st1++){
	    // st1 is to-state (in -ACGT)
            const size_t offsetS = offset + st1 * nstate;
            curr[st1] = trans->data.f[offsetS + 0] + prev[0];
            tb->data.f[tboffset + st1] = 0;
            for(size_t st2=1 ; st2 < nstate ; st2++){
		// st2 is from-state (in -ACGT)
                const float score = trans->data.f[offsetS + st2] + prev[st2];
                if(score > curr[st1]){
                    curr[st1] = score;
                    tb->data.f[tboffset + st1] = st2;
                }
            }
        }
    }

    //  Traceback
    const float score = valmaxf(curr, nstate);
    path[nblk] = argmaxf(curr, nstate);
    for(size_t blk=nblk ; blk > 0 ; blk--){
        const size_t offset = (blk - 1) * tb->stride;
        path[blk - 1] = tb->data.f[offset + path[blk]];
    }

    tb = free_scrappie_imatrix(tb);
    free(mem);

    return score;
}

char * crfpath_to_basecall(int const * path, size_t npos, int * pos){
    RETURN_NULL_IF(NULL == path, NULL);
    RETURN_NULL_IF(NULL == pos, NULL);

    int nbase = 0;
    for(size_t pos=0 ; pos < npos ; pos++){
        if(path[pos] < NBASE){
            nbase += 1;
        }
    }

    char * basecall = calloc(nbase + 1, sizeof(char));
    RETURN_NULL_IF(NULL == basecall, NULL);

    for(size_t pos=0, bpos=0 ; pos < npos ; pos++){
        if(path[pos] < NBASE){
            assert(bpos < nbase);
            basecall[bpos] = base_lookup[path[pos]];
            bpos += 1;
        }
    }

    return basecall;
}


/**  Posterior over states at each block
 *
 *  @param trans.  Constant scrappie matrix containing the (25) energies
 *  for each block.  (order ACGT-, from state in minor, to state major).
 *
 *  @returns scrappie matrix containing the posterior for nblk + 1
 **/
scrappie_matrix posterior_crf(const_scrappie_matrix trans){
    RETURN_NULL_IF(NULL == trans, NULL);

    const size_t nstate = roundf(sqrtf((float)trans->nr));
    assert(nstate * nstate == trans->nr);
    const size_t nblk = trans->nc;

    scrappie_matrix post = make_scrappie_matrix(nstate, nblk + 1);
    RETURN_NULL_IF(NULL == post, NULL);


    //  Forwards pass
    for(size_t st=0 ; st < nstate ; st++){
        // Initialisation
        post->data.f[st] = 0.0f;
    }
    for(size_t blk=0 ; blk < nblk ; blk++){
        const size_t offset = blk * trans->stride;
        const size_t offset_post = blk * post->stride;

        const float * prev = post->data.f + offset_post;
        float * curr = post->data.f + offset_post + post->stride;

        for(size_t st1=0 ; st1 < nstate ; st1++){
            const size_t offsetS = offset + st1 * nstate;
            curr[st1] = trans->data.f[offsetS + 0] + prev[0];
            for(size_t st2=1 ; st2 < nstate ; st2++){
                curr[st1] = logsumexpf(curr[st1], trans->data.f[offsetS + st2] + prev[st2]);
            }
        }
    }

    // Backwards pass
    float * tmpmem = malloc(2 * nstate * sizeof(float));
    float * prev = tmpmem;
    float * curr = tmpmem + nstate;
    for(size_t st=0 ; st < nstate ; st++){
        // Initialisation
        curr[st] = 0.0f;
    }
    // Normalisation of last block
    float tot = 0.0f;
    for(size_t st=0 ; st < nstate ; st++){
        tot = logsumexpf(tot, post->data.f[nblk * post->stride + st]);
    }
    for(size_t st=0 ; st < nstate ; st++){
        post->data.f[nblk * post->stride + st] = expf(post->data.f[nblk * post->stride + st] - tot);
    }
    for(size_t blk=nblk ; blk > 0 ; blk--){
        const size_t blkm1 = blk - 1;
        const size_t offset = blkm1 * trans->stride;
        const size_t offset_post = blkm1 * post->stride;

        {   // Swap
            float * tmp = curr;
            curr = prev;
            prev = tmp;
        }

        for(size_t st=0 ; st < nstate ; st++){
            curr[st] = trans->data.f[offset + st] + prev[0];
        }
        for(size_t st1=1 ; st1 < nstate ; st1++){
            const size_t offsetS = offset + st1 * nstate;
            for(size_t st2=0 ; st2 < nstate ; st2++){
                curr[st2] = logsumexpf(curr[st2], trans->data.f[offsetS + st2] + prev[st1]);
            }
        }

        // Normalisation
        float tot = 0.0f;
        for(size_t st=0 ; st < nstate ; st++){
            post->data.f[offset_post + st] += curr[st];
            tot = logsumexpf(tot, post->data.f[offset_post + st]);
        }
        for(size_t st=0 ; st < nstate ; st++){
            post->data.f[offset_post + st] = expf(post->data.f[offset_post + st] - tot);
        }
    }

    free(tmpmem);


    return post;
}



static float LARGE_VAL = 1e30f;

/**  Map a signal to a predicted squiggle using variant of dynamic time-warping
 *
 *   Uses a local mapping so not all of signal may be mapped and not every position of the
 *   predicted squiggle may be mapped to.
 *
 *   @param signal `raw_table` containing signal to map
 *   @param rate Rate of translocation relative to squiggle model
 *   @param params `scrappie_matrix` containing predicted squiggle
 *   @param prob_back Probability of a backward movement.
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param skip_pen  Penalty for skipping
 *   @param minscore Minimum possible emission for
 *   @param path [OUT] An array containing path.  Length equal to that of FULL signal
 *   @param fwd [OUT]
 *
 *   @returns score
 **/
float squiggle_match_viterbi(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen, float minscore,
                             int32_t * path_padded){
    RETURN_NULL_IF(NULL == signal.raw, NAN);
    RETURN_NULL_IF(NULL == params, NAN);
    RETURN_NULL_IF(NULL == path_padded, NAN);
    assert(signal.start < signal.end);
    assert(signal.end <= signal.n);
    assert(rate > 0.0f);
    assert(prob_back >= 0.0f && prob_back <= 1.0f);

    float final_score = NAN;

    const float * rawsig = signal.raw + signal.start;
    const size_t nsample = signal.end - signal.start;
    const size_t ldp = params->stride;
    const size_t npos = params->nc;
    const size_t nfstate = npos + 2;
    const size_t nstate = npos + nfstate;

    const float move_back_pen = logf(prob_back);
    const float stay_in_back_pen = logf(0.5f);
    const float move_from_back_pen = logf(0.5f);

    float * move_pen = calloc(nfstate, sizeof(float));
    float * fwd = calloc(2 * nstate, sizeof(float));
    float * scale = calloc(npos, sizeof(float));
    float * stay_pen = calloc(nfstate, sizeof(float));
    int32_t * traceback = calloc(nsample * nstate, sizeof(int32_t));
    if(NULL == move_pen || NULL == fwd || NULL == scale ||
       NULL == stay_pen || NULL == traceback){
        goto clean;
    }

    for(size_t pos=0 ; pos < npos ; pos++){
        //  Create array of scales
        scale[pos] = expf(params->data.f[pos * ldp + 1]);
    }
    for(size_t i=0 ; i < signal.n ; i++){
        path_padded[i] = -1;
    }

    //  Only deal with part of path that corresponds to trimmed signal
    int32_t * path = path_padded + signal.start;


    {
        const float lograte = logf(rate);
        float mean_move_pen = 0.0f;
        float mean_stay_pen = 0.0f;
        for(size_t pos=0 ; pos < npos ; pos++){
            const float mp = (1.0f - prob_back) * plogisticf(params->data.f[pos * ldp + 2] + lograte);
            move_pen[pos + 1] = logf(mp);
            stay_pen[pos + 1] = log1pf(-mp - prob_back);
            mean_move_pen += move_pen[pos + 1];
            mean_stay_pen += stay_pen[pos + 1];
        }
        mean_move_pen /= npos;
        mean_stay_pen /= npos;

        move_pen[0] = mean_move_pen;
        move_pen[nfstate - 1] = mean_move_pen;
        stay_pen[0] = mean_stay_pen;
        stay_pen[nfstate - 1] = mean_stay_pen;
    }

    for(size_t st=0 ; st < nstate ; st++){
        // States are start .. positions .. end
        fwd[st] = -LARGE_VAL;
    }
    // Must begin in start state
    fwd[0] = 0.0;


    for(size_t sample=0 ; sample < nsample ; sample++){
        const size_t fwd_prev_off = (sample % 2) * nstate;
        const size_t fwd_curr_off = ((sample + 1) % 2) * nstate;
        const size_t tr_off = sample * nstate;

        for(size_t st=0 ; st < nfstate ; st++){
            //  Stay in start, end or normal position
            fwd[fwd_curr_off + st] = fwd[fwd_prev_off + st] + stay_pen[st];
            traceback[tr_off + st] = st;
        }
        for(size_t st=0 ; st < npos ; st++){
            //  Stay in back position
            const size_t idx = nfstate + st;
            fwd[fwd_curr_off + idx] = fwd[fwd_prev_off + idx] + stay_in_back_pen;
            traceback[tr_off + idx] = idx;
        }
        for(size_t st=1 ; st < nfstate ; st++){
            //  Move to next position
            const float step_score = fwd[fwd_prev_off + st - 1] + move_pen[st - 1];
            if(step_score > fwd[fwd_curr_off + st]){
                fwd[fwd_curr_off + st] = step_score;
                traceback[tr_off + st] = st - 1;
            }
        }
        for(size_t st=2 ; st < nfstate ; st++){
            //  Skip to next position
            const float skip_score = fwd[fwd_prev_off + st - 2] + move_pen[st - 2] - skip_pen;
            if(skip_score > fwd[fwd_curr_off + st]){
                fwd[fwd_curr_off + st] = skip_score;
                traceback[tr_off + st] = st - 2;
            }
        }
        for(size_t destpos=1 ; destpos < npos ; destpos++){
            const size_t destst = destpos + 1;
            //  Move from start into sequence
            const float score = fwd[fwd_prev_off] + move_pen[0] - local_pen * destpos;
            if(score > fwd[fwd_curr_off + destst]){
                fwd[fwd_curr_off + destst] = score;
                traceback[tr_off + destst] = 0;
            }
        }
        for(size_t origpos=0 ; origpos < (npos - 1) ; origpos++){
            const size_t destst = nfstate - 1;
            const size_t origst = origpos + 1;
            const size_t deltapos = npos - 1 - origpos;
            //  Move from sequence into end
            const float score = fwd[fwd_prev_off + origst] + move_pen[origst] - local_pen * deltapos;
            if(score > fwd[fwd_curr_off + destst]){
                fwd[fwd_curr_off + destst] = score;
                traceback[tr_off + destst] = origst;
            }
        }
        for(size_t st=1 ; st < npos ; st++){
            // Move to back
            const float back_score = fwd[fwd_prev_off + st + 1] + move_back_pen;
            if(back_score > fwd[fwd_curr_off + nfstate + st - 1]){
                fwd[fwd_curr_off + nfstate + st - 1] = back_score;
                traceback[tr_off + nfstate + st - 1] = st + 1;
            }
        }
        for(size_t st=1 ; st < npos ; st++){
            // Move from back
            const float back_score = fwd[fwd_prev_off + nfstate + st - 1] + move_from_back_pen;
            if(back_score > fwd[fwd_curr_off + st + 1]){
                fwd[fwd_curr_off + st + 1] = back_score;
                traceback[tr_off + st + 1] = nfstate + st - 1;
            }
        }


        for(size_t pos=0 ; pos < npos ; pos++){
            //  Add on score for samples
            const float location = params->data.f[pos * ldp + 0];
            const float logscale = params->data.f[pos * ldp + 1];
            const float logscore = fmaxf(-minscore, loglaplace(rawsig[sample], location, scale[pos], logscale));
            //  State to add to is offset by one because of start state
            fwd[fwd_curr_off + pos + 1] += logscore;
            fwd[fwd_curr_off + nfstate + pos] += logscore;
        }

        // Score for start and end states
        fwd[fwd_curr_off + 0] -= local_pen;
        fwd[fwd_curr_off + nfstate - 1] -= local_pen;

    }

    //  Score of best path and final states.  Could be either last position or end state
    const size_t fwd_offset = (nsample % 2) * nstate;
    final_score = fmaxf(fwd[fwd_offset + nfstate - 2], fwd[fwd_offset + nfstate - 1]);
    if(fwd[fwd_offset + nfstate - 2] > fwd[fwd_offset + nfstate - 1]){
        path[nsample - 1] = nfstate - 2;
    } else {
        path[nsample - 1] = nfstate - 1;
    }

    for(size_t sample=1 ; sample < nsample ; sample++){
        const size_t rs = nsample - sample;
        const size_t tr_off = rs * nstate;
        path[rs - 1] = traceback[tr_off + path[rs]];
    }

    // Correct path so start and end states are encoded as -1, other states as positions
    {
        size_t sample_min = 0;
        size_t sample_max = nsample;
        for(; sample_min < nsample ; sample_min++){
            if(0 != path[sample_min]){
                break;
            }
            path[sample_min] = -1;
        }
        for(; sample_max > 0 ; sample_max--){
            if(nfstate - 1 != path[sample_max - 1]){
                break;
            }
            path[sample_max - 1] = -1;
        }
        for(size_t sample=sample_min ; sample < sample_max ; sample++){
            assert(path[sample] > 0);
            if(path[sample] >= nfstate){
                path[sample] -= nfstate;
            } else {
                path[sample] -= 1;
            }
        }
    }

clean:
    free(traceback);
    free(stay_pen);
    free(scale);
    free(fwd);
    free(move_pen);

    return final_score;
}


/**  Score a signal against a predicted squiggle using variant of dynamic time-warping
 *
 *   Uses a local mapping so not all of signal may be mapped and not every position of the
 *   predicted squiggle may be mapped to.
 *
 *   @param signal `raw_table` containing signal to map
 *   @param rate Read translocation rate relative to squiggle model
 *   @param params `scrappie_matrix` containing predicted squiggle
 *   @param prob_back Probability of a backward movement.
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param skip_pen  Penalty for skipping
 *   @param minscore Minimum possible emission for
 *
 *   @returns score
 **/
float squiggle_match_forward(const raw_table signal, float rate, const_scrappie_matrix params,
                             float prob_back, float local_pen, float skip_pen, float minscore){
    RETURN_NULL_IF(NULL == signal.raw, NAN);
    RETURN_NULL_IF(NULL == params, NAN);
    assert(signal.start < signal.end);
    assert(signal.end <= signal.n);
    assert(prob_back >= 0.0f && prob_back <= 1.0f);
    assert(rate > 0.0);

    float final_score = NAN;

    const float * rawsig = signal.raw + signal.start;
    const size_t nsample = signal.end - signal.start;
    const size_t ldp = params->stride;
    const size_t npos = params->nc;
    const size_t nfstate = npos + 2;
    const size_t nstate = npos + nfstate;

    const float move_back_pen = logf(prob_back);
    const float stay_in_back_pen = logf(0.5f);
    const float move_from_back_pen = logf(0.5f);

    float * move_pen = calloc(nfstate, sizeof(float));
    float * fwd = calloc(2 * nstate, sizeof(float));
    float * scale = calloc(npos, sizeof(float));
    float * stay_pen = calloc(nfstate, sizeof(float));
    if(NULL == move_pen || NULL == fwd || NULL == scale ||
       NULL == stay_pen){
        goto clean;
    }

    for(size_t pos=0 ; pos < npos ; pos++){
        //  Create array of scales
        scale[pos] = expf(params->data.f[pos * ldp + 1]);
    }

    {
        const float lograte = logf(rate);
        float mean_move_pen = 0.0f;
        float mean_stay_pen = 0.0f;
        for(size_t pos=0 ; pos < npos ; pos++){
            const float mp = (1.0f - prob_back) * plogisticf(params->data.f[pos * ldp + 2] + lograte);
            move_pen[pos + 1] = logf(mp);
            stay_pen[pos + 1] = log1pf(-mp - prob_back);
            mean_move_pen += move_pen[pos + 1];
            mean_stay_pen += stay_pen[pos + 1];
        }
        mean_move_pen /= npos;
        mean_stay_pen /= npos;

        move_pen[0] = mean_move_pen;
        move_pen[nfstate - 1] = mean_move_pen;
        stay_pen[0] = mean_stay_pen;
        stay_pen[nfstate - 1] = mean_stay_pen;
    }

    for(size_t st=0 ; st < nstate ; st++){
        // States are start .. positions .. end
        fwd[st] = -LARGE_VAL;
    }
    // Must begin in start state
    fwd[0] = 0.0;


    for(size_t sample=0 ; sample < nsample ; sample++){
        const size_t fwd_prev_off = (sample % 2) * nstate;
        const size_t fwd_curr_off = ((sample + 1) % 2) * nstate;

        for(size_t st=0 ; st < nfstate ; st++){
            //  Stay in start, end or normal position
            fwd[fwd_curr_off + st] = fwd[fwd_prev_off + st] + stay_pen[st];
        }
        for(size_t st=0 ; st < npos ; st++){
            //  Stay in back position
            const size_t idx = nfstate + st;
            fwd[fwd_curr_off + idx] = fwd[fwd_prev_off + idx] + stay_in_back_pen;
        }
        for(size_t st=1 ; st < nfstate ; st++){
            //  Move to next position
            const float step_score = fwd[fwd_prev_off + st - 1] + move_pen[st - 1];
            fwd[fwd_curr_off + st] = logsumexpf(fwd[fwd_curr_off + st], step_score);
        }
        for(size_t st=2 ; st < nfstate ; st++){
            //  Skip to next position
            const float skip_score = fwd[fwd_prev_off + st - 2] + move_pen[st - 2] - skip_pen;
            fwd[fwd_curr_off + st] = logsumexpf(fwd[fwd_curr_off + st], skip_score);
        }
        for(size_t destpos=1 ; destpos < npos ; destpos++){
            const size_t destst = destpos + 1;
            //  Move from start into sequence
            const float score = fwd[fwd_prev_off] + move_pen[0] - local_pen * destpos;
            fwd[fwd_curr_off + destst] = logsumexpf(fwd[fwd_curr_off + destst], score);
        }
        for(size_t origpos=0 ; origpos < (npos - 1) ; origpos++){
            const size_t destst = nfstate - 1;
            const size_t origst = origpos + 1;
            const size_t deltapos = npos - 1 - origpos;
            //  Move from sequence into end
            const float score = fwd[fwd_prev_off + origst] + move_pen[origst] - local_pen * deltapos;
            fwd[fwd_curr_off + destst] = logsumexpf(fwd[fwd_curr_off + destst], score);
        }
        for(size_t st=1 ; st < npos ; st++){
            // Move to back
            const float back_score = fwd[fwd_prev_off + st + 1] + move_back_pen;
            fwd[fwd_curr_off + nfstate + st - 1] = logsumexpf(fwd[fwd_curr_off + nfstate + st - 1], back_score);
        }
        for(size_t st=1 ; st < npos ; st++){
            // Move from back
            const float back_score = fwd[fwd_prev_off + nfstate + st - 1] + move_from_back_pen;
            fwd[fwd_curr_off + st + 1] = logsumexpf(fwd[fwd_curr_off + st + 1], back_score);
        }

        for(size_t pos=0 ; pos < npos ; pos++){
            //  Add on score for samples
            const float location = params->data.f[pos * ldp + 0];
            const float logscale = params->data.f[pos * ldp + 1];
            const float logscore = fmaxf(-minscore, loglaplace(rawsig[sample], location, scale[pos], logscale));
            //  State to add to is offset by one because of start state
            fwd[fwd_curr_off + pos + 1] += logscore;
            fwd[fwd_curr_off + nfstate + pos] += logscore;
        }

        // Score for start and end states
        fwd[fwd_curr_off + 0] -= local_pen;
        fwd[fwd_curr_off + nfstate - 1] -= local_pen;

    }

    //  Score of best path and final states.  Could be either last position or end state
    const size_t fwd_offset = (nsample % 2) * nstate;
    final_score = logsumexpf(fwd[fwd_offset + nfstate - 2], fwd[fwd_offset + nfstate - 1]);

clean:
    free(stay_pen);
    free(scale);
    free(fwd);
    free(move_pen);

    return final_score;
}


/**  Viterbi score of sequence
 *
 *   Local-global mapping through sequence calculating scores of best path from basecall posterior
 *
 *   Internal states are seq0 ... seq, start, end
 *
 *   @param logpost   Log posterior probability of state at each block.  Stay is last state.
 *   @param stay_pen  Penalty for staying
 *   @param skip_pen  Penalty for skipping
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param seq       Sequence encoded into same history states as basecalls
 *   @param seqlen    Length of seq
 *   @param path      Viterbi path [out].  If NULL, no path is returned
 *
 *   @returns score
 **/
float map_to_sequence_viterbi(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                              float local_pen, int const *seq, size_t seqlen, int *path){
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);

    const size_t nblock = logpost->nc;
    const size_t nst = logpost->nr;
    const size_t STAY = nst - 1;

    const size_t nseqstate = seqlen + 2;
    const size_t START_STATE = seqlen;
    const size_t END_STATE = seqlen + 1;

    // Memory.
    float * cscore = calloc(nseqstate, sizeof(float));
    float * pscore = calloc(nseqstate, sizeof(float));
    scrappie_imatrix traceback = make_scrappie_imatrix(nseqstate, nblock);
    if(NULL == cscore || NULL == pscore || NULL == traceback){
        traceback = free_scrappie_imatrix(traceback);
        free(pscore);
        free(cscore);
        return logscore;
    }


    //  Initialise
    for(size_t pos=0 ; pos < nseqstate ; pos++){
        cscore[pos] = -BIG_FLOAT;
    }
    cscore[START_STATE] = 0.0;

    // Forwards Viterbi
    for(size_t blk=0 ; blk < nblock ; blk++){
        const size_t lpoffset = blk * logpost->stride;
        const size_t toffset = blk * traceback->stride;
        {   // Swap vectors
            float * tmp = pscore;
            pscore = cscore;
            cscore = tmp;
        }

        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + fmaxf(-local_pen, logpost->data.f[lpoffset + STAY]);
        traceback->data.f[toffset + START_STATE] = START_STATE;
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + fmaxf(-local_pen, logpost->data.f[lpoffset + STAY]);
        traceback->data.f[toffset + END_STATE] = END_STATE;

        for(size_t pos=0 ; pos < seqlen ; pos++){
            //  Stay in ordinary state
            cscore[pos] = pscore[pos] - stay_pen + logpost->data.f[lpoffset + STAY];
            traceback->data.f[toffset + pos] = pos;
        }

        for(size_t pos=1 ; pos < seqlen ; pos++){
            //  Step
            const size_t newstate = seq[pos];
            const float step_score = pscore[pos - 1] + logpost->data.f[lpoffset + newstate];
            if(step_score > cscore[pos]){
                cscore[pos] = step_score;
                traceback->data.f[toffset + pos] = pos - 1;
            }
        }

        for(size_t pos=2 ; pos < seqlen ; pos++){
            //  Skip
            const size_t newstate = seq[pos];
            const float skip_score = pscore[pos - 2] - skip_pen + logpost->data.f[lpoffset + newstate];
            if(skip_score > cscore[pos]){
                cscore[pos] = skip_score;
                traceback->data.f[toffset + pos] = pos - 2;
            }
        }

        // Move directly from start to end without mapping
        /*if(pscore[START_STATE] - local_pen > cscore[END_STATE]){
            cscore[END_STATE] = pscore[START_STATE] - local_pen;
            traceback->data.f[toffset + END_STATE] = START_STATE;
        }*/
        // Move from start into sequence
        if(pscore[START_STATE] + logpost->data.f[lpoffset + seq[0]] > cscore[0]){
            cscore[0] = pscore[START_STATE] + logpost->data.f[lpoffset + seq[0]];
            traceback->data.f[toffset + 0] = START_STATE;
        }
        // Move from sequence into end
        if(pscore[seqlen - 1] - local_pen > cscore[END_STATE]){
            cscore[END_STATE] = pscore[seqlen - 1] - local_pen;
            traceback->data.f[toffset + END_STATE] = seqlen - 1;
        }
    }

    logscore = fmaxf(cscore[seqlen - 1], cscore[END_STATE]);
    if(NULL != path){
        path[nblock - 1] = (cscore[seqlen-1] > cscore[END_STATE]) ? (seqlen - 1) : END_STATE;
        for(size_t blk=nblock - 1; blk > 0 ; blk--){
            const size_t toffset = blk * traceback->stride;
            path[blk - 1] = traceback->data.f[toffset + path[blk]];
        }
        for(size_t blk=0 ; blk < nblock ; blk++){
            if(START_STATE == path[blk] || END_STATE == path[blk]){
                path[blk] = -1;
            }
        }
    }

    traceback = free_scrappie_imatrix(traceback);
    free(pscore);
    free(cscore);

    return logscore;
}


/**  Forward  score of sequence
 *
 *   Local-Global mapping through sequence calculating sum of scores over all paths from basecall posterior
 *
 *   @param logpost   Log posterior probability of state at each block.  Stay is last state.
 *   @param stay_pen  Penalty for staying
 *   @param skip_pen  Penalty for skipping
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param seq       Sequence encoded into same history states as basecalls
 *   @param seqlen    Length of seq
 *
 *   @returns score
 **/
float map_to_sequence_forward(const_scrappie_matrix logpost, float stay_pen, float skip_pen,
                              float local_pen, int const *seq, size_t seqlen){
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);

    const size_t nblock = logpost->nc;
    const size_t nst = logpost->nr;
    const size_t STAY = nst - 1;

    const size_t nseqstate = seqlen + 2;
    const size_t START_STATE = seqlen;
    const size_t END_STATE = seqlen + 1;

    // Memory.
    float * cscore = calloc(nseqstate, sizeof(float));
    float * pscore = calloc(nseqstate, sizeof(float));
    if(NULL == cscore || NULL == pscore){
        free(pscore);
        free(cscore);
        return logscore;
    }


    //  Initialise
    for(size_t pos=0 ; pos < nseqstate ; pos++){
        cscore[pos] = -BIG_FLOAT;
    }
    cscore[START_STATE] = 0.0;

    // Forwards pass
    for(size_t blk=0 ; blk < nblock ; blk++){
        const size_t lpoffset = blk * logpost->stride;
        {   // Swap vectors
            float * tmp = pscore;
            pscore = cscore;
            cscore = tmp;
        }

        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + logsumexpf(-local_pen, logpost->data.f[lpoffset + STAY]);
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + logsumexpf(-local_pen, logpost->data.f[lpoffset + STAY]);

        for(size_t pos=0 ; pos < seqlen ; pos++){
            //  Stay
            cscore[pos] = pscore[pos] - stay_pen + logpost->data.f[lpoffset + STAY];
        }

        for(size_t pos=1 ; pos < seqlen ; pos++){
            //  Step
            const size_t newstate = seq[pos];
            const float step_score = pscore[pos - 1] + logpost->data.f[lpoffset + newstate];
            cscore[pos] = logsumexpf(cscore[pos], step_score);
        }


        for(size_t pos=2 ; pos < seqlen ; pos++){
            //  skip
            const size_t newstate = seq[pos];
            const float skip_score = pscore[pos - 2] - skip_pen + logpost->data.f[lpoffset + newstate];
            cscore[pos] = logsumexpf(cscore[pos], skip_score);
        }

        // Move directly from start to end without mapping
        //cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[START_STATE] - local_pen);
        // Move from start into sequence
        cscore[0] = logsumexpf(cscore[0], pscore[START_STATE] + logpost->data.f[lpoffset + seq[0]]);
        // Move from sequence into end
        cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[seqlen - 1] - local_pen);
    }

    logscore = logsumexpf(cscore[seqlen - 1], cscore[END_STATE]);


    free(pscore);
    free(cscore);

    return logscore;
}


/**  Check if sequences of lower and upper bounds are consistent
 *
 *   @param low     Array of lower bounds
 *   @param high    Array of upper bounds
 *   @param nblock  Number of blocks (bounds)
 *   @param seqlen  Length of sequence (maximum upper bound)
 *
 *   @returns bool
 **/
bool are_bounds_sane(size_t const * low, size_t const * high, size_t nblock, size_t seqlen){
    bool ret = true;

    if(NULL == low || NULL == high){
        warnx("One or more bounds are NULL\n");
        // Early return since further tests make no sense
        return false;
    }

    if(low[0] != 0){
        warnx("First bound must include 0 (got %zu)\n", low[0]);
        ret = false;
    }
    if(high[nblock - 1] != seqlen){
        warnx("Last bound must equal seqlen %zu (got %zu)\n", seqlen, high[nblock - 1]);
        ret = false;
    }
    for(size_t i=0 ; i < nblock ; i++){
        if(low[i] > seqlen){
            warnx("Low bound for block %zu exceeds length of sequence (got %zu but seqlen is %zu)\n", i, low[i], seqlen);
            ret = false;
        }
        if(high[i] > seqlen){
            warnx("High bound for block %zu exceeds length of sequence (got %zu but seqlen is %zu)\n", i, high[i], seqlen);
            ret = false;
        }
        if(low[i] > high[i]){
            warnx("Low bound for block %zu exceeds high bound [%zu , %zu).\n", i, low[i], high[i]);
            ret = false;
        }
    }
    for(size_t i=1 ; i < nblock ; i++){
        if(low[i] > high[i - 1]){
            // Allow case where step but not stay is possible (low[i] == high[i-1])
            warnx("Blocks %zu and %zu don't overlap [%zu , %zu) -> [%zu , %zu)\n",
                  i - 1, i , low[i - 1], high[i - 1], low[i], high[i]);
            ret = false;
        }
        if(low[i] < low[i - 1]){
            warnx("Low bounds for blocks %zu and %zu aren't monotonic [%zu , %zu) -> [%zu , %zu)\n",
                  i - 1, i , low[i - 1], high[i - 1], low[i], high[i]);
            ret = false;
        }
        if(high[i] < high[i - 1]){
            warnx("High bounds for blocks %zu and %zu aren't monotonic [%zu , %zu) -> [%zu , %zu)\n",
                  i - 1, i , low[i - 1], high[i - 1], low[i], high[i]);
            ret = false;
        }
    }

    return ret;
}


/**  Viterbi score of sequence, banded
 *
 *   Local-Global mapping through sequence calculating scores of best path from basecall posterior (banded)
 *
 *   @param logpost   Log posterior probability of state at each block.  Stay is last state.
 *   @param stay_pen  Penalty for staying
 *   @param skip_pen  Penalty for skipping
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param seq       Sequence encoded into same history states as basecalls
 *   @param seqlen    Length of seq
 *   @param poslow, poshigh  Arrays of lowest and highest coordinate for each block. Low inclusive, high exclusive
 *
 *   @returns score
 **/
float map_to_sequence_viterbi_banded(const_scrappie_matrix logpost, float stay_pen, float skip_pen, float local_pen,
                                     int const *seq, size_t seqlen, size_t const * poslow, size_t const * poshigh){
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);
    RETURN_NULL_IF(NULL == poslow, logscore);
    RETURN_NULL_IF(NULL == poshigh, logscore);

    const size_t nblock = logpost->nc;
    const size_t nst = logpost->nr;
    const size_t STAY = nst - 1;

    const size_t nseqstate = seqlen + 2;
    const size_t START_STATE = seqlen;
    const size_t END_STATE = seqlen + 1;


    // Verify assumptions about bounds
    RETURN_NULL_IF(!are_bounds_sane(poslow, poshigh, nblock, seqlen), logscore);

    // Memeory. First state is stay in previous position
    float * cscore = calloc(nseqstate, sizeof(float));
    float * pscore = calloc(nseqstate, sizeof(float));
    if(NULL == cscore || NULL == pscore){
        free(pscore);
        free(cscore);
        return logscore;
    }


    //  Initialise
    for(size_t pos=0 ; pos < nseqstate ; pos++){
        pscore[pos] = -BIG_FLOAT;
        cscore[pos] = -BIG_FLOAT;
    }
    pscore[START_STATE] = 0.0;

    // Forwards Viterbi
    {  // First block
        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + fmaxf(-local_pen, logpost->data.f[STAY]);
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + fmaxf(-local_pen, logpost->data.f[STAY]);

        cscore[0] = fmaxf(cscore[0], pscore[0] + logpost->data.f[STAY] - stay_pen);
        if(poshigh[0] > 0){
            // Step
            const size_t stepto = seq[1];
            const float step_score = logpost->data.f[stepto];
            cscore[1] = step_score;
        }
        if(poshigh[0] > 1){
            // Skip
            const size_t stepto = seq[2];
            const float skip_score = logpost->data.f[stepto] - skip_pen;
            cscore[2] = skip_score;
        }
        // Move directly from start to end without mapping -- always allow
        cscore[END_STATE] = fmaxf(cscore[END_STATE], pscore[START_STATE] - local_pen);
        // Move from start into sequence -- lower bound for first block must be zero
        cscore[0] = fmaxf(cscore[0], pscore[START_STATE] + logpost->data.f[seq[0]]);
        // Move from sequence into end
        cscore[END_STATE] = fmaxf(cscore[END_STATE], pscore[seqlen - 1] - local_pen);
    }

    for(size_t blk=1 ; blk < nblock ; blk++){
        const size_t lpoffset = blk * logpost->stride;
        {   // Swap vectors
            float * tmp = pscore;
            pscore = cscore;
            cscore = tmp;
        }

        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + fmaxf(-local_pen, logpost->data.f[lpoffset + STAY]);
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + fmaxf(-local_pen, logpost->data.f[lpoffset + STAY]);


        for(size_t pos=poslow[blk] ; pos < poshigh[blk - 1] ; pos++){
            //  Stay
            cscore[pos] = pscore[pos] - stay_pen + logpost->data.f[lpoffset + STAY];
        }

        const size_t min_step_idx = imax(poslow[blk], poslow[blk - 1] + 1);
        const size_t max_step_idx = imin(poshigh[blk], poshigh[blk - 1] + 1);
        for(size_t pos=min_step_idx ; pos < max_step_idx ; pos++){
            // step -- pos is position going _to_
            const size_t stepto = seq[pos];
            const float step_score = pscore[pos - 1] + logpost->data.f[lpoffset + stepto];
            cscore[pos] = fmaxf(step_score, cscore[pos]);
        }

        const size_t min_skip_idx = imax(poslow[blk], poslow[blk - 1] + 2);
        const size_t max_skip_idx = imin(poshigh[blk], poshigh[blk - 1] + 2);
        for(size_t pos=min_skip_idx ; pos < max_skip_idx ; pos++){
            // skip -- pos is position going _to_
            const size_t skipto = seq[pos];
            const float skip_score = pscore[pos - 2] - skip_pen
                                   + logpost->data.f[lpoffset + skipto];
            cscore[pos] = fmaxf(skip_score, cscore[pos]);
        }

        // Move directly from start to end without mapping -- always allow
        //cscore[END_STATE] = fmaxf(cscore[END_STATE], pscore[START_STATE] - local_pen);
        // Move from start into sequence -- only allowed if lower bound is zero
        if(0 == poslow[blk]){
            cscore[0] = fmaxf(cscore[0], pscore[START_STATE] + logpost->data.f[lpoffset + seq[0]]);
        }
        // Move from sequence into end
        cscore[END_STATE] = fmaxf(cscore[END_STATE], pscore[seqlen - 1] - local_pen);
    }


    logscore = fmax(cscore[seqlen - 1], cscore[END_STATE]);

    free(pscore);
    free(cscore);

    return logscore;
}


/**  Forward score of sequence, banded
 *
 *   Local-Global mapping through sequence calculating scores of best path from basecall posterior (banded)
 *
 *   @param logpost   Log posterior probability of state at each block.  Stay is last state.
 *   @param stay_pen  Penalty for staying
 *   @param skip_pen  Penalty for skipping
 *   @param local_pen Penalty for local mapping (stay in start or ends state)
 *   @param seq       Sequence encoded into same history states as basecalls
 *   @param seqlen    Length of seq
 *   @param poslow, poshigh  Arrays of lowest and highest coordinate for each block. Low inclusive, high exclusive
 *
 *
 *   @returns score
 **/
float map_to_sequence_forward_banded(const_scrappie_matrix logpost, float stay_pen, float skip_pen, float local_pen,
                                     int const *seq, size_t seqlen, size_t const * poslow, size_t const * poshigh){
    float logscore = NAN;
    RETURN_NULL_IF(NULL == logpost, logscore);
    RETURN_NULL_IF(NULL == seq, logscore);
    RETURN_NULL_IF(NULL == poslow, logscore);
    RETURN_NULL_IF(NULL == poshigh, logscore);

    const size_t nblock = logpost->nc;
    const size_t nst = logpost->nr;
    const size_t STAY = nst - 1;

    const size_t nseqstate = seqlen + 2;
    const size_t START_STATE = seqlen;
    const size_t END_STATE = seqlen + 1;


    // Verify assumptions about bounds
    RETURN_NULL_IF(!are_bounds_sane(poslow, poshigh, nblock, seqlen), logscore);

    // Memory. First state is stay in previous position
    float * cscore = calloc(nseqstate, sizeof(float));
    float * pscore = calloc(nseqstate, sizeof(float));
    if(NULL == cscore || NULL == pscore){
        free(pscore);
        free(cscore);
        return logscore;
    }


    //  Initialise
    for(size_t pos=0 ; pos < nseqstate ; pos++){
        pscore[pos] = -BIG_FLOAT;
        cscore[pos] = -BIG_FLOAT;
    }
    pscore[START_STATE] = 0.0f;

    // Forwards pass
    {  // First block
        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + logsumexpf(-local_pen, logpost->data.f[STAY]);
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + logsumexpf(-local_pen, logpost->data.f[STAY]);

        cscore[0] = logsumexpf(cscore[0], pscore[0] + logpost->data.f[STAY] - stay_pen);
        if(poshigh[0] > 0){
            // Step
            const size_t stepto = seq[1];
            const float step_score = logpost->data.f[stepto];
            cscore[1] = step_score;
        }
        if(poshigh[0] > 1){
            // Skip
            const size_t stepto = seq[2];
            const float skip_score = logpost->data.f[stepto] - skip_pen;
            cscore[2] = skip_score;
        }
        // Move directly from start to end without mapping -- always allow
        cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[START_STATE] - local_pen);
        // Move from start into sequence -- lower bound for first block must be zero
        cscore[0] = logsumexpf(cscore[0], pscore[START_STATE] + logpost->data.f[seq[0]]);
        // Move from sequence into end
        cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[seqlen - 1] - local_pen);
    }

    for(size_t blk=1 ; blk < nblock ; blk++){
        const size_t lpoffset = blk * logpost->stride;
        {   // Swap vectors
            float * tmp = pscore;
            pscore = cscore;
            cscore = tmp;
        }

        // Stay in start state (local penalty or stay)
        cscore[START_STATE] = pscore[START_STATE] + logsumexpf(-local_pen, logpost->data.f[lpoffset + STAY]);
        // Stay in end state (local penalty or stay)
        cscore[END_STATE] = pscore[END_STATE] + logsumexpf(-local_pen, logpost->data.f[lpoffset + STAY]);


        for(size_t pos=poslow[blk] ; pos < poshigh[blk - 1] ; pos++){
            //  Stay
            cscore[pos] = pscore[pos] - stay_pen + logpost->data.f[lpoffset + STAY];
        }

        const size_t min_step_idx = imax(poslow[blk], poslow[blk - 1] + 1);
        const size_t max_step_idx = imin(poshigh[blk], poshigh[blk - 1] + 1);
        for(size_t pos=min_step_idx ; pos < max_step_idx ; pos++){
            // step -- pos is position going _to_
            const size_t stepto = seq[pos];
            const float step_score = pscore[pos - 1] + logpost->data.f[lpoffset + stepto];
            cscore[pos] = logsumexpf(step_score, cscore[pos]);
        }

        const size_t min_skip_idx = imax(poslow[blk], poslow[blk - 1] + 2);
        const size_t max_skip_idx = imin(poshigh[blk], poshigh[blk - 1] + 2);
        for(size_t pos=min_skip_idx ; pos < max_skip_idx ; pos++){
            // skip -- pos is position going _to_
            const size_t skipto = seq[pos];
            const float skip_score = pscore[pos - 2] - skip_pen
                                   + logpost->data.f[lpoffset + skipto];
            cscore[pos] = logsumexpf(skip_score, cscore[pos]);
        }

        // Move directly from start to end without mapping -- always allow
        //cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[START_STATE] - local_pen);
        // Move from start into sequence -- only allowed if lower bound is zero
        if(0 == poslow[blk]){
            cscore[0] = logsumexpf(cscore[0], pscore[START_STATE] + logpost->data.f[lpoffset + seq[0]]);
        }
        // Move from sequence into end
        cscore[END_STATE] = logsumexpf(cscore[END_STATE], pscore[seqlen - 1] - local_pen);
    }


    logscore = logsumexpf(cscore[seqlen - 1], cscore[END_STATE]);

    free(pscore);
    free(cscore);

    return logscore;
}
