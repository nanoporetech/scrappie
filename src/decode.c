#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decode.h"
#include "util.h"

#define NBASE 4

#ifndef __SSE2__
#    error "Compilation of function decode_transducer requires a processor that supports at least SSE2"
#endif

#ifndef __SSE4_1
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

float decode_transducer(const scrappie_matrix logpost, float skip_pen, int *seq,
                        bool use_slip) {
    assert(NULL != logpost);
    assert(skip_pen >= 0.0);
    assert(NULL != seq);
    const int nev = logpost->nc;
    const int nstate = logpost->nr;
    const int ldp = logpost->nrq * 4;
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
    if (use_slip) {
        assert((nhistoryqqq % 4) == 0);
    }
    //  Forwards memory + traceback
    scrappie_matrix score = make_scrappie_matrix(nhistory, 1);
    scrappie_matrix prev_score = make_scrappie_matrix(nhistory, 1);
    scrappie_matrix tmp = make_scrappie_matrix(nhistory, 1);
    scrappie_imatrix itmp = make_scrappie_imatrix(nhistory, nev);
    scrappie_imatrix traceback = make_scrappie_imatrix(nhistory, nev);

    //  Initialise
    for (int i = 0; i < nhistoryq; i++) {
        score->data.v[i] = logpost->data.v[i];
    }

    //  Forwards Viterbi iteration
    for (int ev = 1; ev < nev; ev++) {
        const size_t offsetTq = ev * nhistoryq;
        const size_t offsetP = ev * ldp;
        const size_t offsetPq = ev * logpost->nrq;
        // Swap score and previous score
        {
            scrappie_matrix tmptr = score;
            score = prev_score;
            prev_score = tmptr;
        }

        // Stay
        const __m128 stay_m128 =
            _mm_set1_ps(logpost->data.f[offsetP + nhistory]);
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
        if (use_slip) {
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
    }

    //  Viterbi traceback
    float logscore = valmaxf(score->data.f, nhistory);
    int pstate = argmaxf(score->data.f, nhistory);
    for (int ev = 1; ev < nev; ev++) {
        const int iev = nev - ev;
        const int tstate = traceback->data.f[iev * nhistory + pstate];
        if (tstate >= 0) {
            // Non-stay
            seq[iev] = pstate;
            pstate = tstate;
        } else {
            // Move was a stay
            seq[iev] = -1;
        }
    }
    seq[0] = pstate;

    traceback = free_scrappie_imatrix(traceback);
    itmp = free_scrappie_imatrix(itmp);
    tmp = free_scrappie_matrix(tmp);
    prev_score = free_scrappie_matrix(prev_score);
    score = free_scrappie_matrix(score);

    assert(validate_ivector(seq, nev, -1, nhistory - 1, __FILE__, __LINE__));
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

int position_highest_bit(int x) {
    int i = 0;
    for (; x != 0; i++, x >>= 1) ;
    return i;
}

int first_nonnegative(const int *seq, int n) {
    int st;
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
char *overlapper(const int *seq, int n, int nkmer, int *pos) {
    assert(NULL != seq);
    const int kmer_len = position_highest_bit(nkmer) / 2;

    //  Determine length of final sequence
    int length = kmer_len;
    // Find first non-stay
    const int st = first_nonnegative(seq, n);
    assert(st != n);
    int kprev = seq[st];
    for (int k = st + 1; k < n; k++) {
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
    // Fill with first kmer
    for (int kmer = seq[st], k = 1; k <= kmer_len; k++) {
        int b = kmer & 3;
        kmer >>= 2;
        bases[kmer_len - k] = base_lookup[b];
    }

    for (int last_idx = kmer_len - 1, kprev = seq[st], k = st + 1; k < n; k++) {
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

        for (int kmer = seq[k], i = 0; i < ol; i++) {
            int b = kmer & 3;
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
    assert(NULL != seq);
    assert(NULL != dwell);
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
    if (NULL == et.event || NULL == seq) {
        return NULL;
    }
    const int nev = et.end - et.start;
    const int evoffset = et.start;

    const float prior_scale =
        (et.event[nev + evoffset - 1].length +
         et.event[nev + evoffset - 1].start - et.event[evoffset].start)
        / (float)basecall_len;
    int *dwell = calloc(nev, sizeof(int));
    for (int ev = 0; ev < nev; ev++) {
        dwell[ev] = et.event[ev + evoffset].length;
    }

    /*   Calibrate scaling factor for homopolymer estimation.
     *   Simple mean of the dwells of all 'step' movements in
     * the basecall.  Steps within homopolymers are ignored.
     *   A more complex calibration could be used.
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
    const float homo_scale = (prior_scale + tot_step_dwell) / (1.0 + nstep);
    const dwell_model dm = { homo_scale, {0.0f, 0.0f, 0.0f, 0.0f} };

    char *newbases =
        dwell_corrected_overlapper(seq, dwell, nev, nstate - 1, dm);

    free(dwell);

    return newbases;
}
