#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "decode.h"

#define NBASE 4


float decode_transducer(const Mat_rptr logpost, float skip_pen, int * seq, bool use_slip){
	assert(NULL != logpost);
	assert(skip_pen >= 0.0);
	assert(NULL != seq);
	const int nev = logpost->nc;
	const int nstate = logpost->nr;
	const int ldp = logpost->nrq * 4;
	const int nkmer = nstate - 1;
	assert((nkmer %4) == 0);
	const int32_t nkmerq = nkmer / 4;
	const __m128i nkmerqv = _mm_set1_epi32(nkmerq);
	assert((nkmerq % 4) == 0);
	const int32_t nkmerqq = nkmerq / 4;
	const __m128i nkmerqqv = _mm_set1_epi32(nkmerqq);
	assert((nkmerqq % 4) == 0);
	const int32_t nkmerqqq = nkmerqq / 4;
	const __m128i nkmerqqqv = _mm_set1_epi32(nkmerqqq);
	assert((nkmerqqq % 4) == 0);
	const int nkmerqqqq = nkmerqqq / 4;

	//  Forwards memory + traceback
	Mat_rptr score = make_mat(nkmer, 1);
	Mat_rptr prev_score = make_mat(nkmer, 1);
	Mat_rptr tmp = make_mat(nkmer, 1);
	iMat_rptr itmp = make_imat(nkmer, nev);
	iMat_rptr traceback = make_imat(nkmer, nev);

	//  Initialise
	for( int i=0 ; i < nkmerq ; i++){
		score->data.v[i] = logpost->data.v[i];
	}

	//  Forwards Viterbi iteration
	for(int ev=1 ; ev < nev ; ev++){
		const size_t offsetTq = ev * nkmerq;
		const size_t offsetP = ev * ldp;
		const size_t offsetPq = ev * logpost->nrq;
		// Swap score and previous score
		{
			Mat_rptr tmptr = score;
			score = prev_score;
			prev_score = tmptr;
		}

		// Stay
		const __m128 stay_m128 = _mm_set1_ps(logpost->data.f[offsetP + nkmer]);
		const __m128i negone_m128i = _mm_set1_epi32(-1);
		for(int i=0 ; i < nkmerq ; i++){
			// Traceback for stay is negative
			score->data.v[i] = prev_score->data.v[i] + stay_m128;
			traceback->data.v[offsetTq + i] = negone_m128i;
		}

		// Step
		// Following three loops find maximum over suffix and record index
		for(int i=0 ; i<nkmerqq ; i++){
			tmp->data.v[i] = prev_score->data.v[i];
			itmp->data.v[i] = _mm_setzero_si128();
		}
		for(int r=1 ; r<NBASE ; r++){
			const size_t offset = r * nkmerqq;
			const __m128i itmp_fill = _mm_set1_epi32(r);
			for(int i=0 ; i<nkmerqq ; i++){
				__m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i], prev_score->data.v[offset + i]));
				tmp->data.v[i] = _mm_max_ps(tmp->data.v[i], prev_score->data.v[offset + i]);
				itmp->data.v[i] = _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
					                       _mm_and_si128(mask, itmp_fill));
			}
		}
		const __m128i c0123_m128i = _mm_setr_epi32(0, 1, 2, 3);
		for(int i=0 ; i<nkmerqq ; i++){
			itmp->data.v[i] = _mm_add_epi32(
							_mm_mullo_epi32(itmp->data.v[i], nkmerqv),
							_mm_add_epi32(c0123_m128i, _mm_set1_epi32(i * 4)));
		}

		for(int pref=0 ; pref < nkmerq ; pref++){
			const size_t i = pref;
			const __m128 step_score = logpost->data.v[offsetPq + i] + _mm_set1_ps(tmp->data.f[pref]);
			__m128i mask = _mm_castps_si128(_mm_cmplt_ps(score->data.v[i], step_score));
			score->data.v[i] = _mm_max_ps(score->data.v[i], step_score);
			traceback->data.v[offsetTq + i] = _mm_or_si128(_mm_andnot_si128(mask, traceback->data.v[offsetTq + i]),
								       _mm_and_si128(mask, _mm_set1_epi32(itmp->data.f[pref])));
		}



		// Skip
		const __m128 skip_penv = _mm_set1_ps(skip_pen);
		for(int i=0 ; i<nkmerqqq ; i++){
			tmp->data.v[i] = prev_score->data.v[i];
			itmp->data.v[i] = _mm_setzero_si128();
		}
		for(int r=1 ; r<NBASE * NBASE ; r++){
			const size_t offset = r * nkmerqqq;
			const __m128i itmp_fill = _mm_set1_epi32(r);
			for(int i=0 ; i<nkmerqqq ; i++){
				__m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i], prev_score->data.v[offset + i]));
				tmp->data.v[i] = _mm_max_ps(tmp->data.v[i], prev_score->data.v[offset + i]);
				itmp->data.v[i] = _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
					                       _mm_and_si128(mask, itmp_fill));
			}
		}
		for(int i=0 ; i<nkmerqqq ; i++){
			itmp->data.v[i] = _mm_add_epi32(
							_mm_mullo_epi32(itmp->data.v[i], nkmerqqv),
							_mm_add_epi32(c0123_m128i, _mm_set1_epi32(i * 4)));
		}
		for(int pref=0 ; pref < nkmerqq ; pref++){
			for(int i=0 ; i < NBASE ; i++){
				const size_t oi = pref * NBASE + i;
				// This cycling through prefixes
				const __m128 skip_score = logpost->data.v[offsetPq + oi]
                                                        + _mm_set1_ps(tmp->data.f[pref])
							- skip_penv;
				__m128i mask = _mm_castps_si128(_mm_cmplt_ps(score->data.v[oi], skip_score));
				score->data.v[oi] = _mm_max_ps(score->data.v[oi], skip_score);
				traceback->data.v[offsetTq + oi] = _mm_or_si128(_mm_andnot_si128(mask, traceback->data.v[offsetTq + oi]),
									        _mm_and_si128(mask, _mm_set1_epi32(itmp->data.f[pref])));
			}
		}

		// Slip
		if(use_slip){
			const __m128 slip_penv = _mm_set1_ps(2.0 * skip_pen);
			for(int i=0 ; i<nkmerqqqq ; i++){
				tmp->data.v[i] = prev_score->data.v[i];
				itmp->data.v[i] = _mm_setzero_si128();
			}
			for(int r=1 ; r<NBASE * NBASE * NBASE; r++){
				const size_t offset = r * nkmerqqqq;
				const __m128i itmp_fill = _mm_set1_epi32(r);
				for(int i=0 ; i<nkmerqqqq ; i++){
					__m128i mask = _mm_castps_si128(_mm_cmplt_ps(tmp->data.v[i], prev_score->data.v[offset + i]));
					tmp->data.v[i] = _mm_max_ps(tmp->data.v[i], prev_score->data.v[offset + i]);
					itmp->data.v[i] = _mm_or_si128(_mm_andnot_si128(mask, itmp->data.v[i]),
								       _mm_and_si128(mask, itmp_fill));
				}
			}
			for(int i=0 ; i<nkmerqqqq ; i++){
				itmp->data.v[i] = _mm_add_epi32(
								_mm_mullo_epi32(itmp->data.v[i],  nkmerqqqv),
								_mm_add_epi32(c0123_m128i, _mm_set1_epi32(i * 4)));
			}
			for(int pref=0 ; pref < nkmerqqq ; pref++){
				for(int i=0 ; i < NBASE * NBASE; i++){
					const size_t oi = pref * NBASE * NBASE + i;
					// This cycling through prefixes
					const __m128 skip_score = logpost->data.v[offsetPq + oi]
								+ _mm_set1_ps(tmp->data.f[pref])
								- slip_penv;
					__m128i mask = _mm_castps_si128(_mm_cmplt_ps(score->data.v[oi], skip_score));
					score->data.v[oi] = _mm_max_ps(score->data.v[oi], skip_score);
					traceback->data.v[offsetTq + oi] = _mm_or_si128(_mm_andnot_si128(mask, traceback->data.v[offsetTq + oi]),
											_mm_and_si128(mask, _mm_set1_epi32(itmp->data.f[pref])));
				}
			}
		}
	}

	//  Viterbi traceback
	float logscore = valmaxf(score->data.f, nkmer);
	int pstate = argmaxf(score->data.f, nkmer);
	for(int ev=1 ; ev < nev ; ev++){
		const int iev = nev - ev;
		const int tstate = traceback->data.f[iev * nkmer + pstate];
		if(tstate >= 0){
			// Non-stay
			seq[iev] = pstate;
			pstate = tstate;
		} else {
			// Move was a stay
			seq[iev] = -1;
		}
	}
	seq[0] = pstate;


	traceback = free_imat(traceback);
	itmp = free_imat(itmp);
	tmp = free_mat(tmp);
	prev_score = free_mat(prev_score);
	score = free_mat(score);
	return logscore;
}

int overlap(int k1, int k2, int nkmer){
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
	} while(k1 != k2);

	return overlap;
}


int position_highest_bit(int x){
	int i = 0;
	for( ; x != 0 ; i++, x>>=1);
	return i;
}

int first_nonnegative(const int * seq, int n){
	int st;
        for(st=0 ; st < n && seq[st] < 0; st++);
	return st;
}

bool iskmerhomopolymer(int kmer, int klen){
	const int b = kmer & 3;

	for( int k=1 ; k < klen ; k++){
		kmer >>= 2;
		if(b != (kmer & 3)){
			return false;
		}
	}

	return true;
}

const char base_lookup[4] = {'A', 'C', 'G', 'T'};
char * overlapper(const int * seq, int n, int nkmer, int * pos){
	assert(NULL != seq);
	const int kmer_len = position_highest_bit(nkmer) / 2;


	//  Determine length of final sequence
	int length = kmer_len;
 	// Find first non-stay
        const int st = first_nonnegative(seq, n);
	assert(st != n);
	int kprev = seq[st];
	for(int k=st + 1 ; k < n ; k++){
		if(seq[k] < 0){
			// Short-cut stays
			continue;
		}
		assert(seq[k] >= 0);
		length += overlap(kprev , seq[k], nkmer);
		kprev = seq[k];
		assert(kprev >= 0);
	}

	// Initialise basespace sequence with terminating null
	char * bases = calloc(length + 1, sizeof(char));
	// Fill with first kmer
	for(int kmer=seq[st], k=1 ; k <= kmer_len ; k++){
		int b = kmer & 3;
		kmer >>= 2;
		bases[kmer_len - k] = base_lookup[b];
	}



	for(int last_idx=kmer_len - 1, kprev=seq[st], k=st + 1 ; k < n ; k++){
		if(seq[k] < 0){
			// Short-cut stays
			if(NULL != pos){
				pos[k] = pos[k - 1];
			}
			continue;
		}
		int ol = overlap(kprev , seq[k], nkmer);
		if(NULL != pos){
			pos[k] = pos[k - 1] + ol;
		}
		kprev = seq[k];

                for(int kmer=seq[k], i=0 ; i<ol ; i++){
			int b = kmer & 3;
			kmer >>= 2;
			bases[last_idx + ol - i] = base_lookup[b];
		}
		last_idx += ol;
	}

	return bases;
}

int calibrated_dwell(int hdwell, int inhomo, const dwell_model dm){
	const int b = inhomo & 3;
	return (int)roundf(((float)hdwell - dm.base_adj[b]) / dm.scale);
}

char * dwell_corrected_overlapper(const int * seq, const int * dwell, int n, int nkmer, const dwell_model dm){
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
	for(int k=st + 1 ; k < n ; k++){
		/* Simple state machine tagged by inhomo
		 * inhomo < 0  --  not in a homopolymer
		 * inhomo >= 0  -- in homopolymer and value is homopolymer state
                 */
		if(seq[k] < 0){
			// State is stay.  Short circuit rest of logic
			if(inhomo >= 0){
				// Accumate dwell if in homopolymer
				hdwell += dwell[k];
			}
			continue;
		}

		if(seq[k] == inhomo){
			// Not stay but still in same homopolymer
			hdwell += dwell[k];
			continue;
		}

		if(inhomo >= 0){
			// Changed state.  Leave homopolymer
			length += calibrated_dwell(hdwell, inhomo, dm);
			inhomo = -1;
			hdwell = 0;
		}

		assert(seq[k] >= 0);
		length += overlap(kprev , seq[k], nkmer);
		kprev = seq[k];
		assert(kprev >= 0);

		if(iskmerhomopolymer(kprev, kmer_len)){
			// Entered a new homopolymer
			inhomo = kprev;
			hdwell = dwell[k];
		}
	}

	if(inhomo >= 0){
		//  Correction for final homopolymer
		length += calibrated_dwell(hdwell, inhomo, dm);
	}


	// Initialise basespace sequence with terminating null
	char * bases = calloc(length + 1, sizeof(char));
	// Fill with first kmer
	for(int kmer=seq[st], k=1 ; k <= kmer_len ; k++){
		int b = kmer & 3;
		kmer >>= 2;
		bases[kmer_len - k] = base_lookup[b];
	}


	int last_idx = kmer_len - 1;
	inhomo = -1;
	hdwell = 0;
	for(int kprev=seq[st], k=st + 1 ; k < n ; k++){
		if(seq[k] < 0){
			// State is stay.
			if(inhomo >= 0){
				// Accumate dwell if in homopolymer
				hdwell += dwell[k];
			}
			continue;
		}

		if(seq[k] == inhomo){
			// Not stay but still in same homopolymer
			hdwell += dwell[k];
			continue;
		}

		if(inhomo >= 0){
			// Changed state.  Leave homopolymer
			int hlen = calibrated_dwell(hdwell, inhomo, dm);
			char hbase = base_lookup[inhomo & 3];
			for( int i=0 ; i < hlen ; i++, last_idx++){
				bases[last_idx + 1] = hbase;
			}
			inhomo = -1;
			hdwell = 0;
		}

		int ol = overlap(kprev , seq[k], nkmer);
		kprev = seq[k];

                for(int kmer=seq[k], i=0 ; i<ol ; i++){
			int b = kmer & 3;
			kmer >>= 2;
			bases[last_idx + ol - i] = base_lookup[b];
		}
		last_idx += ol;

		if(iskmerhomopolymer(kprev, kmer_len)){
			// Entered a new homopolymer
			inhomo = kprev;
			hdwell += dwell[k];
		}
	}

	if(inhomo >= 0){
		//  Correction for final homopolymer
		int hlen = calibrated_dwell(hdwell, inhomo, dm);
		char hbase = base_lookup[inhomo & 3];
		for( int i=0 ; i < hlen ; i++, last_idx++){
			bases[last_idx] = hbase;
		}
	}
	if(last_idx + 1 != length){
		printf("last_idx %d length %d\n\n", last_idx, length);
		assert(last_idx  + 1 == length);
	}

	return bases;
}
