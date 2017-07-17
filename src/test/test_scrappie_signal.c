#include <CUnit/Basic.h>
#include <err.h>
#include <stdbool.h>

#include "decode.h"
#include "fast5_interface.h"
#include "layers.h"
#include "scrappie_structures.h"
#include "scrappie_util.h"
#include "util.h"

static const char rawsignalfile[] = "raw_signal.crp";
static const char signalfile[] = "trimmed_signal.crp";
static const char normsignalfile[] = "normalised_signal.crp";
static const char posteriorfile[] = "posterior_trimmed.crp";


static scrappie_matrix rawsignal = NULL;
static scrappie_matrix signal = NULL;
static scrappie_matrix normsignal = NULL;
static float * normsig_arr = NULL;



/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_signal(void) {
    rawsignal = read_scrappie_matrix(rawsignalfile);
    if(NULL == rawsignal){
        return 1;
    }

    signal = read_scrappie_matrix(signalfile);
    if(NULL == signal){
        return 1;
    }

    normsignal = read_scrappie_matrix(normsignalfile);
    if(NULL == normsignal){
        return 1;
    }
    normsig_arr = array_from_scrappie_matrix(normsignal);

    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_signal(void) {
    free(normsig_arr);
    (void)free_scrappie_matrix(normsignal);
    (void)free_scrappie_matrix(signal);
    return 0;
}

void test_trim_signal(void) {
    const int winlen = 100;
    raw_table rt = {0};
    rt.raw  = array_from_scrappie_matrix(rawsignal);
    CU_ASSERT_PTR_NOT_NULL_FATAL(rt.raw);
    rt.n = rt.end = rawsignal->nc;

    {   // Scale raw data to pA
        const float range = 1373.41f;
        const float digitisation = 8192.0f;
        const float unit = range / digitisation;
        const float offset = 16.0f;

        for(size_t i=0 ; i < rt.n ; i++){
            rt.raw[i] = (rt.raw[i] + offset) * unit;
        }
    }

    range_t r = trim_raw_by_mad(rt, winlen, 0.0);
    CU_ASSERT_EQUAL(r.start, 0);
    CU_ASSERT_EQUAL(r.end, (rt.n / winlen) * winlen);

    r.start += 200;
    r.end -= 10;

    scrappie_matrix mat_trim = mat_from_array(rt.raw + r.start, 1, r.end - r.start);
    CU_ASSERT_PTR_NOT_NULL_FATAL(mat_trim);
    CU_ASSERT_EQUAL(mat_trim->nc, signal->nc);

    CU_ASSERT_TRUE(equality_scrappie_matrix(mat_trim, signal, 1e-4));

    (void)free_scrappie_matrix(mat_trim);
    free(rt.raw);
}

void test_normalise_signal(void) {
    float * sigarr = array_from_scrappie_matrix(signal);
    size_t n = signal->nc;
    CU_ASSERT_PTR_NOT_NULL_FATAL(sigarr);

    medmad_normalise_array(sigarr, n);
    CU_ASSERT_TRUE(equality_arrayf(sigarr, normsig_arr, n, 1e-5));

    free(sigarr);
}


void test_decode_posterior(void) {
    const float min_prob = 1e-5;
    scrappie_matrix post = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(post);
    //CU_FAIL("Posterior needs rearranging into correct format!");

    robustlog_activation_inplace(post, min_prob);
    const size_t nblock = post->nc;

    int * path = calloc(nblock, sizeof(int));
    //float score = decode_transducer(post, 0.0, path, false);
    float score = sloika_viterbi(post, 0.0f, path);
    fprintf(stdout, "Score = %f\n", score);
    for(int c=0 ; c < (nblock / 10) ; c++){
        fprintf(stdout,"%3d", path[10 * c + 0]);
        for(int i=1 ; i < 10 ; i++){
            fprintf(stdout,", %3d", path[10 * c + i]);
        }
        fputc('\n', stdout);
    }
    fputc('\n', stdout);

    free(path);
    (void)free_scrappie_matrix(post);
}



/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_signal(void) {
    CU_pSuite suite = CU_add_suite("Test manipulating signal",
                                   init_test_signal,
                                   clean_test_signal);
    if (NULL == suite) {
        return CU_get_error();
    }

    if (NULL ==
        CU_add_test(suite, "Normalise trimmed signal", test_normalise_signal)) {
        return CU_get_error();
    }
    if (NULL ==
        CU_add_test(suite, "Trimming of raw signal", test_trim_signal)) {
        return CU_get_error();
    }
    if (NULL ==
        CU_add_test(suite, "Decoding of posterior", test_decode_posterior)) {
        return CU_get_error();
    }

    return 0;
}
