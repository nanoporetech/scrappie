#include <CUnit/Basic.h>
#include <err.h>
#include <stdbool.h>

#include "decode.h"
#include "layers.h"
#include "scrappie_common.h"
#include "scrappie_structures.h"
#include "scrappie_util.h"
#include "test_common.h"
#include "util.h"

static const char rawsignalfile[] = "raw_signal.crp";
static const char signalfile[] = "trimmed_signal.crp";
static const char normsignalfile[] = "normalised_signal.crp";
static const char posteriorfile[] = "posterior_trimmed.crp";
static const char pathfile[] = "path.crp";


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

    rt = trim_raw_by_mad(rt, winlen, 0.0f);
    CU_ASSERT_EQUAL(rt.start, 0);
    CU_ASSERT_EQUAL(rt.end, (rt.n / winlen) * winlen);

    rt.start += 200;
    rt.end -= 10;

    scrappie_matrix mat_trim = mat_from_array(rt.raw + rt.start, 1, rt.end - rt.start);
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

void test_decode_equivalent_helper(float stay_pen, float skip_pen){
    const float min_prob = 1e-5;
    scrappie_matrix post = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(post);

    robustlog_activation_inplace(post, min_prob);
    const size_t nblock = post->nc;

    int * path_vectorised = calloc(nblock, sizeof(int));
    int * path_original = calloc(nblock, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_vectorised);
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_original);
    float score_vectorised = decode_transducer(post, stay_pen, skip_pen, path_vectorised, false);
    float score_original = sloika_viterbi(post, stay_pen, skip_pen, path_original);

    CU_ASSERT_DOUBLE_EQUAL(score_original, score_vectorised, 1e-5);
    CU_ASSERT_TRUE(equality_arrayi(path_original, path_vectorised, nblock));

    free(path_original);
    free(path_vectorised);
    (void)free_scrappie_matrix(post);
}


void test_decode_equivalent(void) {
    test_decode_equivalent_helper(0.0f, 0.0f);
}

void test_decode_with_staypen_equivalent(void) {
    test_decode_equivalent_helper(2.0f, 0.0f);
}

void test_decode_with_skippen_equivalent(void) {
    test_decode_equivalent_helper(0.0f, 2.0f);
}

void test_decode_equivalent_to_sloika(void) {
    const float score_expected = -115.5761f;
    const float min_prob = 1e-5;
    scrappie_matrix post = read_scrappie_matrix(posteriorfile);
    scrappie_matrix path = read_scrappie_matrix(pathfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(post);
    CU_ASSERT_PTR_NOT_NULL_FATAL(path);
    CU_ASSERT_EQUAL(post->nc, path->nc);

    robustlog_activation_inplace(post, min_prob);
    const size_t nblock = post->nc;

    int * path_original = calloc(nblock, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_original);
    float score_original = sloika_viterbi(post, 0.0f, 0.0f, path_original);

    int * path_sloika = calloc(nblock, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_sloika);
    for(int i=0 ; i < nblock ; i++){
        path_sloika[i] = (int)path->data.f[i * path->nrq * 4];
    }
    CU_ASSERT_DOUBLE_EQUAL(score_original, score_expected, 1e-4);
    CU_ASSERT_TRUE(equality_arrayi(path_original, path_sloika, nblock));

    free(path_sloika);
    free(path_original);
    (void)free_scrappie_matrix(path);
    (void)free_scrappie_matrix(post);
}

static test_with_description tests[] = {
    {"Normalise trimmed signal", test_normalise_signal},
    {"Trimming of raw signal", test_trim_signal},
    {"Decoding same as Sloika", test_decode_equivalent_to_sloika},
    {"Decoding of original and vectorised posterior same", test_decode_equivalent},
    {"Decoding of original and vectorised posterior same with stay penalty", test_decode_with_staypen_equivalent},
    {"Decoding of original and vectorised posterior same with skip penalty", test_decode_with_skippen_equivalent},
    {0}};

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_signal(void) {
    return scrappie_register_test_suite("Test manipulating signal", init_test_signal, clean_test_signal, tests);
}
