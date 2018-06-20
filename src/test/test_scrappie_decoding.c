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

static const char posteriorfile[] = "posterior_trimmed.crp";
static const char pathfile[] = "path.crp";


/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_decoding(void) {
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_decoding(void) {
    return 0;
}

void test_decode_equivalent_helper(float stay_pen, float skip_pen, float local_pen){
    const float min_prob = 1e-5;
    scrappie_matrix post = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(post);

    robustlog_activation_inplace(post, min_prob);
    const size_t nblock = post->nc;

    int * path_vectorised = calloc(nblock + 1, sizeof(int));
    int * path_original = calloc(nblock + 1, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_vectorised);
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_original);
    float score_vectorised = decode_transducer(post, stay_pen, skip_pen, local_pen, path_vectorised, false);
    float score_original = sloika_viterbi(post, stay_pen, skip_pen, local_pen, path_original);

    CU_ASSERT_DOUBLE_EQUAL(score_original, score_vectorised, 1e-5);
    CU_ASSERT_TRUE(equality_arrayi(path_original, path_vectorised, nblock));

    free(path_original);
    free(path_vectorised);
    post = free_scrappie_matrix(post);
}


void test_decode_equivalent(void) {
    test_decode_equivalent_helper(0.0f, 0.0f, 2.0f);
}

void test_decode_with_staypen_equivalent(void) {
    test_decode_equivalent_helper(2.0f, 0.0f, 2.0f);
}

void test_decode_with_skippen_equivalent(void) {
    test_decode_equivalent_helper(0.0f, 2.0f, 2.0f);
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

    int * path_original = calloc(nblock + 1, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_original);
    float score_original = sloika_viterbi(post, 0.0f, 0.0f, 100.0f, path_original);

    int * path_sloika = calloc(nblock + 1, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path_sloika);
    for(int i=0 ; i < nblock ; i++){
        path_sloika[i] = (int)path->data.f[i * path->stride];
    }
    CU_ASSERT_DOUBLE_EQUAL(score_original, score_expected, 1e-4);
    // Offset of 1 due to new decoding adding start state on beginning
    CU_ASSERT_TRUE(equality_arrayi(path_original + 1, path_sloika, nblock));

    free(path_sloika);
    free(path_original);
    path = free_scrappie_matrix(path);
    post = free_scrappie_matrix(post);
}

static test_with_description tests[] = {
    {"Decoding same as Sloika", test_decode_equivalent_to_sloika},
    {"Decoding of original and vectorised posterior same", test_decode_equivalent},
    {"Decoding of original and vectorised posterior same with stay penalty", test_decode_with_staypen_equivalent},
    {"Decoding of original and vectorised posterior same with skip penalty", test_decode_with_skippen_equivalent},
    {0}};

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_decoding(void) {
    return scrappie_register_test_suite("Test decoding", init_test_decoding, clean_test_decoding, tests);
}
