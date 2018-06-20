#define BANANA 1
#include <CUnit/Basic.h>
#include <stdbool.h>

#include <decode.h>
#include <layers.h>
#include <util.h>
#include <scrappie_matrix.h>
#include "scrappie_util.h"
#include <test_common.h>

static const char posteriorfile[] = "posterior_trimmed.crp";
static float BIG_VAL = 1.e30f;

static const int32_t mapping_target[] = {
    332, 306, 202, 811, 174, 699, 749, 949, 725, 854,
    364, 432, 704, 771, 15, 61, 245, 980, 851, 335,
    316, 240, 960, 771, 12, 196, 787, 78, 312, 227,
    910, 571, 239, 958, 760, 896, 14, 56, 226, 905,
    548, 147, 591, 319, 254, 994, 906, 554, 686, 699,
    751, 956, 754, 802, 139, 556, 704, 771, 14, 228,
    914, 291, 564, 208, 268, 202, 811, 175, 702, 763,
    949, 727, 861, 372, 464, 835, 270, 225, 900, 529,
    68, 275, 76, 306, 201, 804, 585, 295, 157, 628,
    464, 834, 264, 128, 10, 41, 167, 668, 624, 782,
    57, 229, 916, 592, 263, 28, 115, 463, 828, 243,
    831, 254, 1005, 733, 469, 855, 350, 491, 941, 693,
    727, 860, 450, 777, 36, 147, 591, 316, 242, 803,
    141, 564, 211, 847, 319, 254, 998, 922, 619, 701,
    759, 989, 887, 476, 880, 451, 782, 56, 227, 911,
    575, 252, 1011, 974, 827, 238, 952, 736, 896, 515,
    15, 253, 980};

static const int32_t tightlow[] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
    3, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
    9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,
    12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17,
    17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 21, 21, 22, 23,
    23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25,
    25, 25, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 31, 31, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34,
    34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
    35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
    35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 39, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 48, 48, 49, 49,
    49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51,
    51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53,
    53, 53, 53, 53, 53, 53, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 66, 67, 67, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
    69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72,
    72, 72, 73, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
    75, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78,
    78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
    80, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83,
    83, 84, 84, 84, 84, 84, 84, 84, 84, 84, 86, 86, 87, 88, 88, 88, 88, 88, 88, 88,
    89, 89, 89, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 91, 91, 91, 92, 92,
    92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 94, 94, 94, 95, 96, 96, 96, 97,
    97, 97, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100,
    100, 101, 101, 101, 101, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104,
    104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104,
    104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 106, 106, 106,
    106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 108, 108, 108, 108, 108, 109, 109, 109,
    109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
    110, 110, 110, 110, 110, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 114,
    114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 116, 116, 116, 116, 116,
    116, 116, 116, 116, 116, 117, 117, 117, 117, 118, 118, 118, 118, 119, 119, 119, 119, 119, 120, 121,
    121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 124,
    124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 125, 126, 127, 127, 128, 128, 128, 128, 128,
    128, 128, 129, 129, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130, 131, 132,
    132, 132, 132, 132, 132, 133, 133, 133, 134, 134, 135, 135, 135, 135, 135, 135, 135, 135, 136, 136,
    136, 136, 136, 136, 136, 136, 136, 136, 137, 137, 137, 137, 137, 137, 137, 138, 138, 138, 138, 139,
    140, 140, 140, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 143, 144, 145, 145, 145, 145, 146,
    147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148, 148, 148, 148, 149,
    149, 150, 150, 150, 150, 150, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151,
    151, 152, 152, 152, 152, 152, 153, 153, 154, 154, 154, 154, 154, 155, 155, 155, 156, 156, 156, 156,
    156, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158, 159, 159, 160,
    160, 160, 160, 160, 160, 160, 161, 161, 161, 161, 161, 161, 161, 161, 161, 162, 162, 162, 162, 162};
static int32_t mapping_target_len = 163;


/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_map_to_sequence(void) {
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_map_to_sequence(void) {
    return 0;
}

static size_t nblock = 5;
static size_t seqlen = 6;
static int lower1[] = {0, 1, 2, 3, 3};
static int upper1[] = {1, 2, 3, 3, 6};


void test_bounds_not_null_map_to_sequence(void) {
    CU_ASSERT(!are_bounds_sane(NULL, NULL, nblock, seqlen));
    CU_ASSERT(!are_bounds_sane(lower1, NULL, nblock, seqlen));
    CU_ASSERT(!are_bounds_sane(NULL, upper1, nblock, seqlen));
}

void test_bounds_simple_map_to_sequence(void){
    CU_ASSERT(are_bounds_sane(lower1, upper1, nblock, seqlen));
}

void test_bounds_include_zero_map_to_sequence(void){
    int lower[] = {1, 1, 2, 3, 4};
    CU_ASSERT(!are_bounds_sane(lower, upper1, nblock, seqlen));
}

void test_bounds_include_lastposition_map_to_sequence(void){
    int upper[] = {0, 1, 2, 3, 4};
    CU_ASSERT(!are_bounds_sane(lower1, upper, nblock, seqlen));
}

void test_bounds_overlap_map_to_sequence(void){
    int lower[] = {0, 1, 2, 3, 4};
    CU_ASSERT(!are_bounds_sane(lower, upper1, nblock, seqlen));
}

void test_viterbi_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    float score = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, NULL);
    CU_ASSERT_NOT_EQUAL(score, NAN);

    logpost = free_scrappie_matrix(logpost);
}

void test_forward_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    float score = map_to_sequence_forward(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len);
    CU_ASSERT_NOT_EQUAL(score, NAN);

    logpost = free_scrappie_matrix(logpost);
}

void test_viterbi_with_path_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    const size_t nblock = logpost->nc;
    int * path = calloc(nblock, sizeof(int));
    CU_ASSERT_PTR_NOT_NULL_FATAL(path);

    float score = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, path);
    CU_ASSERT_NOT_EQUAL(score, NAN);

    (void)fputs("Viterbi mapping path\n",stdout);
    fprintf(stdout,"%d",path[0]);
    for(size_t blk=1 ; blk < nblock ; blk++){
        printf(", %d", path[blk]);
        if((blk + 1) % 20 == 0){
            fputc('\n', stdout);
        }
    }

    free(path);
    logpost = free_scrappie_matrix(logpost);
}

void test_forward_better_than_viterbi_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    float score_vit = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, NULL);
    float score_fwd = map_to_sequence_forward(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len);
    printf("vit = %f fwd = %f\n", score_vit, score_fwd);
    CU_ASSERT_TRUE(score_fwd >= score_vit);

    logpost = free_scrappie_matrix(logpost);
}

void test_full_band_viterbi_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    const size_t nblock = logpost->nc;

    int32_t * poslow = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poslow);
    int32_t * poshigh = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poshigh);
    for(size_t i=0 ; i < nblock ; i++){
        // Set bounds
        poslow[i] = 0;
        poshigh[i] = mapping_target_len;
    }

    float score_vit = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, NULL);
    float score_vitB = map_to_sequence_viterbi_banded(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, poslow, poshigh);
    printf("vit = %f vitB = %f\n", score_vit, score_vitB);
    CU_ASSERT_DOUBLE_EQUAL(score_vit, score_vitB, 1e-3);

    free(poshigh);
    free(poslow);
    logpost = free_scrappie_matrix(logpost);
}

void test_full_band_forward_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    const size_t nblock = logpost->nc;

    int32_t * poslow = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poslow);
    int32_t * poshigh = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poshigh);
    for(size_t i=0 ; i < nblock ; i++){
        // Set bounds
        poslow[i] = 0;
        poshigh[i] = mapping_target_len;
    }

    float score_fwd = map_to_sequence_forward(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len);
    float score_fwdB = map_to_sequence_forward_banded(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, poslow, poshigh);
    printf("fwd = %f fwdB = %f\n", score_fwd, score_fwdB);
    CU_ASSERT_DOUBLE_EQUAL(score_fwd, score_fwdB, 1e-3);

    free(poshigh);
    free(poslow);
    logpost = free_scrappie_matrix(logpost);
}


void test_relaxed_band_helper(float local_pen, int32_t band){

    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    const size_t nblock = logpost->nc;

    int32_t * poslow = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poslow);
    int32_t * poshigh = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(poshigh);
    for(size_t i=0 ; i < nblock ; i++){
        // Set bounds
        poslow[i] = imax(tightlow[i] - band, 0);
        poshigh[i] = imin(tightlow[i] + band, mapping_target_len);
    }

    float score_vit = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, local_pen, mapping_target, mapping_target_len, NULL);
    float score_vitB = map_to_sequence_viterbi_banded(logpost, 0.0f, 0.0f, local_pen, mapping_target, mapping_target_len, poslow, poshigh);
    printf("vit = %f vitB = %f\n", score_vit, score_vitB);
    CU_ASSERT_DOUBLE_EQUAL(score_vit, score_vitB, 1e-3);

    free(poshigh);
    free(poslow);
    logpost = free_scrappie_matrix(logpost);
}


void test_relaxed_band2_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(BIG_VAL, 2);
}

void test_relaxed_band3_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(BIG_VAL, 3);
}

void test_relaxed_band4_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(BIG_VAL, 4);
}

void test_relaxed_band5_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(BIG_VAL, 5);
}

static const float test_local_pen = 0.5f;
void test_relaxed_band2_with_local_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(test_local_pen, 2);
}

void test_relaxed_band3_with_local_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(test_local_pen, 3);
}

void test_relaxed_band4_with_local_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(test_local_pen, 4);
}

void test_relaxed_band5_with_local_viterbi_map_to_sequence(void){
    test_relaxed_band_helper(test_local_pen, 5);
}

void test_tight_band_forward_map_to_sequence(void){
    scrappie_matrix logpost = read_scrappie_matrix(posteriorfile);
    CU_ASSERT_PTR_NOT_NULL_FATAL(logpost);
    log_activation_inplace(logpost);

    const size_t nblock = logpost->nc;
    CU_ASSERT_EQUAL(nblock, 1000);

    int32_t * tighthigh = calloc(nblock, sizeof(int32_t));
    CU_ASSERT_PTR_NOT_NULL_FATAL(tightlow);
    for(size_t i=0 ; i < nblock ; i++){
        // Set bound
        tighthigh[i] = tightlow[i] - 1;
    }

    float score_fwd = map_to_sequence_forward(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len);
    float score_fwdB = map_to_sequence_forward_banded(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len,
                                                      tightlow, tighthigh);
    float score_vit = map_to_sequence_viterbi(logpost, 0.0f, 0.0f, BIG_VAL, mapping_target, mapping_target_len, NULL);
    printf("vit = %f fwd = %f fwdB = %f\n", score_vit, score_fwd, score_fwdB);
    CU_ASSERT_DOUBLE_EQUAL(score_fwdB, score_vit, 1e-3);
    CU_ASSERT_TRUE(score_fwd > score_fwdB);

    free(tighthigh);
    logpost = free_scrappie_matrix(logpost);
}


static test_with_description tests[] = {
    {"Test assumptions about bounds -- not null", test_bounds_not_null_map_to_sequence},
    {"Test assumptions about bounds -- simple pass", test_bounds_simple_map_to_sequence},
    {"Test assumptions about bounds -- includes zero", test_bounds_include_zero_map_to_sequence},
    {"Test assumptions about bounds -- includes last position", test_bounds_include_lastposition_map_to_sequence},
    {"Test assumptions about bounds -- overlap", test_bounds_overlap_map_to_sequence},
    {"Test mapping -- viterbi", test_viterbi_map_to_sequence},
    {"Test mapping -- forward", test_forward_map_to_sequence},
    {"Test mapping -- viterbi with path", test_viterbi_with_path_map_to_sequence},
    {"Test mapping -- forwards score exceeds Viterbi", test_forward_better_than_viterbi_map_to_sequence},
    {"Test mapping -- forwards with full band equals forwards", test_full_band_forward_map_to_sequence},
    {"Test mapping -- Viterbi with full band equals Viterbi", test_full_band_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band2 equals Viterbi", test_relaxed_band2_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band3 equals Viterbi", test_relaxed_band3_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band4 equals Viterbi", test_relaxed_band4_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band5 equals Viterbi", test_relaxed_band5_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band2 equals Viterbi, local pen", test_relaxed_band2_with_local_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band3 equals Viterbi, local pen", test_relaxed_band3_with_local_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band4 equals Viterbi, local pen", test_relaxed_band4_with_local_viterbi_map_to_sequence},
    {"Test mapping -- Viterbi with relaxed band5 equals Viterbi, local pen", test_relaxed_band5_with_local_viterbi_map_to_sequence},
    {0}};

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_map_to_sequence(void) {
    return scrappie_register_test_suite("Test map to sequence code, including banded routines",
                                        init_test_map_to_sequence, clean_test_map_to_sequence, tests);
    return 0;
}
