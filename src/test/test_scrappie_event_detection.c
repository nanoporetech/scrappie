#define BANANA 1
#include <CUnit/Basic.h>
#include <err.h>
#include <stdbool.h>
#include <stdlib.h>

#include <scrappie_structures.h>
#include "test_common.h"
#include <util.h>

void compute_sum_sumsq(const float *data, double *sum,
                       double *sumsq, size_t d_length);
float *compute_tstat(const double *sum, const double *sumsq,
                     size_t d_length, size_t w_length);
event_table create_events(size_t const *peaks, double const *sums,
                          double const *sumsqs, size_t nsample);

static event_table events = {0};
const size_t nevent = 10;


/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_eventdetection(void) {
    const float raw[] = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f,
                         3.0f, 4.0f, 4.0f, 5.0f, 5.0f,
                         6.0f, 6.0f, 7.0f, 7.0f, 8.0f,
                         8.0f, 9.0f, 9.0f, 10.0f, 10.0f};
    const size_t peaks[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const size_t n = 20;
    double * sum = calloc(n + 1, sizeof(double));
    double * sumsq = calloc(n + 1, sizeof(double));

    compute_sum_sumsq(raw, sum, sumsq, n);
    events = create_events(peaks, sum, sumsq, n);

    free(sumsq);
    free(sum);

    if(NULL == sum || NULL == sumsq){
        //  Allocation failure occurred
        return 1;
    }
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_eventdetection(void) {
    free(events.event);
    return 0;
}

void test_cumulative_sums(void) {
    const float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const double expt_sum[] = {0.0f, 1.0f, 3.0f, 6.0f, 10.f, 15.0f,
                               21.0f, 28.0f, 36.0f, 45.0f, 55.0f};
    const double expt_sumsq[] = {0.0f, 1.0f, 5.0f, 14.0f, 30.0f, 55.0f,
                                 91.0f, 140.0f, 204.0f, 285.0f, 385.0f};
    const size_t n = 10;

    double * sum = calloc(n + 1, sizeof(double));
    double * sumsq = calloc(n + 1, sizeof(double));
    CU_ASSERT_PTR_NOT_NULL_FATAL(sum);
    CU_ASSERT_PTR_NOT_NULL_FATAL(sumsq);

    compute_sum_sumsq(data, sum, sumsq, n);

    CU_ASSERT_TRUE(equality_array(sum, expt_sum, n + 1, 0.0));
    CU_ASSERT_TRUE(equality_array(sumsq, expt_sumsq, n + 1, 0.0));

    free(sumsq);
    free(sum);
}

void test_calculation_tstat(void){
#define _BOUNDARY 1.15470054f, 2.0f, 3.46410162f, 100.0f, 3.46410162f, 2.0f, 1.15470054f
    const int winlen = 4;
    const float data[] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float expt[] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        _BOUNDARY,
        0.0f, 0.0f, 0.0f,
        _BOUNDARY,
        0.0f, 0.0f, 0.0f,
        _BOUNDARY,
        0.0f, 0.0f, 0.0f,
        _BOUNDARY,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const size_t n = 50;

    double * sum = calloc(n + 1, sizeof(double));
    double * sumsq = calloc(n + 1, sizeof(double));
    CU_ASSERT_PTR_NOT_NULL_FATAL(sum);
    CU_ASSERT_PTR_NOT_NULL_FATAL(sum);

    compute_sum_sumsq(data, sum, sumsq, n);
    float * tstat = compute_tstat(sum, sumsq, n, winlen);
    CU_ASSERT_PTR_NOT_NULL_FATAL(tstat);

    // Squash large values
    for(size_t i=0 ; i < n ; i++){
        if(tstat[i] > 100.0){
            tstat[i] = 100.0;
        }
    }
    CU_ASSERT_TRUE(equality_arrayf(tstat, expt, n, 1e-5));

    free(tstat);
    free(sumsq);
    free(sum);
}

void test_correct_number_of_events(void){
    CU_ASSERT_EQUAL(events.n, nevent);
}

void test_correct_starts(void){
    for(size_t i=0 ; i < events.n ; i++){
        warnx("start: %zu %zu %zu", i, events.event[i].start, i * 2);
        CU_ASSERT_EQUAL(events.event[i].start, i * 2);
    }
}

void test_correct_lengths(void){
    for(size_t i=0 ; i < events.n ; i++){
        warnx("start: %zu %f %f", i, events.event[i].length, 2.0f);
        CU_ASSERT_EQUAL(events.event[i].length, 2.0f);
    }
}

void test_correct_means(void){
    for(size_t i=0 ; i < events.n ; i++){
        warnx("start: %zu %f %f", i, events.event[i].mean, (float)(i+1));
        CU_ASSERT_EQUAL(events.event[i].mean, (float)(i+1));
    }
}

void test_correct_stdv(void){
    for(size_t i=0 ; i < events.n ; i++){
        warnx("start: %zu %f %f", i, events.event[i].stdv, 0.0f);
        CU_ASSERT_EQUAL(events.event[i].stdv, 0.0f);
    }
}


static test_with_description tests[] = {
    {"Cumulative sum and sums", test_cumulative_sums},
    {"Calculation of t-statistic", test_calculation_tstat},
    {"Correct number of events", test_correct_number_of_events},
    {"Correct event start times", test_correct_starts},
    {"Correct event length", test_correct_lengths},
    {"Correct event means", test_correct_means},
    {"Correct event stdv", test_correct_stdv},
    {0}};


/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_eventdetection(void) {
    return scrappie_register_test_suite("Tests for event detection ported from Dragonet", init_test_eventdetection, clean_test_eventdetection, tests);
}
