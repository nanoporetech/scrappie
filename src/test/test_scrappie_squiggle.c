#define BANANA 1
#include <CUnit/Basic.h>
#include <stdbool.h>

#include <test_common.h>

#include <networks.h>

static const int sequence[100] = {
        1, 0, 3, 3, 2, 1, 0, 1, 3, 1, 1, 0, 2, 1, 1, 3, 2, 1, 3, 2,
        2, 2, 3, 2, 0, 1, 0, 2, 2, 2, 3, 2, 0, 2, 0, 1, 3, 1, 1, 0,
        3, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 3, 0, 3, 3,
        0, 3, 3, 0, 0, 0, 0, 0, 3, 1, 0, 0, 2, 3, 3, 3, 1, 1, 1, 2};
static const size_t nseqbase = 100;

/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_squiggle(void) {
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_squiggle(void) {
    return 0;
}

void test_short_squiggle_original_units(void) {
    scrappie_matrix squiggle = squiggle_r94(sequence, nseqbase, false);
    CU_ASSERT_PTR_NOT_NULL_FATAL(squiggle);
    fprint_scrappie_matrix(stdout, "#  Squiggle with network parameters", squiggle, 0, 0, false);

    free_scrappie_matrix(squiggle);
    CU_ASSERT_PTR_NULL(squiggle);
}

void test_short_squiggle_transformed_units(void) {
    scrappie_matrix squiggle = squiggle_r94(sequence, nseqbase, true);
    CU_ASSERT_PTR_NOT_NULL_FATAL(squiggle);
    fprint_scrappie_matrix(stdout, "#  Squiggle with transformed parameters", squiggle, 0, 0, false);

    free_scrappie_matrix(squiggle);
    CU_ASSERT_PTR_NULL(squiggle);
}

static test_with_description tests[] = {
    {"Short sequence to squiggle with network parameterisation", test_short_squiggle_original_units},
    {"Short sequence to squiggle with transformed parameterisation", test_short_squiggle_transformed_units},
    {0}};

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_squiggle(void) {
    return scrappie_register_test_suite("Test sequence-to-squiggle", init_test_squiggle, clean_test_squiggle, tests);
    return 0;
}
