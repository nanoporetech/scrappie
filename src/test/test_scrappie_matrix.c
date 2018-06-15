#define BANANA 1
#include <CUnit/Basic.h>
#include <stdbool.h>

#include <scrappie_matrix.h>
#include <test_common.h>

/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_scrappie_matrix(void) {
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_scrappie_matrix(void) {
    return 0;
}

void test_rownormalise_scrappie_matrix_helper(int nr) {
    scrappie_matrix mat = make_scrappie_matrix(nr, 1);
    CU_ASSERT_PTR_NOT_NULL_FATAL(mat);
    const int stride = mat->stride;
    for(int i=0 ; i < stride ; i++){
        mat->data.f[i] = 1.0f;
    }
    row_normalise_inplace(mat);

    const float expected = 1.0f / nr;
    for(int i=0 ; i < nr ; i++){
        CU_ASSERT_DOUBLE_EQUAL(mat->data.f[i], expected, 1e-5);
    }
    free_scrappie_matrix(mat);
}

void test_rownormalise_nr08scrappie_matrix(void){
    test_rownormalise_scrappie_matrix_helper(8);
}
void test_rownormalise_nr09scrappie_matrix(void){
    test_rownormalise_scrappie_matrix_helper(9);
}
void test_rownormalise_nr10scrappie_matrix(void){
    test_rownormalise_scrappie_matrix_helper(10);
}
void test_rownormalise_nr11scrappie_matrix(void){
    test_rownormalise_scrappie_matrix_helper(11);
}

static test_with_description tests[] = {
    {"Row normalisation edge case nr  8", test_rownormalise_nr08scrappie_matrix},
    {"Row normalisation edge case nr  9", test_rownormalise_nr09scrappie_matrix},
    {"Row normalisation edge case nr 10", test_rownormalise_nr10scrappie_matrix},
    {"Row normalisation edge case nr 11", test_rownormalise_nr11scrappie_matrix},
    {0}};

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_matrix(void) {
    return scrappie_register_test_suite("Functions on scrappie matrices", init_test_scrappie_matrix, clean_test_scrappie_matrix, tests);
    return 0;
}
