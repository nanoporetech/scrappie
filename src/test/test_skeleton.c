#include <CUnit/Basic.h>
#include <stdbool.h>

/**  Initialise test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int init_test_skeleton(void) {
    return 0;
}

/**  Clean up after test
 *
 *   @returns 0 on success, non-zero on failure
 **/
int clean_test_skeleton(void) {
    return 0;
}

void test_nop_skeleton(void) {
    CU_ASSERT(true);
}

/**   Register tests with CUnit
 *
 *    @returns 0 on success, non-zero on failure
 **/
int register_test_skeleton(void) {
    CU_pSuite suite = CU_add_suite("Skeleton set of tests for modification",
                                   init_test_skeleton,
                                   clean_test_skeleton);
    if (NULL == suite) {
        return CU_get_error();
    }

    if (NULL ==
        CU_add_test(suite, "Skeleton test doing no-op", test_nop_skeleton)) {
        return CU_get_error();
    }

    return 0;
}
