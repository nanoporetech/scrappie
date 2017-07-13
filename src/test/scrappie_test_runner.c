#include <stdlib.h>
#include <CUnit/Basic.h>

int register_scrappie_matrix_util(void);
int register_test_skeleton(void);
int register_test_convolution(void);

int (*test_suites[]) (void) = {
    register_test_skeleton, register_scrappie_matrix_util, register_test_convolution, NULL      // Last element of array should be NULL
};

int main(void) {

    // Initialise CUnit
    if (CUE_SUCCESS != CU_initialize_registry()) {
        return CU_get_error();
    }
    // Register test suites
    for (int i = 0; NULL != test_suites[i]; ++i) {
        int err = test_suites[i] ();
        if (err) {
            CU_cleanup_registry();
            return err;
        }
    }

    // Run all tests using the CUnit Basic interface
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    int failures = CU_get_number_of_failures();
    CU_cleanup_registry();

    return (failures > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
