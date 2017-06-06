#include <CUnit/Basic.h>
#include "test_scrappie_matrix_util.h"

int main(void){

	// Initialise CUnit
	if (CUE_SUCCESS != CU_initialize_registry()){
		return CU_get_error();
	}

	//  Register test suites
	CU_SuiteInfo suites[] = {
		scrappie_matrix_util_suite,
		CU_SUITE_INFO_NULL};

	CU_register_suites(suites);


	// Run all tests using the CUnit Basic interface
	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	CU_cleanup_registry();
	return CU_get_error();
}

