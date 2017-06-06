#include <CUnit/Basic.h>

int register_scrappie_matrix_util(void);

int main(void){

	// Initialise CUnit
	if (CUE_SUCCESS != CU_initialize_registry()){
		return CU_get_error();
	}

	int err = register_scrappie_matrix_util();
	if(err){
		CU_cleanup_registry();
		return err;
	}

	// Run all tests using the CUnit Basic interface
	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	CU_cleanup_registry();
	return CU_get_error();
}

