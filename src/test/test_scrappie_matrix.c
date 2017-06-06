#include "scrappie_matrix_util.h"

#include <CUnit/Basic.h>
#include <err.h>
#include <stdio.h>


static const char testfile[] = "test_matrix.crp";

FILE * infile = NULL;
FILE * outfile = NULL;
scrappie_matrix mat = NULL;


/**  Initialise scrappie matrix test
 *
 *   Opens a test file for reading and a temporary file for writing.
 *   Creates a random scrappie matrix
 *
 *  @returns 0 on success, non-zero on failure
 **/
int init_test_scrappie_matrix(void){
	infile = fopen(testfile, "r");
	if(NULL == infile){
		warnx("Failed to open %s to read matrix from.\n", testfile);
	}

	//outfile = tmpfile();
	outfile = tmpfile();
	if(NULL == outfile){
		warnx("Failed to open temporary file to write to.\n");
	}

	mat = random_scrappie_matrix(5, 9, -1.0, 1.0);
	if(NULL == mat){
		warnx("Failed to create random scrappie matrix.\n");
	}

	return (NULL != infile && NULL != outfile && NULL != mat) ? 0 : -1;
}

/**  Clean up after scrappie matrix test
 *
 *   Closes file handles
 *
 *  @returns 0 on success, non-zero on failure
 **/
int clean_test_scrappie_matrix(void){
	int ret = fclose(infile);
	ret |= fclose(outfile);
	(void) free_scrappie_matrix(mat);
	return ret;
}

void test_read_scrappie_matrix(void){
	scrappie_matrix mat = read_scrappie_matrix(infile);
	CU_ASSERT(NULL != mat);
	CU_ASSERT(validate_scrappie_matrix(mat, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
	mat = free_scrappie_matrix(mat);
}

void test_write_scrappie_matrix(void){
	const int nelt = mat->nc * mat->nr;
	int ret = write_scrappie_matrix(outfile, mat);
	CU_ASSERT(ret == nelt);
}

void test_roundtrip_scrappie_matrix(void){
	rewind(outfile);
	scrappie_matrix mat_in = read_scrappie_matrix(outfile);
	CU_ASSERT(NULL != mat_in);
	CU_ASSERT(equality_scrappie_matrix(mat_in, mat, 0.0));
	mat = free_scrappie_matrix(mat_in);
}


int main(void){
	CU_pSuite pSuite = NULL;

	// Initialise CUnit
	if (CUE_SUCCESS != CU_initialize_registry()){
		return CU_get_error();
	}

	// Add test to suite
	pSuite = CU_add_suite("Scrappie matrix IO tests", init_test_scrappie_matrix,
			      clean_test_scrappie_matrix);
	if (NULL == pSuite){
		CU_cleanup_registry();
		return CU_get_error();
	}

	// Add tests to the suite
	if ((NULL == CU_add_test(pSuite, "Reading scrappie_matrix from file", test_read_scrappie_matrix)) ||
	    (NULL == CU_add_test(pSuite, "Writing scrappie_matrix to file", test_write_scrappie_matrix)) ||
	    (NULL == CU_add_test(pSuite, "Round-trip scrappie_matrix to / from file", test_roundtrip_scrappie_matrix))){
		CU_cleanup_registry();
		return CU_get_error();
	}

	// Run all tests using the CUnit Basic interface
	CU_basic_set_mode(CU_BRM_VERBOSE);
	CU_basic_run_tests();
	CU_cleanup_registry();
	return CU_get_error();
}
