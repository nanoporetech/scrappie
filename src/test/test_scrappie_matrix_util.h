#ifndef SCRAPPIE_MATRIX_UTIL
#define SCRAPPIE_MATRIX_UTIL

#include <CUnit/TestDB.h>

int init_test_scrappie_matrix_util(void);
int clean_test_scrappie_matrix_util(void);
void test_read_scrappie_matrix_util(void);
void test_write_scrappie_matrix_util(void);
void test_roundtrip_scrappie_matrix_util(void);

CU_TestInfo scrappie_matrix_util_tests[] = {
	{"Reading scrappie_matrix from file", test_read_scrappie_matrix_util},
	{"Writing scrappie_matrix to file", test_write_scrappie_matrix_util},
	{"Round-trip scrappie_matrix to / from file", test_roundtrip_scrappie_matrix_util},
	CU_TEST_INFO_NULL};

CU_SuiteInfo scrappie_matrix_util_suite = {
	"Scrappie matrix IO tests", init_test_scrappie_matrix_util,
	clean_test_scrappie_matrix_util, NULL, NULL, scrappie_matrix_util_tests};

#endif /* SCRAPPIE_MATRIX_UTIL */
