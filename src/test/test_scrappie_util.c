// Needed for mkstemp and fdopen
#define BANANA 1
#define _BSD_SOURCE 1
#define _POSIX_SOURCE 1

#include "scrappie_util.h"

#include <CUnit/Basic.h>
#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "scrappie_util.h"
#include "test_common.h"

static const char testfile[] = "test_matrix.crp";

static FILE *infile = NULL;
static FILE *outfile = NULL;
static scrappie_matrix mat = NULL;
char scrappie_matrix_tmpfile_name[] = "scrappie_matrix_file_XXXXXX";

/**  Initialise scrappie matrix test
 *
 *   Opens a test file for reading and a temporary file for writing.
 *   Creates a random scrappie matrix
 *
 *  @returns 0 on success, non-zero on failure
 **/
int init_test_scrappie_util(void) {
    infile = fopen(testfile, "r");
    if (NULL == infile) {
        warnx("Failed to open %s to read matrix from.\n", testfile);
    }

    (void)umask(022);
    int outfileno = mkstemp(scrappie_matrix_tmpfile_name);
    if (-1 != outfileno) {
        outfile = fdopen(outfileno, "w+b");
    }
    if (-1 == outfileno || NULL == outfile) {
        warnx("Failed to open temporary file to write to.\n");
    }

    mat = random_scrappie_matrix(5, 9, -1.0, 1.0);
    if (NULL == mat) {
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
int clean_test_scrappie_util(void) {
    int ret = fclose(infile);
    ret |= fclose(outfile);
    ret |= remove(scrappie_matrix_tmpfile_name);
    (void)free_scrappie_matrix(mat);
    return ret;
}

void test_read_matrix_scrappie_util(void) {
    scrappie_matrix mat_in = read_scrappie_matrix_from_handle(infile);
    CU_ASSERT_FATAL(NULL != mat_in);
    CU_ASSERT(validate_scrappie_matrix
              (mat_in, -1.0, 1.0, 0.0, true, __FILE__, __LINE__));
    mat_in = free_scrappie_matrix(mat_in);
}

void test_write_matrix_scrappie_util(void) {
    const int nelt = mat->nc * mat->nr;
    int ret = write_scrappie_matrix_to_handle(outfile, mat);
    CU_ASSERT(ret == nelt);
}

void test_roundtrip_matrix_scrappie_util(void) {
    rewind(outfile);
    scrappie_matrix mat_in = read_scrappie_matrix_from_handle(outfile);
    CU_ASSERT_FATAL(NULL != mat_in);
    CU_ASSERT(equality_scrappie_matrix(mat_in, mat, 0.0));
    mat_in = free_scrappie_matrix(mat_in);
}

void test_copy_matrix_scrappie_util(void) {
    scrappie_matrix mat_cpy = copy_scrappie_matrix(mat);
    CU_ASSERT_FATAL(NULL != mat_cpy);
    CU_ASSERT(equality_scrappie_matrix(mat_cpy, mat, 0.0));
    free_scrappie_matrix(mat_cpy);
}

void test_tofrom_array_scrappie_util(void) {
    float * array = array_from_scrappie_matrix(mat);
    CU_ASSERT_FATAL(NULL != array);
    scrappie_matrix mat_cpy = mat_from_array(array, mat->nr, mat->nc);
    CU_ASSERT_FATAL(NULL != mat_cpy);
    CU_ASSERT(equality_scrappie_matrix(mat_cpy, mat, 0.0));
    (void)free_scrappie_matrix(mat_cpy);
    free(array);
}


static test_with_description tests[] = {
    {"Reading scrappie_matrix from file", test_read_matrix_scrappie_util},
    {"Writing scrappie_matrix to file", test_write_matrix_scrappie_util},
    {"Round-trip scrappie_matrix to / from file", test_roundtrip_matrix_scrappie_util},
    {"Copy scrappie_matrix", test_copy_matrix_scrappie_util},
    {"Round-trip scrappie_matrix to / from array", test_tofrom_array_scrappie_util},
    {0}};


int register_scrappie_util(void) {
    // Would be preferable to contruct CU_SuiteInfo and use CU_register_suites
    // but this is incompatible between Trusty (CUnit 2.1-2) and Xenial (CUnit 2.1-3)
    // due to the addition of setup and teardown functions to the CU_SuiteInfo structure

    return scrappie_register_test_suite("Scrappie matrix IO tests", init_test_scrappie_util, clean_test_scrappie_util, tests);
}
