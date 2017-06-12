#include "scrappie_matrix_util.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (3 != argc) {
        fputs("Usage: write_crp nr nc\n", stderr);
        return EXIT_FAILURE;
    }

    int nr = atoi(argv[1]);
    int nc = atoi(argv[2]);
    assert(nr > 1);
    assert(nc > 1);

    scrappie_matrix mat = random_scrappie_matrix(nr, nc, -1.0, 1.0);
    int ret = write_scrappie_matrix(stdout, mat);
    assert(mat->nr * mat->nc == ret);
    free_scrappie_matrix(mat);

    return EXIT_SUCCESS;
}
