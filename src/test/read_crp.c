#include <err.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "scrappie_util.h"

int main(int argc, char * argv[]){
    if(4 != argc){
        fprintf(stderr, "Usage: read_crp filename nrow ncol\n");
        exit(EXIT_FAILURE);
    }
    scrappie_matrix mat = read_scrappie_matrix(argv[1]);
    if(NULL == mat){
        errx(EXIT_FAILURE, "Failed to read matrix from '%s'\n", argv[1]);
    }

    int nr = atoi(argv[2]);
    int nc = atoi(argv[3]);
    fprint_scrappie_matrix(stdout, NULL, mat, nr, nc, false);

    return EXIT_SUCCESS;
}
