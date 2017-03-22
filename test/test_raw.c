#include <assert.h>
#include <err.h>
#include <stdlib.h>

#include "fast5_interface.h"

int main(int argc, char * argv[]){
	if(1 == argc){
		errx(EXIT_FAILURE, "Please give files on commandline,");
	}

	for(int i=1 ; i < argc ; i++){
		raw_table rawtbl_unscaled = read_raw(argv[i], false);
		raw_table rawtbl_scaled = read_raw(argv[i], true);
		assert(rawtbl_unscaled.n == rawtbl_scaled.n);
		assert(rawtbl_unscaled.start == rawtbl_scaled.start);
		assert(rawtbl_unscaled.end == rawtbl_scaled.end);

		printf("* File %s has %lu samples (%lu, %lu).\n", argv[i], rawtbl_unscaled.n, rawtbl_unscaled.start, rawtbl_unscaled.end);
		for(int j=0 ; j < 10 ; j++){
			printf("  %f %f\n", rawtbl_unscaled.raw[i], rawtbl_scaled.raw[i]);
		}
		fputs("   ...\n", stdout);
		for(int j=0 ; j < 10 ; j++){
			const int idx = rawtbl_unscaled.n - j - 1;
			printf("  %f %f\n", rawtbl_unscaled.raw[idx], rawtbl_scaled.raw[idx]);
		}
		fputc('\n', stdout);
	}

	return EXIT_SUCCESS;
}

