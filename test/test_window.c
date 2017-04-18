#include <assert.h>
#include <stdio.h>

#include <layers.h>
#include <util.h>

#define NCOL 100

int main(int argc, char * argv[]){
	assert(argc == 3);
	int w = atoi(argv[1]);
	int stride = atoi(argv[2]);

	Mat_rptr mat = make_mat(1, NCOL);
	for(int i=0 ; i<NCOL ; i++){
		mat->data.f[i * mat->nrq * 4] = i + 1;
		//mat->data.f[i * mat->nrq * 4 + 1] = -i - 1;
	}

	Mat_rptr matW = window(mat, w, stride);

	fputs("* Input matrix\n", stdout);
	fprint_mat(stdout, NULL, mat, -1, -1);

	fputs("* Window matrix\n", stdout);
	fprint_mat(stdout, NULL, matW, -1, -1);
}
