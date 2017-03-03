#include <assert.h>
#include <cblas.h>
#include <stdlib.h>

#include <layers.h>

int main(void){

	Mat_rptr X = make_mat(3, 5);
        X->data.f[0] = 0.87459307; X->data.f[1] = -0.02721545; X->data.f[2] = 0.15777617; X->data.f[3] = 0.0;
        X->data.f[4] = -0.2574465; X->data.f[5] = 0.7703190; X->data.f[6] = 1.4487959; X->data.f[7] = 0.0;
        X->data.f[8] = -0.2223679; X->data.f[9] = 0.2092048; X->data.f[10] = -1.2084769; X->data.f[11] = 0.0;
        X->data.f[12] = -1.53680934; X->data.f[13] = -0.07612422; X->data.f[14] = 0.13584755; X->data.f[15] = 0.0;
        X->data.f[16] = 0.7246051; X->data.f[17] = -0.2746767; X->data.f[18] = 0.1908710; X->data.f[19] = 0.0;
	fprint_mat(stdout, "Input", X, 0, 0);

 	Mat_rptr W = make_mat(7, 2);
        // Filter 1
        W->data.f[0] = -1.0; W->data.f[1] = -1.0; W->data.f[2] = -1.0; W->data.f[3] = 0.0;
	W->data.f[4] = 1.0; W->data.f[5] = 1.0; W->data.f[6] = 1.0; W->data.f[7] = 0.0;
        // Filter 2
	W->data.f[8] = 1.0; W->data.f[9] = 1.0; W->data.f[10] = 1.0; W->data.f[11] = 0.0;
	W->data.f[12] = -1.0; W->data.f[13] = -1.0; W->data.f[14] = -1.0; W->data.f[15] = 0.0;
	fprint_mat(stdout, "Filters", W, 0, 0);


	Mat_rptr C = Convolution(X, W, 1, NULL);
	fprint_mat(stdout, "Output1 -- stride 1", C, 0, 0);
	zero_mat(C);
	C = Convolution(X, W, 2, C);
	fprint_mat(stdout, "Output1 -- stride 2", C, 0, 0);
	zero_mat(C);
	C = Convolution(X, W, 3, C);
	fprint_mat(stdout, "Output1 -- stride 3", C, 0, 0);


 	Mat_rptr W2 = make_mat(11, 2);
        // Filter 1
        W2->data.f[0] = -1.0; W2->data.f[1] = -1.0; W2->data.f[2] = -1.0; W2->data.f[3] = 0.0;
	W2->data.f[4] = 1.0; W2->data.f[5] = 1.0; W2->data.f[6] = 1.0; W2->data.f[7] = 0.0;
	W2->data.f[8] = 1.0; W2->data.f[9] = 1.0; W2->data.f[10] = 1.0; W2->data.f[11] = 0.0;
        // Filter 2
	W2->data.f[12] = 1.0; W2->data.f[13] = 1.0; W2->data.f[14] = 1.0; W2->data.f[15] = 0.0;
	W2->data.f[16] = -1.0; W2->data.f[17] = -1.0; W2->data.f[18] = -1.0; W2->data.f[19] = 0.0;
	W2->data.f[20] = -1.0; W2->data.f[21] = -1.0; W2->data.f[22] = -1.0; W2->data.f[23] = 0.0;
	fprint_mat(stdout, "Filters2", W2, 0, 0);

	C = Convolution(X, W2, 1, C);
	fprint_mat(stdout, "Output2 -- stride 1", C, 0, 0);
	zero_mat(C);
	C = Convolution(X, W2, 2, C);
	fprint_mat(stdout, "Output2 -- stride 2", C, 0, 0);
	zero_mat(C);
	C = Convolution(X, W2, 3, C);
	fprint_mat(stdout, "Output2 -- stride 3", C, 0, 0);

	free_mat(&C);
	free_mat(&W2);
	free_mat(&W);
	free_mat(&X);
	return EXIT_SUCCESS;
}

