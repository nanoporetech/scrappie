#ifndef LAYERS_H
#define LAYERS_H

#include "util.h"

Mat * window(const Mat * input, int w);
Mat * feedforward_linear(const Mat * X, const Mat * W,
		         const Mat * b, Mat * C);
Mat * feedforward_tanh(const Mat * X, const Mat * W,
		       const Mat * b, Mat * C);
Mat * feedforward_exp(const Mat * X, const Mat * W,
		      const Mat * b, Mat * C);
Mat * softmax(const Mat * X, const Mat * W,
	      const Mat * b, Mat * C);

Mat * feedforward2_tanh(const Mat * Xf, const Mat * Xb,
		        const Mat * Wf, const Mat * Wb,
			const Mat * b, Mat * C);

Mat * gru_forward(const Mat * X, const Mat * iW, const Mat * sW, const Mat * sW2, const Mat * b, Mat * res);
Mat * gru_backward(const Mat * X, const Mat * iW, const Mat * sW, const Mat * sW2, const Mat * b, Mat * res);
void gru_step(const Mat * x, const Mat * istate,
	      const Mat * xW, const Mat * sW, const Mat * sW2, const Mat * bias,
	      Mat * xF, Mat * ostate);

Mat * lstm_forward(const Mat * X, const Mat * iW, const Mat * sW, const Mat * b, const Mat * p, Mat * output);
Mat * lstm_backward(const Mat * X, const Mat * iW, const Mat * sW, const Mat * b, const Mat * p, Mat * output);
void lstm_step(const Mat * x, const Mat * out_prev,
	       const Mat * xW, const Mat * sW, const Mat * bias, const Mat * peep,
	       Mat * xF, Mat * state, Mat * output);
#endif /* LAYERS_H */
