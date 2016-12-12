#ifndef LAYERS_H
#define LAYERS_H

#include "util.h"

Mat_rptr window(const Mat_rptr input, int w);
Mat_rptr feedforward_linear(const Mat_rptr X, const Mat_rptr W,
		            const Mat_rptr b, Mat_rptr C);
Mat_rptr feedforward_tanh(const Mat_rptr X, const Mat_rptr W,
		          const Mat_rptr b, Mat_rptr C);
Mat_rptr feedforward_exp(const Mat_rptr X, const Mat_rptr W,
		         const Mat_rptr b, Mat_rptr C);
Mat_rptr softmax(const Mat_rptr X, const Mat_rptr W,
	         const Mat_rptr b, Mat_rptr C);

Mat_rptr feedforward2_tanh(const Mat_rptr Xf, const Mat_rptr Xb,
		           const Mat_rptr Wf, const Mat_rptr Wb,
			   const Mat_rptr b, Mat_rptr C);

Mat_rptr gru_forward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW,
		     const Mat_rptr sW2, const Mat_rptr b, Mat_rptr res);
Mat_rptr gru_backward(const Mat_rptr X, const Mat_rptr iW, const Mat_rptr sW,
		      const Mat_rptr sW2, const Mat_rptr b, Mat_rptr res);
void gru_step(const Mat_rptr x, const Mat_rptr istate,
	      const Mat_rptr xW, const Mat_rptr sW, const Mat_rptr sW2,
	      const Mat_rptr bias, Mat_rptr xF, Mat_rptr ostate);

Mat_rptr lstm_forward(const Mat_rptr X, const Mat_rptr sW,
		      const Mat_rptr p, Mat_rptr output);
Mat_rptr lstm_backward(const Mat_rptr X, const Mat_rptr sW,
		       const Mat_rptr p, Mat_rptr output);
void lstm_step(const Mat_rptr x, const Mat_rptr out_prev,
	       const Mat_rptr sW,
	       const Mat_rptr peep, Mat_rptr xF, Mat_rptr state, Mat_rptr output);
#endif /* LAYERS_H */
