#ifndef LAYERS_H
#define LAYERS_H

#include "util.h"

scrappie_matrix window(const scrappie_matrix input, int w, int stride);
scrappie_matrix Convolution(const scrappie_matrix X, const scrappie_matrix W, const scrappie_matrix b,
		     int stride, scrappie_matrix C);
scrappie_matrix feedforward_linear(const scrappie_matrix X, const scrappie_matrix W,
		            const scrappie_matrix b, scrappie_matrix C);
scrappie_matrix feedforward_tanh(const scrappie_matrix X, const scrappie_matrix W,
		          const scrappie_matrix b, scrappie_matrix C);
scrappie_matrix feedforward_exp(const scrappie_matrix X, const scrappie_matrix W,
		         const scrappie_matrix b, scrappie_matrix C);
scrappie_matrix softmax(const scrappie_matrix X, const scrappie_matrix W,
	         const scrappie_matrix b, scrappie_matrix C);

scrappie_matrix feedforward2_tanh(const scrappie_matrix Xf, const scrappie_matrix Xb,
		           const scrappie_matrix Wf, const scrappie_matrix Wb,
			   const scrappie_matrix b, scrappie_matrix C);

scrappie_matrix gru_forward(const scrappie_matrix X, const scrappie_matrix iW, const scrappie_matrix sW,
		     const scrappie_matrix sW2, const scrappie_matrix b, scrappie_matrix res);
scrappie_matrix gru_backward(const scrappie_matrix X, const scrappie_matrix iW, const scrappie_matrix sW,
		      const scrappie_matrix sW2, const scrappie_matrix b, scrappie_matrix res);
void gru_step(const scrappie_matrix x, const scrappie_matrix istate,
	      const scrappie_matrix xW, const scrappie_matrix sW, const scrappie_matrix sW2,
	      const scrappie_matrix bias, scrappie_matrix xF, scrappie_matrix ostate);

scrappie_matrix lstm_forward(const scrappie_matrix X, const scrappie_matrix sW,
		      const scrappie_matrix p, scrappie_matrix output);
scrappie_matrix lstm_backward(const scrappie_matrix X, const scrappie_matrix sW,
		       const scrappie_matrix p, scrappie_matrix output);
void lstm_step(const scrappie_matrix x, const scrappie_matrix out_prev,
	       const scrappie_matrix sW,
	       const scrappie_matrix peep, scrappie_matrix xF, scrappie_matrix state, scrappie_matrix output);
#endif /* LAYERS_H */
