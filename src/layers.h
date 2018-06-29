#pragma once
#ifndef LAYERS_H
#    define LAYERS_H

#    include "scrappie_matrix.h"

void tanh_activation_inplace(scrappie_matrix C);
void exp_activation_inplace(scrappie_matrix C);
void log_activation_inplace(scrappie_matrix C);
void elu_activation_inplace(scrappie_matrix C);
void robustlog_activation_inplace(scrappie_matrix C, float min_prob);

scrappie_matrix embedding(int const * index, size_t n, const_scrappie_matrix E,
                          scrappie_matrix C);
scrappie_matrix window(const_scrappie_matrix input, size_t w, size_t stride);
scrappie_matrix convolution(const_scrappie_matrix X, const_scrappie_matrix W,
                            const_scrappie_matrix b, size_t stride,
                            scrappie_matrix C);
scrappie_matrix feedforward_linear(const_scrappie_matrix X,
                                   const_scrappie_matrix W,
                                   const_scrappie_matrix b, scrappie_matrix C);
scrappie_matrix feedforward_tanh(const_scrappie_matrix X,
                                 const_scrappie_matrix W,
                                 const_scrappie_matrix b, scrappie_matrix C);
scrappie_matrix feedforward_exp(const_scrappie_matrix X,
                                const_scrappie_matrix W,
                                const_scrappie_matrix b, scrappie_matrix C);
scrappie_matrix residual(const_scrappie_matrix X, const_scrappie_matrix fX,
                         scrappie_matrix C);
void residual_inplace(const_scrappie_matrix X, scrappie_matrix fX);
scrappie_matrix softmax(const_scrappie_matrix X, const_scrappie_matrix W,
                        const_scrappie_matrix b, scrappie_matrix C);
scrappie_matrix softmax_with_temperature(scrappie_matrix X, const_scrappie_matrix W,
                                         const_scrappie_matrix b, float tempW, float tempb,
                                         scrappie_matrix C);

scrappie_matrix feedforward2_tanh(const_scrappie_matrix Xf,
                                  const_scrappie_matrix Xb,
                                  const_scrappie_matrix Wf,
                                  const_scrappie_matrix Wb,
                                  const_scrappie_matrix b, scrappie_matrix C);

scrappie_matrix gru_forward(const_scrappie_matrix X, const_scrappie_matrix sW,
                            const_scrappie_matrix sW2, scrappie_matrix res);
scrappie_matrix gru_backward(const_scrappie_matrix X, const_scrappie_matrix sW,
                             const_scrappie_matrix sW2, scrappie_matrix res);
void gru_step(const_scrappie_matrix x, const_scrappie_matrix istate,
              const_scrappie_matrix sW, const_scrappie_matrix sW2,
              scrappie_matrix xF, scrappie_matrix ostate);

scrappie_matrix grumod_forward(const_scrappie_matrix X, const_scrappie_matrix sW,
                               scrappie_matrix res);
scrappie_matrix grumod_backward(const_scrappie_matrix X, const_scrappie_matrix sW,
                                scrappie_matrix res);
void grumod_step(const_scrappie_matrix x, const_scrappie_matrix istate,
                 const_scrappie_matrix sW, scrappie_matrix xF,
                 scrappie_matrix ostate);

scrappie_matrix lstm_forward(const_scrappie_matrix X, const_scrappie_matrix sW,
                             const_scrappie_matrix p, scrappie_matrix output);
scrappie_matrix lstm_backward(const_scrappie_matrix X, const_scrappie_matrix sW,
                              const_scrappie_matrix p, scrappie_matrix output);
void lstm_step(const_scrappie_matrix x, const_scrappie_matrix out_prev,
               const_scrappie_matrix sW,
               const_scrappie_matrix peep, scrappie_matrix xF,
               scrappie_matrix state, scrappie_matrix output);


scrappie_matrix globalnorm(const_scrappie_matrix X, const_scrappie_matrix W,
                           const_scrappie_matrix b, scrappie_matrix C);
float crf_partition_function(const_scrappie_matrix C);
#endif                          /* LAYERS_H */
