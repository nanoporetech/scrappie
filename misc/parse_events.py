#!/usr/bin/env python3
import pickle
import math
import numpy as np
import sys


model_file = sys.argv[1]

def process_column(v, pad):
    """ process and pad """
    return [str(f) for f in v] + ["0.0"] * pad


def cformatM(fh, name, X):
    nrq = int(math.ceil(X.shape[1] / 4.0))
    pad = nrq * 4 - X.shape[1]
    lines = map(lambda v: ', '.join(process_column(v, pad)), X)

    fh.write('float {}[] = {}\n'.format('__' + name, '{'))
    fh.write('\t' + ',\n\t'.join(lines))
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', X.shape[1], nrq, X.shape[0], '__' + name, '}'))
    fh.write('const scrappie_matrix {} = &{};\n\n'.format(name, '_' + name))


def cformatV(fh, name, X):
    nrq = int(math.ceil(X.shape[0] / 4.0))
    pad = nrq * 4 - X.shape[0]
    lines = ', '.join(list(map(lambda f: str(f), X)) + ["0.0"] * pad)
    fh.write('float {}[] = {}\n'.format( '__' + name, '{'))
    fh.write('\t' + lines)
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', X.shape[0], nrq, 1, '__' + name, '}'))
    fh.write('const scrappie_matrix {} = &{};\n\n'.format(name, '_' + name))


def reshape_lstmM(mat):
	_, isize = mat.shape
	return mat.reshape((-1, 4, isize)).transpose([1, 0, 2]).reshape((-1, isize))


def reshape_lstmV(mat):
	return mat.reshape((-1, 4)).transpose().reshape(-1)


with open(model_file, 'rb') as fh:
    network = pickle.load(fh, encoding='latin1')

sys.stdout.write("""#pragma once
#ifndef NANONET_EVENTS_MODEL_H
#define NANONET_EVENTS_MODEL_H
#include "util.h"
""")

""" First LSTM layer
"""

bilstm1 = network.layers[1]
lstm = bilstm1.layers[0]
cformatM(sys.stdout, 'lstmF1_iW', reshape_lstmM(lstm.iW.get_value()))
cformatM(sys.stdout, 'lstmF1_sW', reshape_lstmM(lstm.sW.get_value()))
cformatV(sys.stdout, 'lstmF1_b', reshape_lstmV(lstm.b.get_value().reshape(-1)))
cformatV(sys.stdout, 'lstmF1_p', lstm.p.get_value().reshape(-1))

lstm = bilstm1.layers[1].layer
cformatM(sys.stdout, 'lstmB1_iW', reshape_lstmM(lstm.iW.get_value()))
cformatM(sys.stdout, 'lstmB1_sW', reshape_lstmM(lstm.sW.get_value()))
cformatV(sys.stdout, 'lstmB1_b', reshape_lstmV(lstm.b.get_value().reshape(-1)))
cformatV(sys.stdout, 'lstmB1_p', lstm.p.get_value().reshape(-1))


""" First feed forward layer
"""
size = network.layers[2].insize // 2
cformatM(sys.stdout, 'FF1_Wf', network.layers[2].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF1_Wb', network.layers[2].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF1_b', network.layers[2].b.get_value())


""" Second LSTM layer
"""
bilstm2 = network.layers[3]
lstm = bilstm2.layers[0]
cformatM(sys.stdout, 'lstmF2_iW', reshape_lstmM(lstm.iW.get_value()))
cformatM(sys.stdout, 'lstmF2_sW', reshape_lstmM(lstm.sW.get_value()))
cformatV(sys.stdout, 'lstmF2_b', reshape_lstmV(lstm.b.get_value().reshape(-1)))
cformatV(sys.stdout, 'lstmF2_p', lstm.p.get_value().reshape(-1))

lstm = bilstm2.layers[1].layer
cformatM(sys.stdout, 'lstmB2_iW', reshape_lstmM(lstm.iW.get_value()))
cformatM(sys.stdout, 'lstmB2_sW', reshape_lstmM(lstm.sW.get_value()))
cformatV(sys.stdout, 'lstmB2_b', reshape_lstmV(lstm.b.get_value().reshape(-1)))
cformatV(sys.stdout, 'lstmB2_p', lstm.p.get_value().reshape(-1))


""" Second feed forward layer
"""
size = network.layers[4].insize // 2
cformatM(sys.stdout, 'FF2_Wf', network.layers[4].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF2_Wb', network.layers[4].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF2_b', network.layers[4].b.get_value())


""" Softmax layer
"""
nstate = network.layers[5].W.get_value().shape[0]
shuffle = np.append(np.arange(nstate - 1) + 1, 0)
cformatM(sys.stdout, 'FF3_W', network.layers[5].W.get_value()[shuffle])
cformatV(sys.stdout, 'FF3_b', network.layers[5].b.get_value()[shuffle])

sys.stdout.write('#endif /* NANONET_EVENTS_MODEL_H */')
