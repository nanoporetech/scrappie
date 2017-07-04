#!/usr/bin/env python3
import pickle
import math
import numpy as np
import sys


model_file = sys.argv[1]


def process_column(v, pad):
    """ process and pad """
    return [str(f) for f in v] + ["0.0"] * pad


def cformatM(fh, name, X, nr=None, nc=None):
    nrq = int(math.ceil(X.shape[1] / 4.0))
    pad = nrq * 4 - X.shape[1]
    lines = map(lambda v: ', '.join(process_column(v, pad)), X)

    if nr is None:
        nr = X.shape[1]
    else:
        nrq = int(math.ceil(nr / 4.0))
    if nc is None:
        nc = X.shape[0]

    fh.write('float {}[] = {}\n'.format('__' + name, '{'))
    fh.write('\t' + ',\n\t'.join(lines))
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', nr, nrq, nc, '__' + name, '}'))
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



with open(model_file, 'rb') as fh:
    network = pickle.load(fh, encoding='latin1')

sys.stdout.write("""#pragma once
#ifndef NANONET_RAW_MODEL_H
#define NANONET_RAW_MODEL_H
#include <assert.h>
#include "util.h"
""")

""" First LSTM layer
"""

filterW =  network.layers[0].W.get_value()
nfilter, _ , winlen = filterW.shape
cformatM(sys.stdout, 'conv_raw_W', filterW.reshape(-1, 1), nr = winlen * 4 - 3, nc=nfilter)
cformatV(sys.stdout, 'conv_raw_b', network.layers[0].b.get_value().reshape(-1))
sys.stdout.write("const int conv_raw_stride = {};\n".format(network.layers[0].stride))
sys.stdout.write("""const size_t _conv_nfilter = {};
const size_t _conv_winlen = {};
""".format(nfilter, winlen))

bigru1 = network.layers[1]
gru = bigru1.layers[0]
cformatM(sys.stdout, 'gruF1_raw_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruF1_raw_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruF1_raw_sW2', gru.sW2.get_value())
cformatV(sys.stdout, 'gruF1_raw_b', gru.b.get_value().reshape(-1))

gru = bigru1.layers[1].layer
cformatM(sys.stdout, 'gruB1_raw_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruB1_raw_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruB1_raw_sW2', gru.sW2.get_value())
cformatV(sys.stdout, 'gruB1_raw_b', gru.b.get_value().reshape(-1))


""" First feed forward layer
"""
assert(network.layers[2].insize % 2 == 0)
size = network.layers[2].insize // 2
cformatM(sys.stdout, 'FF1_raw_Wf', network.layers[2].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF1_raw_Wb', network.layers[2].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF1_raw_b', network.layers[2].b.get_value())


""" Second GRU layer
"""
bigru1 = network.layers[3]
gru = bigru1.layers[0]
cformatM(sys.stdout, 'gruF2_raw_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruF2_raw_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruF2_raw_sW2', gru.sW2.get_value())
cformatV(sys.stdout, 'gruF2_raw_b', gru.b.get_value().reshape(-1))

gru = bigru1.layers[1].layer
cformatM(sys.stdout, 'gruB2_raw_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruB2_raw_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruB2_raw_sW2', gru.sW2.get_value())
cformatV(sys.stdout, 'gruB2_raw_b', gru.b.get_value().reshape(-1))


""" Second feed forward layer
"""
size = network.layers[4].insize // 2
assert(network.layers[4].insize % 2 == 0)
cformatM(sys.stdout, 'FF2_raw_Wf', network.layers[4].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF2_raw_Wb', network.layers[4].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF2_raw_b', network.layers[4].b.get_value())


""" Softmax layer
"""
nstate = network.layers[5].W.get_value().shape[0]
shuffle = np.append(np.arange(nstate - 1) + 1, 0)
cformatM(sys.stdout, 'FF3_raw_W', network.layers[5].W.get_value()[shuffle])
cformatV(sys.stdout, 'FF3_raw_b', network.layers[5].b.get_value()[shuffle])

sys.stdout.write('#endif /* NANONET_RAW_MODEL_H */')
