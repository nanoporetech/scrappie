#!/usr/bin/env python3
import pickle
import math
import numpy as np
import re
import sys


model_file = sys.argv[1]

trim_trailing_zeros = re.compile('0+p')

def small_hex(f):
    hf = float(f).hex()
    return trim_trailing_zeros.sub('p', hf)


def process_column(v, pad):
    """ process and pad """
    return [small_hex(f) for f in v] + [small_hex(0.0)] * pad


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
    lines = ', '.join(list(map(lambda f: small_hex(f), X)) + [small_hex(0.0)] * pad)
    fh.write('float {}[] = {}\n'.format( '__' + name, '{'))
    fh.write('\t' + lines)
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', X.shape[0], nrq, 1, '__' + name, '}'))
    fh.write('const scrappie_matrix {} = &{};\n\n'.format(name, '_' + name))



with open(model_file, 'rb') as fh:
    network = pickle.load(fh, encoding='latin1')

sys.stdout.write("""#pragma once
#ifndef NANONET_RGR_MODEL_H
#define NANONET_RGR_MODEL_H
#include <assert.h>
#include "util.h"
""")

""" Convolution layer
"""

filterW =  network.sublayers[0].W.get_value()
nfilter, _ , winlen = filterW.shape
cformatM(sys.stdout, 'conv_rgr_W', filterW.reshape(-1, 1), nr = winlen * 4 - 3, nc=nfilter)
cformatV(sys.stdout, 'conv_rgr_b', network.sublayers[0].b.get_value().reshape(-1))
sys.stdout.write("const int conv_rgr_stride = {};\n".format(network.sublayers[0].stride))
sys.stdout.write("""const size_t _conv_rgr_nfilter = {};
const size_t _conv_rgr_winlen = {};
""".format(nfilter, winlen))

"""  Backward GRU (first layer)
"""
gru1 = network.sublayers[1].sublayers[0]
cformatM(sys.stdout, 'gruB1_rgr_iW', gru1.iW.get_value())
cformatM(sys.stdout, 'gruB1_rgr_sW', gru1.sW.get_value())
cformatM(sys.stdout, 'gruB1_rgr_sW2', gru1.sW2.get_value())
cformatV(sys.stdout, 'gruB1_rgr_b', gru1.b.get_value().reshape(-1))

"""  Forward GRU (second layer)
"""
gru2 = network.sublayers[2]
cformatM(sys.stdout, 'gruF2_rgr_iW', gru2.iW.get_value())
cformatM(sys.stdout, 'gruF2_rgr_sW', gru2.sW.get_value())
cformatM(sys.stdout, 'gruF2_rgr_sW2', gru2.sW2.get_value())
cformatV(sys.stdout, 'gruF2_rgr_b', gru2.b.get_value().reshape(-1))



""" backward GRU(third layer)
"""
gru3 = network.sublayers[3].sublayers[0]
cformatM(sys.stdout, 'gruB3_rgr_iW', gru3.iW.get_value())
cformatM(sys.stdout, 'gruB3_rgr_sW', gru3.sW.get_value())
cformatM(sys.stdout, 'gruB3_rgr_sW2', gru3.sW2.get_value())
cformatV(sys.stdout, 'gruB3_rgr_b', gru3.b.get_value().reshape(-1))

""" Softmax layer
"""
nstate = network.sublayers[4].W.get_value().shape[0]
shuffle = np.append(np.arange(nstate - 1) + 1, 0)
cformatM(sys.stdout, 'FF_rgr_W', network.sublayers[4].W.get_value()[shuffle])
cformatV(sys.stdout, 'FF_rgr_b', network.sublayers[4].b.get_value()[shuffle])

sys.stdout.write('#endif /* NANONET_RGR_MODEL_H */')
