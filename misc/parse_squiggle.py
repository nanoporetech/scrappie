#!/usr/bin/env python3
import argparse
import pickle
import math
import numpy as np
import re
import sys

import sloika.layers

parser = argparse.ArgumentParser()
parser.add_argument('--id', default='' , help='Identifier for model names')
parser.add_argument('model', help='Pickle to read model from')

EMBEDDING_MATRIX = np.array([[1.0, 0.0, -1.0 / np.sqrt(2.0)],
                             [-1.0, 0.0, -1.0 / np.sqrt(2.0)],
                             [0.0, 1.0, 1.0 / np.sqrt(2.0)],
                             [0.0, -1.0, 1.0 / np.sqrt(2.0)]], dtype=np.float32)


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

    nelt = 4 * nrq * nc

    fh.write('float {}[{}] = {}\n'.format('__' + name, nelt, '{'))
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


def process_convolution(layer, layerid):
    assert isinstance(layer, sloika.layers.Convolution)

    filterW =  layer.W.get_value()
    nfilter, nfeature , winlen = filterW.shape
    nfeatureCeil4 = 4 * int(math.ceil(nfeature / 4.0))
    cformatM(sys.stdout, '{}W'.format(layerid), filterW.transpose(0, 2, 1).reshape(-1, nfeature),
             nr = (winlen - 1) * nfeatureCeil4 + nfeature, nc=nfilter)
    cformatV(sys.stdout, '{}b'.format(layerid), layer.b.get_value().reshape(-1))
    sys.stdout.write("const int {}stride = {};\n".format(layerid, layer.stride))
    sys.stdout.write("""const size_t _{}nfilter = {};
const size_t _{}winlen = {};
""".format(layerid, nfilter, layerid, winlen))


if __name__ == '__main__':
    args = parser.parse_args()
    modelid = args.id + '_'

    with open(args.model, 'rb') as fh:
        network = pickle.load(fh)
    assert network.major_version == 2


    sys.stdout.write("""#pragma once
#ifndef NANONET_SQUIGGLE_{}MODEL_H
    #define NANONET_SQUIGGLE_{}MODEL_H
    #include <assert.h>
    #include "../util.h"
""".format(modelid.upper(), modelid.upper()))


    #  Embedding layer
    cformatM(sys.stdout, 'embed_squiggle_{}W'.format(modelid), EMBEDDING_MATRIX)


    """ first convolution
    """
    process_convolution(network.sublayers[0], 'conv1_squiggle_{}'.format(modelid))
    assert isinstance(network.sublayers[1], sloika.layers.Residual)
    process_convolution(network.sublayers[1].sublayers[0], 'conv2_squiggle_{}'.format(modelid))
    assert isinstance(network.sublayers[2], sloika.layers.Residual)
    process_convolution(network.sublayers[2].sublayers[0], 'conv3_squiggle_{}'.format(modelid))
    assert isinstance(network.sublayers[3], sloika.layers.Residual)
    process_convolution(network.sublayers[3].sublayers[0], 'conv4_squiggle_{}'.format(modelid))
    assert isinstance(network.sublayers[4], sloika.layers.Residual)
    process_convolution(network.sublayers[4].sublayers[0], 'conv5_squiggle_{}'.format(modelid))
    process_convolution(network.sublayers[5], 'conv6_squiggle_{}'.format(modelid))

    sys.stdout.write('\n#endif /* NANONET_SQUIGGLE_{}MODEL_H */\n'.format(modelid.upper()))
