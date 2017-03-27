#!/usr/bin/env python
import cPickle
import numpy as np
import sys


model_file = sys.argv[1]


def cformatM(fh, name, X):
    lines = map(lambda v: ', '.join(map(lambda f: str(f), v)), X)
    fh.write('const float {}[] = {}\n'.format('_' + name, '{'))
    fh.write('\t' + ',\n\t'.join(lines))
    fh.write('};\n')
    fh.write('Mat_rptr {} = NULL;\n\n'.format(name));
    # Mat object
    return '{} = mat_from_array({}, {}, {});\n'.format(name, '_' + name, X.shape[1], X.shape[0])


def cformatV(fh, name, X):
    lines = ', '.join(map(lambda f: str(f), X))
    fh.write('const float {}[] = {}\n'.format( '_' + name, '{'))
    fh.write('\t' + lines)
    fh.write('};\n')
    fh.write('Mat_rptr {} = NULL;\n\n'.format(name));
    # Mat object
    return '{} = mat_from_array({}, {}, {});\n'.format(name, '_' + name, X.shape[0], 1)


def write_setup(fh, setup):
    fh.write('void scrappie_gru_raw_setup(void){\n')
    fh.write(''.join(map(lambda x: '\t' + x, setup)))
    fh.write('}\n')


def reshape_lstmM(mat):
	_, isize = mat.shape
	return mat.reshape((-1, 4, isize)).transpose([1, 0, 2]).reshape((-1, isize))


def reshape_lstmV(mat):
	return mat.reshape((-1, 4)).transpose().reshape(-1)


with open(model_file, 'r') as fh:
    network = cPickle.load(fh)

sys.stdout.write("""#ifndef NANONET_RAW_MODEL_H
#define NANONET_RAW_MODEL_H
#include "util.h"
""")

""" First LSTM layer
"""
setup = []

setup.append(
    cformatM(sys.stdout, 'conv_raw_W', network.layers[0].W.get_value().reshape(-1, 1)))
setup.append(
    cformatV(sys.stdout, 'conv_raw_b', network.layers[0].b.get_value().reshape(-1)))
sys.stdout.write("const int conv_raw_stride = {};\n".format(network.layers[0].stride))


bigru1 = network.layers[1]
gru = bigru1.layers[0]
setup.append(
	cformatM(sys.stdout, 'gruF1_raw_iW', gru.iW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruF1_raw_sW', gru.sW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruF1_raw_sW2', gru.sW2.get_value()))
setup.append(
	cformatV(sys.stdout, 'gruF1_raw_b', gru.b.get_value().reshape(-1)))

gru = bigru1.layers[1].layer
setup.append(
	cformatM(sys.stdout, 'gruB1_raw_iW', gru.iW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruB1_raw_sW', gru.sW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruB1_raw_sW2', gru.sW2.get_value()))
setup.append(
	cformatV(sys.stdout, 'gruB1_raw_b', gru.b.get_value().reshape(-1)))


""" First feed forward layer
"""
assert(network.layers[2].insize % 2 == 0)
size = network.layers[2].insize // 2
setup.append(
	cformatM(sys.stdout, 'FF1_raw_Wf', network.layers[2].W.get_value()[:, : size]))
setup.append(
	cformatM(sys.stdout, 'FF1_raw_Wb', network.layers[2].W.get_value()[:, size : 2 * size]))
setup.append(
	cformatV(sys.stdout, 'FF1_raw_b', network.layers[2].b.get_value()))


""" Second GRU layer
"""
bigru1 = network.layers[3]
gru = bigru1.layers[0]
setup.append(
	cformatM(sys.stdout, 'gruF2_raw_iW', gru.iW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruF2_raw_sW', gru.sW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruF2_raw_sW2', gru.sW2.get_value()))
setup.append(
	cformatV(sys.stdout, 'gruF2_raw_b', gru.b.get_value().reshape(-1)))

gru = bigru1.layers[1].layer
setup.append(
	cformatM(sys.stdout, 'gruB2_raw_iW', gru.iW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruB2_raw_sW', gru.sW.get_value()))
setup.append(
	cformatM(sys.stdout, 'gruB2_raw_sW2', gru.sW2.get_value()))
setup.append(
	cformatV(sys.stdout, 'gruB2_raw_b', gru.b.get_value().reshape(-1)))


""" Second feed forward layer
"""
size = network.layers[4].insize // 2
assert(network.layers[4].insize % 2 == 0)
setup.append(
	cformatM(sys.stdout, 'FF2_raw_Wf', network.layers[4].W.get_value()[:, : size]))
setup.append(
	cformatM(sys.stdout, 'FF2_raw_Wb', network.layers[4].W.get_value()[:, size : 2 * size]))
setup.append(
	cformatV(sys.stdout, 'FF2_raw_b', network.layers[4].b.get_value()))


""" Softmax layer
"""
nstate = network.layers[5].W.get_value().shape[0]
shuffle = np.append(np.arange(nstate - 1) + 1, 0)
setup.append(
	cformatM(sys.stdout, 'FF3_raw_W', network.layers[5].W.get_value()[shuffle]))
setup.append(
	cformatV(sys.stdout, 'FF3_raw_b', network.layers[5].b.get_value()[shuffle]))

write_setup(sys.stdout, setup)

sys.stdout.write('#endif /* NANONET_RAW_MODEL_H */')
