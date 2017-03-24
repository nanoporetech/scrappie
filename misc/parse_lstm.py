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
    fh.write('void scrappie_network_setup(void){\n')
    fh.write(''.join(map(lambda x: '\t' + x, setup)))
    fh.write('}\n')


def reshape_lstmM(mat):
	_, isize = mat.shape
	return mat.reshape((-1, 4, isize)).transpose([1, 0, 2]).reshape((-1, isize))


def reshape_lstmV(mat):
	return mat.reshape((-1, 4)).transpose().reshape(-1)


with open(model_file, 'r') as fh:
    network = cPickle.load(fh)

sys.stdout.write("""#ifndef NANONET_EVENTS_MODEL_H
#define NANONET_EVENTS_MODEL_H
#include "util.h"
""")

""" First LSTM layer
"""
setup = []

bilstm1 = network.layers[1]
lstm = bilstm1.layers[0]
setup.append(
	cformatM(sys.stdout, 'lstmF1_iW', reshape_lstmM(lstm.iW.get_value())))
setup.append(
	cformatM(sys.stdout, 'lstmF1_sW', reshape_lstmM(lstm.sW.get_value())))
setup.append(
	cformatV(sys.stdout, 'lstmF1_b', reshape_lstmV(lstm.b.get_value().reshape(-1))))
setup.append(
	cformatV(sys.stdout, 'lstmF1_p', lstm.p.get_value().reshape(-1)))

lstm = bilstm1.layers[1].layer
setup.append(
	cformatM(sys.stdout, 'lstmB1_iW', reshape_lstmM(lstm.iW.get_value())))
setup.append(
	cformatM(sys.stdout, 'lstmB1_sW', reshape_lstmM(lstm.sW.get_value())))
setup.append(
	cformatV(sys.stdout, 'lstmB1_b', reshape_lstmV(lstm.b.get_value().reshape(-1))))
setup.append(
	cformatV(sys.stdout, 'lstmB1_p', lstm.p.get_value().reshape(-1)))


""" First feed forward layer
"""
size = network.layers[2].insize // 2
setup.append(
	cformatM(sys.stdout, 'FF1_Wf', network.layers[2].W.get_value()[:, : size]))
setup.append(
	cformatM(sys.stdout, 'FF1_Wb', network.layers[2].W.get_value()[:, size : 2 * size]))
setup.append(
	cformatV(sys.stdout, 'FF1_b', network.layers[2].b.get_value()))


""" Second LSTM layer
"""
bilstm2 = network.layers[3]
lstm = bilstm2.layers[0]
setup.append(
	cformatM(sys.stdout, 'lstmF2_iW', reshape_lstmM(lstm.iW.get_value())))
setup.append(
	cformatM(sys.stdout, 'lstmF2_sW', reshape_lstmM(lstm.sW.get_value())))
setup.append(
	cformatV(sys.stdout, 'lstmF2_b', reshape_lstmV(lstm.b.get_value().reshape(-1))))
setup.append(
	cformatV(sys.stdout, 'lstmF2_p', lstm.p.get_value().reshape(-1)))

lstm = bilstm2.layers[1].layer
setup.append(
	cformatM(sys.stdout, 'lstmB2_iW', reshape_lstmM(lstm.iW.get_value())))
setup.append(
	cformatM(sys.stdout, 'lstmB2_sW', reshape_lstmM(lstm.sW.get_value())))
setup.append(
	cformatV(sys.stdout, 'lstmB2_b', reshape_lstmV(lstm.b.get_value().reshape(-1))))
setup.append(
	cformatV(sys.stdout, 'lstmB2_p', lstm.p.get_value().reshape(-1)))


""" Second feed forward layer
"""
size = network.layers[4].insize // 2
setup.append(
	cformatM(sys.stdout, 'FF2_Wf', network.layers[4].W.get_value()[:, : size]))
setup.append(
	cformatM(sys.stdout, 'FF2_Wb', network.layers[4].W.get_value()[:, size : 2 * size]))
setup.append(
	cformatV(sys.stdout, 'FF2_b', network.layers[4].b.get_value()))


""" Softmax layer
"""
nstate = network.layers[5].W.get_value().shape[0]
shuffle = np.append(np.arange(nstate - 1) + 1, 0)
setup.append(
	cformatM(sys.stdout, 'FF3_W', network.layers[5].W.get_value()[shuffle]))
setup.append(
	cformatV(sys.stdout, 'FF3_b', network.layers[5].b.get_value()[shuffle]))

write_setup(sys.stdout, setup)

sys.stdout.write('#endif /* NANONET_EVENTS_MODEL_H */')
