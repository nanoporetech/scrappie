import cPickle
import sys

model_file = 'gru_obese/model_checkpoint_00006.pkl'


def cformatM(fh, name, X):
    lines = map(lambda v: ', '.join(map(lambda f: str(f), v)), X)
    fh.write('const float {}[] = {}\n'.format(name, '{'))
    fh.write(',\n\t'.join(lines))
    fh.write('};\n\n')

def cformatV(fh, name, X):
    lines = ', '.join(map(lambda f: str(f), X))
    fh.write('const float {}[] = {}\n'.format(name, '{'))
    fh.write('\t' + lines)
    fh.write('};\n\n')


with open(model_file, 'r') as fh:
    network = cPickle.load(fh)

sys.stdout.write("""#ifndef NANONET_MODEL_H
#define NANONET_MODEL_H

""")

""" First GRU layer
"""
bigru1 = network.layers[1]
gru = bigru1.layers[0]
cformatM(sys.stdout, 'gruF1_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruF1_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruF1_sW2', gru.sW2.get_value())
cformatM(sys.stdout, 'gruF1_b', gru.b.get_value().reshape(3,-1))

gru = bigru1.layers[1].layer
cformatM(sys.stdout, 'gruB1_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruB1_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruB1_sW2', gru.sW2.get_value())
cformatM(sys.stdout, 'gruB1_b', gru.b.get_value().reshape(3, -1))


""" First feed forward layer
"""
size = network.layers[2].insize // 2
cformatM(sys.stdout, 'FF1_Wf', network.layers[2].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF1_Wb', network.layers[2].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF1_b', network.layers[2].b.get_value())


""" Second GRU layer
"""
bigru2 = network.layers[3]
gru = bigru2.layers[0]
cformatM(sys.stdout, 'gruF2_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruF2_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruF2_sW2', gru.sW2.get_value())
cformatM(sys.stdout, 'gruF2_b', gru.b.get_value().reshape(3,-1))

gru = bigru2.layers[1].layer
cformatM(sys.stdout, 'gruB2_iW', gru.iW.get_value())
cformatM(sys.stdout, 'gruB2_sW', gru.sW.get_value())
cformatM(sys.stdout, 'gruB2_sW2', gru.sW2.get_value())
cformatM(sys.stdout, 'gruB2_b', gru.b.get_value().reshape(3, -1))


""" Second feed forward layer
"""
size = network.layers[2].insize // 2
cformatM(sys.stdout, 'FF2_Wf', network.layers[4].W.get_value()[:, : size])
cformatM(sys.stdout, 'FF2_Wb', network.layers[4].W.get_value()[:, size : 2 * size])
cformatV(sys.stdout, 'FF2_b', network.layers[4].b.get_value())

""" Softmax layer
"""
cformatM(sys.stdout, 'FF3_W', network.layers[5].W.get_value())
cformatV(sys.stdout, 'FF3_b', network.layers[5].b.get_value())

sys.stdout.write('#endif /* NANONET_MODEL_H */')
