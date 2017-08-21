import cPickle
import numpy as np
from sloika import features
from untangled import fast5

modelfile = 'model_final.pkl'
fn = 'reads/MINICOL228_20161012_FNFAB42578_MN17976_mux_scan_HG_52221_ch271_read66_strand.fast5'

with open(modelfile, 'r') as fh:
	network = cPickle.load(fh)

iWf = network.layers[1].layers[0].iW.get_value()
sWf = network.layers[1].layers[0].sW.get_value()
sW2f = network.layers[1].layers[0].sW2.get_value()
bf = network.layers[1].layers[0].b.get_value()
iWb = network.layers[1].layers[1].layer.iW.get_value()
sWb = network.layers[1].layers[1].layer.sW.get_value()
sW2b = network.layers[1].layers[1].layer.sW2.get_value()
bb = network.layers[1].layers[1].layer.b.get_value()

with fast5.Reader(fn) as f5:
    ev = f5.get_section_events("template", analysis='Segment_Linear')
    sn = f5.filename_short

ev = ev[50:]
print "Input"
print ev[:10]

features = features.from_events(ev, tag='')
print "Features"
print features[:10]

#  Windowing
w = 3
inMat = features
zeros = np.zeros((w // 2, features.shape[1]))
padMat = np.concatenate([zeros, inMat, zeros], axis=0)
tmp = np.concatenate([padMat[i : 1 + i - w] for i in xrange(w - 1)], axis=1)
feature3 = np.concatenate([tmp, padMat[w - 1 :]], axis=1)
print "Feature 3"
print feature3[:10]

#  GRU forward
def gatefun(x):
	return 1.0 / (1.0 + np.exp(-x))
size = sWf.shape[1]
state = np.zeros(size)
outF = np.zeros((len(feature3), size))
for i, f in enumerate(feature3):
        vI = np.tensordot(f, iWf, axes=(0,1)) + bf
        vS = np.tensordot(state, sWf, axes=(0,1))
        vT = vI[:2 * size] + vS
        vT = vT.reshape((2, -1))

        z = gatefun(vT[0])
        r = gatefun(vT[1])
        y = np.tensordot(r * state, sW2f, axes=(0,1))
        hbar = np.tanh(vI[2 * size:] + y)
        state = z * state + (1 - z) * hbar
	outF[i] = state
print "GRU forward"
print outF[:10,:8]

state = np.zeros(size)
outB = np.zeros((len(feature3), size))
for i, f in enumerate(feature3[::-1]):
        vI = np.tensordot(f, iWb, axes=(0,1)) + bb
        vS = np.tensordot(state, sWb, axes=(0,1))
        vT = vI[:2 * size] + vS
        vT = vT.reshape((2, -1))

        z = gatefun(vT[0])
        r = gatefun(vT[1])
        y = np.tensordot(r * state, sW2b, axes=(0,1))
        hbar = np.tanh(vI[2 * size:] + y)
        state = z * state + (1 - z) * hbar
        outB[i] = state
outB = outB[::-1]
print "GRU backward"
print outB[:10,:8]

inFF = np.concatenate((outF, outB), axis=1)

W1 = network.layers[2].W.get_value()
b1 = network.layers[2].b.get_value()
outFF1 = np.tanh(np.tensordot(inFF, W1, axes=(1,1)) + b1)
print "FF1"
print outFF1[:10,:8]
