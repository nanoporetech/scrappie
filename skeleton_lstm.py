import cPickle
import numpy as np
from sloika import features
from untangled import fast5

modelfile = '/mnt/data/human/training/lstm_obese/model_final.pkl'
fn = '/home/OXFORDNANOLABS/tmassingham/git/crappie/data/MINICOL228_20161012_FNFAB42578_MN17976_mux_scan_HG_52221_ch271_read66_strand.fast5'

with open(modelfile, 'r') as fh:
	network = cPickle.load(fh)

iWf = network.layers[1].layers[0].iW.get_value()
sWf = network.layers[1].layers[0].sW.get_value()
bf = network.layers[1].layers[0].b.get_value()
pf = network.layers[1].layers[0].p.get_value()

iWb = network.layers[1].layers[1].layer.iW.get_value()
sWb = network.layers[1].layers[1].layer.sW.get_value()
bb = network.layers[1].layers[1].layer.b.get_value()
pb = network.layers[1].layers[1].layer.p.get_value()

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

#  LSTM forward
def gatefun(x):
	return 1.0 / (1.0 + np.exp(-x))
size = sWf.shape[1]
state = np.zeros(size)
lastout = np.zeros(size)
outF = np.zeros((len(feature3), size))
for i, f in enumerate(feature3):
        vW = np.tensordot(f, iWf, axes=(0, 1))
        outW = np.tensordot(lastout, sWf, axes=(0, 1))
        sumW = vW + outW  + bf
        sumW = sumW.reshape((size, 4))

        #  Forget gate activation
        forget = state * gatefun(sumW[:,2] + state * pf[1])
        #  Update state with input
        update = np.tanh(sumW[:,0]) * gatefun(sumW[:,1] + state * pf[0])
	state = forget + update
        #  Output gate activation
        lastout = np.tanh(state) * gatefun(sumW[:,3] + state * pf[2])
	outF[i] = lastout

print "LSTM forward"
print outF[:10,:8]

state = np.zeros(size)
lastout = np.zeros(size)
outB = np.zeros((len(feature3), size))
for i, f in enumerate(feature3[::-1]):
        vW = np.tensordot(f, iWb, axes=(0, 1))
        outW = np.tensordot(lastout, sWb, axes=(0, 1))
        sumW = vW + outW  + bb
        sumW = sumW.reshape((size, 4))

        #  Forget gate activation
        forget = state * gatefun(sumW[:,2] + state * pb[1])
        #  Update state with input
        update = np.tanh(sumW[:,0]) * gatefun(sumW[:,1] + state * pb[0])
	state = forget + update
        #  Output gate activation
        lastout = np.tanh(state) * gatefun(sumW[:,3] + state * pb[2])
	outB[i] = lastout
outB = outB[::-1]
print "LSTM backward"
print outB[:10,:8]

inFF = np.concatenate((outF, outB), axis=1)

W1 = network.layers[2].W.get_value()
b1 = network.layers[2].b.get_value()
outFF1 = np.tanh(np.tensordot(inFF, W1, axes=(1,1)) + b1)
print "FF1"
print outFF1[:10,:8]
