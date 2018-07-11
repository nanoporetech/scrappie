#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import sys

from sloika.cmdargs import AutoBool, FileExists, Positive
import sloika.layers as layers
from sloika.helpers import objwalk, set_at_path


parser = argparse.ArgumentParser('Hack temperature into sloika model')
parser.add_argument('--temperature1', default=1.0, type=Positive(float),
                    help='Temperature for softmax weights')
parser.add_argument('--temperature2', default=1.0, type=Positive(float),
                    help='Temperature for softmax bias')
parser.add_argument('model', action=FileExists, help='Pickled sloika model to upgrade')
parser.add_argument('output', help='Output file to write to')


def change_softmax(network, weight_f, bias_f):

    for path, value in objwalk(network, types=(layers.Layer)):
        if isinstance(value, layers.Softmax):
            W = value.W.get_value(borrow=True)
            W *= weight_f
            b = value.b.get_value(borrow=True)
            b *= bias_f
        set_at_path(network, path, value)

    return network


if __name__ == '__main__':
    args = parser.parse_args()

    try:
        with open(args.model, 'rb') as fh:
            network = pickle.load(fh)
        sys.stdout.write('Loaded python 3 pickle from file `{}`\n'.format(args.model))
    except UnicodeDecodeError:
        with open(args.model, 'rb') as fh:
            network = pickle.load(fh, encoding='latin1')
        sys.stdout.write('Loaded python 2 pickle from file `{}`\n'.format(args.model))

    top_level_object_name = type(network).__name__
    assert top_level_object_name == 'Serial', top_level_object_name

    network = change_softmax(network,
                             np.reciprocal(args.temperature1),
                             np.reciprocal(args.temperature2))

    with open(args.output, 'wb') as fo:
        pickle.dump(network, fo)

    sys.stdout.write('Written python 3 pickle to file `{}`\n'.format(args.output))
