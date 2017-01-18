#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from untangled.bio import all_kmers, kmer_mapping, kmers_to_sequence
from untangled.cmdargs import Positive, Vector

parser = argparse.ArgumentParser()
parser.add_argument('--base_adj', default=np.array([2.1587565, 2.9004156, 6.9480399, 0.5906636]), nargs=4, type=Vector(float))
parser.add_argument('--factor', default=9.8382457, type=float)
parser.add_argument('--kmer', default=5, type=Positive(int))
parser.add_argument('hdf5')

BASES = ['A', 'C', 'G', 'T']

if __name__ == '__main__':
    args = parser.parse_args()
    kmer = np.array(all_kmers(args.kmer))
    kmer_to_state = kmer_mapping(args.kmer)

    kidx = args.kmer // 2

    with h5py.File(args.hdf5, 'r') as h5:
        for read in h5.keys():
            events = h5[read][()]
            #  Filter events unassigned to position
            events = events[events['pos'] >= 0]
            nev = len(events)

            homostates = {1 + kmer_to_state[b * args.kmer] : i for i, b in enumerate(BASES)}

            states = [] 
            ev = 0
            while ev < nev:
                if events['state'][ev] not in homostates:
                    states.append(events['state'][ev])
                    ev += 1
                    continue

                hstate = events['state'][ev]
                bidx = homostates[hstate]
                dwell = 0
                while ev < nev and events['state'][ev] in [0, hstate]:
                    dwell += events['length'][ev]
                    ev += 1

                rep_est = int(round((dwell - args.base_adj[bidx]) / args.factor))
                states += [hstate] * rep_est
                
            # Code to emit bases
            seq = kmers_to_sequence([kmer[st - 1] for st in states if st > 0], always_move=True)
            print ">{}\n{}".format(read, seq)
