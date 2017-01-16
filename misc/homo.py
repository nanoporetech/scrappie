#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from untangled.bio import all_kmers, kmer_mapping
from untangled.cmdargs import Positive

parser = argparse.ArgumentParser()
parser.add_argument('--kmer', default=5, type=Positive(int))
parser.add_argument('hdf5')

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

            for base in ['A', 'C', 'G', 'T']:
                homostate = 1 + kmer_to_state[base * args.kmer]

                ev = 0
                while ev < nev:
                    while ev < nev and events['state'][ev] != homostate:
                        #  Increment ev until next homopolymer found
                        ev += 1
                    if ev == nev:
                        break

                    hstart, hend = ev, ev + 1
                    for evi in range(hstart, 0, -1):
                        # Track back until find start.
                        evst = events['state'][evi - 1]
                        if evst == 0:
                            continue
                        #print "ev {} has kmer {}".format(evi - 1, kmer[evst -1 ])
                        if kmer[evst - 1][kidx] != base:
                            break
                        hstart = evi - 1
                    for evi in range(ev + 1, nev):
                        # Track forward until find end.
                        evst = events['state'][evi]
                        hend = evi + 1
                        if evst == 0:
                            continue
                        #print "ev {} has kmer {} ({})".format(evi, kmer[evst - 1], kmer[evst - 1][kidx])
                        if kmer[evst - 1][kidx] != base:
                            hend = evi
                            #print '... break' 
                            break

                    print read, base, hstart, hend - 1,
                    dwell = np.sum(events['length'][hstart : hend])
                    hl = events['pos'][hend - 1] - events['pos'][hstart] + 1
                    #print events[hstart:hend]
                    print events['pos'][hstart], hl, dwell

                    ev = hend
