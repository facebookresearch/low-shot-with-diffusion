# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import argparse





basedir = os.getenv('DDIR')

def parse_val():
    pass

def parse_test_run(fname): 
    acc, acc_novel = np.nan, np.nan
    if os.path.exists(fname):
        for l in open(fname):
            if 'top-5 accuracy' in l:                
                fi = l.split()
                assert fi[0] in ('iter', 'intial'), l
                acc = float(fi[6])
                acc_novel = float(fi[7][1:])
        # keep the last seen acc
    return acc, acc_novel

    
def parse_test():

    for nbg in 10**6, 10**7, 10**8: 
        print "============ nbg=", nbg
        for nl in 1, 2, 5, 10, 20:
            accuracies = np.ones((5, 2)) * np.nan
            for seed in 1, 2, 3, 4, 5:
                fname = '%s/logs/diffusion_test/nl%d.nbg%d.seed%d.stdout' % (
                    basedir, nl, nbg, seed)                
                accuracies[seed - 1] = parse_test_run(fname)

            print "nl=%d: %.2f +/- %.3f, novel %.2f +/- %.3f " % (
                nl,
                100 * accuracies[:, 0].mean(),
                100 * accuracies[:, 0].std(),
                100 * accuracies[:, 1].mean(),
                100 * accuracies[:, 1].std(),
            )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=['val', 'test'], default='test',
        help='validation or test'
    )
    args = parser.parse_args()
    mode = args.mode

    if mode == 'val':
        parse_val()
    else:
        parse_test()

    
