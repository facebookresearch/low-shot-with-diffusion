# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import argparse



def to_ndarray(h):

    """converts a set of parameters + eval result into a N-dim atrray of
    results, where N is the nb of parameters """
    components = None
    for k in h:
        if components is None:
            components = [set([ki]) for ki in k]
        for i, ki in enumerate(k):
            components[i].add(ki)

    components = [tuple(sorted(c)) for c in components]

    sizes = [len(c) for c in components]
    a1 = np.ones(sizes) * np.nan
    a2 = np.ones(sizes) * np.nan
    print sizes
    for k, (v1, v2) in h.iteritems():
        idx = tuple([ci.index(ki) for ci, ki in zip(components, k)])
        # print k, idx
        a1[idx] = v1
        a2[idx] = v2

    return components, a1, a2



def parse_logreg_result(key, prefix, res):
    nmiss, ninc = 0, 0
    for seed in range(1, 6):
        
        fname = '%s.seed%d.stdout' % (prefix, seed)
        if not os.path.exists(fname): 
            # print "missing", fname
            nmiss += 1
            continue
        last_it = -1
        for l in open(fname):
            if 'iteration' in l:
                fi = l.split()
                if len(fi) != 9: break
                it = int(fi[2])
                err5_1 = float(fi[-4])
                err5_2 = float(fi[-3][1:])
                res[key + (seed, it)] = (err5_1, err5_2)
                last_it = it
        if last_it == -1:
            ninc += 1
    return nmiss, ninc





def nanmean(a, axis=None):
    return np.nansum(a, axis=axis) / np.nansum(a * 0.0 + 1.0, axis=axis)


basedir = os.getenv('DDIR')

def parse_val():
    top5s = {}

    for nl in 1, 2, 5, 10, 20:
        print '=========== nl=', nl
        for lr in 0.1, 0.01, 0.001, 0.0001:
            print 'lr=%8g' % lr,
            for wd in 0, 0.01, 0.1:
                print '   wd=%-4g' % wd,
                for bs in 16, 64, 128:
                    prefix = '%s/logs/logreg_val/nl%d.lr%g.wd%g.bs%d' % (basedir, nl, lr, wd, bs)
                    nmiss, ninc = parse_logreg_result((nl, lr, wd, bs), prefix , top5s)
                    print '%d+%d' % (nmiss, ninc),
            print
    components, top5s_all, top5s_novel = to_ndarray(top5s)



    onlynovel = False
    top5s = top5s_novel if onlynovel else top5s_all

def parse_test_run(fname): 
    if os.path.exists(fname):
        for l in open(fname):
            if 'top-5 accuracy' in l:
                fi = l.split()
                acc = float(fi[5])
                acc_novel = float(fi[6][1:])
                return acc, acc_novel
    return np.nan, np.nan

    
def parse_test():

    for nl in 1, 2, 5, 10, 20:
        accuracies = np.ones((5, 2)) * np.nan
        for seed in 1, 2, 3, 4, 5:
            fname = '%s/logs/logreg_test/nl%d.seed%d.stdout' % (
                basedir, nl, seed)

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

    
