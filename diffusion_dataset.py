# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function
import sys
import os

import numpy as np
import faiss
import pdb
import cPickle
import time

from multiprocessing.dummy import Pool as ThreadPool


def normalize_features(M):
    sy = M.sum(axis=1)
    sy[sy == 0] = 1
    M /= sy.reshape(-1, 1)
    return M


def parse_ndis(distractors):
    assert distractors.startswith('flickr')
    distractors = distractors[6:]
    mult = 1
    if distractors[-1] == 'M':
        mult = 10**6
        distractors = distractors[:-1]
    if distractors[-1] == 'k':
        mult = 10**3
        distractors = distractors[:-1]
    return int(distractors) * mult


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


##########################################################
# Choose between the two datasets
##########################################################

class Opt:
    pass

class BharathEval:

    def __init__(self, Yte, novel_classes):
        self.Yte = Yte
        self.novel_classes = novel_classes

    def remap_labels(self, label_map):
        self.Yte = label_map[self.Yte]
        self.novel_classes = label_map[self.novel_classes]

    def compute_accuracies(self, val_L):
        assert val_L.shape[0] == self.Yte.size, pdb.set_trace()
        _, labels5 = faiss.kmax(val_L, 5)
        # binary vector of elements that are ok in top5
        ok = (labels5 == self.Yte.reshape(-1, 1)).sum(axis=1)
        N, nclasses = val_L.shape
        acc_full = ok.sum() / float(N)
        nc_mask = np.zeros(nclasses, dtype=bool)
        nc_mask[self.novel_classes] = True
        nc_examples = nc_mask[self.Yte]
        acc_novel_classes = ok[nc_examples].sum() / float(nc_examples.sum())

        return acc_full, acc_novel_classes


    def do_eval(self, val_L):
        acc_full, acc_novel_classes = self.compute_accuracies(val_L)
        return 'top-5 accuracy: %.3f (%.3f on novel)' % (
            acc_full, acc_novel_classes)

    def __str__(self):
        return '%d examples, %d classes, %d novel classes' % (
            self.Yte.size, np.bincount(self.Yte).astype(bool).sum(),
            self.novel_classes.size)

fv3_dir = os.getenv('DDIR') + '/features/'


def load_bharath_fv3(class_set, include_base_class, split='train'):
    label_info = eval(open("./label_idx.json").read())
    labels_mask = np.zeros(1000, dtype=bool)
    eval_classes_subset = np.array(label_info['novel_classes_%d' % class_set]) - 1
    labels_mask[eval_classes_subset] = True
    print("nb of eval classes", eval_classes_subset.size)
    if include_base_class:
        labels_mask[np.array(label_info['base_classes_%d' % class_set]) - 1] = True

    h5_filename = fv3_dir + split + '.hdf5'
    import h5py
    print('open ', h5_filename)        
    f = h5py.File(h5_filename, 'r')

    count= f['count'][0]
    labels = f['all_labels'][:count]

    if True: 
        npy_fname = h5_filename[:-5] + '_features.npy'
        if not os.path.exists(npy_fname):     
            features = f['all_feats'][:count]
            print("write ", npy_fname)
            np.save(npy_fname, features)
        else:
            print("read ", npy_fname)
            t0 = time.time()
            features = np.load(npy_fname, mmap_mode='r')
            features = np.array(features)
    else: 
        features = f['all_feats'][:count]

    print('   features loaded in %.3f s' % (time.time() - t0))
    mask = labels_mask[labels]
    return features[mask], labels[mask], eval_classes_subset


def load_traintest(nl, class_set=1, seed=1234,
                   include_base_class=True,
                   pca256=False, nnonlabeled=0):

    # imagenet validation images    
    Xte, Yte, eval_classes_subset = load_bharath_fv3(
        class_set, include_base_class, 'val')
    # imagenet train images
    Xtr, Ytr, eval_classes_subset = load_bharath_fv3(
        class_set, include_base_class, 'train')

    Yte = BharathEval(Yte, eval_classes_subset)

    # reduce labels to consecutive numbers that start at 0
    label_map = np.cumsum(np.bincount(Ytr).astype(bool)) - 1
    Ytr = label_map[Ytr]
    Yte.remap_labels(label_map)
    eval_classes_subset = label_map[eval_classes_subset]
    
    nclasses = Ytr.max() + 1

    print("selecting images, seed=%d" % seed)
    rs = np.random.RandomState(seed)

    perm1 = []
    perm0 = []
    base = []
    for cl in range(nclasses):
        imnos = (Ytr == cl).nonzero()[0]
        if cl in eval_classes_subset:
            rs.shuffle(imnos)
            perm1.append(imnos[:nl])
            perm0.append(imnos[nl:])
        else:
            base.append(imnos)

    if nnonlabeled == 0:
        perm = np.hstack(base + perm1)
    else:
        perm = np.hstack(base + perm1 + perm0)
        nnonlabeled = np.hstack(perm0).size

    Ytr = Ytr[perm]
    if nnonlabeled != 0:
        Ytr[-nnonlabeled:] = -1
    Xtr = Xtr[perm]

    if pca256:

        pca_fname = fv3_dir + 'PCAR256.vt'
        print("load", pca_fname)
        pcar = faiss.read_VectorTransform(pca_fname)
        Xtr = pcar.apply_py(Xtr)
        Xte = pcar.apply_py(Xte)

    return Xtr, Ytr, Xte, Yte

###############################################################
# Same as above, instead of descriptors returns imnet filenames

def load_bharath_ids(opt, split='train'):
    """ Load Bharath's imagenet descriptors and train/test split """

    # if 'prn' in os.getenv('HOSTNAME'):
    bharath_dataset_dir = '/mnt/vol/gfsai-local/ai-group/users/matthijs/bharath-dataset'

    from torch.utils.serialization import load_lua
    label_info = eval(open(bharath_dataset_dir + "/label_idx.json").read())

    feat_basedir = bharath_dataset_dir + '/baseline50_' + split

    features = []
    labels = []
    labels_mask = np.zeros(1000, dtype=bool)

    # novel_classes_2 are the test-test
    # novel_classes_1 are the test-val
    eval_classes_subset = np.array(label_info['novel_classes_%d' % opt.class_set]) - 1
    labels_mask[eval_classes_subset] = True
    print("nb of eval classes", eval_classes_subset.size)
    if opt.include_base_class:
        labels_mask[np.array(label_info['base_classes_%d' % opt.class_set]) - 1] = True

    from torch.utils.serialization import load_lua
    label_info = eval(open(bharath_dataset_dir + "/label_idx.json").read())

    feat_basedir = bharath_dataset_dir + '/baseline50_' + split
    if split == 'train':
        r = range(1, 128 + 1)
    else:
        r = range(5)

    Tmeta = torchfile.load(
        '/mnt/vol/gfsai-east/ai-group/users/bharathh/imagenet_meta/' + split + '.t7')

    image_names = Tmeta['image_names']

    count = 0
    def load_chunk(chunk_no):
        fname = '%s/feats_%d.t7' % (feat_basedir, chunk_no)
        # print ("load", fname, 'count=', count, '   \r', end=' ')
        print("load", chunk_no, '   \r', end=' ')
        sys.stdout.flush()
        return chunk_no, load_lua(fname)

    pool = ThreadPool(10)
    i0 = 0
    for chunk_no, F in pool.imap(load_chunk, r):
        # F = load_lua(fname)

        Flabels = F.labels.numpy().astype(int) - 1

        # pdb.set_trace()

        # Ffeats = F.feats.numpy()
        i1 = i0 + len(Flabels)
        idx = F.idx.numpy()
        Ffeats = image_names[idx - 1]
        i0 = i1

        mask = labels_mask[Flabels]

        features.append(Ffeats[mask])
        labels.append(Flabels[mask])
        count += mask.sum()
    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels, eval_classes_subset


def load_bharath_traintest_ids(nl, class_set=1, seed=1234,
                               include_base_class=True,
                               nnonlabeled=0):

    opt = Opt()
    opt.bharath256 = True
    opt.bharath = False
    opt.class_set = class_set
    opt.include_base_class = include_base_class

    Xte, Yte, eval_classes_subset = load_bharath_ids(opt, 'val')
    Xtr, Ytr, eval_classes_subset = load_bharath_ids(opt, 'train')

    Yte = BharathEval(Yte, eval_classes_subset)

    # reduce labels to consecutive numbers that start at 0
    label_map = np.cumsum(np.bincount(Ytr).astype(bool)) - 1
    Ytr = label_map[Ytr]
    # Yte = label_map[Yte]
    Yte.remap_labels(label_map)
    eval_classes_subset = label_map[eval_classes_subset]

    nclasses = Ytr.max() + 1

    rs = np.random.RandomState(seed)

    perm1 = []
    perm0 = []
    base = []
    for cl in range(nclasses):
        imnos = (Ytr == cl).nonzero()[0]
        if cl in eval_classes_subset:
            rs.shuffle(imnos)
            perm1.append(imnos[:nl])
            perm0.append(imnos[nl:])
        else:
            base.append(imnos)

    if nnonlabeled == 0:
        perm = np.hstack(base + perm1)
    else:
        perm = np.hstack(base + perm1 + perm0)
        nnonlabeled = np.hstack(perm0).size

    Ytr = Ytr[perm]
    if nnonlabeled != 0:
        Ytr[-nnonlabeled:] = -1
    Xtr = Xtr[perm]


    return Xtr, Ytr, Xte, Yte


def load_bharath_distractors(ndis):
    fname = fv3_dir + '/f100m/concatenated_PCAR256.raw'
    print("memmapping ", fname)
    return np.memmap(fname, shape=(ndis, 256), dtype='float32')
