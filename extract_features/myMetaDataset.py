# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

from zipfile import ZipFile
from cStringIO import StringIO

identity = lambda x:x

class MetaDataset:
    def __init__(self, rootdir='/mnt/fair/imagenet-256', meta='/home/bharathh/imagenet_meta/train.json', transform=transforms.ToTensor(), target_transform=identity):
        with open(meta, 'r') as f:
            self.meta = json.load(f)
        self.rootdir=rootdir
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.rootdir, self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class F100MDataset:

    def __init__(self, i0, i1, transform=transforms.ToTensor(), target_transform=identity):
        if i1 == -1: i1 = 10**8
        self.i0, self.i1 = i0, i1

        self.transform = transform
        self.target_transform = target_transform

        self.basedir = os.getenv('DDIR') + '/yfcc100m'
        self.zips = {}

    def load_img(self, i):

        zipno = i / 1000 
        imno = i % 1000
        
        if zipno not in self.zips:
            fname = '%s/%02d/%03d.zip' % (
                self.basedir, zipno / 1000, zipno % 1000)
            self.zips[zipno] = ZipFile(fname, 'r')

        try: 
            img = Image.open(self.zips[zipno].open('%03d.jpg' % imno))
            return img.convert('RGB')
        except Exception, e:
            print "bad image", i, zipno, e
            return Image.new('RGB', (256, 256))
            
    def __getitem__(self, i):
        img = self.load_img(i)
        img = self.transform(img)
        target = self.target_transform(0)
        return img, target

    def __len__(self):
        return self.i1 - self.i0
