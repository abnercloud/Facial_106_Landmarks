# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 106
        self.nJoints = 106

        self.accIdxs = tuple(range(1,107))

        self.flipRef = ((1, 33), (2, 32), (3, 31), (4, 30), (5, 29), (6, 28), (7, 27), (8, 26), (9, 25), (10, 24), (11, 23),
                        (12, 22), (13, 21), (14, 20), (15, 19), (16, 18),
                        (34, 47), (35, 46), (36, 45), (37, 44), (38, 43), (39, 51), (40, 50), (41, 49), (42, 48),
                        (52, 65), (53, 64), (54, 63), (55, 62), (56, 67), (57, 66), (58, 68), (59, 69), (60, 70), (61, 71),
                        (76, 77), (78, 79), (80, 81), (82, 86), (83, 85),
                        (87, 91), (97, 98), (93, 94), (96, 95), (88, 90), (99, 101), (102, 104), (105, 106))

        # create train/val split
        with h5py.File('../data/coco/annot_coco.h5', 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:90131]
            self.bndbox_coco_train = annot['bndbox'][:90131]
            self.part_coco_train = annot['part'][:90131]
            # val
            self.imgname_coco_val = annot['imgname'][90131:101397]
            self.bndbox_coco_val = annot['bndbox'][90131:101397]
            self.part_coco_val = annot['part'][90131:101397]

        self.size_train = self.imgname_coco_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)

        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
