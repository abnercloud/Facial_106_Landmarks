# -----------------------------------------------------
# Copyright (c) Datatang.com. All rights reserved.
# Written by wduo(wangduo@datatang.com)
# ON Ubuntu 10.10.9.195.
# -----------------------------------------------------
import os
import glob
import json
import h5py
import numpy as np
from PIL import Image
from collections import defaultdict


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def imgname_parse(one_img_name, subdir, prefix):
    # one_img_name_id = one_img_name.split('-')[0]
    # one_img_name_ext = one_img_name.split('.')[-1]
    # one_img_name = one_img_name_id + '.' + one_img_name_ext
    # one_img_name = subdir + one_img_name

    one_img_name = prefix + one_img_name
    name_char_list = [ord(char) for char in one_img_name]

    # fixed_len = 37
    # if len(name_char_list) < fixed_len:
    #     for _ in range(len(name_char_list) + 1, fixed_len + 1):
    #         name_char_list.append(ord(' '))

    return name_char_list


def keypoints_parser(keypoints_json, img):
    width, height = img.size
    points_num = 106
    points = keypoints_json["DataList"]

    exist_points = [int(point["id"]) for point in points]
    points_list = []
    for ii in range(1, points_num + 1):
        if ii in exist_points:
            idx = exist_points.index(ii)
            points_list.append(points[idx]["coordinates"])
        else:
            points_list.append([0, 0])

    points_for_bndbox = [point["coordinates"] for point in points]
    coordinates = np.array(points_for_bndbox)
    upleft_x = min(coordinates[:, 0])
    upleft_y = min(coordinates[:, 1])
    bottomright_x = max(coordinates[:, 0])
    bottomright_y = max(coordinates[:, 1])

    # Expand ~1/10
    width_expand = (bottomright_x - upleft_x) * .1
    height_expand = (bottomright_y - upleft_y) * .1
    upleft_x = np.clip(upleft_x - width_expand, 0, width)
    upleft_y = np.clip(upleft_y - height_expand, 0, height)
    bottomright_x = np.clip(bottomright_x + width_expand, 0, width)
    bottomright_y = np.clip(bottomright_y + height_expand, 0, height)

    return points_list, [[upleft_x, upleft_y, bottomright_x, bottomright_y]]


def write_h5(annot_coco_h5_dict, annot_h5_dir, idxes):
    f = h5py.File(os.path.join(annot_h5_dir, 'annot_coco.h5'), 'w')
    f['imgname'] = annot_coco_h5_dict['imgname'][idxes]
    f['bndbox'] = annot_coco_h5_dict['bndbox'][idxes]
    f['part'] = annot_coco_h5_dict['part'][idxes]
    f.close()


def show_h5(annot_h5_dir):
    annot = h5py.File(os.path.join(annot_h5_dir, 'annot_coco.h5'))
    for k in annot.keys():
        print(k)

    bndboxes = annot['bndbox'][:]
    print(bndboxes.shape)
    imgnames = annot['imgname'][:]
    print(imgnames.shape)
    parts = annot['part'][:]
    print(parts.shape)


corrupt_imgs = ['/home/wd/assets/mounts/complex_emotion/session09/021532.jpg',
                '/home/wd/assets/mounts/complex_emotion/session10/024815.jpg',
                '/home/wd/assets/mounts/complex_emotion/session10/025415.jpg',
                '/home/wd/assets/mounts/complex_emotion/session10/027251.jpg',
                '/home/wd/assets/mounts/complex_emotion/session11/030398.jpg',
                '/home/wd/assets/mounts/complex_emotion/session13/045457.jpg'
                ]

Corrupt_EXIF_data = ['/home/wd/assets/mounts/simple_emotion/session05/018806.jpg',
                     '/home/wd/assets/mounts/simple_emotion/session05/019553.jpg'
                     ]

Malformed_MPO_file = ['/home/wd/assets/mounts/complex_emotion/session12/041114.jpg',
                      '/home/wd/assets/mounts/complex_emotion/session12/041139.jpg',
                      '/home/wd/assets/mounts/complex_emotion/session13/044169.jpg',
                      '/home/wd/assets/mounts/complex_emotion/session13/045412.jpg'
                      ]


def read_annot(data_dir, annot_h5_dir):
    """"""
    annot_coco_h5_dict = defaultdict(list)
    pwd = os.getcwd()

    annot_dir = data_dir
    original_img_paths = glob.glob(os.path.join(annot_dir, '*.[jJp][pPn][gG]'))  # All imgs
    os.chdir(annot_dir)
    for one_img_path in original_img_paths:
        img = Image.open(one_img_path)
        one_img_name = os.path.basename(one_img_path)

        # Read keypoints json file.
        one_img_name_no_ext = os.path.splitext(one_img_name)[0]
        keypoints_json_file = one_img_name_no_ext + '.json'
        keypoints_json = load_json(keypoints_json_file)

        # Add imgname field.
        name_char_list = imgname_parse(one_img_name, subdir='', prefix='')
        annot_coco_h5_dict['imgname'].append(name_char_list)

        # Add part field.
        keypoints, bndbox = keypoints_parser(keypoints_json, img)
        annot_coco_h5_dict['part'].append(keypoints)

        # Add bndbox field.
        annot_coco_h5_dict['bndbox'].append(bndbox)

    for k in annot_coco_h5_dict.keys():
        annot_coco_h5_dict[k] = np.array(annot_coco_h5_dict[k])
    print('All done.')

    os.chdir(pwd)
    # Write annot_coco_h5_dict to h5 file.
    idxes = np.array(range(annot_coco_h5_dict["imgname"].shape[0]))
    np.random.shuffle(idxes)  # For shuffle simple and complex emotion set
    write_h5(annot_coco_h5_dict, annot_h5_dir, idxes)
    # Show annot_coco.h5
    show_h5(annot_h5_dir)

    pass


if __name__ == '__main__':
    data_dir = r'D:\Users\wduo\Desktop\Facial_106_Landmarks\data'
    annot_h5_dir = 'facial_h5'

    read_annot(data_dir, annot_h5_dir)
