# ------------------------------------------------
# Generate train/val/test file list to txt file.
# Draw gt and det res in face imgs.
# Written by wduo. wangduo@datatang.com
# ------------------------------------------------
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import h5py
from functools import reduce
from tqdm import tqdm


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def read_h5(data_dir, file_name_h5, train_val_test):
    """Create train/val/test split.s"""
    imgnames, bndboxes, parts = [], [], []
    with h5py.File(os.path.join(data_dir, file_name_h5), 'r') as annot:
        if train_val_test == 'train':
            imgnames = annot['imgname'][:90131]
            bndboxes = annot['bndbox'][:90131]
            parts = annot['part'][:90131]
        elif train_val_test == 'val':
            imgnames = annot['imgname'][90131:101397]
            bndboxes = annot['bndbox'][90131:101397]
            parts = annot['part'][90131:101397]
        elif train_val_test == 'test':
            imgnames = annot['imgname'][101397:]
            bndboxes = annot['bndbox'][101397:]
            parts = annot['part'][101397:]
        else:
            print('train_val_test error.')

    return imgnames, bndboxes, parts


def generate_file_list(imgnames, data_dir, file_list_txt_generated):
    """Generate train/val/test file list to txt file."""
    imgnames_list = []
    for imgname in imgnames:
        imgname_ = reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), imgname))
        imgnames_list.append(imgname_)

    if not os.path.exists(os.path.join(data_dir, file_list_txt_generated)):
        imgnames_list_txt = '\n'.join(imgnames_list) + '\n'
        with open(os.path.join(data_dir, file_list_txt_generated), 'w') as f:
            f.write(imgnames_list_txt)
    else:
        print('Test file list existed:', os.path.join(data_dir, file_list_txt_generated))

    return imgnames_list


def show_keypoints_annots(img_dir, gt, res_json_file, facial_render_dir, kps_num):
    """Draw gt and det res in face imgs, and save."""
    if not os.path.exists(facial_render_dir):
        os.makedirs(facial_render_dir)

    # gt
    imgnames_list, parts = gt
    assert parts.shape[1] == kps_num, 'gt data error, please choose right gt data.'
    gt_formatted = defaultdict(list)
    for ii, imgname in enumerate(imgnames_list):
        gt_formatted[imgname].append(parts[ii])

    # det res
    res_kps_data = load_json(res_json_file)
    assert len(res_kps_data[0]["keypoints"]) == kps_num * 3, 'det data error, please choose right det data.'
    res_kps_data_formatted = defaultdict(list)
    for one_face_keypoints in res_kps_data:
        res_kps_data_formatted[one_face_keypoints['image_id']].append(one_face_keypoints)

    # Show according det res
    dpi = 6
    for one_img_name, res_one_img_keypoints in tqdm(res_kps_data_formatted.items()):
        if os.path.exists(os.path.join(facial_render_dir, one_img_name)):
            print('Exist:', os.path.join(facial_render_dir, one_img_name))
            continue

        img = Image.open(os.path.join(img_dir, one_img_name))
        width, height = img.size
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.imshow(img)

        # Shot gt
        for gt_kps in gt_formatted[one_img_name]:
            # Draw gt points.
            for idx in range(kps_num):
                plt.plot(np.clip(gt_kps[idx, 0], 0, width), np.clip(gt_kps[idx, 1], 0, height),
                         marker='o', color='g', ms=width / 6)
                plt.text(np.clip(gt_kps[idx, 0], 0, width), np.clip(gt_kps[idx, 1], 0, height),
                         str(idx + 1), fontsize=width / 6, color='cyan')

        # Show det res
        for res_one_img_keypoint in res_one_img_keypoints:
            res_keypoints = np.array(res_one_img_keypoint['keypoints'])

            # Draw det points.
            for idx in range(kps_num):
                plt.plot(np.clip(res_keypoints[idx * 3], 0, width), np.clip(res_keypoints[idx * 3 + 1], 0, height),
                         marker='o', color='r', ms=width / 6)
                idx_conf = str(idx + 1) + '\n%.3f' % res_keypoints[idx * 3 + 2]
                plt.text(np.clip(res_keypoints[idx * 3], 0, width),
                         np.clip(res_keypoints[idx * 3 + 1], 0, height),
                         idx_conf, fontsize=width / 6, color='blue')

        plt.axis('off')
        ax = plt.gca()
        ax.set_xlim([0, width])
        ax.set_ylim([height, 0])
        # plt.show()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(facial_render_dir, one_img_name), pad_inches=0.0, bbox_inches=extent, dpi=dpi)
        plt.close(fig)

        pass


if __name__ == '__main__':
    # Raw imgs and gt.sss
    data_dir = 'train_sppe/data/coco/'
    imgs_dir_name = 'images'
    file_name_h5 = 'annot_coco.h5'

    # Generate file list txt for inference.
    train_val_test = 'test'
    file_list_txt_generated = 'file_list_for_' + train_val_test + '.txt'
    imgnames, bndboxes, parts = read_h5(data_dir, file_name_h5, train_val_test)
    imgnames_list_txt = generate_file_list(imgnames, data_dir, file_list_txt_generated)

    # Show gt and det res with id and conf.
    res_dir_name = 'examples/20190704/output_mini_101/'
    res_json_file = 'alphapose-results.json'
    facial_render_dir = 'facial_render'
    gt = (imgnames_list_txt, parts)
    kps_num = 106
    show_keypoints_annots(img_dir=os.path.join(data_dir, imgs_dir_name), gt=gt,
                          res_json_file=os.path.join(res_dir_name, res_json_file),
                          facial_render_dir=os.path.join(res_dir_name, facial_render_dir),
                          kps_num=kps_num)
