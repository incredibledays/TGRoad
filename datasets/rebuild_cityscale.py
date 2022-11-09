import pickle
import cv2
import numpy as np
import math
from tqdm import tqdm
import shutil


def neighbor_to_integer(n_in):
    n_out = {}
    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))
        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []
        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))
            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)
        n_out[nk] = nv
    return n_out


input_dir = './cityscale/'
output_dir = './cityscale/'
dataset_image_size = 2048
size = 512
stride = 256

for i in tqdm(range(180)):
    sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
    neighbors = neighbor_to_integer(pickle.load(open(input_dir + 'region_%d_refine_gt_graph.p' % i, 'rb')))

    seg = np.zeros((dataset_image_size, dataset_image_size))
    ach = np.zeros((dataset_image_size, dataset_image_size, 3))
    ori = np.zeros((dataset_image_size, dataset_image_size, 1))

    for loc, n_locs in neighbors.items():
        if len(n_locs) == 1:
            ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 0] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        elif len(n_locs) == 2:
            ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 1] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        elif len(n_locs) > 2:
            ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 2] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        for n_loc in n_locs:
            angle = math.atan2(loc[0] - n_loc[0], n_loc[1] - loc[1]) / math.pi
            cv2.line(seg, (loc[1], loc[0]), (n_loc[1], n_loc[0]), 1)
            cv2.line(ori, (loc[1], loc[0]), (n_loc[1], n_loc[0]), angle)

    seg = np.expand_dims(cv2.dilate(seg, np.ones((3, 3), np.uint8)), axis=2)
    ori = np.expand_dims(cv2.dilate(ori, np.ones((3, 3), np.uint8)), axis=2)
    sao = np.concatenate([seg, ach, ori], axis=2)

    if i % 10 < 8:
        output_dir = '../datasets/train/'
    if i % 20 == 18:
        output_dir = '../datasets/valid/'
    if i % 20 == 8 or i % 10 == 9:
        output_dir = '../datasets/test/'
        shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
        shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
        cv2.imwrite(output_dir + 'region_%d_sao.png' % i, sao * 255)
        continue

    maxx = int((dataset_image_size - size) / stride)
    maxy = int((dataset_image_size - size) / stride)
    for x in range(maxx + 1):
        for y in range(maxy + 1):
            sat_block = sat[x * stride:x * stride + size, y * stride:y * stride + size, :]
            sao_block = sao[x * stride:x * stride + size, y * stride:y * stride + size, :]
            pof_block = pof[x * stride:x * stride + size, y * stride:y * stride + size, :]

            cv2.imwrite(output_dir + '{}_{}_{}_sat.png'.format(i, x, y), sat_block)
            cv2.imwrite(output_dir + '{}_{}_{}_sao.png'.format(i, x, y), sao_block * 255)
            cv2.imwrite(output_dir + '{}_{}_{}_pof.png'.format(i, x, y), pof_block * 255)
