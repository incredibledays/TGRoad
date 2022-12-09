import pickle
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from afm_op import afm
import torch


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
output_root = './cityscale/'
output_dir = ''
dataset_image_size = 2048
size = 512
stride = 256

for i in tqdm(range(180)):
    sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
    neighbors = neighbor_to_integer(pickle.load(open(input_dir + 'region_%d_refine_gt_graph.p' % i, 'rb')))

    seg = np.zeros((dataset_image_size, dataset_image_size))
    ach = np.zeros((dataset_image_size, dataset_image_size, 2))

    lines = []
    for loc, n_locs in neighbors.items():
        if len(n_locs) < 3:
            ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 0] = np.ones((3, 3))
        else:
            ach[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 1] = np.ones((3, 3))

        for n_loc in n_locs:
            cv2.line(seg, (loc[1], loc[0]), (n_loc[1], n_loc[0]), 1)
            if loc[0] < n_loc[0]:
                line = [loc[1], loc[0], n_loc[1], n_loc[0]]
            elif loc[0] > n_loc[0]:
                line = [n_loc[1], n_loc[0], loc[1], loc[0]]
            else:
                if loc[1] < n_loc[1]:
                    line = [loc[1], loc[0], n_loc[1], n_loc[0]]
                else:
                    line = [n_loc[1], n_loc[0], loc[1], loc[0]]
            if line not in lines:
                lines.append(line)

    shape_info = np.array([[0, len(lines), dataset_image_size, dataset_image_size]])
    lines = torch.Tensor(lines).cuda()
    shape_info = torch.IntTensor(shape_info).cuda()
    afmap, aflabel = afm(lines, shape_info, dataset_image_size, dataset_image_size)

    seg = np.expand_dims(cv2.dilate(seg, np.ones((3, 3), np.uint8)), axis=2)
    pal = np.concatenate([seg, ach], axis=2) * 255
    ori = np.flip(afmap[0].data.cpu().numpy().transpose([1, 2, 0]), 0)
    ori[:, :, 1] = - ori[:, :, 1]
    # ori = np.rot90(afmap[0].data.cpu().numpy().transpose([1, 2, 0]))
    # x = ori[:, :, 0]
    # y = ori[:, :, 1]
    # ori[:, :, 0] = y
    # ori[:, :, 1] = - x
    # from dataset import random_shift_scale_rotate
    # ori = afmap[0].data.cpu().numpy().transpose([1, 2, 0])
    # sat, pal, ori = random_shift_scale_rotate(sat, pal, ori, shift_limit=(-0.1, 0.1), scale_limit=(-0, 0), aspect_limit=(-0, 0), rotate_limit=(-0, 0))
    # lab = aflabel[0].data.cpu().numpy().transpose([1, 2, 0])
    # lab = lab / np.max(lab) * 255

    # import matplotlib.pyplot as plt
    # xx, yy = np.meshgrid(range(2048), range(2048))
    # afx = np.sign(ori[:, :, 0]) * np.exp(- np.abs(ori[:, :, 0])) * 2048 + xx
    # afy = np.sign(ori[:, :, 1]) * np.exp(- np.abs(ori[:, :, 1])) * 2048 + yy
    # plt.plot(afx, afy, 'r.', markersize=0.5)
    # # plt.imshow(seg * 255)
    # plt.show()
    # break

    if i % 10 < 8:
        output_dir = output_root + 'train/'
    if i % 20 == 18:
        output_dir = output_root + 'valid/'
    if i % 20 == 8 or i % 10 == 9:
        output_dir = output_root + 'test/'
        shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
        shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
        cv2.imwrite(output_dir + 'region_%d_pal.png' % i, pal)
        continue

    maxx = int((dataset_image_size - size) / stride)
    maxy = int((dataset_image_size - size) / stride)
    for x in range(maxx + 1):
        for y in range(maxy + 1):
            # x = 6
            # y = 6
            #
            # seg_block = seg[x * stride:x * stride + size, y * stride:y * stride + size]
            # cv2.imwrite('./{}_{}_{}_seg.png'.format(i, x, y), seg_block * 255)
            # ach_block = pal[x * stride:x * stride + size, y * stride:y * stride + size, :]
            # ach_block[:, :, 0] = 0
            # lab_block = lab[x * stride:x * stride + size, y * stride:y * stride + size, :]
            # cv2.imwrite('./{}_{}_{}_ach.png'.format(i, x, y), ach_block)
            # cv2.imwrite('./{}_{}_{}_lab.png'.format(i, x, y), lab_block)
            # break

            sat_block = sat[x * stride:x * stride + size, y * stride:y * stride + size, :]
            pal_block = pal[x * stride:x * stride + size, y * stride:y * stride + size, :]
            ori_block = ori[x * stride:x * stride + size, y * stride:y * stride + size, :]
            cv2.imwrite(output_dir + '{}_{}_{}_sat.png'.format(i, x, y), sat_block)
            cv2.imwrite(output_dir + '{}_{}_{}_pal.png'.format(i, x, y), pal_block)
            pickle.dump(ori_block, open(output_dir + '{}_{}_{}_ori.pkl'.format(i, x, y), 'wb'))
