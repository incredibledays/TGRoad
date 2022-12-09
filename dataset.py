import os
import glob
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data


def random_hue_saturation_value(sat, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(sat)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        sat = cv2.merge((h, s, v))
        sat = cv2.cvtColor(sat, cv2.COLOR_HSV2BGR)
    return sat


def random_horizontal_flip(sat, pal, ori, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 1)
        pal = np.flip(pal, 1)
        ori = np.flip(ori, 1)
        ori[:, :, 0] = - ori[:, :, 0]
    return sat, pal, ori


def random_vertical_flip(sat, pal, ori, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 0)
        pal = np.flip(pal, 0)
        ori = np.flip(ori, 0)
        ori[:, :, 1] = - ori[:, :, 1]
    return sat, pal, ori


class TGRoadDataset(data.Dataset):
    def __init__(self, root_dir):
        self.sample_list = list(map(lambda x: x[:-8], glob.glob(root_dir + '*_sat.png')))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sat = cv2.imread(os.path.join('{}_sat.png').format(self.sample_list[item]))
        pal = cv2.imread(os.path.join('{}_pal.png').format(self.sample_list[item]))
        ori = pickle.load(open(os.path.join('{}_ori.pkl').format(self.sample_list[item]), 'rb'))

        sat = random_hue_saturation_value(sat, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        sat, pal, ori = random_horizontal_flip(sat, pal, ori)
        sat, pal, ori = random_vertical_flip(sat, pal, ori)

        sat = torch.Tensor(np.array(sat, np.float32).transpose((2, 0, 1))) / 255.0 * 3.2 - 1.6
        seg = torch.Tensor(np.array(np.expand_dims(pal[:, :, 0], axis=2), np.float32).transpose((2, 0, 1))) / 255.0
        ach = torch.Tensor(np.array(pal[:, :, 1:], np.float32).transpose((2, 0, 1))) / 255.0
        ori = torch.Tensor(np.array(ori, np.float32).transpose((2, 0, 1)))

        return {'sat': sat, 'seg': seg, 'ach': ach, 'ori': ori}
