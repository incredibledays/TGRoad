import os
import glob
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


def random_shift_scale_rotate(sat, sao, pof, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), border_mode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = sat.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        sat = cv2.warpPerspective(sat, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        sao = cv2.warpPerspective(sao, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        pof = cv2.warpPerspective(pof, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
    return sat, sao, pof


def random_horizontal_flip(sat, sao, pof, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 1)
        sao = np.flip(sao, 1)
        sao[:, :, 2] = 255 - sao[:, :, 2]
        pof = np.flip(pof, 1)
        pof[:, :, 1] = - pof[:, :, 1]
    return sat, sao, pof


def random_vertical_flip(sat, sao, pof, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 0)
        sao = np.flip(sao, 0)
        sao[:, :, 2] = 255 - sao[:, :, 2]
        pof = np.flip(pof, 0)
        pof[:, :, 2] = - pof[:, :, 2]
    return sat, sao, pof


def random_rotate_90(sat, sao, pof, u=0.5):
    if np.random.random() < u:
        sat = np.rot90(sat)
        sao = np.rot90(sao)
        sao[sao[:, :, 2] < 128, 2] += 128
        sao[sao[:, :, 2] >= 128, 2] -= 128
        pof = np.rot90(pof)
        tmp = pof[:, :, 1]
        pof[:, :, 1] = -pof[:, :, 2]
        pof[:, :, 2] = tmp
    return sat, sao, pof


class RoadExtractionDataset(data.Dataset):
    def __init__(self, root_dir):
        self.sample_list = list(map(lambda x: x[:-8], glob.glob(root_dir + '*_sat.png')))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sat = cv2.imread(os.path.join('{}_sat.png').format(self.sample_list[item]))
        sao = cv2.imread(os.path.join('{}_sao.png').format(self.sample_list[item]))
        pof = cv2.imread(os.path.join('{}_pof.png').format(self.sample_list[item]))

        sat = random_hue_saturation_value(sat, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        sat, sao, pof = random_shift_scale_rotate(sat, sao, pof, shift_limit=(-0.1, 0.1), scale_limit=(-0, 0), aspect_limit=(-0, 0), rotate_limit=(-0, 0))
        sat, sao, pof = random_horizontal_flip(sat, sao, pof)
        sat, sao, pof = random_vertical_flip(sat, sao, pof)
        sat, sao, pof = random_rotate_90(sat, sao, pof)

        sat = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat = torch.Tensor(sat)
        sao = np.array(sao, np.float32).transpose((2, 0, 1)) / 255.0
        sao = torch.Tensor(sao)
        pof = np.array(pof, np.float32).transpose((2, 0, 1)) / 255.0
        pof = torch.Tensor(pof)
        return sat, sao, pof
