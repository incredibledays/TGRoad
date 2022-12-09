import cv2
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data

from dataset import TGRoadDataset
from extractor import Extractor


def train_exp9():
    """DLinkNet34 as backbone, including segmentation branch, junction and anchor point branch, orientation branch."""
    from network import TGDLinkNet34
    from loss import TGLoss

    data_dir = './datasets/cityscale/train/'
    checkpoint_dir = './checkpoints/exp2/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 16
    num_workers = 4
    total_epoch = 200

    dataset = TGRoadDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(TGDLinkNet34, TGLoss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        sao_vis = torch.cat((pre['seg'][0], (0.5 * pre['ach'][0][0] + pre['ach'][0][1]).unsqueeze(0), pre['ori'][0]), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_sao.jpg'.format(epoch), np.uint8(sao_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()


def train_exp1():
    """DLASeg as backbone, including segmentation branch, anchor point branch."""
    from network import TGDLABaseline
    from loss import TGBaselineLoss

    data_dir = './datasets/cityscale/train/'
    checkpoint_dir = './checkpoints/exp1/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 8
    num_workers = 4
    total_epoch = 200

    dataset = TGRoadDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(TGDLABaseline, TGBaselineLoss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        pre_vis = torch.cat((pre['seg'][0], pre['ach'][0], pre['ach'][0]), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_pre.jpg'.format(epoch), np.uint8(pre_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()


def train_exp2():
    """DLASeg as backbone, including segmentation branch, junction and anchor point branch."""
    from network import TGDLAPoint
    from loss import TGPointLoss

    data_dir = './datasets/cityscale/train/'
    checkpoint_dir = './checkpoints/exp2/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 8
    num_workers = 4
    total_epoch = 200

    dataset = TGRoadDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(TGDLAPoint, TGPointLoss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        pre_vis = torch.cat((pre['seg'][0], pre['ach'][0][0].unsqueeze(0), pre['ach'][0][1].unsqueeze(0)), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_pre.jpg'.format(epoch), np.uint8(pre_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()


def train_exp3():
    """DLASeg as backbone, including segmentation branch, junction and anchor point branch and orientation branch."""
    from network import TGDLA
    from loss import TGLoss

    data_dir = './datasets/cityscale/train/'
    checkpoint_dir = './checkpoints/exp3/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    batch_size = 8
    num_workers = 4
    total_epoch = 200

    dataset = TGRoadDataset(data_dir)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = Extractor(TGDLA, TGLoss)
    epoch_to_start = model.load(checkpoint_dir + 'best.th')

    for epoch in range(epoch_to_start, total_epoch + 1):
        dataloader_iter = iter(dataloader)
        train_epoch_loss = 0
        for data_batch in tqdm(dataloader_iter):
            model.set_input(data_batch)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(dataloader)
        print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.lr())

        sat, pre = model.visual()
        sat_vis = (sat[0] + 1.6) / 3.2 * 255
        pre_vis = torch.cat((pre['seg'][0], pre['ach'][0][0].unsqueeze(0), pre['ach'][0][1].unsqueeze(0)), 0) * 255
        cv2.imwrite(checkpoint_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
        cv2.imwrite(checkpoint_dir + '{}_pre.jpg'.format(epoch), np.uint8(pre_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

        model.save(checkpoint_dir + str(epoch) + '.th', epoch)
        model.save(checkpoint_dir + 'best.th', epoch)

        model.update_learning_rate()


if __name__ == '__main__':
    train_exp3()
