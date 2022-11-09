import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, batch=True):
        super(DiceBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


class SOAPLoss(nn.Module):
    def __init__(self, batch=True):
        super(SOAPLoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cos_similarity = nn.CosineSimilarity()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, seg_gt, seg_pre, ach_gt, ach_pre, ori_gt, ori_pre, dis_gt, dis_pre, dir_gt, dir_pre):
        seg_1 = self.bce_loss(seg_pre, seg_gt)
        seg_2 = self.soft_dice_loss(seg_gt, seg_pre)
        ach = self.bce_loss(ach_pre, ach_gt) * 10
        ori = self.bce_loss(ori_pre, ori_gt) * 10
        dis = self.smooth_l1(dis_pre, dis_gt) * 100
        dir = (1 - self.cos_similarity(dir_pre, dir_gt)).mean()
        loss = seg_1 + seg_2 + ach + ori + dis + dir
        # print('total loss: ', loss.item(), 'seg_1_loss: ', seg_1.item(), 'seg_2_loss: ', seg_2.item(), 'ach_loss: ', ach.item(), 'ori_loss: ', ori.item(), 'dis_loss: ', dis.item(), 'dir_loss: ', dir.item())
        return loss


class TGLoss(nn.Module):
    def __init__(self):
        super(TGLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.cos_similarity = nn.CosineSimilarity()

    @staticmethod
    def soft_dice_loss(y_pred, y_true):
        smooth = 0.0
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
        score = (2. * intersection + smooth) / (i + j + smooth)
        loss = 1 - score.mean()
        return loss

    @staticmethod
    def abs_cosine_similarity_loss(y_pred, y_true, mask):
        loss = 1 - torch.abs(torch.cos((y_pred * mask - y_true * mask) * torch.pi))
        loss = loss.sum() / torch.sum(mask)
        return loss

    def __call__(self, seg_pre, seg_gt, ach_pre, ach_gt, ori_pre, ori_gt, dis_pre, dis_gt, dir_pre, dir_gt):
        seg_loss = self.bce_loss(seg_pre, seg_gt) + self.soft_dice_loss(seg_pre, seg_gt)
        ach_loss = self.bce_loss(ach_pre, ach_gt) * 20
        ori_loss = self.abs_cosine_similarity_loss(ori_pre, ori_gt, seg_gt) * 2
        dis_loss = self.smooth_l1(dis_pre, dis_gt)
        dir_loss = 1 - self.cos_similarity(dir_pre, dir_gt)
        print(seg_loss.item(), ach_loss.item(), ori_loss.item(), dis_loss.item(), dir_loss.mean().item())
        return seg_loss + ach_loss + ori_loss + dis_loss + dir_loss.mean()
