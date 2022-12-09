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


class TGLoss(nn.Module):
    def __init__(self):
        super(TGLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def __call__(self, pre, gt):
        seg_loss = self.bce_loss(pre['seg'], gt['seg']) * 10
        ach_loss = self.bce_loss(pre['ach'], gt['ach']) * 100
        rst_loss = self.bce_loss(pre['rst'], gt['seg']) * 10
        ori_loss = self.l1_loss(pre['ori'], gt['ori']) * 0.2
        # print(seg_loss.item(), ach_loss.item(), rst_loss.item(), ori_loss.item())
        return seg_loss + ach_loss + rst_loss + ori_loss


class TGBaselineLoss(nn.Module):
    def __init__(self):
        super(TGBaselineLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def __call__(self, pre, gt):
        seg_loss = self.bce_loss(pre['seg'], gt['seg'])
        ach_loss = self.bce_loss(pre['ach'], torch.sum(gt['ach'], dim=1).unsqueeze(1)) * 5
        # print(seg_loss.item(), ach_loss.item())
        return seg_loss + ach_loss


class TGPointLoss(nn.Module):
    def __init__(self):
        super(TGPointLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def __call__(self, pre, gt):
        seg_loss = self.bce_loss(pre['seg'], gt['seg'])
        ach_loss = self.bce_loss(pre['ach'], gt['ach'])
        rst_loss = self.bce_loss(pre['rst'], gt['seg'])
        # print(seg_loss.item(), ach_loss.item(), rst_loss.item())
        return seg_loss + ach_loss + rst_loss
