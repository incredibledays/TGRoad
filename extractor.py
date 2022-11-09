import torch
from torch.autograd import Variable as V


class Extractor:
    def __init__(self, net, loss=None, eval_mode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.sat = None
        self.seg_gt = None
        self.ach_gt = None
        self.ori_gt = None
        self.dis_gt = None
        self.dir_gt = None
        self.seg_pre = None
        self.ach_pre = None
        self.ori_pre = None
        self.dis_pre = None
        self.dir_pre = None
        if eval_mode:
            self.net.eval()
        else:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=2e-4)
            self.loss = loss()
            self.old_lr = 2e-4

    def set_input(self, sat_batch, sao_batch=None, pof_batch=None):
        self.sat = V(sat_batch.cuda())
        if sao_batch is not None:
            self.seg_gt, self.ach_gt, self.ori_gt = sao_batch.split([1, 1, 1], dim=1)
            self.seg_gt = V(self.seg_gt.cuda())
            self.ach_gt = V(self.ach_gt.cuda())
            self.ori_gt = V(self.ori_gt.cuda())
        if pof_batch is not None:
            self.dis_gt, self.dir_gt = pof_batch.split([1, 2], dim=1)
            self.dis_gt = V(self.dis_gt.cuda())
            self.dir_gt = V(self.dir_gt.cuda())

    def optimize(self):
        self.optimizer.zero_grad()
        self.seg_pre, self.ach_pre, self.ori_pre, self.dis_pre, self.dir_pre = self.net.forward(self.sat)
        loss = self.loss(self.seg_pre, self.seg_gt, self.ach_pre, self.ach_gt, self.ori_pre, self.ori_gt, self.dis_pre, self.dis_gt, self.dir_pre, self.dir_gt)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, sat):
        seg_pre, ach_pre, ori_pre, _, _ = self.net.forward(sat)
        return seg_pre, ach_pre, ori_pre

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self):
        new_lr = self.old_lr * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.old_lr = new_lr

    def visual(self):
        sat = (self.sat[0] + 1.6) / 3.2 * 255
        sao = torch.cat((self.seg_pre[0], self.ach_pre[0], self.ori_pre[0] * self.seg_pre[0]), 0) * 255
        return sat, sao
