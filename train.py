import cv2
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

from dataset import RoadExtractionDataset
from extractor import Extractor
from network import TGDLinkNet34
from loss import TGLoss

root_dir = './datasets/train/'
weight_dir = './checkpoints/exp3/'

dataset = RoadExtractionDataset(root_dir)
dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)

model = Extractor(TGDLinkNet34, TGLoss)

total_epoch = 300
train_epoch_best_loss = 100.
no_optim = 0
for epoch in range(0, total_epoch + 1):
    dataloader_iter = iter(dataloader)
    train_epoch_loss = 0
    for sat, sao, pof in tqdm(dataloader_iter):
        model.set_input(sat, sao, pof)
        train_loss = model.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(dataloader)
    print('epoch:', epoch, ' train_epoch_loss:', train_epoch_loss, ' lr:', model.old_lr)

    sat_vis, sao_vis = model.visual()
    cv2.imwrite(weight_dir + '{}_sat.jpg'.format(epoch), np.uint8(sat_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))
    cv2.imwrite(weight_dir + '{}_sao.jpg'.format(epoch), np.uint8(sao_vis.cpu().float().detach().numpy().transpose((1, 2, 0))))

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        model.save(weight_dir + str(epoch) + '.th')
        model.save(weight_dir + 'best.th')
        train_epoch_best_loss = train_epoch_loss
    if no_optim > 6:
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if model.old_lr < 5e-7:
            break
        model.load(weight_dir + 'best.th')
        model.update_lr()
