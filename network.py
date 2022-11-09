import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


class DilateBlock(nn.Module):
    def __init__(self, channel):
        super(DilateBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(1, 1), padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(2, 2), padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(4, 4), padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(8, 8), padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out), inplace=True)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, (1, 1))
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, (1, 1))
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(DLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DilateBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1))
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, (3, 3), padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


class SOAPDLinkNet34(nn.Module):
    def __init__(self):
        super(SOAPDLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DilateBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.seg_branch = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.ach_branch = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.ori_branch = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 8, (3, 3), padding=1))

        self.dis_branch = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.dir_branch = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1))

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        seg = torch.sigmoid(self.seg_branch(d1))
        ach = torch.sigmoid(self.ach_branch(d1))
        ori = torch.sigmoid(self.ori_branch(d1))
        dis = torch.sigmoid(self.dis_branch(d1))
        dir = torch.sigmoid(self.dir_branch(d1))
        return seg, ach, ori, dis, dir


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        self.res_path1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), padding=1)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1)),
            nn.BatchNorm2d(64)
        )
        self.res_block2 = ResBlock([64, 128], [2, 1])
        self.res_block3 = ResBlock([128, 256], [2, 1])
        self.res_block4 = ResBlock([256, 512], [2, 1])
        self.res_block5 = ResBlock([768, 256], [1, 1])
        self.res_block6 = ResBlock([384, 128], [1, 1])
        self.res_block7 = ResBlock([192, 64], [1, 1])
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.res_path1(x) + self.shortcut1(x)
        block2 = self.res_block2(block1)
        block3 = self.res_block3(block2)
        block4 = self.res_block4(block3)
        block5 = self.res_block5(torch.cat([F.interpolate(block4, scale_factor=2, mode='bilinear'), block3], 1))
        block6 = self.res_block6(torch.cat([F.interpolate(block5, scale_factor=2, mode='bilinear'), block2], 1))
        block7 = self.res_block7(torch.cat([F.interpolate(block6, scale_factor=2, mode='bilinear'), block1], 1))

        return self.output(block7)


class ResBlock(nn.Module):
    def __init__(self, channels, strides):
        super(ResBlock, self).__init__()
        self.res_path = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), stride=strides[0], padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=(3, 3), stride=strides[1], padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(1, 1), stride=strides[0]),
            nn.BatchNorm2d(channels[1])
        )

    def forward(self, x):
        residual = self.res_path(x)
        x = self.shortcut(x)

        return x + residual


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class TGDLinkNet34(nn.Module):
    def __init__(self):
        super(TGDLinkNet34, self).__init__()

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DilateBlock(512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.seg_branch = nn.Sequential(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.ach_branch = nn.Sequential(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.ori_branch = nn.Sequential(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.dis_branch = nn.Sequential(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1))

        self.dir_branch = nn.Sequential(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1))

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        seg = torch.sigmoid(self.seg_branch(d1))
        ach = torch.sigmoid(self.ach_branch(d1))
        ori = torch.sigmoid(self.ori_branch(d1))
        dis = torch.sigmoid(self.dis_branch(d1))
        dir = torch.sigmoid(self.dir_branch(d1))
        return seg, ach, ori, dis, dir


class TGUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TGUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, n_classes[0], (3, 3), padding=1), nn.Sigmoid())
        self.ach_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, n_classes[1], (3, 3), padding=1), nn.Sigmoid())
        self.ori_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, n_classes[2], (3, 3), padding=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        seg = self.seg_branch(x)
        ach = self.ach_branch(x)
        ori = self.ori_branch(x)
        return seg, ach, ori
