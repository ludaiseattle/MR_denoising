#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class UNet1(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1):
        """Initializes U-Net."""

        super(UNet1, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        #print("")
        #print("x", x.shape)
        pool1 = self._block1(x)
        #print("pool1", pool1.shape)
        pool2 = self._block2(pool1)
        #print("pool2", pool2.shape)
        pool3 = self._block2(pool2)
        #print("pool3", pool3.shape)
        pool4 = self._block2(pool3)
        #print("pool4", pool4.shape)
        pool5 = self._block2(pool4)
        #print("pool5", pool5.shape)

        # Decoder
        upsample5 = self._block3(pool5)
        #print("upsample5", upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), dim=0)
        #print("concat5", concat5.shape)
        upsample4 = self._block4(concat5)
        #print("upsample4", upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), dim=0)
        #print("concat4", concat4.shape)
        upsample3 = self._block5(concat4)
        #print("upsample3", upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), dim=0)
        #print("concat3", concat3.shape)
        upsample2 = self._block5(concat3)
        #print("upsample2", upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), dim=0)
        #print("concat2", concat2.shape)
        upsample1 = self._block5(concat2)
        #print("upsample1", upsample1.shape)
        concat1 = torch.cat((upsample1, x), dim=0)
        #print("concat1", concat1.shape)
        out = self._block6(concat1)
        #print("out", out.shape)

        # Final activation
        return out 

class UNet2(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=1, out_channels=1):
        """Initializes U-Net."""

        super(UNet2, self).__init__()

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True))

        self._block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True))

        self._block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True))

        self._block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True))

        self._block5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0, output_padding=0))

        self._block6 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0, output_padding=0))

        self._block7 = nn.Sequential(
            nn.Conv2d(512, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0))

        self._block8 = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0))

        self._block9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True))

        self._block10 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=1, padding=0),
            nn.Conv2d(32, 1, 1, padding=0),
            nn.Conv2d(1, 1, 1, padding=0))

        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        #print("")
        #print("x", x.shape)
        pool1 = self._block1(x)
        #print("pool1", pool1.shape)
        pool2 = self._block2(pool1)
        #print("pool2", pool2.shape)
        pool3 = self._block3(pool2)
        #print("pool3", pool3.shape)
        pool4 = self._block4(pool3)
        #print("pool4", pool4.shape)

        up1 = self._block5(pool4)
        #print("up1", up1.shape)
        concat1 = torch.cat((up1, pool4), dim=0)
        #print("concat1", concat1.shape)
        up2 = self._block6(concat1)
        #print("up2", up2.shape)
        concat2 = torch.cat((up2, pool3), dim=0)
        #print("concat2", concat2.shape)
        up3 = self._block7(concat2)
        #print("up3", up3.shape)
        concat3 = torch.cat((up3, pool2), dim=0)
        #print("concat3", concat3.shape)
        up4 = self._block8(concat3)
        #print("up4", up4.shape)
        concat4 = torch.cat((up4, pool1), dim=0)
        #print("concat4", concat4.shape)
        outcov = self._block9(concat4)
        #print("outcov", outcov.shape)
        out = self._block10(outcov)
        #print("out", out.shape)

        # Final activation
        return out 
