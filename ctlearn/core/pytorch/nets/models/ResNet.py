import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from sys import path
from os import getcwd
import math

# path.append(getcwd() + "/../deep_learning_helper")
# path.append(getcwd() + "/../Training")

from ctlearn_helper.ModelHelper import ModelHelper
from nets.block.cnn_blocks import ResBlock


class ResNet(nn.Module):
    def __init__(self, class_names, num_output, conv_output=False):

        super(ResNet, self).__init__()
        self.conv_output = conv_output
        self.class_names = class_names
        self.num_output = num_output

        kernel_size_1 = 7
        kernel_size_2 = 5
        kernel_size_3 = 3
        conv_drop_pro = 0.1
        fc_drop_pro = 0.2

        resblock_1_out_size = 16
        resblock_2_out_size = 32
        resblock_3_out_size = 64
        resblock_4_out_size = 128
        resblock_5_out_size = 256
        # --------------------------------------------------------------------
        self.resblock1 = ResBlock(
            n_chans_in=1,
            n_chans_out=resblock_1_out_size,
            kernel_size=kernel_size_1,
            conv_drop_pro=conv_drop_pro,
        )
        self.resblock2 = ResBlock(
            n_chans_in=resblock_1_out_size,
            n_chans_out=resblock_2_out_size,
            kernel_size=kernel_size_2,
            conv_drop_pro=conv_drop_pro,
        )
        self.resblock3 = ResBlock(
            n_chans_in=resblock_2_out_size,
            n_chans_out=resblock_3_out_size,
            kernel_size=kernel_size_3,
            conv_drop_pro=conv_drop_pro,
        )
        self.resblock4 = ResBlock(
            n_chans_in=resblock_3_out_size,
            n_chans_out=resblock_4_out_size,
            kernel_size=kernel_size_3,
            conv_drop_pro=conv_drop_pro,
        )
        self.resblock5 = ResBlock(
            n_chans_in=resblock_4_out_size,
            n_chans_out=resblock_5_out_size,
            kernel_size=kernel_size_3,
            conv_drop_pro=conv_drop_pro,
        )
        # --------------------------------------------------------------------

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_conv_0 = nn.Linear(resblock_5_out_size, 256)
        self.fc_act_0 = nn.PReLU()
        self.fc_conv_1 = nn.Linear(256, 256)
        self.fc_conv_2 = nn.Linear(256, self.num_output)

        if self.conv_output:
            self.conv1x1_1 = nn.Conv1d(256 * 9 * 9, 32, kernel_size=1)
            self.conv1x1_batchnorm_1 = nn.BatchNorm1d(num_features=32)
            self.act6 = nn.LeakyReLU()

            self.conv1x1_2 = nn.Conv1d(32, 256, kernel_size=1)
            self.conv1x1_batchnorm_2 = nn.BatchNorm1d(num_features=256)
            self.act7 = nn.LeakyReLU()

            self.conv1x1_3 = nn.Conv1d(256, 256, kernel_size=1)
            self.conv1x1_batchnorm_3 = nn.BatchNorm1d(num_features=256)
            self.act8 = nn.LeakyReLU()

            self.conv1x1_final = nn.Conv1d(256, self.num_output, 1)

            torch.nn.init.constant_(self.conv1x1_batchnorm_1.weight, 0.5)
            torch.nn.init.constant_(self.conv1x1_batchnorm_2.weight, 0.5)
            torch.nn.init.constant_(self.conv1x1_batchnorm_3.weight, 0.5)

        else:
            self.fc1 = nn.Linear(256 * 9 * 9, 1024)
            self.act6 = nn.LeakyReLU()
            self.fc1_dropout = nn.Dropout(p=fc_drop_pro)
            self.fc2 = nn.Linear(1024, self.num_output)

    def forward(self, x):

        out = self.resblock1(x)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)

        out = self.avg_pooling(out)
        fcsize = out.shape[1] * out.shape[2] * out.shape[3]
        # --------------------------------------------------------------------
        # Convolution output
        if self.conv_output:
            out = out.view(-1, fcsize, 1)
            out = self.act6(self.conv1x1_batchnorm_1(self.conv1x1_1(out)))
            out = self.act7(self.conv1x1_batchnorm_2(self.conv1x1_2(out)))
            out = self.act8(self.conv1x1_batchnorm_3(self.conv1x1_3(out)))
            out = self.conv1x1_final(out)
            out = out.view(out.shape[0], self.num_output)
            ii=0
            # out = torch.log_softmax(out, dim=1)
        else:
            # Full Connected

            out = out.view(-1, fcsize)
            out = self.fc_conv_0(out)

            out = self.fc_conv_1(out)
            out = self.fc_conv_2(out)
            # out = torch.log_softmax(out_features, dim=1)
        return out
