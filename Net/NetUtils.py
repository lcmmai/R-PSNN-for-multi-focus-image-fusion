import torch
import torch.nn as nn

class conv_block_1(nn.Module):                                                                                          #single conv block
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv1_block_1(nn.Module):                                                                                          #single conv block
    def __init__(self, ch_in, ch_out):
        super(conv1_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv7_block_1(nn.Module):                                                                                         #single conv block, kernel_size = 7
    def __init__(self, ch_in, ch_out):
        super(conv7_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class res_block_1(nn.Module):                                                                                           #single residual conv block
    def __init__(self, ch_in, ch_out):
        super(res_block_1, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        res_x = self.res(x)
        return res_x + x

class RACP_block_1(nn.Module):                                                                                          #single residual RACP block
    def __init__(self, ch_in, ch_out):
        super(RACP_block_1, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv1 = nn.Conv2d(ch_in * 3, ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(ch_in * 3, ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(ch_in * 3, ch_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv1_2(x))
        buffer_1.append(self.conv1_3(x))
        b_1 = self.conv1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv2_1(x))
        buffer_2.append(self.conv2_2(x))
        buffer_2.append(self.conv2_3(x))
        b_2 = self.conv2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv3_1(x))
        buffer_3.append(self.conv3_2(x))
        buffer_3.append(self.conv3_3(x))
        b_3 = self.conv3(torch.cat(buffer_3, 1))

        return x + b_1 + b_2 + b_3

