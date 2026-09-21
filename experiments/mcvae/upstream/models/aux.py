import torch
import torch.nn as nn


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_func, mid_channels=None, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            act_func(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            act_func()
        )

    def forward(self, x):
        conv = self.double_conv(x)
        if self.skip_connection and x.shape[1] <= conv.shape[1]:
            return pad_channels(x, conv.shape[1]) + conv
        else:
            return conv


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, act_func):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, act_func)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, act_func, upsampling=True, size=None):
        super().__init__()

        # if upsampling, use the normal convolutions to reduce the number of channels
        if upsampling:
            if size is None:
                self.up = nn.Upsample(scale_factor=2, mode='nearest')  # , align_corners=True
            else:
                self.up = nn.Upsample(size=size, mode='nearest')  # align_corners=True
            self.conv = DoubleConv(in_channels, out_channels, act_func, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, act_func)

    def forward(self, x):
        x = self.conv(self.up(x))
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ULA_nn(nn.Module):
    def __init__(self, input=2, output=2, hidden=(50,), h_dim=None):
        super().__init__()
        if h_dim is None:
            h_dim = 0
        self.l1 = nn.Linear(in_features=input + h_dim, out_features=hidden[0])
        self.act_func = nn.LeakyReLU()
        self.hidden_part = nn.ModuleList([])
        for i in range(1, len(hidden)):
            self.hidden_part.append(nn.Linear(in_features=hidden[i - 1], out_features=hidden[i]))
            self.hidden_part.append(self.act_func)
        self.llast = nn.Linear(in_features=hidden[-1], out_features=output)

    def forward(self, x):
        h = self.act_func(self.l1(x))
        for layer in self.hidden_part:
            h = layer(h)
        return self.llast(h)


class ULA_nn_sm(nn.Module):
    def __init__(self, input=2, output=2, hidden=(50,), h_dim=None):
        super().__init__()
        if h_dim is None:
            h_dim = 0
        self.l1 = nn.Linear(in_features=input + h_dim, out_features=hidden[0])
        self.act_func = nn.LeakyReLU()
        self.hidden_part = nn.ModuleList([])
        for i in range(1, len(hidden)):
            self.hidden_part.append(nn.Linear(in_features=hidden[i - 1], out_features=hidden[i]))
            self.hidden_part.append(self.act_func)
        self.llast = nn.Linear(in_features=hidden[-1], out_features=output)

    def forward(self, z, x):
        if len(x.shape) > 2:
            x_ = x.view((z.shape[0], -1))
            h = torch.cat([z, x_], dim=1)
        else:
            h = torch.cat([z, x], dim=1)
        h = self.act_func(self.l1(h))
        for layer in self.hidden_part:
            h = layer(h)  ###potentially good to write here layer(torch.cat([z,h], dim = 1)) ??
        return self.llast(h)
