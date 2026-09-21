import torch.nn as nn

from models.aux import Up, OutConv


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class FC_decoder_mnist(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(hidden_dim, 400),
            # act_func(),
            nn.Linear(hidden_dim, 784)
        )
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


class CONV_decoder(nn.Module):
    def __init__(self, act_func, hidden_dim, n_channels, shape, upsampling=True):
        super().__init__()
        self.upsampling = upsampling
        factor = 2 if upsampling else 1
        num_maps = 16
        shape_init = shape
        shape = shape // 8
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, (8 * num_maps // factor) * (shape ** 2)),
            View(((8 * num_maps // factor), shape, shape)),
            Up(8 * num_maps // factor, 4 * num_maps // factor, act_func, upsampling, (shape * 2, shape * 2)),
            Up(4 * num_maps // factor, 2 * num_maps // factor, act_func, upsampling, (shape * 4, shape * 4)),
            Up(2 * num_maps // factor, num_maps, act_func, upsampling, (shape_init, shape_init)),
            OutConv(num_maps, n_channels)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


class FC_decoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 400),
            act_func(),
            nn.Linear(400, 3072)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


def get_decoder(net_type, act_func, hidden_dim, dataset, shape):
    if str(dataset).lower().find('mni') > -1:
        if net_type == 'fc':
            return FC_decoder_mnist(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_decoder(act_func=act_func,
                                hidden_dim=hidden_dim, n_channels=1, shape=shape)
    else:
        if net_type == 'fc':
            return FC_decoder_cifar(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_decoder(act_func=act_func,
                                hidden_dim=hidden_dim, n_channels=3, shape=shape)
