import torch
import torch.nn as nn
from pyro.distributions.transforms import AffineAutoregressive, BlockAutoregressive, AffineCoupling
from pyro.nn import AutoRegressiveNN, DenseNN


class NormFlow(nn.Module):
    def __init__(self, flow_type, num_flows, hidden_dim=20, need_permute=False):
        super(NormFlow, self).__init__()
        self.need_permute = need_permute
        if flow_type == 'IAF':
            self.flow = nn.ModuleList(
                [AffineAutoregressive(AutoRegressiveNN(hidden_dim, [2 * hidden_dim]), stable=True) for _ in
                 range(num_flows)])
        elif flow_type == 'BNAF':
            self.flow = nn.ModuleList(
                [BlockAutoregressive(input_dim=hidden_dim) for _ in
                 range(num_flows)])
        elif flow_type == 'RNVP':
            split_dim = hidden_dim // 2
            param_dims = [hidden_dim - split_dim, hidden_dim - split_dim]
            params = min(hidden_dim, 15)
            self.flow = nn.ModuleList(
                [AffineCoupling(split_dim, DenseNN(split_dim, [params], param_dims)) for _ in range(num_flows)])
        else:
            raise NotImplementedError
        even = [i for i in range(0, hidden_dim, 2)]
        odd = [i for i in range(1, hidden_dim, 2)]
        undo_eo = [i // 2 if i % 2 == 0 else (i // 2 + len(even)) for i in range(hidden_dim)]
        undo_oe = [(i // 2 + len(odd)) if i % 2 == 0 else i // 2 for i in range(hidden_dim)]
        self.register_buffer('eo', torch.tensor(even + odd, dtype=torch.int64))
        self.register_buffer('oe', torch.tensor(odd + even, dtype=torch.int64))
        self.register_buffer('undo_eo', torch.tensor(undo_eo, dtype=torch.int64))
        self.register_buffer('undo_oe', torch.tensor(undo_oe, dtype=torch.int64))

    def permute(self, z, i, undo=False):
        if not undo:
            if i % 2 == 0:
                z = torch.index_select(z, 1, self.eo)
            else:
                z = torch.index_select(z, 1, self.oe)
        else:
            if i % 2 == 0:
                z = torch.index_select(z, 1, self.undo_eo)
            else:
                z = torch.index_select(z, 1, self.undo_oe)
        return z

    def forward(self, z):
        log_jacob = torch.zeros_like(z[:, 0], dtype=torch.float32)
        for i in range(len(self.flow)):
            if self.need_permute:
                z = self.permute(z, i)
            z_new = self.flow[i](z)
            log_jacob += self.flow[i].log_abs_det_jacobian(z, z_new)
            if self.need_permute:
                z_new = self.permute(z_new, i, undo=True)
            z = z_new
        return z, log_jacob
