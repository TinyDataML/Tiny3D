import torch
from torch.nn import Module, Linear, Sequential

import numpy as np

from .utils import FullyConnected


class GPool(Module):

    def __init__(self, n, dim, use_mlp=False, mlp_activation='relu'):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.pre = Sequential(
                FullyConnected(dim, dim // 2, bias=True, activation=mlp_activation),
                FullyConnected(dim // 2, dim // 4, bias=True, activation=mlp_activation),
            )
            self.p = Linear(dim // 4, 1, bias=True)
        else:
            self.p = Linear(dim, 1, bias=True)
        self.n = n

    def forward(self, pos, x):
        # pos       : B * N * 3
        # x         : B * N * Fin
        batchsize = x.size(0)
        if self.n < 1:
            k = int(x.size(1) * self.n)
        else:
            k = self.n

        if self.use_mlp:
            y = self.pre(x)
        else:
            y = x

        y = (self.p(y) / torch.norm(self.p.weight, p='fro')).squeeze(-1)  # B * N

        top_idx = torch.argsort(y, dim=1, descending=True)[:, 0:k]  # B * k 
        y = torch.gather(y, dim=1, index=top_idx)  # B * k
        y = torch.sigmoid(y)

        pos = torch.gather(pos, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, 3))
        x = torch.gather(x, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, x.size(-1)))
        x = x * y.unsqueeze(-1).expand_as(x)

        return top_idx, pos, x


class RandomPool(Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_choice(self, batch, num_points):
        if self.n < 1:
            n = int(num_points * self.n)
        else:
            n = self.n
        choice = np.arange(0, num_points)
        np.random.shuffle(choice)
        choice = torch.from_numpy(choice[:n]).long()

        return choice.unsqueeze(0).repeat(batch, 1)

    def forward(self, pos, x):
        B, N, _ = pos.size()
        idx = self.get_choice(B, N).to(device=pos.device)     # (B, K)

        pos = torch.gather(pos, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 3))
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, x.size(-1)))

        return idx, pos, x


if __name__ == '__main__':

    pool = RandomPool(100)
    print(pool.get_choice(2, 20))
