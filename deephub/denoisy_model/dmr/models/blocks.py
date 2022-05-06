import torch
from torch.nn import Module, ModuleList, Identity, ReLU, Parameter, Sequential

from .utils import *
from .conv import *
from .pool import *

class FeatureExtraction(Module):

    def __init__(self, in_channels=3, dynamic_graph=True, conv_channels=24, num_convs=4, conv_num_layers=3, conv_layer_out_dim=12, conv_knn=16, conv_aggr='max', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.dynamic_graph = dynamic_graph
        self.num_convs = num_convs

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        for i in range(num_convs):
            if i == 0:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=None)
            else:
                trans = FullyConnected(in_channels, conv_channels, bias=True, activation=activation)
            conv = DenseEdgeConv(conv_channels, num_layers=conv_num_layers, layer_out_dim=conv_layer_out_dim, knn=conv_knn, aggr=conv_aggr, activation=activation)
            self.transforms.append(trans)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def dynamic_graph_forward(self, x):
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, x)
        return x

    def static_graph_forward(self, pos):
        x = pos
        for i in range(self.num_convs):
            x = self.transforms[i](x)
            x = self.convs[i](x, pos)
        return x 

    def forward(self, x):
        if self.dynamic_graph:
            return self.dynamic_graph_forward(x)
        else:
            return self.static_graph_forward(x)
        

class Downsampling(Module):

    def __init__(self, feature_dim, ratio=0.5):
        super().__init__()
        self.pool = GPool(ratio, dim=feature_dim)

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        idx, pos, x = self.pool(pos, x)
        return idx, pos, x


class DownsampleAdjust(Module):
    def __init__(self, feature_dim, ratio=0.5, use_mlp=False, activation='relu', random_pool=False, pre_filter=True):
        super().__init__()
        self.pre_filter = pre_filter
        if random_pool:
            self.pool = RandomPool(ratio)
        else:
            self.pool = GPool(ratio, dim=feature_dim, use_mlp=use_mlp, mlp_activation=activation)
        self.mlp = Sequential(
            FullyConnected(feature_dim, feature_dim // 2, activation=activation),
            FullyConnected(feature_dim // 2, feature_dim // 4, activation=activation),
            FullyConnected(feature_dim // 4, 3, activation=None)
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        idx, pos, x = self.pool(pos, x)
        if self.pre_filter:
            pos = pos + self.mlp(x)
        return idx, pos, x


class Upsampling(Module):

    def __init__(self, feature_dim, mesh_dim=1, mesh_steps=2, use_random_mesh=False, activation='relu'):
        super().__init__()
        self.mesh_dim = mesh_dim
        self.mesh_steps = mesh_steps
        self.mesh = Parameter(get_mesh(dim=mesh_dim, steps=mesh_steps), requires_grad=False)    # Regular mesh
        self.use_random_mesh = use_random_mesh
        if use_random_mesh:
            self.ratio = mesh_steps
            print('[INFO] Using random mesh.')
        else:
            self.ratio = mesh_steps ** mesh_dim


        self.folding = Sequential(
            FullyConnected(feature_dim+mesh_dim, 128, bias=True, activation=activation),
            FullyConnected(128, 128, bias=True, activation=activation),
            FullyConnected(128, 64, bias=True, activation=activation),
            FullyConnected(64, 3, bias=True, activation=None),
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        batchsize, n_pts, _ = x.size()
        x_tiled = x.repeat(1, self.ratio, 1)

        if self.use_random_mesh:
            mesh = get_sample_points(dim=self.mesh_dim, samples=self.mesh_steps, num_points=n_pts, num_batch=batchsize).to(device=x.device)
            x_expanded = torch.cat([x_tiled, mesh], dim=-1)   # (B, rN, d+d_mesh)
        else:        
            mesh_tiled = self.mesh.unsqueeze(-1).repeat(1, 1, n_pts).transpose(1, 2).reshape(1, -1, self.mesh_dim).repeat(batchsize, 1, 1)
            x_expanded = torch.cat([x_tiled, mesh_tiled], dim=-1)   # (B, rN, d+d_mesh)

        residual = self.folding(x_expanded) # (B, rN, 3)

        upsampled = pos.repeat(1, self.ratio, 1) + residual
        return upsampled
        

if __name__ == '__main__':
    batchsize = 3
    n_pts = 3

    x = torch.arange(0, batchsize * n_pts * 3).reshape(batchsize, n_pts, 3).float()
    print(x)

    ratio = 2
    mesh = get_1d_mesh(steps=ratio)

    x_tiled = x.repeat(1, ratio, 1)
    mesh_tiled = mesh.unsqueeze(-1).repeat(1, n_pts).flatten() # (rN, )
    mesh_tiled = mesh_tiled.unsqueeze(0).unsqueeze(-1).repeat(batchsize, 1, 1)  # (B, rN, 1)
    print(x_tiled.size(), mesh_tiled.size())

    
    x_expanded = torch.cat([x_tiled, mesh_tiled], dim=-1)
    print(x_expanded)
