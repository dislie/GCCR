import torch
import math
from torch import nn
import numpy as np
from torch.nn import Sequential as Seq
import torch.nn.functional as F



class BatchCosGraphConv(nn.Module):
    def __init__(self, feature_dim, topk):
        super(BatchCosGraphConv, self).__init__()

        self.topk = topk
        self.nn = nn.Sequential(
            nn.Linear(feature_dim*self.topk, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )

    def forward(self, x):  # x: B, N, C

        B, N, C = x.shape
        x = x.permute(1, 0, 2)  # N, B, C

        cos_sim = x.bmm(x.permute(0, 2, 1).contiguous())  # N, B, B

        topk_weight, topk_index = torch.topk(cos_sim, k=self.topk+1, dim=-1)
        topk_weight = topk_weight[:, :, 1:]  # N, B, K (exclude self-loop)
        topk_index = topk_index[:, :, 1:]  # N, B, K

        topk_index = topk_index.to(torch.long)

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(N).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]
        selected_features = x[batch_indices, topk_index, :]  # N, B, K, C

        topk_weight = F.softmax(topk_weight, dim=2)  # N, B, K
        # w
        x = torch.mul(topk_weight.unsqueeze(-1), selected_features) #  N, B, K, C

        x = self.nn(x.reshape(N, B, -1)) # N, B, C

        x = x.permute(1, 0, 2)

        return x

class CCL(nn.Module):
    def __init__(self, in_channels, topk=2):
        super(CCL, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_channels, in_channels))
        self.GCN = BatchCosGraphConv(feature_dim=in_channels, topk=topk)
        self.fc2 = nn.Sequential(nn.Linear(in_channels, in_channels))

    def forward(self, x):

        x = self.fc1(x)
        x = self.GCN(x)
        x = self.fc2(x)

        return x

class Batch_GNN(nn.Module):
    def __init__(self, in_channels,batch_size,depth):
        super(Batch_GNN, self).__init__()

        self.batch_size_padding = 3

        self.downsample = nn.Sequential(*[
            nn.Linear(in_channels,256,bias=False),
            nn.LayerNorm(256)
        ])

        blocks = []
        for i in range(depth):
            blocks.append(CCL(in_channels=256, topk=2))
        self.GCN = nn.ModuleList(blocks)

        self.upsample = nn.Sequential(*[
            nn.Linear(256,in_channels, bias=False),
            nn.LayerNorm(in_channels)
        ])


    def forward(self, x):

        B, N, C = x.shape
        x_4 = x
        short_cut = x_4.mean(dim=1) # BxC
        x_4 = self.downsample(x_4)  # B, N, C

        ## batch
        padding = False
        if x_4.shape[0] < self.batch_size_padding:
            padding = True
        if padding:
            ori_batch_size = B
            x_4 = torch.cat([x_4, torch.zeros((self.batch_size_padding-x_4.shape[0],*x_4.shape[1:]), dtype=x_4.dtype, device=x_4.device)], dim=0)
            B = x_4.shape[0]

        ## Batch Graph Convolution
        for block in self.GCN:
            x_4 = block(x_4)
        if padding:
            x_4 = x_4[:ori_batch_size]

        x_4 = self.upsample(x_4)  # B, N, C

        x_4 = x_4.mean(dim=1)
        x_4 += short_cut

        return x_4
