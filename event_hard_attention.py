import torch
from torch import nn
from scnn import SCNN
from vit import ViT

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class strans(nn.Module):
    def __init__(self, cfg_cnn):
        super().__init__()

        self.snn = SCNN(cfg_cnn=cfg_cnn)

        self.trans = ViT(image_size = 16,
                         patch_size = 4,
                         dim = 1024,
                         depth = 6,
                         heads = 16,
                         mlp_dim = 2048,
                         dropout = 0.1,
                         emb_dropout = 0.1,
                         channels = 64)

    def forward(self, x):

        x = self.snn(x)
        print(' snn output tensor', x.shape)

        # batch C H W
        x = self.trans(x)
        print(' attention tensor', x.shape)

        return x