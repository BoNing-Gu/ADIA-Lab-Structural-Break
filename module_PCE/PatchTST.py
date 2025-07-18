__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from module.layers.PatchTST_backbone import PatchTST_backbone
from module.layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        
        super().__init__()
        
        # load parameters
        n_layers = config.encoder_layers
        d_model = config.d_model
        n_heads = config.n_heads
        d_k = config.d_k
        d_v = config.d_v
        d_ff = config.d_ff
        norm = config.norm
        attn_dropout = config.attn_dropout
        dropout = config.dropout
        fc_dropout = config.fc_dropout
        head_dropout = config.head_dropout
        act = config.act
        res_attention = config.res_attention
        pre_norm = config.pre_norm
        store_attn = config.store_attn
        pe = config.pe
        learn_pe = config.learn_pe
        verbose = config.verbose
    
        # model
        self.decomposition = config.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(config.kernel_size)
            self.model_trend = PatchTST_backbone(
                config=config, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout, act=act, 
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe,
                verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(
                config=config, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout, act=act, 
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, 
                verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(
                config=config, 
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, fc_dropout=fc_dropout, head_dropout=head_dropout, act=act, 
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, 
                verbose=verbose, **kwargs)
    
    
    def forward(self, x, period=None, key_padding_mask=None):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init, period, key_padding_mask)
            trend = self.model_trend(trend_init, period, key_padding_mask)
            x = res + trend
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x, period, key_padding_mask)  # x: [Batch]
        return x
