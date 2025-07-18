__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=1, sci_mode=False)

#from collections import OrderedDict
from module.layers.PatchTST_layers import *
from module.layers.RevIN import RevIN

from module.utils import plot_debug_zseq, plot_debug_attn

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, config, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., fc_dropout:float=0., head_dropout = 0, act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        self.config = config
        self.debug = self.config.debug
        # RevIn
        if self.config.revin: 
            self.revin_layer = RevIN(self.config.n_vars, affine=config.affine, subtract_last=config.subtract_last)
        
        # Patching
        self.patch_num = int((self.config.seq_length - self.config.patch_len) / self.config.stride + 1)
        if self.config.padding_patch_method == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.config.stride)) 
            self.patch_num += 1
        # CLS Token & SEP Token
        self.register_buffer("CLS", torch.full((self.config.n_vars, self.config.patch_len, 1), 1.0))  # [nvars, patch_len, 1]
        self.register_buffer("SEP", torch.full((self.config.n_vars, self.config.patch_len, 1), 0.0))  # [nvars, patch_len, 1]
        # self.CLS = nn.Parameter(torch.full((self.config.n_vars, self.config.patch_len, 1), 1.0))  # [nvars, patch_len, 1]
        # self.SEP = nn.Parameter(torch.full((self.config.n_vars, self.config.patch_len, 1), 0.0))  # [nvars, patch_len, 1]
        
        # Backbone 
        self.backbone = TSTiEncoder(
            config=config, debug=self.debug, 
            patch_num=self.patch_num+3, 
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
            norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act, 
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs
        )

        # Classification Head (based on CLS token)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.config.n_vars, d_model), 
            nn.ReLU(), 
            nn.Dropout(fc_dropout),
            nn.Linear(d_model, 1),  # Binary classification
        )

        # # Classification Head (based on Flatten token)
        # self.cls_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(d_model* (self.patch_num + 3) * self.config.n_vars, d_model), 
        #     nn.ReLU(), 
        #     nn.Dropout(fc_dropout),
        #     nn.Linear(d_model, 1),  # Binary classification
        # )
    
    def forward(self, z, period=None, key_padding_mask=None):           
        # z: [bs, nvars, seq_len]
        # key_padding_mask: [bs, seq_len] (True indicates padding)
        # period: [bs, seq_len]  
        # print(f'[DEBUG]: z:{z}, z.shape:{z.shape}')
        # print(f'[DEBUG]: period:{period}, period.shape:{period.shape}')
        # print(f'[DEBUG]: key_padding_mask:{key_padding_mask}, key_padding_mask.shape:{key_padding_mask.shape}')
        if self.debug:
            plot_debug_zseq(z, period, key_padding_mask)

        # RevIn norm
        if self.config.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        bs, n_vars, seq_len = z.shape
        if self.debug:
            print(f'[DEBUG]: bs, n_vars, seq_len: {bs, n_vars, seq_len}')

        z_list = []
        token_type_ids_list = []
        key_padding_mask_list = []

        def patch_single(z_part: torch.Tensor) -> torch.Tensor:
            if self.config.padding_patch_method == 'end':
                # ReplicationPad1d requires input of shape [N, C, L]
                z_part = self.padding_patch_layer(z_part.unsqueeze(0)).squeeze(0)  # [n_vars, len_padded]
            patches = z_part.unfold(dimension=-1, size=self.config.patch_len, step=self.config.stride)  # [n_vars, patch_num, patch_len]
            patches = patches.permute(0, 2, 1)  # [n_vars, patch_len, patch_num]
            return patches

        for i in range(bs):
            z_i = z[i]                         # [n_vars, seq_len]
            mask_i = key_padding_mask[i]       # [seq_len]
            valid_z = z_i[:, ~mask_i]          # [n_vars, valid_len]
            period_i = period[i][~mask_i]      # [valid_len]
            z0_i = valid_z[:, period_i == 0]   # [n_vars, len0]
            z1_i = valid_z[:, period_i == 1]   # [n_vars, len1]
            
            # do patching
            z0_patch = patch_single(z0_i)      # [n_vars, patch_len, patch_num0]
            z1_patch = patch_single(z1_i)      # [n_vars, patch_len, patch_num1]

            # ---------- Special Tokens ----------
            cls_token = self.CLS  # [n_vars, patch_len, 1]  .clone().detach()
            sep_token = self.SEP
            
            # compute total length
            patch_len_0 = z0_patch.shape[2]
            patch_len_1 = z1_patch.shape[2]
            total_len = patch_len_0 + patch_len_1

            # truncate
            if total_len > self.patch_num:
                drop_len = total_len - self.patch_num
                z1_patch = z1_patch[:, :, :-drop_len]
                patch_len_1 = z1_patch.shape[2]
                total_len = patch_len_0 + patch_len_1
            # padding
            if total_len < self.patch_num:
                pad_len = self.patch_num - total_len
                padding = torch.zeros(n_vars, self.config.patch_len, pad_len, device=z.device)
            else:
                padding = torch.empty(0, device=z.device)

            # ---------- Concat ----------
            z_i = torch.cat([cls_token, z0_patch, sep_token, z1_patch, sep_token, padding], dim=2)  # [n_vars, patch_len, patch_num + 3 (+ pad)]
            z_list.append(z_i)

            # ---------- Token Type IDs ----------
            type_ids_0 = torch.zeros(n_vars, 1, 1 + patch_len_0 + 1, device=z.device, dtype=torch.long)
            type_ids_1 = torch.ones(n_vars, 1, patch_len_1 + 1, device=z.device, dtype=torch.long)
            type_ids_pad = torch.zeros(n_vars, 1, padding.shape[-1], device=z.device, dtype=torch.long)
            token_type_i = torch.cat([type_ids_0, type_ids_1, type_ids_pad], dim=2)  # [n_vars, 1, total_patches]
            token_type_ids_list.append(token_type_i)

            # ---------- Key Padding Mask ----------
            mask_0 = torch.zeros(n_vars, 1, 1 + patch_len_0 + 1 + patch_len_1 + 1, device=z.device, dtype=torch.bool)  # CLS + z0 + SEP + z1 + SEP
            mask_1 = torch.ones(n_vars, 1, padding.shape[-1], device=z.device, dtype=torch.bool)  # padding 部分
            key_mask = torch.cat([mask_0, mask_1], dim=2)  # [n_vars, 1, total_patches]
            key_padding_mask_list.append(key_mask)

        # ---------- Stack ----------
        z = torch.stack(z_list, dim=0)                                # [bs, n_vars, patch_len, patch_num + 3]
        token_type_ids = torch.stack(token_type_ids_list, dim=0)      # [bs, n_vars, 1, patch_num + 3]
        key_padding_mask = torch.stack(key_padding_mask_list, dim=0)  # [bs, n_vars, 1, patch_num + 3]

        # print(f'[DEBUG]: z:{z}, z.shape:{z.shape}')
        # print(f'[DEBUG]: token_type_ids:{token_type_ids}, token_type_ids.shape:{token_type_ids.shape}')
        # print(f'[DEBUG]: key_padding_mask:{key_padding_mask}, key_padding_mask.shape:{key_padding_mask.shape}')
        
        # model
        z = self.backbone(z, token_type_ids, key_padding_mask)             # z: [bs, nvars, d_model, patch_num]

        # print(f'[DEBUG]: z:{z}, z.shape:{z.shape}')
        
        # Extract CLS token embedding
        cls_emb = z[:, :, :, 0]  # [bs, nvars, d_model]
        cls_emb = cls_emb.permute(0, 2, 1).reshape(bs, -1)  # flatten: [bs, d_model * nvars]
        # print(f'[DEBUG]: cls_emb:{cls_emb}, cls_emb.shape:{cls_emb.shape}')
        logits = self.cls_head(cls_emb)  # [bs, 1]
        
        # # Extract flatten patch embedding
        # z = z.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, d_model]
        # z = z.reshape(bs, -1)      # [bs, nvars * patch_num * d_model]
        # logits = self.cls_head(z)  # [bs, 1]
        
        return logits.squeeze(-1)  # [bs]


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, config, debug, patch_num,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,d_ff=256, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu",
                 res_attention=True, pre_norm=False, store_attn=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        super().__init__()
        self.config = config
        self.debug = debug
        # Input encoding
        self.seq_len = patch_num
        self.W_P = nn.Linear(self.config.patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.seq_len, d_model)

        # Token type embedding
        self.token_type_embedding = nn.Embedding(2, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            self.seq_len, debug = debug,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
            norm=norm, attn_dropout=attn_dropout, dropout=dropout, activation=act,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn
        )
        
    def forward(self, x, token_type_ids, key_padding_mask) -> Tensor:  
        # x: [bs, nvars, patch_len, patch_num]
        # token_type_ids: [bs, nvars, 1, patch_num]
        # key_padding_mask: [bs, nvars, 1, patch_num]

        bs, n_vars, patch_len, patch_num = x.shape

        # Token type embedding
        token_type_ids = token_type_ids.permute(0,1,3,2)                         # token_type_ids: [bs, nvars, patch_num, 1]
        token_type_ids = torch.reshape(
            token_type_ids, (
                token_type_ids.shape[0]*token_type_ids.shape[1], 
                token_type_ids.shape[2], 
                token_type_ids.shape[3]
        ))                                                                       # token_type_ids: [bs * nvars, patch_num, 1]
        token_type_ids = token_type_ids.squeeze(-1)                              # token_type_ids: [bs * nvars, patch_num]
        token_type_emb = self.token_type_embedding(token_type_ids)               # token_type_emb: [bs * nvars, patch_num, d_model]

        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                                                          # x: [bs, nvars, patch_num, d_model]
        u = torch.reshape(
            x, (
                x.shape[0]*x.shape[1],
                x.shape[2],
                x.shape[3]
        ))                                                                       # u: [bs * nvars, patch_num, d_model]
        u = self.dropout(u + self.W_pos + token_type_emb)                        # u: [bs * nvars, patch_num, d_model]

        # Key padding mask
        key_padding_mask = key_padding_mask.permute(0,1,3,2)                     # key_padding_mask: [bs, nvars, patch_num, 1]
        key_padding_mask = torch.reshape(
            key_padding_mask, (
                key_padding_mask.shape[0]*key_padding_mask.shape[1], 
                key_padding_mask.shape[2], 
                key_padding_mask.shape[3]
        ))                                                                       # key_padding_mask: [bs * nvars, patch_num, 1]
        key_padding_mask = key_padding_mask.squeeze(-1)                          # key_padding_mask: [bs * nvars, patch_num]

        # Encoder
        z = self.encoder(u, key_padding_mask)                                    # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    

# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, debug,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=None, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, pre_norm=False, store_attn=False):
        super().__init__()

        self.debug = debug
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                q_len, debug,
                d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                norm=norm, attn_dropout=attn_dropout, dropout=dropout, activation=activation, 
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn
            ) for i in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, debug, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        self.debug = debug
        
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(debug, d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class _MultiheadAttention(nn.Module):
    def __init__(self, debug, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()

        self.debug = debug
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(debug, d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        if self.debug:
            plot_debug_attn(q_s, "Query")
            plot_debug_attn(k_s, "Key")
            plot_debug_attn(v_s, "Value")

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, debug, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()

        self.debug = debug
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if self.debug:
            plot_debug_attn(attn_scores, "Attention Scores (before mask)")
        # mask_row = key_padding_mask.unsqueeze(2)  # [bs, seq_len, 1]
        # mask_col = key_padding_mask.unsqueeze(1)  # [bs, 1, seq_len]
        # key_padding_mask = mask_row | mask_col           # [bs, seq_len, seq_len]
        # key_padding_mask = key_padding_mask.unsqueeze(1) # [bs, 1, seq_len, seq_len]
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)       # [bs, 1, 1, seq_len]
        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask, -np.inf)
        if self.debug:
            plot_debug_attn(attn_scores, "Attention Scores (after mask)")

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.debug:
            plot_debug_attn(output, title="Attention Output")
            plot_debug_attn(attn_weights, title="Attention Map")

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x