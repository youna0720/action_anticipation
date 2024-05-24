import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import copy
from typing import Optional, List
#from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from models.attn import Attention
import pdb

import math
import random
import time as Time


#%time_emb = get_timestep_embedding(t, self.time_emb_dim)
def get_timestep_embedding(timesteps, embedding_dim): # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def swish(x):
    return x * torch.sigmoid(x)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def normalize(x, scale): # [0,1] > [-scale, scale]
    x = (x * 2 - 1.) * scale
    return x

def denormalize(x, scale): #  [-scale, scale] > [0,1]
    x = ((x / scale) + 1) / 2  
    return x


class Transformer(nn.Module):

    def __init__(self, device, n_class, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, rel_pos_type=None, rel_only=False, time_emb_dim = 512):
        super().__init__()
        
        self.device = device
        self.n_class = n_class
        self.time_emb_dim = time_emb_dim
        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])        
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, d_model)
        
        timesteps = 1000
        self.sampling_timesteps = 25
        self.ddim_sampling_eta = 1.0
        self.snr_scale = 0.5
        self.cond_types = ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1']
        self.detach_decoder = False
        
        betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.scale = 1.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ##############################################################
        #decoder_params['input_dim'] = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params['num_f_maps']
        #if -1 in encoder_params['feature_layer_indices']: # -1 means "video feature"
        #    decoder_params['input_dim'] += encoder_params['input_dim']
        #if -2 in encoder_params['feature_layer_indices']: # -2 means "encoder prediction"
        #    decoder_params['input_dim'] += self.num_classes

        self.d_head = d_head = d_model // nhead

        if rel_pos_type == 'rotary' :
            assert (d_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embedding'
            pos_emb = RotaryEmbedding(d_head // 2)
        else :
            pos_emb = None

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, pos_emb, rel_pos_type)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, pos_emb, rel_pos_type)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        self.d_model = d_model
        self.nhead = nhead
        self.rel_pos_type = rel_pos_type
        self.rel_only = rel_only
        
        self.conv_in =  nn.Conv1d(n_class, d_model, 1)   # nn.Conv1d   B C T
        self.conv_out = nn.Conv1d(d_model, n_class, 1)   #
        
        self._reset_parameters()
 


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, past_label, src, tgt, mask, tgt_mask, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed):
        # memory =  T B F  [8 32 512]       src 8,32,512    tgt 1,32,512  t b c
        ###########################
        '''<Diffact> input_dim = 2048   T = 1602  num_class = 19
        feature [1,2048,1602]  label [1,1602]
        encoder_out [1,19,1602]  backbone_feats=[1,192,1602]
        event_gt = event_diffused = event_out [1, 19, 1602] '''
        #fra는 conv_in으로 event_diffused.float 의 19를 24(d_model, hidden)
        #backbone_feats는 동일
        '''encoder_out, backbone_feats = self.encoder(video_feats=src, get_features=True)'''
         #encoder_out = [B C T] backbone_feats = [B F합 T]
         
        if self.rel_only :
            pos_embed = None
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # memory =  t b c  [8 32 512]
        
        event_diffused, noise, t = self.prepare_targets(past_label)
        
        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)        
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)  
        #time_emb 1,512  512= time_emb_dim
        
        fra = self.conv_in(event_diffused.float()) 
        #fra = self.module(fra, backbone_feats, time_emb)
        
        if time_emb is not None:
            fra = fra + self.time_proj(swish(time_emb))[:,:,None]      # 동일하게 1 24 1602
 
        #for layer in self.layers:
        #    fra = layer(fra, backbone_feats) #B H 24 T, B F합(192) T        레이어 돌아도 fra 동일'''

        fra = rearrange(fra, 'b h c -> c b h')   #원래 tgt =1 32 512처럼 1 24 1602 -> 1602 1 24
        fra = self.decoder(fra, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
        
        '''event_out = self.conv_out(fra)     #이 과정 필요없나? 이미 fra= B C T  C=nclass=19'''
        #pdb.set_trace()
        return memory, fra   #8 32 512    1 32 512
        #DETR_next: src_emb, output 

   ######################################################################
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def q_sample(self, x_start, t, noise=None): # forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def model_predictions(self, backbone_feats, x, t):              

        x_m = torch.clamp(x, min=-1 * self.scale, max=self.scale) # [-scale, +scale]
        x_m = denormalize(x_m, self.scale)                        # [0, 1]

        assert(x_m.max() <= 1 and x_m.min() >= 0)
        x_start = self.decoder(backbone_feats, t, x_m.float()) # torch.Size([1, C, T])
        x_start = F.softmax(x_start, 1)
        assert(x_start.max() <= 1 and x_start.min() >= 0)

        x_start = normalize(x_start, self.scale)                              # [-scale, +scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return pred_noise, x_start

    def prepare_targets(self, event_gt):

        # event_gt: normalized [0, 1]
        
        scale = 0.5 #1
        #assert(event_gt.max() <= 1 and event_gt.min() >= 0)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long() #device=self.device
        #self.num_timesteps-1 범위 내 랜덤,디퓨징 과정의 가능한 모든 타임스텝을 무작위로 샘플링가능

        noise = torch.randn(size=event_gt.shape, device=self.device)
        x_start = (event_gt * 2. - 1.) * scale  #[-scale, +scale]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * scale, max=scale)
        event_diffused = ((x / scale) + 1) / 2.           # normalized [0, 1]

        return event_diffused, noise, t
    #############################################################################
    

class TransformerEncoder(nn.Module) :

    def __init__(self, encoder_layer, num_layers, norm=None) :
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, pos_emb=None, rel_pos_type=None):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        if rel_pos_type == 'abs' :
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else :
            self.self_attn = Attention(d_model, d_model, nheads=nhead, dropout=dropout,
                                       pos_embed=pos_emb, rel_pos_type=rel_pos_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = v = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, pos_emb=False, rel_pos_type=None):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        if rel_pos_type == 'abs' :
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else :
            self.self_attn = Attention(d_model, d_model, nheads=nhead, dropout=dropout, pos_embed=pos_emb, rel_pos_type=rel_pos_type)
            self.multihead_attn = Attention(d_model, d_model, nheads=nhead, dropout=dropout, pos_embed=pos_emb, rel_pos_type=rel_pos_type)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] =None):
        q = k = v = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=self.with_pos_embed(memory, pos),
#                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

   