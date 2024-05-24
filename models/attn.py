import torch as tr
import torch.nn as nn
from einops import rearrange
from math import sqrt
import torch.nn.functional as F
import pdb
#from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

class Attention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        nheads=8,
        kernel_size=1,
        pos_embed=None,
        stride=(1,1,1),
        dropout=0.1,
        rel_pos_type=None
    ):
        super(Attention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid = d_out // nheads
        self.nheads = nheads
        self.kernel_size = kernel_size
        self.ksize = ksize = kernel_size
        self.pos_embed = pos_embed
        self.rel_pos_type = rel_pos_type
        self.stride = stride
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = 2000
        #q, k, v embedding
#        self.q = nn.Linear(d_in, d_hid*nheads)
#        self.k = nn.Linear(d_in, d_hid*nheads)
#        self.v = nn.Linear(d_in, d_hid*nheads)
        if rel_pos_type == 'conv' :
            kernel_size = 5
        else :
            kernel_size = 1
        pad_size = int((kernel_size - 1)/2)
        self.q = nn.Conv1d(d_in, d_hid*nheads, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.k = nn.Conv1d(d_in, d_hid*nheads, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.v = nn.Conv1d(d_in, d_hid*nheads, kernel_size=kernel_size, padding=pad_size, bias=False)

        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
#        if pos_embed:
#            rel_lengths = tuple(2*k - 1 for k in kernel_size)
#            self.rel_pos = self._calc_rel_pos(kernel_size)
#            self.embed_luv = nn.Parameter(tr.randn(rel_lengths[0], d_hid))
#            self._weight_initialization(self.embed_luv,fan=ksize*d_hid)
#            pdb.set_trace()
        if rel_pos_type == 'rel' :
            rel_lengths = 2*self.max_seq_len - 1
            self.rel_pos = self._calc_rel_pos(kernel_size)
            self.embed_luv = nn.Parameter(tr.randn(rel_lengths, d_hid))
            self._weight_initialization(self.embed_luv, fan=kernel_size*d_hid)
        elif rel_pos_type == 'rotary' :
            self.max_seq_len = 2000
            self.rotary_emb = pos_embed

        self.proj = nn.Conv1d(d_out, d_out, 1, bias=False)
#        self.proj = nn.Linear(d_out, d_out)
        nn.init.xavier_uniform_(self.proj.weight)


        # Downsampling layer
        if max(self.stride) > 1:
            self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def extra_repr(self) -> str:
        return f'dim={self.d_in}, stride={self.stride}, kernel_size={self.kernel_size}, pos_embed={self.pos_embed}, nheads={self.nheads}, d_hid={self.d_hid}'

    def _calc_rel_pos(self, ks):
        pos = tr.arange(ks).unsqueeze(-1)
#         pos = rearrange(tr.stack(pos), 'n i-> i n')  # [n, 2] pos[n] = (i)
        rel_pos = pos[None, :] - pos[:, None]                  # [n, n, 2] rel_pos[n] = (rel_n)
        rel_pos[:,:,0] += ks -1                             # shift value range from [-n+1, n-1] to [0, 2n-2]
        return rel_pos

    def _weight_initialization(self, weight, fan=None):
        gain = sqrt(2.0)
        std = gain / sqrt(fan)
        with tr.no_grad():
            return weight.normal_(0, std)

    def _L2norm(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, query, key, value, position=None, attn_mask=None, key_padding_mask=None):
        query = rearrange(query, 's b c -> b c s')
        key = rearrange(key, 's b c -> b c s')
        value = rearrange(value, 's b c -> b c s')
        B,C,Q = query.size()
        _,_,K = key.size()
        '''q, k, v projection'''
        embed_q = self.q(query)
        embed_k = self.k(key)
        embed_v = self.v(value)

#        Q,B,C = query.siz()
#        K,_,_ = key.size()
#        B,_,S = embed_q.size()
#
#        embed_q = rearrange(embed_q, 's b (n c) -> b n s c', n=self.nheads)
#        embed_k = rearrange(embed_k, 's b (n c) -> b n s c', n=self.nheads)
#        embed_v = rearrange(embed_v, 's b (n c) -> b n s c', n=self.nheads)

        embed_q = rearrange(embed_q, 'b (n c) s -> b n s c',n=self.nheads)
        embed_k = rearrange(embed_k, 'b (n c) s -> b n s c',n=self.nheads)
        embed_v = rearrange(embed_v, 'b (n c) s -> b n s c',n=self.nheads)
        '''Normalization'''
        embed_k = embed_k / sqrt(self.d_hid)
        '''
        q = b h n k
        k = b h m k
        v = b h m v
        '''
        '''Self-Attention'''
        attn = tr.einsum('bhnk,bhmk->bhnm', embed_q, embed_k)

        if self.rel_pos_type == 'rel':
            #qP
            rel_pos_k = self._calc_rel_pos(K).unbind(dim=-1)
            rel_pos_emb_k = self.embed_luv[rel_pos_k]
#            n = self.rel_pos.unbind(dim = -1)
#            rel_pos_emb = self.embed_luv[n] # N,M,C
            qP = tr.einsum('bhqc, kkc->bhqk',embed_q, rel_pos_emb_k)

#            #kP
#            rel_pos_q = self._calc_rel_pos(Q).unbind(dim=-1)
#            rel_pos_emb_q = self.embed_luv[rel_pos_q]
#            kP = tr.einsum('bhkc, qqc -> bhqk', embed_k, rel_pos_emb_q)

            #PP
#            PP = tr.einsum('qkc, qkc -> qk', rel_pos_emb_k, rel_pos_emb_q)

            attn = attn + qP

        elif self.rel_pos_type == 'rotary' :
            freqs = self.pos_embed(tr.arange(self.max_seq_len, device=key.device), cache_key = self.max_seq_len)
            if Q == K :
                freqs = rearrange(freqs[:Q], 'n d -> () () n d' )
                embed_q = apply_rotary_emb(freqs.to(embed_q.device), embed_q)
                embed_k = apply_rotary_emb(freqs.to(embed_k.device), embed_k)
#                embed_v = apply_rotary_emb(freqs.to(embed_v.device), embed_v)
            else :
                freqs_q = rearrange(freqs[:Q], 'n d -> () () n d' )
                freqs_k = rearrange(freqs[:K], 'n d -> () () n d' )
                embed_q = apply_rotary_emb(freqs_q.to(embed_q.device), embed_q)
                embed_k = apply_rotary_emb(freqs_k.to(embed_k.device), embed_k)
#                embed_v = apply_rotary_emb(freqs_k.to(embed_v.device), embed_v)

        #attention mask
        if key_padding_mask is not None :
            B, k = key_padding_mask.size()
            key_padding_mask = key_padding_mask.view(B, 1, 1, k).expand(-1,self.nheads, -1, -1)

            if attn_mask is None :
                attn_mask = key_padding_mask
            elif attn_mask.dtype == tr.bool :
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else :
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == tr.bool:
            new_attn_mask = tr.zeros_like(attn_mask, dtype=tr.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if attn_mask is not None:
            attn += attn_mask


        attn = F.softmax(attn, dim=-1)
        out = tr.einsum('bhnm,bhmv->bhnv',attn,embed_v)
#        out = rearrange(out, 'b n s c -> b s (n c)', s=Q)
        out = rearrange(out, 'b n s c -> b (n c) s', s=Q)
#        if max(self.stride) > 1:
#            out = self.avgpool(out)
#        if self.dropout_p > 0.0 :
#            attn = self.dropout(attn)

        out = self.proj(out)
#        out = rearrange(out, 'b s c -> s b c')
        out = rearrange(out, 'b c s -> s b c')
        return out, attn
