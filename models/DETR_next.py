import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
import time as Time
import random
from einops import repeat, rearrange
from models.position import PositionalEncoding
from models.transformer_ import Transformer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


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


class Detr_next(nn.Module):
    def __init__(self, n_class, hidden_dim, pad_idx, device, args, nheads=8, num_encoder_layers=6, num_decoder_layers=6, \
                 dropout=0.1, input_seq_len=1, feature_dim=1024, time_emb_dim = 512):
        super().__init__()
        
        #Basic Parameters from opts
        self.n_class = n_class
        self.pad_idx = pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.args = args
        self.input_seq_len = int(input_seq_len)
        
        self.time_emb_dim = time_emb_dim
        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])        
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        
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
        
        self.conv_in =  nn.Conv1d(n_class, hidden_dim, 1)   # nn.Conv1d   B C T
        self.conv_out = nn.Conv1d(hidden_dim, n_class, 1)   #

        #Basic Layers
        self.input_embed = nn.Linear(feature_dim, hidden_dim)
        #layer initialization
        nn.init.xavier_uniform_(self.input_embed.weight)

        #Transformer
        if args.qk_attn :
            self.transformer = Transformer(device, n_class, hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
                                           hidden_dim*4, normalize_before=False, rel_pos_type=args.rel_pos_type,
                                           rel_only=args.rel_only)
        else :
            self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
                                              dim_feedforward=hidden_dim*4, dropout=dropout)

        #Dropout Module
        self.dropout_src = nn.Dropout(0.8)
        self.dropout_tgt = nn.Dropout(0.8)
        self.dropout_feat = nn.Dropout(0.8)

        #feature_loss for next anticipation
        if args.feat_loss :
            self.feature_predict = nn.Linear(hidden_dim, feature_dim)
            nn.init.xavier_uniform_(self.feature_predict.weight)

        #Query Embedding for the decoder
        self.query_embed = nn.Embedding(self.input_seq_len, hidden_dim)

        if args.pos_emb :
            #Positional Embedding Param
#        self.pos_embedding = nn.Parameter(torch.zeros(1, self.input_seq_len, hidden_dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, 20, hidden_dim))
            nn.init.xavier_uniform_(self.pos_embedding)
        else :
            #Positional Encoding layer
            self.pos_enc = PositionalEncoding(hidden_dim)

        if args.next :
            self.fc_next = nn.Linear(hidden_dim, self.n_class)
            nn.init.xavier_uniform_(self.fc_next.weight)

#        if args.feat :
#            #feature loss layer
#            self.fc_feat = nn.Linear(hidden_dim, feature_dim)
#            nn.init.xavier_uniform_(self.fc_feat.weight)

        if args.feat_seg :
            self.fc_feat_seg = nn.Linear(feature_dim, self.n_class)
            nn.init.xavier_uniform_(self.fc_feat_seg.weight)

        if args.seg :
            #Segmentation Module
            self.fc_seg = nn.Linear(hidden_dim, self.n_class)
            nn.init.xavier_uniform_(self.fc_seg.weight)

    def forward(self, past_label, inputs):
        #src: feature, src_pad: pad mask
        src = inputs           #feature = [B, S, C]
#        src_key_padding_mask = self.get_pad_mask(src_pad, self.pad_idx).to(self.device)
#        src_mask = (src_pad != self.pad_idx).to(self.device)

        #Get size of tensors
        B, S, C = src.size()

        #Features to hidden dim
        src = self.input_embed(src)                 #[B, H, S]
        #pdb.set_trace()            #src = features = [32,8,1024] [B S C]  -> [B S H] [32, 8, 512])
        src = F.relu(src)

        #DETR decoder target setting
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)       #[B, T, H]
        tgt = torch.zeros_like(query_embed)         #[B, T, H]                      [32 1 512] 왜 차원이 이렇지? 한 frame에 대한 추측씩 뽑아내서 그런가

        if self.args.qk_attn :
            if self.args.pos_emb :
                pos_embed = self.pos_embedding[:, :S,].repeat(B, 1, 1)
                pos_embed = rearrange(pos_embed, 'b t c -> t b c')
            else :
                pos_embed = None
                src = self.pos_enc(src)
            #pdb.set_trace()
            src = rearrange(src, 'b t c -> t b c')                      #8 32 512
            tgt = rearrange(tgt, 'b t c -> t b c')                      #1 32 512
            query_embed = rearrange(query_embed, 'b t c -> t b c')      #1 32 512
            src_key_padding_mask = None
            tgt_mask = None
            src_emb, output = self.transformer(past_label, src, tgt, src_key_padding_mask, tgt_mask, None, query_embed, pos_embed, None)
            #8 32 512 / 1 32 512
            #pdb.set_trace()
        else :
            #Positional Embedding
            if self.args.pos_emb:
                src = src + self.pos_embedding[:,:src.size(1),]
            else:
                #sinusodal pos encoding
                src = self.pos_enc(src)

            query_pos = query_embed
            tgt = tgt + query_pos                       #[B, T, H]

            #Dropout
#            src = self.dropout_src(src)                 #[B, S, H]
#            tgt = self.dropout_tgt(tgt)                 #[B, T, H]

            #Transformer Layer
            src = src.permute(1, 0, 2).contiguous()     #[S, B, H]
            src_emb = self.transformer.encoder(src, src_key_padding_mask=None)

            tgt = tgt.permute(1, 0, 2).contiguous()     #[T, B, H]
            output = self.transformer.decoder(tgt, src_emb).to(self.device)

        #Output reshape
        output = output.permute(1, 0, 2).contiguous()   #[B, T, H]      32 1 512
        src_emb = src_emb.permute(1, 0, 2).contiguous() #[B, S, H]      32 8 512
        
        #Output generation
        outputs = dict()
        if self.args.feat_loss:
            feat_emb = self.dropout_feat(src_emb)
            output_feat = self.feature_predict(feat_emb)
            outputs['feat'] = output_feat

            if self.args.feat_seg :
                output_seg = self.fc_feat_seg(output_feat) #[1024 -> N]
                outputs['seg'] = output_seg

        if self.args.seg :
            #Segmentation Result
            seg_emb = self.dropout_src(src_emb)
            output_seg = self.fc_seg(src_emb)      
            outputs['seg'] = output_seg

        if self.args.next : #이때 output은 그냥 decoder 출력(T B H) permute한 것(B T H)
            output = self.dropout_tgt(output)  # nn.Dropout(0.8)
            output_next = self.fc_next(output)   #nn.Linear(hidden_dim, self.n_class)
            outputs['next'] = output_next 
#        if self.args.feature_one_loss == True:
#            output_feature = self.fc_feat(output)
#            outputs['feature'] = output_feature
#            output_next = self.fc_next(output_feature[:,-1,:])
#            outputs['next'] = output_next
#
#        elif self.args.feature_loss==False:
#        #Next action anticipation Result
#            output_next = self.fc_next(output)
##            output_next = self.fc_next(output[:,-1,:])      #[B, 1, H], Ta=1 when input_seq_len=4
#            outputs['next'] = output_next
#        else:
#        #previous 4 tokens answer
#            output_feature = self.fc_feat(output)
#            outputs['feature'] = output_feature
#            output_next = self.fc_next(output_feature[:,-1,:])
#            outputs['next'] = output_next

        return outputs  #[B, T, H]      32 1 512


    def get_pad_mask(self, seq, pad_idx):
        return (seq == pad_idx)
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        
    @torch.no_grad()
    def ddim_sample(self, past_label, inputs, seed=None):
        src = inputs           #feature = [B, S, C] [1,2048,1602] 이건 diffact고, 뒤에 순서바뀜 B S F [32 8 1024]
        B, S, C = src.size()

        #Features to hidden dim
        src = self.input_embed(src)                #[32 8 1024] -> [32 8 512]
        src = F.relu(src)

        #DETR decoder target setting
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)       #[B, T, H] -> [32 1 512]
        tgt = torch.zeros_like(query_embed)         #[B, T, H]                      [32 1 512] 

        if self.args.pos_emb :
            pos_embed = self.pos_embedding[:, :S,].repeat(B, 1, 1)
            pos_embed = rearrange(pos_embed, 'b t c -> t b c')
        else :
            pos_embed = None
            src = self.pos_enc(src)
        src = rearrange(src, 'b t c -> t b c')                      #[8 32 512]
        video_feats = src
        tgt = rearrange(tgt, 'b t c -> t b c')                      #[1 32 512]
        query_embed = rearrange(query_embed, 'b t c -> t b c')      #[1 32 512]
        src_key_padding_mask = None
        tgt_mask = None
        tgt_key_padding_mask = None
        tgt_pos_embed = None
        
        #encoder_out, backbone_feats = self.encoder(video_feats, get_features=True)
        memory = self.transformer.encoder(video_feats, src_key_padding_mask=None, pos=None) #이거 none해도되나 #인,아웃 둘다 [8 32 512]

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # torch.Size([1, 19, 4847])
        shape = (video_feats.shape[1], self.n_class, 1)              # B class T (B=1,19, 1) 튜플    [32 2513 1]
        #total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta   #1000 25
        total_timesteps, sampling_timesteps, eta = 1000, 25, 1.0
        
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        ## -1.,  39.,  79., 119., 159., 199., 239., 279., 319., 359., 399., 439.,
        #479., 519., 559., 599., 639., 679., 719., 759., 799., 839., 879., 919.,
        #959., 999.
        
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))     #길이 26
        ##[999, 959, 919, 879, 839, 799, 759, 719, 679, 639, 599, 559, 519, 479, 439, 399, 359, 319, 279, 239, 199, 159, 119, 79, 39, -1]
        
        # [999, 749, 499, 249, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)] #길이 25
        # [(999, 749), (749, 499), (499, 249), (249, -1)]
        ##[(999, 959), (959, 919), (919, 879), (879, 839), (839, 799), (799, 759), (759, 719), (719, 679), (679, 639), (639, 599), (599, 559), (559, 519), (519, 479), (479, 439), (439, 399), (399, 359), (359, 319), (319, 279), (279, 239), (239, 199), (199, 159), (159, 119), (119, 79), (79, 39), (39, -1)]
        x_time = torch.randn(shape, device=self.device)    #torch.Size([1, 19, 4749])   [32 2513 1]

        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)     #torch 1차원 원소 1개 (그건 999 인덱스/스텝값)

            '''pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)'''
            #([1, 19, 1212]), ([1, 19, 1212])
            
            
            #pdb.set_trace()
            x_m = torch.clamp(x_time, min=-1 * self.scale, max=self.scale) # [-scale, +scale]   ([1, 19, 1212]     [32 2513 1] 
            #[[[-0.9247],[-0.4253],[-1.0000],
            x_m = denormalize(x_m, self.scale)                        # [0, 1]      
            #[[[0.0377],[0.2873],[0.0000],                               

            assert(x_m.max() <= 1 and x_m.min() >= 0)
            '''x_start = self.decoder(backbone_feats, time_cond, x_m.float()) # torch.Size([1, C, T])'''
            ###########################decoder start################################3
            time_emb = get_timestep_embedding(time_cond, self.time_emb_dim)
            time_emb = self.time_in[0](time_emb)        
            time_emb = swish(time_emb)
            time_emb = self.time_in[1](time_emb)
            #time_emb [1 512]  512= time_emb_dim   -> 다 음수? 근데 맞겠지
            
            fra = self.conv_in(x_m.float())    #nn.Conv1d(num_classes, num_f_maps, 1)  # [32 512 1]  ???????????
            '''    num_f_maps = hidden_dim = 512라서 그런가?
            [[[-1.1654e-01],
         [-3.3465e-01],
         [-1.6965e-01],
         ...,
         [ 8.6595e-01],
         [ 2.2803e-01],
         [ 2.9976e-02]],

        [[ 2.5625e-02],
         [-3.6557e-01],
         [ 7.7835e-03],
         ...,
         [ 9.1813e-01],
         [-1.7810e-01],
         [-5.0378e-02]],
            '''
            
            #fra = self.module(fra, backbone_feats, time_emb)  [-0.0655],[-0.2447],[ 0.2324]...[ 0.5326],[ 0.2755],[ 0.1702]]
            
            # self.time_proj = nn.Linear(time_emb_dim, dodel)
            if time_emb is not None:
                fra = fra + self.time_proj(swish(time_emb))[:,:,None]      # 동일하게 1 24 1602    [32 512 1]

            #fra = [[[-6.6541e-02],  [-3.0230e-01],[-1.7794e-01],
            #pdb.set_trace()
            #for layer in self.layers:
            #    fra = layer(fra, backbone_feats) #B H 24 T, B F합(192) T        레이어 돌아도 fra 동일
            
            fra = rearrange(fra, 'b h c -> c b h')   #원래 tgt =1 32 512처럼 1 24 1602 -> 1602 1 24     # [1 32 512]      [-0.0155], [-0.2124],[ 0.2241],
            #pdb.set_trace()
            
            fra = self.transformer.decoder(fra, memory, tgt_mask=tgt_mask, memory_key_padding_mask= src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)    # [1 32 512]
            #fra [1 32 512]             #[[[-0.0155, -0.2124,  0.2241,  ...,  0.4946,  0.3426,  0.2462],
            '''
            fra 
            [[[ 0.3255,  0.1578,  0.3357,  ...,  0.1758,  0.2529,  0.0120],
            [ 0.4148, -0.4167,  0.1255,  ...,  0.0617,  0.8981,  0.7286],
            '''
            
            fra = rearrange(fra, 'c b h -> b h c')  #[32 512 1]                     [[[ 0.1635],   [ 0.0566],    [ 0.5965],
            #fra = fra.permute(1, 0, 2).contiguous()   #[B, T, H]      32 1 512
            
            x_start = self.conv_out(fra)   #[32 512 1] -> [32 2513 1]          [[[-0.6582], [-0.6588], [ 0.0841],       
            '''
            [[[-0.5237],
            [-0.6955],
            [ 0.2386],
            ...,
            '''
            
            #x_start = rearrange(x_start, 't b c -> b c t')  #32 2513 8
            #return memory, fra   # 8 32 512        1 32 512
            #<<<일단 diffact에서는 b t h 32 2513 1 이 맞긴함>>>>

            #################decoder end########################
            x_start = F.softmax(x_start, 1)          # [32 2513 1]    #[[[0.0002],[0.0002],[0.0004],  ...,[0.0001],[0.0004], [0.0010]],...
            assert(x_start.max() <= 1 and x_start.min() >= 0) 
            '''
            tensor([[[0.0002],
            [0.0002],
            [0.0004],
            ...,
            '''
            x_start = normalize(x_start, self.scale)                #[32 2513 1]    [[[-0.9997],[-0.9997],[-0.9993],          # [-scale, +scale]
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)    #[32 2315 1]
            #pdb.set_trace()
            #RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 2
            
            pred_noise = self.predict_noise_from_start(x_time, time_cond, x_start)  #[32 2513 1]
            '''
            pred_noise
            tensor([[[-2.3966],
            [-0.6238],
            [-0.8775],
            ...,
            [ 0.9068],
            [ 1.4329],              '''
            #([1, 19, 1212]), x_start ([1, 19, 1212])
            #x_time ([32, 2513, 8]) 이므로 fra=x_start도 맞춰줘야함.(현재 8 32 512)   
            
            x_return = torch.clone(x_start)       #[32 2513 1]
            #([1, 19, 1212])
            
            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time] #float값 작음
            alpha_next = self.alphas_cumprod[time_next] #0.0039

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()  #(0.9981,
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)  #[1, 19, 1212]     [32 2513 1]

            x_time = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
                            #([1, 19, 1212])       [32 2513 1]

        #pdb.set_trace()
        '''
        [[[-0.9996],
         [-0.9996],
         [-0.9992],
         ...,
         [-0.9998],
         [-0.9992],
         [-0.9981]], ...
        '''
        x_return = denormalize(x_return, self.scale)   #([1, 19, 1212])     
        #x_return: [32 2513 1]
        #pdb.set_trace()
        '''
        [배치32
            클래스1[[1.7893e-04],
            클래스2[1.8933e-04],
            클래스3[3.8907e-04],
        '''
        
        if seed is not None:
            t = 1000 * Time.time() # current time in milliseconds
            t = int(t) % 2**16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

#([[[5.5280e-04, 4.2853e-04, 3.6073e-04,  ..., 1.0490e-02,
#          1.1286e-02, 1.2366e-02],
#         [2.6078e-03, 2.2621e-03, 2.3722e-03,  ..., 2.3127e-03,
#          3.0355e-03, 2.8653e-03],
#         [1.8767e-03, 1.4889e-03, 1.3571e-03,  ..., 6.4812e-01,
#          6.2417e-01, 6.4046e-01]]])
        return x_return        #([1, 19, 1212])
    
    
    '''
        output = output.permute(1, 0, 2).contiguous()   #[B, T, H]      32 1 512
        #src_emb = src_emb.permute(1, 0, 2).contiguous() #[B, S, H]      32 8 512
        
        #Output generation
        outputs = dict()
        '''
#################################################################################################################3
    def ddim_func(self, past_label, inputs):
        src = inputs           #feature = [B, S, C]
        B, S, C = src.size()
        
        #Features to hidden dim
        src = self.input_embed(src)              
        src = F.relu(src)

        #DETR decoder target setting
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)       #[B, T, H]
        tgt = torch.zeros_like(query_embed)         #[B, T, H]                      [32 1 512] 왜 차원이 이렇지? 한 frame에 대한 추측씩 뽑아내서 그런가

        if self.args.qk_attn :
            if self.args.pos_emb :
                pos_embed = self.pos_embedding[:, :S,].repeat(B, 1, 1)
                pos_embed = rearrange(pos_embed, 'b t c -> t b c')
            else :
                pos_embed = None
                src = self.pos_enc(src)
            #pdb.set_trace()
            src = rearrange(src, 'b t c -> t b c')                      #8 32 512
            tgt = rearrange(tgt, 'b t c -> t b c')                      #1 32 512
            query_embed = rearrange(query_embed, 'b t c -> t b c')      #1 32 512
            src_key_padding_mask = None
            tgt_mask = None
            tgt_key_padding_mask = None
            tgt_pos_emd = None
            output = self.transformer.ddim_sample(past_label, src, tgt, src_key_padding_mask, tgt_mask, None, query_embed, pos_embed, None)
            
            
            #pdb.set_trace()
            #8 32 512 / 1 32 512
            #pdb.set_trace()
        else :
            #Positional Embedding
            if self.args.pos_emb:
                src = src + self.pos_embedding[:,:src.size(1),]
            else:
                #sinusodal pos encoding
                src = self.pos_enc(src)

            query_pos = query_embed
            tgt = tgt + query_pos                       #[B, T, H]

            #Transformer Layer
            src = src.permute(1, 0, 2).contiguous()     #[S, B, H]
            src_emb = self.transformer.encoder(src, src_key_padding_mask=None)

            tgt = tgt.permute(1, 0, 2).contiguous()     #[T, B, H]
            output = self.transformer.decoder(tgt, src_emb).to(self.device)

        #Output reshape
        output = output.permute(1, 0, 2).contiguous()   #[B, T, H]      32 1 512
        #src_emb = src_emb.permute(1, 0, 2).contiguous() #[B, S, H]      32 8 512
        
        #Output generation
        outputs = dict()
        #if self.args.feat_loss:
        #    feat_emb = self.dropout_feat(src_emb)
        #    output_feat = self.feature_predict(feat_emb)
        #    outputs['feat'] = output_feat

        #    if self.args.feat_seg :
        #        output_seg = self.fc_feat_seg(output_feat) #[1024 -> N]
        #        outputs['seg'] = output_seg

        if self.args.seg :
            #Segmentation Result
            seg_emb = self.dropout_src(src_emb)
            output_seg = self.fc_seg(src_emb)
            outputs['seg'] = output_seg

        if self.args.next :
            output = self.dropout_tgt(output)
            output_next = self.fc_next(output)      ##############^^
            outputs['next'] = output_next
            
            #RuntimeError: mat1 and mat2 shapes cannot be multiplied (80416x8 and 512x2513)

        return output  #[B, T, H]      32 1 class