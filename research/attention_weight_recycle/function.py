import math
import torch
import torch.nn.functional as F 
from src.nn.rtdetr.hybrid_encoder import HybridEncoder, TransformerEncoderLayer
import torch.nn as nn 

def attentionWeight_twice_matmul_type1(weight, feature, scale_factor):
    """
    Warning: Using too much cuda memory
    """
    N, Q, K = weight.shape
    QH = QW = int(math.sqrt(Q))
    KH = KW = int(math.sqrt(K))

    # 4 dimension interpolate
    attentionWeight_twice = weight.reshape(N, QH, 1, QW, 1, KH, 1, KW, 1).repeat(1, 1, scale_factor,1,scale_factor,1,scale_factor,1,scale_factor)
    attentionWeight_twice = attentionWeight_twice.reshape(N, Q * scale_factor * scale_factor, K * scale_factor * scale_factor)

    N, C, H, W = feature.shape
    result = torch.einsum('nij, ncj->nci', attentionWeight_twice, feature.flatten(2))
    result = result.reshape(N, C, H, W)
    return result


try:
    from research.attention_weight_recycle import attention_weight_twice_matmul
    def attentionWeight_twice_matmul_type1_c(weight, feature, scale_factor):
        return attention_weight_twice_matmul.attentionWeight_twice_matmul_type1(weight, feature, scale_factor)
except:
    print("Please run the command: `python setup.py build_ext --inplace`")


def attentionWeight_twice_matmul_type2(weight, feature, original_shape):
    H, W = original_shape
    N, Q, K = weight.shape
    N, C, FH, FW = feature.shape
    SH, SW = FH // H, FW // W

    feature = feature.reshape(N, C, H, SH, W, SW).permute(0, 1, 3, 5, 2, 4).reshape(N, C, SH * SW, H*W).permute(0, 1, 3, 2)
    result = torch.matmul(weight.reshape(N,1,Q,K), feature)
    result = result.sum(axis=3)
    result = result.reshape(N, C, H, 1, W, 1).repeat(1, 1, 1, SH, 1, SW).reshape(N, C, H * SH, W * SW)
    return result


def attentionWeight_twice_matmul_type3(weight, feature, original_shape):
    """
    Current best implementation
    """
    H, W = original_shape
    N, Q, K = weight.shape
    N, C, FH, FW = feature.shape
    SH, SW = FH // H, FW // W

    feature = feature.reshape(N, C, H, SH, W, SW).permute(0, 1, 3, 5, 2, 4).reshape(N, C, SH * SW, H*W).permute(0, 1, 3, 2)
    result = torch.einsum('nij, ncjq->nciq', weight, feature)
    result = result.sum(axis=3)
    result = result.reshape(N, C, H, 1, W, 1).repeat(1, 1, 1, SH, 1, SW).reshape(N, C, H * SH, W * SW)
    return result


def TransformerEncoderLayer_forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    q = k = self.with_pos_embed(src, pos_embed)


    src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    return src, _


class HybridEncoder1(HybridEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b1 = nn.BatchNorm2d(self.hidden_dim)
        self.b2 = nn.BatchNorm2d(self.hidden_dim)
        
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory, _ = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # attention weight recycle
        original_shape = (proj_feats[2].shape[2], proj_feats[2].shape[3])
        proj_feats[1] = self.b1(proj_feats[1] + attentionWeight_twice_matmul_type3(_, proj_feats[1], original_shape))
        proj_feats[0] = self.b2(proj_feats[0] + attentionWeight_twice_matmul_type3(_, proj_feats[0], original_shape))


        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
    

class attentionWeightRecycle(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward, dropout):
        super(attentionWeightRecycle, self).__init__()
        self.b1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.ac = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.b2 = nn.LayerNorm(hidden_dim)

    def forward(self, weight, feature, original_shape):

        residual = feature.reshape(feature.shape[0], feature.shape[1], feature.shape[2]*feature.shape[3]).transpose(1, 2)
        x = attentionWeight_twice_matmul_type3(weight, feature, original_shape)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).transpose(1, 2)
        x = self.b1(residual + self.dropout1(x))


        residual = x
        x = self.linear2(self.dropout2(self.ac(self.linear1(x))))
        x = self.b2(residual + self.dropout3(x))

        return x

class HybridEncoder2(HybridEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim_feedforward, dropout = kwargs['dim_feedforward'], kwargs['dropout']
        self.s4attenRecycle = attentionWeightRecycle(self.hidden_dim, dim_feedforward, dropout)
        self.s5attenRecycle = attentionWeightRecycle(self.hidden_dim, dim_feedforward, dropout)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory, _ = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # attention weight recycle
        original_shape = (proj_feats[2].shape[2], proj_feats[2].shape[3])
        proj_feats[1] = self.s4attenRecycle(_, proj_feats[1], original_shape)
        proj_feats[1] = proj_feats[1].permute(0,2,1).reshape(-1, self.hidden_dim, original_shape[0] * 2 , original_shape[1] * 2)
        proj_feats[0] = self.s5attenRecycle(_, proj_feats[0], original_shape)
        proj_feats[0] = proj_feats[0].permute(0,2,1).reshape(-1, self.hidden_dim, original_shape[0] * 4 , original_shape[1] * 4)

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
    
import convNd


def ConvTranspose4d(in_channels: int, out_channels: int, kernel_size:int=2,
                    stride:int=1, padding:int = 0, padding_mode: str ="zeros", 
                    bias: bool = True, groups: int = 1, dilation: int = 1):
    w = torch.rand(1)[0]
    if bias:
        b = torch.zeros(1)[0]
    return convNd.convNd(in_channels=in_channels, out_channels=out_channels,
                           num_dims=4,kernel_size=kernel_size, 
                           stride=(stride,stride,stride,stride), padding=padding, 
                           padding_mode=padding_mode, output_padding=0,
                           is_transposed=True, use_bias=bias, groups=groups,  dilation = dilation, 
                           kernel_initializer=lambda x: torch.nn.init.constant_(x, w),  
                           bias_initializer=lambda x: torch.nn.init.constant_(x, b))