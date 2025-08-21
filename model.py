#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class Text2MotionTransformer(nn.Module):
    """
    Text-conditioned seq2seq Transformer for KIT-21 normalized XYZ.
    No pretrained encoders. Supports CFG at inference.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 joints: int = 21, output_scale: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.joints = int(joints)
        self.motion_dim = self.joints * 3
        self.output_scale = output_scale

        # text
        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = PositionalEncoding(d_model, dropout)

        # motion
        self.motion_in = nn.Linear(self.motion_dim, d_model)
        self.motion_pos = PositionalEncoding(d_model, dropout)

        self.tr = nn.Transformer(d_model=d_model, nhead=nhead,
                                 num_encoder_layers=num_encoder_layers,
                                 num_decoder_layers=num_decoder_layers,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout, batch_first=True)

        self.out = nn.Linear(d_model, self.motion_dim)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor, motion_in: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        text_ids: (B, Lt)
        text_mask: (B, Lt) True for valid tokens
        motion_in: (B, Lm, J, 3)
        returns: (B, Lm, J, 3)
        """
        B, Lm, J, C = motion_in.shape
        assert J == self.joints and C == 3

        txt = self.text_pos(self.text_embed(text_ids) * math.sqrt(self.d_model))
        mem_kpm = memory_key_padding_mask if memory_key_padding_mask is not None else (~text_mask.bool())

        mot = self.motion_pos(self.motion_in(motion_in.reshape(B, Lm, self.motion_dim)))

        h = self.tr(src=txt, tgt=mot,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=mem_kpm,
                    tgt_key_padding_mask=tgt_key_padding_mask)
        y = self.out(h).reshape(B, Lm, self.joints, 3)
        # keep outputs bounded; dataset is normalized to ~unit scale
        return torch.tanh(y) * self.output_scale

    @torch.no_grad()
    def generate(self, text_ids: torch.Tensor, text_mask: torch.Tensor, max_frames: int = 120) -> torch.Tensor:
        """Standard greedy autoregressive decoding."""
        self.eval()
        device = text_ids.device
        B = text_ids.size(0)
        J = self.joints

        txt = self.text_pos(self.text_embed(text_ids) * math.sqrt(self.d_model))
        mem_kpm = ~text_mask.bool()

        cur = torch.zeros(B, 1, J, 3, device=device)
        outs = []
        for _ in range(max_frames):
            Lm = cur.size(1)
            tgt_mask = torch.triu(torch.ones(Lm, Lm, device=device), diagonal=1).bool()
            tgt_kpm = torch.zeros(B, Lm, dtype=torch.bool, device=device)
            mot = self.motion_pos(self.motion_in(cur.reshape(B, Lm, J * 3)))
            h = self.tr(src=txt, tgt=mot,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=mem_kpm,
                        tgt_key_padding_mask=tgt_kpm)
            y = self.out(h)[:, -1].view(B, J, 3)
            y = torch.tanh(y) * self.output_scale
            cur = torch.cat([cur, y.unsqueeze(1)], dim=1)
            outs.append(y.unsqueeze(1))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def generate_cfg(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                     guidance_scale: float = 3.0, max_frames: int = 120,
                     noise_std: float = 0.0) -> torch.Tensor:
        """
        Classifier-Free Guidance for regression:
          y = y_uncond + s * (y_cond - y_uncond)
        where 'uncond' uses an empty prompt (BOS/EOS only).
        """
        self.eval()
        device = text_ids.device
        B = text_ids.size(0)
        J = self.joints

        # cond text
        txt_c = self.text_pos(self.text_embed(text_ids) * math.sqrt(self.d_model))
        mem_c = ~text_mask.bool()

        # uncond text (BOS + EOS, rest PAD)
        Lt = text_ids.size(1)
        uncond_ids = torch.zeros_like(text_ids)
        uncond_mask = torch.zeros_like(text_mask)
        if Lt >= 1:
            uncond_ids[:, 0] = 1  # BOS
            uncond_mask[:, 0] = True
        if Lt >= 2:
            uncond_ids[:, 1] = 2  # EOS
            uncond_mask[:, 1] = True
        txt_u = self.text_pos(self.text_embed(uncond_ids) * math.sqrt(self.d_model))
        mem_u = ~uncond_mask.bool()

        cur_c = torch.zeros(B, 1, J, 3, device=device)
        cur_u = torch.zeros(B, 1, J, 3, device=device)

        outs = []
        for _ in range(max_frames):
            # conditional step
            Lm = cur_c.size(1)
            tgt_mask = torch.triu(torch.ones(Lm, Lm, device=device), diagonal=1).bool()
            tgt_kpm = torch.zeros(B, Lm, dtype=torch.bool, device=device)
            mot_c = self.motion_pos(self.motion_in(cur_c.reshape(B, Lm, J * 3)))
            h_c = self.tr(src=txt_c, tgt=mot_c,
                          tgt_mask=tgt_mask,
                          memory_key_padding_mask=mem_c,
                          tgt_key_padding_mask=tgt_kpm)
            y_c = self.out(h_c)[:, -1].view(B, J, 3)
            y_c = torch.tanh(y_c) * self.output_scale

            # uncond step
            mot_u = self.motion_pos(self.motion_in(cur_u.reshape(B, Lm, J * 3)))
            h_u = self.tr(src=txt_u, tgt=mot_u,
                          tgt_mask=tgt_mask,
                          memory_key_padding_mask=mem_u,
                          tgt_key_padding_mask=tgt_kpm)
            y_u = self.out(h_u)[:, -1].view(B, J, 3)
            y_u = torch.tanh(y_u) * self.output_scale

            # guidance
            y_g = y_u + guidance_scale * (y_c - y_u)
            if noise_std > 0:
                y_g = y_g + torch.randn_like(y_g) * noise_std

            outs.append(y_g.unsqueeze(1))

            # feed guided for next step
            cur_c = torch.cat([cur_c, y_g.unsqueeze(1)], dim=1)
            cur_u = torch.cat([cur_u, y_g.unsqueeze(1)], dim=1)

        return torch.cat(outs, dim=1)

