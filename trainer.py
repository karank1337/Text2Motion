#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# type hint only; not importing MotionLoss from model
from model import Text2MotionTransformer  # noqa: F401
from dataset_flex import collate_batch, BaseTextMotionDataset


class MotionLoss(nn.Module):
    def __init__(self, pos_w: float = 1.0, vel_w: float = 1.0):
        super().__init__()
        self.pos_w = pos_w
        self.vel_w = vel_w
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        pred/target: (B,T,J,3)
        mask:       (B,T) True for valid
        """
        B, T, J, C = target.shape
        m = mask.unsqueeze(-1).unsqueeze(-1).float()  # (B,T,1,1)

        pos = self.l1(pred, target) * m               # (B,T,J,3)

        # velocities
        pred_v = pred[:, 1:] - pred[:, :-1]
        targ_v = target[:, 1:] - target[:, :-1]
        m_v = mask[:, 1:].unsqueeze(-1).unsqueeze(-1).float()
        vel = self.l1(pred_v, targ_v) * m_v

        # reduce
        pos_loss = pos.mean()
        vel_loss = vel.mean()
        loss = self.pos_w * pos_loss + self.vel_w * vel_loss
        return loss, pos_loss.detach(), vel_loss.detach()


class TrainConfig:
    def __init__(self):
        self.d_model = 256
        self.nhead = 8
        self.enc_layers = 4
        self.dec_layers = 4
        self.ffn = 1024
        self.dropout = 0.1
        self.batch_size = 16
        self.epochs = 20
        self.lr = 1e-4
        self.grad_clip = 0.5
        self.max_text_len = 64
        self.max_motion_len = 196
        self.joints = 21
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    def __init__(self, model: Text2MotionTransformer, tokenizer,
                 train_ds: BaseTextMotionDataset, val_ds: BaseTextMotionDataset,
                 cfg: TrainConfig, ckpt_path: str, log_callback=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.log = log_callback or (lambda s: None)

        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.criterion = MotionLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=0, collate_fn=collate_batch)
        self.val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=0, collate_fn=collate_batch) if val_ds else None

    def save_ckpt(self, epoch: int, stats: dict):
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'tokenizer': {
                'min_freq': self.tokenizer.min_freq,
                'max_vocab': self.tokenizer.max_vocab,
                'idx2word': self.tokenizer.idx2word
            },
            'cfg': self.cfg.__dict__,
            'stats': stats
        }, self.ckpt_path)

    def train(self):
        best_val = float('inf')
        stats = {'train_loss': [], 'val_loss': []}
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running = []
            t0 = time.time()
            for batch in self.train_loader:
                text_ids = batch['text_ids'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)
                motion = batch['motion'].to(self.device)
                motion_mask = batch['motion_mask'].to(self.device)

                if torch.isnan(motion).any():
                    continue

                B, Tm, J, _ = motion.shape
                zero = torch.zeros((B, 1, J, 3), device=self.device)
                motion_in = torch.cat([zero, motion[:, :-1]], dim=1)

                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(Tm).to(self.device)
                pred = self.model(text_ids, text_mask, motion_in,
                                   tgt_mask=tgt_mask,
                                   memory_key_padding_mask=~text_mask.bool(),
                                   tgt_key_padding_mask=~motion_mask.bool())

                loss, _, _ = self.criterion(pred, motion, motion_mask)
                if torch.isnan(loss):
                    continue
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                running.append(loss.item())

            tr_loss = float(np.mean(running)) if running else 0.0

            val_loss = None
            if self.val_loader:
                self.model.eval()
                v_losses = []
                with torch.no_grad():
                    for batch in self.val_loader:
                        text_ids = batch['text_ids'].to(self.device)
                        text_mask = batch['text_mask'].to(self.device)
                        motion = batch['motion'].to(self.device)
                        motion_mask = batch['motion_mask'].to(self.device)

                        B, Tm, J, _ = motion.shape
                        zero = torch.zeros((B, 1, J, 3), device=self.device)
                        motion_in = torch.cat([zero, motion[:, :-1]], dim=1)
                        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(Tm).to(self.device)
                        pred = self.model(text_ids, text_mask, motion_in,
                                           tgt_mask=tgt_mask,
                                           memory_key_padding_mask=~text_mask.bool(),
                                           tgt_key_padding_mask=~motion_mask.bool())
                        loss, _, _ = self.criterion(pred, motion, motion_mask)
                        v_losses.append(loss.item())
                val_loss = float(np.mean(v_losses)) if v_losses else None

            stats['train_loss'].append(tr_loss)
            stats['val_loss'].append(val_loss if val_loss is not None else float('nan'))
            t1 = time.time()

            msg = f"Epoch {epoch}/{self.cfg.epochs} | train {tr_loss:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            msg += f" | {t1 - t0:.1f}s"
            self.log(msg)

            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.save_ckpt(epoch, stats)
                self.log("Saved best checkpoint.")
            elif val_loss is None:
                self.save_ckpt(epoch, stats)

        return stats

