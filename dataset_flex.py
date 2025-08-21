#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---- Tokenizer ----

class WordTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, min_freq: int = 1, max_vocab: int = 20000):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.word2idx: Dict[str, int] = {
            '<pad>': self.PAD, '<bos>': self.BOS, '<eos>': self.EOS, '<unk>': self.UNK
        }
        self.idx2word: List[str] = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.fitted = False

    def normalize(self, s: str) -> str:
        return ' '.join(s.lower().strip().split())

    def fit(self, texts: List[str]):
        freqs: Dict[str, int] = {}
        for t in texts:
            for w in self.normalize(t).split():
                freqs[w] = freqs.get(w, 0) + 1
        words = sorted([w for w, f in freqs.items() if f >= self.min_freq],
                       key=lambda w: (-freqs[w], w))
        space_left = max(0, self.max_vocab - len(self.idx2word))
        words = words[:space_left]
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.idx2word) - 1
        self.fitted = True

    def encode(self, s: str, max_len: int) -> Tuple[np.ndarray, int]:
        assert self.fitted
        words = self.normalize(s).split()
        ids = [self.BOS] + [self.word2idx.get(w, self.UNK) for w in words] + [self.EOS]
        length = min(len(ids), max_len)
        arr = np.full((max_len,), self.PAD, dtype=np.int64)
        arr[:length] = np.array(ids[:length], dtype=np.int64)
        return arr, length

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

# ---- Motion helpers ----

def _to_TJC(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        J = arr.shape[1] // 3
        arr = arr.reshape(arr.shape[0], J, 3)
    if not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"Unexpected motion shape: {arr.shape}")
    return arr.astype(np.float32)

# KIT-21 edges (pelvis=0; legs from 0; arms from 2)
_KIT_EDGES = [(0,1),(1,2),(2,3),(3,4),
              (2,5),(5,6),(6,7),
              (2,8),(8,9),(9,10),
              (0,12),(12,13),(13,14),(14,15),
              (0,17),(17,18),(18,19),(19,20),
              (0,11),(0,16)]

def _bone_lengths(frame: np.ndarray, edges=_KIT_EDGES) -> np.ndarray:
    L = []
    for a,b in edges:
        if a < frame.shape[0] and b < frame.shape[0]:
            L.append(float(np.linalg.norm(frame[b]-frame[a]) + 1e-8))
        else:
            L.append(0.0)
    return np.asarray(L, dtype=np.float32)

def _seq_scale_normalize(motion: np.ndarray, target_mean_len=1.0) -> np.ndarray:
    # center pelvis 0, scale whole sequence so first-frame mean bone len ~ target
    m = motion.copy()
    m -= m[:, :1, :]
    bl = _bone_lengths(m[0])
    mean_len = float(np.mean(bl)) if bl.size else 1.0
    s = target_mean_len / (mean_len + 1e-6)
    return m * s

# ---- Dataset ----

@dataclass
class MotionSample:
    text: str
    text_ids: np.ndarray
    text_len: int
    motion: np.ndarray
    motion_len: int

class BaseTextMotionDataset(Dataset):
    def __init__(self, index_json: str, tokenizer: WordTokenizer,
                 max_text_len: int = 64, max_motion_len: int = 196,
                 joints: Optional[int] = 21):
        super().__init__()
        assert os.path.isfile(index_json), f"Index file not found: {index_json}"
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_motion_len = max_motion_len
        self.joints = int(joints) if joints is not None else 21  # lock to KIT-21

        with open(index_json, 'r', encoding='utf-8') as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> MotionSample:
        rec = self.records[idx]
        text = rec['text']
        text_ids, text_len = self.tokenizer.encode(text, self.max_text_len)

        # load and normalize to KIT-21 normalized space
        arr = np.load(rec['motion_path'], allow_pickle=False)
        m = _to_TJC(arr)                                 # (T,J,3)
        if m.shape[1] != self.joints:
            if m.shape[1] > self.joints:
                m = m[:, :self.joints, :]
            else:
                pad = np.zeros((m.shape[0], self.joints-m.shape[1], 3), dtype=m.dtype)
                m = np.concatenate([m, pad], axis=1)

        m = _seq_scale_normalize(m, target_mean_len=1.0)

        T = min(m.shape[0], self.max_motion_len)
        motion_len = T
        buf = np.zeros((self.max_motion_len, self.joints, 3), dtype=np.float32)
        buf[:T] = m[:T]

        return MotionSample(text=text, text_ids=text_ids, text_len=text_len,
                            motion=buf, motion_len=motion_len)

class HumanML3DDataset(BaseTextMotionDataset):
    pass

class KITMLDataset(BaseTextMotionDataset):
    pass

def collate_batch(batch: List[MotionSample]):
    max_text = max(b.text_len for b in batch)
    max_motion = max(b.motion_len for b in batch)
    B = len(batch)
    J = batch[0].motion.shape[1]

    text_ids = np.full((B, max_text), WordTokenizer.PAD, dtype=np.int64)
    text_mask = np.zeros((B, max_text), dtype=np.bool_)
    motion = np.zeros((B, max_motion, J, 3), dtype=np.float32)
    motion_mask = np.zeros((B, max_motion), dtype=np.bool_)

    for i, b in enumerate(batch):
        text_ids[i, :b.text_len] = b.text_ids[:b.text_len]
        text_mask[i, :b.text_len] = 1
        motion[i, :b.motion_len] = b.motion[:b.motion_len]
        motion_mask[i, :b.motion_len] = 1

    return {
        'text_ids': torch.from_numpy(text_ids),
        'text_mask': torch.from_numpy(text_mask),
        'motion': torch.from_numpy(motion),
        'motion_mask': torch.from_numpy(motion_mask),
    }

