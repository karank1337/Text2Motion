#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import numpy as np

SUPPORTED_EXTS = ('.npy', '.npz')


def gather_motion_files(root):
    files = []
    root = os.path.abspath(root)
    for dp, _, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            if p.lower().endswith(SUPPORTED_EXTS):
                files.append(p)
    return sorted(files)


def npz_first_array(z):
    for k in ['arr_0', 'poses', 'data', 'motion', 'X', 'array', 'pose']:
        if k in z.files:
            return z[k]
    return z[z.files[0]]


def valid_motion_shape(path, allow_pickle=False):
    try:
        if path.endswith('.npy'):
            arr = np.load(path, allow_pickle=allow_pickle)
        else:
            z = np.load(path, allow_pickle=allow_pickle)
            arr = npz_first_array(z)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 3 and arr.shape[2] == 3:
                return int(arr.shape[0])
            if arr.ndim == 2 and arr.shape[1] % 3 == 0:
                return int(arr.shape[0])
    except Exception:
        return None
    return None


def read_caption(path_no_ext, texts_dir=None):
    txt_path = path_no_ext + '.txt'
    if os.path.isfile(txt_path):
        try:
            return open(txt_path, 'r', encoding='utf-8').read().strip()
        except Exception:
            pass
    if texts_dir:
        base = os.path.basename(path_no_ext) + '.txt'
        cand = os.path.join(texts_dir, base)
        if os.path.isfile(cand):
            try:
                return open(cand, 'r', encoding='utf-8').read().strip()
            except Exception:
                pass
    # fallback
    return os.path.basename(path_no_ext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Folder with motion .npy/.npz (e.g., new_joints)')
    ap.add_argument('--texts-dir', default=None, help='Folder with captions .txt (filenames match, e.g., 00001.txt)')
    ap.add_argument('--split', type=float, default=0.9)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--allow-pickle', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = gather_motion_files(args.root)
    records = []
    for p in files:
        # ignore stats files like Mean.npy / Std.npy at a different folder level
        name = os.path.basename(p).lower()
        if name in ('mean.npy', 'std.npy'):
            continue
        T = valid_motion_shape(p, allow_pickle=args.allow_pickle)
        if T is None:
            continue
        base, _ = os.path.splitext(p)
        cap = read_caption(base, args.texts_dir)
        records.append({'text': cap, 'motion_path': os.path.abspath(p), 'nframes': int(T)})

    if not records:
        print('No valid motion arrays found.')
        return

    random.shuffle(records)
    k = max(1, int(len(records) * args.split))
    train_recs, val_recs = records[:k], records[k:]

    tr = os.path.join(args.outdir, 'train_index.json')
    vl = os.path.join(args.outdir, 'val_index.json')
    with open(tr, 'w', encoding='utf-8') as f:
        json.dump(train_recs, f, ensure_ascii=False, indent=2)
    with open(vl, 'w', encoding='utf-8') as f:
        json.dump(val_recs, f, ensure_ascii=False, indent=2)

    print(f'train_index: {tr} ({len(train_recs)})')
    print(f'val_index:   {vl} ({len(val_recs)})')


if __name__ == '__main__':
    main()
