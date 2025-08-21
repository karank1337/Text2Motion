#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, os, re
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# --------- minimal TF-IDF over tokenizer vocab (excludes special tokens <4)

class TextDB:
    def __init__(self):
        self.N = 0
        self.df: Dict[int, int] = {}
        self.docs: List[Dict[int, int]] = []     # list of {token_id: count}
        self.paths: List[str] = []
        self.texts: List[str] = []
        self.idf: Dict[int, float] = {}
        self.doc_norm: List[float] = []          # L2 norms of tf-idf vectors

def _tokenize(tokenizer, text: str) -> List[int]:
    s = tokenizer.normalize(text)
    ids = []
    for w in s.split():
        i = tokenizer.word2idx.get(w, tokenizer.UNK)
        if i >= 4:  # skip specials
            ids.append(i)
    return ids

def build_text_db(index_json: str, tokenizer) -> TextDB:
    with open(index_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    db = TextDB()
    db.N = len(records)

    # pass 1: df counts
    for r in records:
        ids = set(_tokenize(tokenizer, r["text"]))
        for i in ids:
            db.df[i] = db.df.get(i, 0) + 1

    # idf
    for i, df in db.df.items():
        db.idf[i] = math.log((db.N + 1) / (df + 1)) + 1.0

    # pass 2: doc tf and norms
    for r in records:
        ids = _tokenize(tokenizer, r["text"])
        tf: Dict[int, int] = {}
        for i in ids:
            tf[i] = tf.get(i, 0) + 1
        db.docs.append(tf)
        db.paths.append(r["motion_path"])
        db.texts.append(r["text"])
    # norms
    for tf in db.docs:
        s = 0.0
        for i, c in tf.items():
            w = (1.0 + math.log(c)) * db.idf.get(i, 0.0)
            s += w * w
        db.doc_norm.append(math.sqrt(s) if s > 0 else 1.0)

    return db

# --------- keyword extraction & boosted scoring

_CANON = {
    "walk": ["walk", "stroll"],
    "run": ["run", "jog", "sprint"],
    "jump": ["jump", "hop"],
    "wave": ["wave", "hello"],
    "turn": ["turn", "rotate"],
    "kick": ["kick"],
    "sit":  ["sit"],
    "stand":["stand", "idle"],
    "left": ["left"],
    "right":["right"],
    "back": ["back", "backward"],
    "forward":["forward", "ahead"],
}

def extract_keywords(text: str) -> List[str]:
    t = text.lower()
    found = []
    for key, syns in _CANON.items():
        if any(re.search(r"\b"+re.escape(s)+r"\b", t) for s in syns):
            found.append(key)
    return found

def _to_int(x: Union[int, float, np.integer, np.floating]) -> int:
    try:
        return int(x)
    except Exception:
        return 1

def query_topk(db: TextDB, tokenizer, text: str, k: int = 3,
               keywords: Optional[List[str]] = None, boost: float = 0.25) -> List[Tuple[int, float]]:
    ids = _tokenize(tokenizer, text)
    if not ids:
        return []
    # query tf-idf
    q_tf: Dict[int, int] = {}
    for i in ids:
        q_tf[i] = q_tf.get(i, 0) + 1
    q_w: Dict[int, float] = {}
    for i, c in q_tf.items():
        q_w[i] = (1.0 + math.log(c)) * db.idf.get(i, 0.0)
    q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0

    kw_set = set(keywords or [])
    scores: List[Tuple[int, float]] = []
    for di, tf in enumerate(db.docs):
        # cosine
        dot = 0.0
        for i, qv in q_w.items():
            if i in tf:
                dv = (1.0 + math.log(tf[i])) * db.idf.get(i, 0.0)
                dot += qv * dv
        denom = (q_norm * db.doc_norm[di]) or 1.0
        s = dot / denom

        # keyword bonus if doc text contains any action keywords
        if kw_set:
            low = db.texts[di].lower()
            hits = 0
            for kw_key in kw_set:  # FIX: do not shadow 'k'
                for syn in _CANON.get(kw_key, []):
                    if syn in low:
                        hits += 1
                        break
            if hits:
                s += boost * min(hits, 3)

        scores.append((di, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    k_int = max(1, _to_int(k))  # FIX: ensure integer slice
    return scores[:k_int]

