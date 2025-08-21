#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, hashlib, random
import numpy as np
import torch
from PyQt5 import QtWidgets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed
from dataset_flex import WordTokenizer, HumanML3DDataset
from trainer import Trainer, TrainConfig
from viewer import SkeletonViewer
from model import Text2MotionTransformer
from retrieval import build_text_db, query_topk, extract_keywords
from controllers import apply_keyword_controls

DEFAULT_JOINTS = 21

_EDGES = [(0,1),(1,2),(2,3),(3,4),
          (2,5),(5,6),(6,7),
          (2,8),(8,9),(9,10),
          (0,12),(12,13),(13,14),(14,15),
          (0,17),(17,18),(18,19),(19,20),
          (0,11),(0,16)]

def _bone_lengths(f0):
    L=[]
    for a,b in _EDGES:
        if a<f0.shape[0] and b<f0.shape[0]:
            L.append(float(np.linalg.norm(f0[b]-f0[a])+1e-8))
        else:
            L.append(0.0)
    return np.asarray(L, dtype=np.float32)

def _normalize_motion_TJC(m):
    m = m.astype(np.float32).copy()
    m -= m[:, :1, :]
    bl = _bone_lengths(m[0]); mean_len = float(np.mean(bl)) if bl.size else 1.0
    s = 1.0 / (mean_len + 1e-6)
    return m * s

def _load_motion_npy(path, J=DEFAULT_JOINTS, Tcap=120):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        arr = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
    if not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"Unexpected motion shape at {path}: {arr.shape}")
    T, Jgot, _ = arr.shape
    if Jgot > J: arr = arr[:, :J, :]
    elif Jgot < J:
        pad = np.zeros((T, J - Jgot, 3), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    if arr.shape[0] > Tcap: arr = arr[:Tcap]
    return _normalize_motion_TJC(arr)

def _seed_from_text(text: str) -> int:
    h = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
    return int(h[:8], 16)

class Text2MotionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text2Motion App"); self.resize(1240, 840)
        self.central = QtWidgets.QTabWidget(); self.setCentralWidget(self.central)

        self.tokenizer=None; self.trainer=None; self.model=None
        self.viewer = SkeletonViewer(joints=DEFAULT_JOINTS)

        self.text_db = None
        self.train_index_path = None

        self._init_generate_tab()
        self._init_train_tab()
        self._init_dataset_tab()

    def log(self,s): print(s)

    # ---------- Tabs ----------

    def _init_generate_tab(self):
        tab=QtWidgets.QWidget(); self.central.addTab(tab,"Generate")
        lay=QtWidgets.QVBoxLayout(tab)

        row = QtWidgets.QHBoxLayout()
        self.text_prompt=QtWidgets.QLineEdit(); self.text_prompt.setPlaceholderText("Type a motion description…")
        self.btn_generate=QtWidgets.QPushButton("Generate Motion")
        self.btn_generate.clicked.connect(self.on_generate)
        row.addWidget(self.text_prompt); row.addWidget(self.btn_generate)

        ctrl = QtWidgets.QHBoxLayout()
        # CFG
        self.cfg_chk = QtWidgets.QCheckBox("Use CFG"); self.cfg_chk.setChecked(True)
        self.guid_spin = QtWidgets.QDoubleSpinBox(); self.guid_spin.setRange(0.0, 7.0); self.guid_spin.setSingleStep(0.1); self.guid_spin.setValue(3.0)
        self.noise_spin = QtWidgets.QDoubleSpinBox(); self.noise_spin.setRange(0.0, 0.2); self.noise_spin.setSingleStep(0.005); self.noise_spin.setValue(0.01)
        # Retrieval
        self.ret_chk = QtWidgets.QCheckBox("Use retrieval mix"); self.ret_chk.setChecked(True)
        self.ret_k_spin = QtWidgets.QSpinBox(); self.ret_k_spin.setRange(1, 5); self.ret_k_spin.setValue(2)
        self.mix_spin = QtWidgets.QDoubleSpinBox(); self.mix_spin.setRange(0.0, 1.0); self.mix_spin.setSingleStep(0.05); self.mix_spin.setValue(0.6)
        # Controllers
        self.ctrl_chk = QtWidgets.QCheckBox("Keyword controls"); self.ctrl_chk.setChecked(True)
        self.ctrl_strength = QtWidgets.QDoubleSpinBox(); self.ctrl_strength.setRange(0.0, 1.0); self.ctrl_strength.setSingleStep(0.05); self.ctrl_strength.setValue(0.5)
        self.axis_combo = QtWidgets.QComboBox(); self.axis_combo.addItems(["x","z"])
        self.ground_spin = QtWidgets.QDoubleSpinBox(); self.ground_spin.setRange(-1.0, 1.0); self.ground_spin.setSingleStep(0.01); self.ground_spin.setValue(0.0)

        for w in (self.cfg_chk, QtWidgets.QLabel("Guidance"), self.guid_spin,
                  QtWidgets.QLabel("Noise"), self.noise_spin,
                  self.ret_chk, QtWidgets.QLabel("TopK"), self.ret_k_spin,
                  QtWidgets.QLabel("Mix"), self.mix_spin,
                  self.ctrl_chk, QtWidgets.QLabel("Ctrl Strength"), self.ctrl_strength,
                  QtWidgets.QLabel("Forward Axis"), self.axis_combo,
                  QtWidgets.QLabel("Ground Y"), self.ground_spin):
            ctrl.addWidget(w)
        ctrl.addStretch(1)

        lay.addLayout(row)
        lay.addLayout(ctrl)
        lay.addWidget(self.viewer)

    def _init_train_tab(self):
        tab=QtWidgets.QWidget(); self.central.addTab(tab,"Train")
        lay=QtWidgets.QVBoxLayout(tab)
        self.btn_train=QtWidgets.QPushButton("Start Training"); self.btn_train.clicked.connect(self.on_train)
        self.btn_load_ckpt=QtWidgets.QPushButton("Load Checkpoint"); self.btn_load_ckpt.clicked.connect(self.on_load_ckpt)
        lay.addWidget(self.btn_train); lay.addWidget(self.btn_load_ckpt)

    def _init_dataset_tab(self):
        tab=QtWidgets.QWidget(); self.central.addTab(tab,"Dataset")
        lay=QtWidgets.QVBoxLayout(tab)
        self.path_train_idx=QtWidgets.QLineEdit(); self.path_train_idx.setPlaceholderText("Path to train_index.json")
        self.path_val_idx=QtWidgets.QLineEdit(); self.path_val_idx.setPlaceholderText("Path to val_index.json (optional)")
        self.btn_browse_train=QtWidgets.QPushButton("Browse Train Index"); self.btn_browse_val=QtWidgets.QPushButton("Browse Val Index")
        self.btn_build_vocab=QtWidgets.QPushButton("Build Vocab")
        self.btn_browse_train.clicked.connect(self._browse_train); self.btn_browse_val.clicked.connect(self._browse_val)
        self.btn_build_vocab.clicked.connect(self.on_build_vocab)
        lay.addWidget(self.path_train_idx); lay.addWidget(self.btn_browse_train)
        lay.addWidget(self.path_val_idx); lay.addWidget(self.btn_browse_val); lay.addWidget(self.btn_build_vocab)

    # ---------- Dataset tab handlers ----------

    def _browse_train(self):
        p,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select train index.json",filter="JSON (*.json)")
        if p: self.path_train_idx.setText(p)

    def _browse_val(self):
        p,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select val index.json",filter="JSON (*.json)")
        if p: self.path_val_idx.setText(p)

    def on_build_vocab(self):
        p=self.path_train_idx.text().strip()
        if not os.path.isfile(p): self.log("Invalid train index path."); return
        with open(p,"r",encoding="utf-8") as f: recs=json.load(f)
        texts=[r['text'] for r in recs]
        self.tokenizer=WordTokenizer(min_freq=1)
        self.tokenizer.fit(texts)
        self.train_index_path = p
        self.text_db = build_text_db(self.train_index_path, self.tokenizer)
        self.log(f"Built vocab size {self.tokenizer.vocab_size} | TextDB N={self.text_db.N}")

    # ---------- Train tab handlers ----------

    def on_train(self):
        if self.tokenizer is None: self.log("Build vocab first."); return
        tr_idx=self.path_train_idx.text().strip(); vl_idx=self.path_val_idx.text().strip()
        if not os.path.isfile(tr_idx): self.log("Missing train index."); return

        train_ds=HumanML3DDataset(tr_idx,self.tokenizer,joints=DEFAULT_JOINTS)
        val_ds=HumanML3DDataset(vl_idx,self.tokenizer,joints=DEFAULT_JOINTS) if os.path.isfile(vl_idx) else None

        cfg=TrainConfig(); cfg.joints=DEFAULT_JOINTS; cfg.lr=1e-4; cfg.grad_clip=0.5
        self.model=Text2MotionTransformer(vocab_size=self.tokenizer.vocab_size,
                                          d_model=cfg.d_model,nhead=cfg.nhead,
                                          num_encoder_layers=cfg.enc_layers,num_decoder_layers=cfg.dec_layers,
                                          dim_feedforward=cfg.ffn,dropout=cfg.dropout,joints=cfg.joints)

        self.trainer=Trainer(model=self.model,tokenizer=self.tokenizer,
                             train_ds=train_ds,val_ds=val_ds,cfg=cfg,
                             ckpt_path="./checkpoints/text2motion_best.pt",
                             log_callback=self.log)
        self.trainer.train()

    def on_load_ckpt(self):
        if self.tokenizer is None: self.log("Build vocab first (Dataset tab → Build Vocab)."); return
        path="./checkpoints/text2motion_best.pt"
        if not os.path.isfile(path): self.log(f"Checkpoint not found at {path}."); return
        try: ckpt=torch.load(path,map_location="cpu",weights_only=True)
        except TypeError: ckpt=torch.load(path,map_location="cpu")

        tok=ckpt.get("tokenizer") if isinstance(ckpt,dict) else None
        if tok:
            self.tokenizer=WordTokenizer(min_freq=tok["min_freq"],max_vocab=tok["max_vocab"])
            self.tokenizer.idx2word=tok["idx2word"]
            self.tokenizer.word2idx={w:i for i,w in enumerate(self.tokenizer.idx2word)}
            self.tokenizer.fitted=True

        cfg=TrainConfig(); cfg.joints=DEFAULT_JOINTS
        self.model=Text2MotionTransformer(vocab_size=self.tokenizer.vocab_size,
                                          d_model=cfg.d_model,nhead=cfg.nhead,
                                          num_encoder_layers=cfg.enc_layers,num_decoder_layers=cfg.dec_layers,
                                          dim_feedforward=cfg.ffn,dropout=cfg.dropout,joints=cfg.joints)
        state=ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
        missing,unexpected=self.model.load_state_dict(state,strict=False)
        if missing or unexpected: self.log(f"Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        if self.trainer is not None:
            self.trainer.model=self.model.to(self.trainer.device); self.trainer.model.eval()
        else:
            device="cuda" if torch.cuda.is_available() else "cpu"
            self.model=self.model.to(device); self.model.eval()
        self.log("Checkpoint loaded.")

    # ---------- Generate ----------

    def _set_prompt_seed(self, text: str):
        s = _seed_from_text(text)
        random.seed(s)
        np.random.seed(s & 0xffffffff)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _generate_model(self, text: str):
        net=self.trainer.model if (self.trainer and getattr(self.trainer,"model",None)) else self.model
        if net is None: return None
        # make generation depend on prompt deterministically
        self._set_prompt_seed(text)
        ids,L=self.tokenizer.encode(text,64)
        device=next(net.parameters()).device
        text_ids=torch.from_numpy(ids).unsqueeze(0).to(device)
        text_mask=torch.zeros_like(text_ids).bool(); text_mask[0,:L]=True
        with torch.no_grad():
            if self.cfg_chk.isChecked():
                # small noise helps diversify; prompt-seeded above
                motion=net.generate_cfg(text_ids,text_mask,guidance_scale=float(self.guid_spin.value()),
                                        max_frames=120,noise_std=float(self.noise_spin.value()))
            else:
                motion=net.generate(text_ids,text_mask,max_frames=120)
        return motion.squeeze(0).cpu().numpy()

    def _generate_retrieval(self, text: str):
        if self.text_db is None:
            p=self.path_train_idx.text().strip()
            if os.path.isfile(p) and self.tokenizer is not None:
                self.text_db = build_text_db(p, self.tokenizer)
        if self.text_db is None:
            return None
        kw = extract_keywords(text)
        top = query_topk(self.text_db, self.tokenizer, text, k=int(self.ret_k_spin.value()),
                         keywords=kw, boost=0.3)
        if not top: return None
        clips=[]; Tcap=120
        for idx,_ in top:
            mp=self.text_db.paths[idx]
            if not os.path.isfile(mp): continue
            try:
                m=_load_motion_npy(mp, J=DEFAULT_JOINTS, Tcap=Tcap)
                clips.append(m)
            except Exception:
                continue
        if not clips: return None
        T=min(min(c.shape[0] for c in clips), Tcap)
        clips=[c[:T] for c in clips]
        return np.mean(np.stack(clips,axis=0),axis=0)

    def on_generate(self):
        if self.tokenizer is None: self.log("Build vocab first."); return
        text=self.text_prompt.text().strip()
        if not text: return

        kw = extract_keywords(text)

        m_model=self._generate_model(text)
        m_ret=self._generate_retrieval(text) if self.ret_chk.isChecked() else None

        if m_model is None and m_ret is None:
            self.log("No model and no retrieval available."); return

        # per-action mix defaults
        alpha = float(self.mix_spin.value())
        if "wave" in kw:
            alpha = 0.0
        elif "turn" in kw:
            alpha = min(alpha, 0.35)
        elif "jump" in kw:
            alpha = min(alpha, 0.45)

        if m_model is None:
            motion_np = m_ret
        elif m_ret is None:
            motion_np = m_model
        else:
            T=min(m_model.shape[0], m_ret.shape[0])
            motion_np = (1.0-alpha)*m_model[:T] + alpha*m_ret[:T]

        # ensure strong controller effect when keywords present
        eff_strength = float(self.ctrl_strength.value())
        if kw:
            need = 0.9 if "wave" in kw else 0.75 if ("walk" in kw or "run" in kw) else 0.7
            if eff_strength < need:
                eff_strength = need

        if self.ctrl_chk.isChecked():
            motion_np, tags = apply_keyword_controls(
                motion_np, text, fps=30,
                strength=eff_strength,
                forward_axis=self.axis_combo.currentText(),
                ground=float(self.ground_spin.value())
            )
            if tags:
                self.log(f"[controllers] applied: {', '.join(tags)}")
            else:
                self.log("[controllers] no tags matched")

        self.log(f"[debug] generated motion shape: {motion_np.shape}, min={motion_np.min():.3f}, max={motion_np.max():.3f}")

        T,J,_=motion_np.shape
        if J!=DEFAULT_JOINTS:
            if J>DEFAULT_JOINTS: motion_np=motion_np[:,:DEFAULT_JOINTS,:]
            else:
                pad=np.zeros((T,DEFAULT_JOINTS-J,3),dtype=motion_np.dtype)
                motion_np=np.concatenate([motion_np,pad],axis=1)
        if motion_np.shape[0]<2: motion_np=np.repeat(motion_np,60,axis=0)
        self.viewer.set_motion(motion_np,fps=30)

def main():
    set_seed(1337)
    app=QtWidgets.QApplication([])
    win=Text2MotionApp(); win.show()
    app.exec_()

if __name__=="__main__":
    main()

