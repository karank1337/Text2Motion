#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

CHAINS = [
    [0,1,2,3,4],
    [2,5,6,7],
    [2,8,9,10],
    [0,12,13,14,15],
    [0,17,18,19,20],
    [0,11],
    [0,16],
]
def _edges(chains):
    e=[]
    for ch in chains:
        for a,b in zip(ch, ch[1:]): e.append((a,b))
    return e
EDGES = _edges(CHAINS)

def smooth(m, win=5):
    if m is None or len(m)==0: return m
    k=max(1,win//2); pad=np.pad(m,((k,k),(0,0),(0,0)),mode="edge"); out=np.empty_like(m)
    for t in range(m.shape[0]): out[t]=pad[t:t+2*k+1].mean(axis=0)
    return out

def enforce(frame, edges, rest):
    f=frame.copy()
    for i,(a,b) in enumerate(edges):
        if a>=f.shape[0] or b>=f.shape[0]: continue
        v=f[b]-f[a]; n=np.linalg.norm(v); 
        if n<1e-8: continue
        f[b]=f[a]+(rest[i]*(v/n))
    return f

def rest_lengths(f0, edges):
    L=[]
    for a,b in edges:
        if a<f0.shape[0] and b<f0.shape[0]:
            L.append(float(np.linalg.norm(f0[b]-f0[a])+1e-8))
        else: L.append(0.0)
    return np.asarray(L, dtype=np.float32)

class SkeletonViewer(QtWidgets.QWidget):
    def __init__(self, joints=21, parent=None):
        super().__init__(parent)
        self.view=gl.GLViewWidget(self); self.view.setCameraPosition(distance=6.0,elevation=25,azimuth=35)
        grid=gl.GLGridItem(); grid.setSize(10,10,1); self.view.addItem(grid)

        self.btn_play=QtWidgets.QPushButton("Play")
        self.btn_pause=QtWidgets.QPushButton("Pause")
        self.btn_stop=QtWidgets.QPushButton("Stop")
        self.fps_spin=QtWidgets.QSpinBox(); self.fps_spin.setRange(1,120); self.fps_spin.setValue(30)
        ctrls=QtWidgets.QHBoxLayout()
        for w in (self.btn_play,self.btn_pause,self.btn_stop,QtWidgets.QLabel("FPS"),self.fps_spin): ctrls.addWidget(w)
        ctrls.addStretch(1)
        lay=QtWidgets.QVBoxLayout(self); lay.addLayout(ctrls); lay.addWidget(self.view)

        self.btn_play.clicked.connect(self.start); self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop); self.fps_spin.valueChanged.connect(self._on_fps)

        self._timer=QtCore.QTimer(self); self._timer.timeout.connect(self._tick)
        self.fps=30; self.ptr=0; self.motions=None; self.joints=joints; self.edges=EDGES; self._rest=None

        self.scat=gl.GLScatterPlotItem(pos=np.zeros((self.joints,3),dtype=np.float32),size=8,color=(1,1,1,1))
        self.view.addItem(self.scat)
        self.lines=[]; self._rebuild_lines()

    def _rebuild_lines(self):
        for it in self.lines:
            try: self.view.removeItem(it)
            except Exception: pass
        self.lines=[]
        for _ in self.edges:
            lp=gl.GLLinePlotItem(); self.view.addItem(lp); self.lines.append(lp)

    def _on_fps(self,v):
        self.fps=int(v)
        if self._timer.isActive(): self._timer.start(max(1,int(1000/self.fps)))

    def _tick(self):
        if self.motions is None: return
        self.ptr=(self.ptr+1)%len(self.motions); self.update_frame()

    def set_motion(self, motion: np.ndarray, fps:int=30):
        if motion is None or motion.ndim!=3 or motion.shape[1]!=21 or motion.shape[2]!=3: return
        m=motion.astype(np.float32).copy()
        m-=m[:, :1, :]   # center pelvis
        m=smooth(m, win=5)
        self._rest=rest_lengths(m[0], self.edges)
        self.motions=m; self.ptr=0; self.fps=int(fps); self.fps_spin.setValue(self.fps)
        self.update_frame(); self.start()

    def update_frame(self):
        if self.motions is None: return
        f=self.motions[self.ptr]
        f=enforce(f, self.edges, self._rest)
        self.scat.setData(pos=f)
        J=f.shape[0]
        for (a,b), lp in zip(self.edges, self.lines):
            if a<J and b<J:
                pts=np.array([f[a],f[b]],dtype=np.float32)
                lp.setData(pos=pts,color=(0,1,0,1),width=2,antialias=True)
            else:
                lp.setData(pos=np.zeros((0,3),dtype=np.float32))

    def start(self):
        if self.motions is None or len(self.motions)==0: return
        self._timer.start(max(1,int(1000/self.fps)))
    def pause(self): self._timer.stop()
    def stop(self): self._timer.stop(); self.ptr=0; self.update_frame()

