#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from typing import List, Tuple

# KIT-21 indices
J_PEL = 0
J_HEAD = 4
L_SHO, R_SHO = 5, 8
L_ELBOW, R_ELBOW = 6, 9
L_WRIST, R_WRIST = 7, 10
L_HIP, R_HIP = 12, 17
L_KNEE, R_KNEE = 14, 19
L_ANK, R_ANK = 15, 20
SPINE2 = 2

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}

def _safe(m):
    if m.ndim != 3 or m.shape[2] != 3:
        raise ValueError(f"motion must be (T, J, 3), got {m.shape}")
    return m.astype(np.float32, copy=True)

def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float32)

def _apply_rotation_y(frame, angle):
    return frame @ _rot_y(angle).T

def _normalize_clip_center(m: np.ndarray) -> np.ndarray:
    return m - m[:, J_PEL:J_PEL+1, :]

def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    if a is None: return b
    if b is None: return a
    T = min(a.shape[0], b.shape[0])
    return (1.0 - alpha) * a[:T] + alpha * b[:T]

def _dir_flags(text: str):
    t = text.lower()
    return {
        "left": "left" in t,
        "right": "right" in t,
        "back": ("back" in t) or ("backward" in t),
        "forward": ("forward" in t) or ("ahead" in t)
    }

# -------------------- GAIT --------------------

def gait_walk_like(m: np.ndarray, fps: int, forward_axis="x", speed_hz=1.8, step_len=0.25,
                   step_h=0.08, drift=0.02, strength=0.6, backward=False, start_left=True):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    ax = _AXIS_IDX.get(forward_axis, 0)
    sign = -1.0 if backward else 1.0
    w = 2.0 * math.pi * max(0.1, speed_hz) / max(fps, 1)
    l_base = out[0, L_ANK].copy()
    r_base = out[0, R_ANK].copy()
    phase_offset = 0.0 if start_left else math.pi

    for t in range(T):
        s = t * w
        out[t, J_PEL, 1] += 0.04 * strength * math.sin(s)
        out[t, :, ax] += drift * strength * sign

        ls = math.sin(s + phase_offset)
        rs = math.sin(s + phase_offset + math.pi)

        out[t, L_KNEE, 1] += step_h * strength * max(0.0, ls)
        out[t, R_KNEE, 1] += step_h * strength * max(0.0, rs)

        out[t, L_ANK, ax] += step_len * strength * ls
        out[t, R_ANK, ax] += step_len * strength * rs

        if ls < 0.0: out[t, L_ANK, ax] = 0.5 * out[t, L_ANK, ax] + 0.5 * l_base[ax]
        if rs < 0.0: out[t, R_ANK, ax] = 0.5 * out[t, R_ANK, ax] + 0.5 * r_base[ax]

        out[t, L_WRIST, ax] -= 0.20 * strength * ls
        out[t, R_WRIST, ax] -= 0.20 * strength * rs

    return _normalize_clip_center(out)

# -------------------- UPPER BODY --------------------

def _arm_indices(side: str):
    if side == "left":
        return L_SHO, L_ELBOW, L_WRIST, +1.0
    else:
        return R_SHO, R_ELBOW, R_WRIST, -1.0

def ctrl_wave(m: np.ndarray, fps: int, strength: float = 0.9,
              speed_hz: float = 2.0, side: str = "right", freeze_lower: bool = True):
    """
    Upper-body wave: freeze hips/legs, pre-raise arm (wrist above shoulder), bend elbow, wrist arc.
    """
    m = _safe(m); T = m.shape[0]; out = m.copy()
    SHO, ELB, WRI, zsign = _arm_indices(side)
    w = 2.0 * math.pi * speed_hz / max(fps, 1)

    if freeze_lower:
        base = out[0].copy()
        freeze_ids = [L_HIP, L_KNEE, L_ANK, R_HIP, R_KNEE, R_ANK]
        for t in range(T):
            for j in freeze_ids:
                out[t, j] = base[j]
            out[t, J_PEL] = base[J_PEL]

    # pre-pose arm higher and slightly out
    for t in range(T):
        out[t, SHO, 1] += 0.22 * strength
        out[t, ELB, 1] += 0.30 * strength
        out[t, WRI, 1] += 0.40 * strength

        out[t, SHO, 2] += 0.10 * strength * zsign
        out[t, ELB, 2] += 0.16 * strength * zsign
        out[t, WRI, 2] += 0.24 * strength * zsign

        sh_y = out[t, SHO, 1]
        if out[t, WRI, 1] < sh_y + 0.14 * strength:
            out[t, WRI, 1] = sh_y + 0.14 * strength

        vec_sw = out[t, WRI] - out[t, SHO]
        out[t, ELB] = out[t, SHO] + 0.55 * vec_sw
        out[t, ELB, 1] += 0.06 * strength

    # wrist waving arc
    for t in range(T):
        s = t * w
        out[t, WRI, 0] += 0.12 * strength * math.sin(s)
        out[t, WRI, 2] += 0.09 * strength * zsign * math.cos(s)
        out[t, WRI, 1] += 0.07 * strength * (0.5 + 0.5 * math.sin(s))

    return _normalize_clip_center(out)

def ctrl_jump(m: np.ndarray, fps: int, strength: float = 0.8, hops: float = 1.0):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    for t in range(T):
        ph = (t / max(T - 1, 1)) * math.pi * hops
        y = max(0.0, math.sin(ph))
        out[t, :, 1] += 0.42 * strength * y
        out[t, L_WRIST, 1] += 0.20 * strength * y
        out[t, R_WRIST, 1] += 0.20 * strength * y
    return _normalize_clip_center(out)

def ctrl_turn(m: np.ndarray, fps: int, strength: float = 0.85, degrees: float = 90.0):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    total = math.radians(degrees) * strength
    for t in range(T):
        a = total * (t / max(T - 1, 1))
        out[t] = _apply_rotation_y(out[t], a)
    return _normalize_clip_center(out)

def ctrl_kick(m: np.ndarray, fps: int, strength: float = 0.8, side: str = "right"):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    ank = R_ANK if side == "right" else L_ANK
    knee = R_KNEE if side == "right" else L_KNEE
    for t in range(T):
        phase = t / max(T - 1, 1)
        burst = math.sin(math.pi * min(1.0, phase * 2.0))
        out[t, ank, 0] += 0.36 * strength * burst
        out[t, knee, 1] += 0.12 * strength * burst
    return _normalize_clip_center(out)

def ctrl_sit(m: np.ndarray, fps: int, strength: float = 0.85):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    for t in range(T):
        phase = min(1.0, t / max(T*0.3, 1.0))
        out[t, :, 1] -= 0.30 * strength * phase
        out[t, L_KNEE, 1] -= 0.17 * strength * phase
        out[t, R_KNEE, 1] -= 0.17 * strength * phase
    return _normalize_clip_center(out)

def ctrl_stand(m: np.ndarray, fps: int, strength: float = 0.65):
    m = _safe(m); T = m.shape[0]; out = m.copy()
    for t in range(T):
        out[t, J_HEAD, 1] += 0.05 * strength
        out[t, SPINE2, 1] += 0.03 * strength
    return _normalize_clip_center(out)

# -------------------- ENTRY --------------------

def apply_keyword_controls(motion: np.ndarray, prompt: str, fps: int,
                           strength: float = 0.5,
                           forward_axis: str = "x",
                           ground: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    if motion is None: return None, []
    text = prompt.lower()
    tags: List[str] = []
    out = motion.copy()

    dirs = _dir_flags(text)
    forward_ax = "x" if forward_axis not in ("x", "z") else forward_axis

    def add(mod, tag):
        nonlocal out, tags
        out = _blend(out, mod, strength)
        tags.append(tag)

    # walk / run
    if ("walk" in text) or ("stroll" in text) or ("step" in text) or ("walking" in text):
        add(gait_walk_like(out, fps, forward_axis=forward_ax,
                           speed_hz=1.8, step_len=0.26, step_h=0.09,
                           drift=0.018, strength=0.75,
                           backward=dirs["back"] and not dirs["forward"],
                           start_left=not dirs["right"]), "walk")
    if ("run" in text) or ("jog" in text) or ("sprint" in text) or ("running" in text):
        add(gait_walk_like(out, fps, forward_axis=forward_ax,
                           speed_hz=2.5, step_len=0.32, step_h=0.11,
                           drift=0.030, strength=0.85,
                           backward=False, start_left=True), "run")

    # turn
    if ("turn" in text) or ("rotate" in text) or ("turning" in text):
        deg = -90.0 if dirs["right"] else (90.0 if dirs["left"] else 60.0)
        add(ctrl_turn(out, fps, strength=0.85, degrees=deg), f"turn({deg:.0f})")

    # jump
    if ("jump" in text) or ("hop" in text) or ("jumping" in text):
        add(ctrl_jump(out, fps, strength=0.8, hops=1.2), "jump")

    # wave (upper-body)
    if ("wave" in text) or ("hello" in text) or ("waving" in text) or ("greet" in text):
        side = "left" if dirs["left"] else ("right" if dirs["right"] else "right")
        add(ctrl_wave(out, fps, strength=0.9, speed_hz=2.0, side=side, freeze_lower=True), f"wave-{side}")

    # kick
    if "kick" in text or "kicking" in text:
        side = "right" if dirs["right"] else ("left" if dirs["left"] else "right")
        add(ctrl_kick(out, fps, strength=0.8, side=side), f"kick-{side}")

    # sit / stand
    if "sit" in text or "sitting" in text: add(ctrl_sit(out, fps, strength=0.85), "sit")
    if "stand" in text or "idle" in text or "standing" in text: add(ctrl_stand(out, fps, strength=0.65), "stand")

    # ground reference
    out = out.copy()
    out[:, :, 1] -= (out[:, J_PEL:J_PEL+1, 1] - ground)
    return out, tags

