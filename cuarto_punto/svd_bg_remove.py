#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svd_bg_remove.py
----------------
Extrae el fondo dominante de un vídeo mediante SVD y genera
dos salidas:
  • <nombre>_bg.mp4  → fondo reconstruido
  • <nombre>_fg.mp4  → objetos en movimiento (fondo removido)

Opciones clave
--------------
--rank/-k    : número de componentes singulares para el fondo (≥1)
--resize W H : reescalar los fotogramas antes de procesar
--fps        : submuestrear a esta tasa de cuadros
--no-center  : *no* sustraer la media antes de la SVD
"""
import cv2
import time
import argparse
import os
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

# ---------------------------------------------------------------------
def cargar_frames(path, resize=None, fps_out=None):
    """
    Lee los fotogramas, opcionalmente redimensiona y submuestrea la tasa FPS.
    Devuelve un cubo H×W×T (float32 en rango [0,1]).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir {path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    keep_each = max(1, int(round(orig_fps / fps_out))) if fps_out else 1

    frames, idx = [], 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                desc="Leyendo video", unit="frm")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % keep_each == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frames.append(gray)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return np.stack(frames, axis=2)  # H x W x T

# ---------------------------------------------------------------------
def separar_fondo_svd(cube, r=1, center=True):
    """
    Aplica SVD sobre la matriz (H·W)×T y devuelve:
      fondo   : H×W×T reconstruido con rango r
      moviles : |frame − fondo| con clipping [0,1]
    """
    h, w, T = cube.shape
    M = cube.reshape(h * w, T)

    if center:
        mu = M.mean(axis=1, keepdims=True)
        M0 = M - mu
    else:
        mu = 0.0
        M0 = M

    U, S, Vt = randomized_svd(M0, n_components=r, random_state=0)
    fondo = (mu + (U @ np.diag(S) @ Vt)).reshape(h, w, T)
    moviles = np.clip(np.abs(cube - fondo), 0.0, 1.0)
    return fondo, moviles

# ---------------------------------------------------------------------
def guardar_video(frames, path, fps):
    """Codifica frames (H×W×T, float [0,1]) como MP4 en escala de grises."""
    h, w, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)
    for i in tqdm(range(frames.shape[2]), desc=f"→ {os.path.basename(path)}",
                  unit="frm"):
        f8 = (frames[:, :, i] * 255).astype(np.uint8)
        vw.write(cv2.merge([f8, f8, f8]))  # 3 canales para compatibilidad
    vw.release()

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="video.mp4", help="archivo de vídeo")
    ap.add_argument("--rank", "-k", type=int, default=1,
                    help="rango para reconstruir fondo")
    ap.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                    help="nuevo tamaño en píxeles")
    ap.add_argument("--fps", type=int, help="fps de salida / submuestreo")
    ap.add_argument("--no-center", action="store_true",
                    help="no sustraer la media antes de SVD")
    args = ap.parse_args()

    # ---------- Lectura -----------------
    t0 = time.perf_counter()
    cube = cargar_frames(args.input,
                         resize=tuple(args.resize) if args.resize else None,
                         fps_out=args.fps)
    print(f"Frames: {cube.shape[2]} | Resolución: {cube.shape[1]}×{cube.shape[0]}")
    print(f"Lectura + preproc: {time.perf_counter()-t0:.2f} s")

    # ---------- SVD ---------------------
    t1 = time.perf_counter()
    fondo, moviles = separar_fondo_svd(cube,
                                       r=args.rank,
                                       center=not args.no_center)
    t_svd = time.perf_counter() - t1
    print(f"SVD: {t_svd:.2f} s  |  fps efectivo ≈ {cube.shape[2]/t_svd:.1f}")

    # ---------- Guardado ----------------
    base = os.path.splitext(args.input)[0]
    fps_out = args.fps or 30
    guardar_video(fondo,   f"{base}_bg.mp4", fps_out)
    guardar_video(moviles, f"{base}_fg.mp4", fps_out)

if __name__ == "__main__":
    main()
