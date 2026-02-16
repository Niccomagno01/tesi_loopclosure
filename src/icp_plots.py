import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Utility salvataggio

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _savefig(path: str, dpi: int = 140):
    # Disabilita temporaneamente la modalità interattiva per non mostrare finestre
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        _ensure_dir(os.path.dirname(path))
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
    finally:
        # Ripristina lo stato interattivo originale
        if was_interactive:
            plt.ion()


# 1) Schema concettuale corrispondenze (subset)

def save_concept_correspondences(res: Dict, title: str, out_path: str, max_lines: int = 120):
    tgt = np.asarray(res['tgt_local'])
    src = np.asarray(res['src_local'])
    if tgt.size == 0 or src.size == 0:
        return

    # Costruisci NN su subset semplice (euclideo O(N*M) sul subset)
    n = min(len(src), len(tgt), max_lines)
    src_sub = src[:n]
    # Abbina naive: per ogni src, trova NN su tgt
    idxs = []
    for p in src_sub:
        d2 = np.sum((tgt - p) ** 2, axis=1)
        idxs.append(int(np.argmin(d2)))

    plt.figure(figsize=(5, 4))
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c='tab:blue', label='Target (k-1)')
    plt.scatter(src[:, 0], src[:, 1], s=10, c='tab:red', label='Source (k)')

    # Disegna le linee di corrispondenza in nero (rappresentano la situazione reale)
    for i, j in enumerate(idxs):
        xs = [src_sub[i, 0], tgt[j, 0]]
        ys = [src_sub[i, 1], tgt[j, 1]]
        plt.plot(xs, ys, c='black', lw=0.6, alpha=0.7)

    # Aggiungi una linea dummy per la legenda delle corrispondenze
    plt.plot([], [], c='black', lw=0.6, alpha=0.7, label='Corrispondenze (situazione reale)')

    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)


# 2) Effetto dell'inizializzazione: overlay finali

def save_alignment_overlays(res: Dict, title: str, out_path: str):
    tgt = np.asarray(res['tgt_local'])
    src_none = np.asarray(res['none']['src_transformed'])
    src_raw_none = np.asarray(res['raw_none']['src_transformed'])
    plt.figure(figsize=(6, 5))
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c='k', label='Target (k-1)')
    plt.scatter(src_none[:, 0], src_none[:, 1], s=8, c='tab:red', alpha=0.7, label='ICP (filtrato)')
    plt.scatter(src_raw_none[:, 0], src_raw_none[:, 1], s=8, c='tab:orange', alpha=0.5, label='RAW')
    plt.axis('equal'); plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend(loc='upper right', fontsize=8)
    _savefig(out_path)


# 3) Curve di convergenza (RMSE per iterazione)

def save_convergence_curves(res: Dict, title: str, out_path: str):
    # Compatibilità: il nuovo ICP usa 'errors', il vecchio usava 'errs'
    e_none = np.asarray(res['none'].get('errors', res['none'].get('errs', [])))
    e_rawn = np.asarray(res['raw_none'].get('errors', res['raw_none'].get('errs', [])))
    plt.figure(figsize=(6, 4))
    if e_none.size: plt.plot(e_none, label='ICP (filtrato)')
    if e_rawn.size: plt.plot(e_rawn, '--', label='RAW')
    plt.xlabel('Iterazione'); plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(alpha=0.3); plt.legend()
    _savefig(out_path)


# 9) Frecce stima (Δx, Δy) e α

def save_motion_arrows(res: Dict, title: str, out_path: str):
    def ang_deg(r_mat):
        return float(np.degrees(np.arctan2(r_mat[1, 0], r_mat[0, 0])))

    # Prepara le frecce: Ground Truth + ICP filtrato + RAW
    ests = []

    # Aggiungi Ground Truth se disponibile (situazione reale)
    gt_r = res.get('gt_R')
    gt_t = res.get('gt_t')
    if gt_r is not None and gt_t is not None:
        ests.append(('Situazione Reale (GT)', gt_t, gt_r, 'black'))

    # Aggiungi ICP filtrato e RAW
    ests.append(('ICP (filtrato)', res['none']['t'], res['none']['R'], 'tab:red'))
    ests.append(('RAW', res['raw_none']['t'], res['raw_none']['R'], 'tab:orange'))

    plt.figure(figsize=(6, 6))

    # Calcola i limiti necessari per mostrare tutte le frecce
    max_x, max_y = 0.0, 0.0
    for name, t, r_est, col in ests:
        t = np.asarray(t)
        r_est = np.asarray(r_est)
        # Freccia più spessa per la situazione reale (nero)
        width = 0.006 if col == 'black' else 0.004
        plt.quiver(0, 0, t[0], t[1], angles='xy', scale_units='xy', scale=1,
                  color=col, width=width,
                  label=f'{name} (α={ang_deg(r_est):+.2f}°)')
        # Traccia i limiti massimi
        max_x = max(max_x, abs(t[0]))
        max_y = max(max_y, abs(t[1]))

    # Imposta limiti degli assi con margine ottimale per mostrare frecce complete
    margin = 1.5  # Margine del 50% extra - equilibrio tra visibilità e compattezza
    max_val = max(max_x, max_y) * margin

    # Imposta limiti PRIMA di chiamare gca() per aspect ratio
    if max_val > 0:
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)

    # Usa set_aspect invece di axis('equal') per non sovrascrivere i limiti
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.grid(alpha=0.3)
    plt.xlabel('Δx [m]', fontsize=10)
    plt.ylabel('Δy [m]', fontsize=10)
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    _savefig(out_path)


# 14) Confronto RAW vs Filtrato (overlay target + raw vs filtrato, per una coppia)


# 15) Errore di posizione e orientazione nel tempo

def save_error_over_time(real_traj: np.ndarray, icp_traj: np.ndarray, raw_traj: np.ndarray,
                         title: str, out_path: str, dt: float = 0.1):
    """
    Grafico degli errori ICP vs ground truth nel tempo.

    Args:
        real_traj: Traiettoria reale (ground truth) [N x 3] con (x, y, theta)
        icp_traj: Traiettoria stimata dall'ICP filtrato [N x 3]
        raw_traj: Traiettoria stimata dall'ICP RAW [N x 3]
        title: Titolo del grafico
        out_path: Percorso dove salvare l'immagine
        dt: Intervallo temporale tra i campioni (secondi)
    """
    real_traj = np.asarray(real_traj, dtype=float)
    icp_traj = np.asarray(icp_traj, dtype=float)
    raw_traj = np.asarray(raw_traj, dtype=float)

    # Assicuriamoci che le traiettorie abbiano la stessa lunghezza
    n = min(len(real_traj), len(icp_traj), len(raw_traj))
    if n < 2:
        return  # Non abbastanza punti per plottare

    real_traj = real_traj[:n]
    icp_traj = icp_traj[:n]
    raw_traj = raw_traj[:n]

    # Calcola il tempo
    time = np.arange(n) * dt

    # Calcola errore di posizione (distanza euclidea)
    pos_error_icp = np.sqrt((real_traj[:, 0] - icp_traj[:, 0])**2 +
                             (real_traj[:, 1] - icp_traj[:, 1])**2)
    pos_error_raw = np.sqrt((real_traj[:, 0] - raw_traj[:, 0])**2 +
                             (real_traj[:, 1] - raw_traj[:, 1])**2)

    # Calcola errore di orientazione (differenza angolare normalizzata in [-pi, pi])
    def angle_diff(a1, a2):
        diff = a1 - a2
        return np.arctan2(np.sin(diff), np.cos(diff))

    orient_error_icp = np.abs(angle_diff(real_traj[:, 2], icp_traj[:, 2]))
    orient_error_raw = np.abs(angle_diff(real_traj[:, 2], raw_traj[:, 2]))

    # Converti orientazione in gradi per visualizzazione
    orient_error_icp_deg = np.degrees(orient_error_icp)
    orient_error_raw_deg = np.degrees(orient_error_raw)

    # Crea il grafico con due subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Errore di posizione
    ax1.plot(time, pos_error_icp, 'r-', linewidth=1.5, label='ICP (filtrato)')
    ax1.plot(time, pos_error_raw, 'orange', linestyle='--', linewidth=1.5, label='RAW')
    ax1.set_xlabel('Tempo [s]', fontsize=11)
    ax1.set_ylabel('Errore di Posizione [m]', fontsize=11)
    ax1.set_title(f'{title} - Errore di Posizione', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Subplot 2: Errore di orientazione
    ax2.plot(time, orient_error_icp_deg, 'r-', linewidth=1.5, label='ICP (filtrato)')
    ax2.plot(time, orient_error_raw_deg, 'orange', linestyle='--', linewidth=1.5, label='RAW')
    ax2.set_xlabel('Tempo [s]', fontsize=11)
    ax2.set_ylabel('Errore di Orientazione [°]', fontsize=11)
    ax2.set_title(f'{title} - Errore di Orientazione', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    _savefig(out_path)

def save_scan2map_overlay(map_pts, scan_aligned, title, out_path):

    map_pts = np.asarray(map_pts, float)
    scan_aligned = np.asarray(scan_aligned, float)
    if map_pts.size == 0 or scan_aligned.size == 0:
        return

    plt.figure(figsize=(8,6))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=2, alpha=0.6, label="Map")
    plt.scatter(scan_aligned[:,0], scan_aligned[:,1], s=6, alpha=0.9, label="Scan aligned")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.legend(loc="best")
    _savefig(out_path, dpi=160)

def save_scan2map_trajectories(
    traj_gt: np.ndarray,
    traj_init: np.ndarray,
    traj_raw: np.ndarray,
    title: str,
    out_path: str
):
    plt.figure(figsize=(7, 7))
    if traj_gt is not None:
        traj_gt = np.asarray(traj_gt, dtype=float)
        if traj_gt.size:
            plt.plot(traj_gt[:, 0], traj_gt[:, 1], "k-", linewidth=2.0, label="Ground Truth")

    if traj_init is not None:
        traj_init = np.asarray(traj_init, dtype=float)
        if traj_init.size:
            plt.plot(traj_init[:, 0], traj_init[:, 1], "r-", linewidth=2.0, label="ICP (filtrato)")

    if traj_raw is not None:
        traj_raw = np.asarray(traj_raw, dtype=float)
        if traj_raw.size:
            plt.plot(traj_raw[:, 0], traj_raw[:, 1], "--", linewidth=2.0, label="RAW")

    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend(loc="best")
    _savefig(out_path, dpi=160)
