from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
import open3d as o3d
from environment_presets import setup_environments_per_trajectory
from lidar import Lidar  # sensore LiDAR
import argparse
from icp import run_icp_scan_to_map_pair, compute_relative_transform_from_odometry
import time  # per ETA nella barra di progresso
from tqdm import tqdm as _tqdm  # progress bar esterna con ETA
from icp_plots import (
    save_error_over_time,
    save_scan2map_overlay,
    save_scan2map_trajectories,
)
from typing import List, Optional, Tuple, Dict
from environment import Environment
import re
import math
import numpy as np
import sys, os
import datetime as _dt

# Nuovo: inizializza colorama per garantire rendering ANSI su Windows
try:
    from colorama import init as _colorama_init
except ImportError:
    _colorama_init = None
else:
    _colorama_init()

# Helper slugify locale (evita warning su uso di funzione privata) e precompila regex
_slugify_re = re.compile(r'[^a-z0-9_\-]')

# Piccolo helper: wrapping angolare in [-pi,pi)
_def_pi = math.pi
_def_2pi = 2.0 * math.pi


def _wrap_angle(a: float) -> float:
    return (float(a) + _def_pi) % _def_2pi - _def_pi


def _apply_world_transform(traj: Optional[np.ndarray], base_pose: np.ndarray) -> Optional[np.ndarray]:
    """Applica la trasformazione di mondo (R0,t0) derivata dalla prima posa reale base_pose = (x0,y0,theta0)
    a una traiettoria (x,y,theta) locale ricostruita dal log. Ritorna una nuova array, o None se traj è None.
    """
    if traj is None:
        return None
    arr = np.asarray(traj, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    x0, y0, th0 = map(float, base_pose[:3])
    c, s = math.cos(th0), math.sin(th0)
    r0 = np.array([[c, -s], [s, c]], dtype=float)
    arr_xy = arr[:, :2] @ r0.T + np.array([x0, y0], dtype=float)
    arr_th = np.array([_wrap_angle(th0 + t) for t in arr[:, 2]], dtype=float)
    out = arr.copy()
    out[:, 0:2] = arr_xy
    out[:, 2] = arr_th
    return out


# Regex per sopprimere nel file le righe di progress bar e "Salvataggio immagini"
# - Righe che iniziano con "Salvataggio immagini" (tqdm o ASCII fallback)
# - Righe di tqdm con percentuale e barra (es. " 42%|####...") ovunque nella riga
_prog_re_salva = re.compile(r"^\s*Salvataggio immagini\b")
_prog_re_tqdm = re.compile(r"\b\d{1,3}%\|")


# Tee per duplicare stdout/stderr su file e console, filtrando le progress nel file
class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary
        self.encoding = getattr(primary, 'encoding', 'utf-8')
        self._buf = ''
        self._suppressed_last = False  # evita righe vuote dopo soppressione

    @staticmethod
    def _should_suppress_line(line: str) -> bool:
        if not line:
            return False
        s = line.lstrip('\r')
        if _prog_re_salva.match(s):
            return True
        if _prog_re_tqdm.search(s):
            return True
        return False

    def _write_to_secondary_filtered(self, data: str) -> None:
        # Normalizza solo CRLF in LF; lasciamo i CR come separatori di update trattandoli come fine linea
        text = data.replace('\r\n', '\n')
        # Spezza su CR per catturare aggiornamenti in-place senza introdurre '\n' spurii
        parts = text.split('\r')
        for idxp, part in enumerate(parts):
            self._buf += part
            # processa linee complete terminate da \n
            while True:
                nl = self._buf.find('\n')
                if nl == -1:
                    break
                line = self._buf[:nl]
                self._buf = self._buf[nl + 1:]
                if self._should_suppress_line(line):
                    self._suppressed_last = True
                    continue
                # scarta righe vuote immediatamente dopo soppressione
                if self._suppressed_last and line.strip() == '':
                    # mantieni flag finché non arriva una riga non vuota
                    continue
                self._secondary.write(line + '\n')
                self._suppressed_last = False
            # Se non è l'ultimo pezzo, abbiamo avuto un CR che segnala aggiornamento riga: tratta il contenuto accumulato come linea completa
            if idxp < len(parts) - 1:
                line_cr = self._buf
                self._buf = ''
                if self._should_suppress_line(line_cr):
                    self._suppressed_last = True
                    continue
                if self._suppressed_last and (line_cr.strip() == ''):
                    continue
                # Scrive la linea derivata da CR senza aggiungere newline extra (usa \n per chiudere la linea corrente)
                self._secondary.write(line_cr)
                self._suppressed_last = False

    def write(self, data):
        try:
            self._primary.write(data)
        finally:
            try:
                self._write_to_secondary_filtered(str(data))
                # flush immediato del file per non perdere dati in caso di terminazione improvvisa
                if hasattr(self._secondary, 'flush'):
                    self._secondary.flush()
            except (IOError, OSError, AttributeError):
                # Gestisce errori di I/O o problemi con l'oggetto file
                self._secondary.write(str(data))
                try:
                    self._secondary.flush()
                except (IOError, OSError, AttributeError):
                    pass
        return len(data)

    def flush(self):
        if self._buf:
            rem = self._buf
            self._buf = ''
            if not self._should_suppress_line(rem):
                if not (self._suppressed_last and rem.strip() == ''):
                    self._secondary.write(rem)
            # se era soppressa o vuota dopo soppressione, non scrivere nulla
        try:
            self._primary.flush()
        finally:
            self._secondary.flush()

    def isatty(self):
        return bool(getattr(self._primary, 'isatty', lambda: False)())

    def fileno(self):
        if hasattr(self._primary, 'fileno'):
            return self._primary.fileno()
        raise OSError('fileno non disponibile')


def _slugify_local(text: str) -> str:
    base = (text or '').lower().strip() or 'case'
    base = re.sub(r'\s+', '_', base)
    return _slugify_re.sub('', base)


def build_simulator() -> Simulator:
    """Crea un simulatore con un robot di default."""
    return Simulator(robot=Robot())


def reset_robot_default(sim: Simulator, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
    """Reimposta il robot del simulatore alla posa iniziale di default (x,y,theta)."""
    sim.reset_robot(x=x, y=y, theta=theta)


# ------------------ Collisione via LiDAR ------------------

def _support_distance_rect(delta: float, a: float, b: float) -> float:
    """Distanza dal centro alla frontiera di un rettangolo axis-aligned (semiassi a=half-length, b=half-width)
    lungo la direzione con angolo delta nel frame del robot. Formula del supporto: a*|cos δ| + b*|sin δ|."""
    c = abs(math.cos(delta))
    s = abs(math.sin(delta))
    return a * c + b * s


def _lidar_clearance_measure(pose: np.ndarray, lidar: Lidar, env: Environment, body_length: float,
                             body_width: float) -> float:
    """Ritorna la minima differenza (range - supporto_rettangolo) sui raggi del LiDAR per la posa.
    Se <= 0 si considera contatto (il corpo tocca l'ostacolo)."""
    # semi-dimensioni del rettangolo corpo (metri)
    a = 0.5 * float(body_length)
    b = 0.5 * float(body_width)
    # Angoli relativi dei raggi nel frame del robot (come in Lidar.scan)
    half = 0.5 * float(lidar.angle_span)
    rel_angles = np.linspace(-half, half, num=lidar.n_rays, endpoint=True)
    # Scansione attuale
    _, ranges = lidar.scan(pose, env, return_ranges=True)
    # Misura di clearance: range meno distanza bordo corpo su ciascun raggio
    supports = np.array([_support_distance_rect(float(da), a, b) for da in rel_angles], dtype=float)
    diffs = ranges - supports
    return float(np.min(diffs))


def _first_collision_via_lidar(history: np.ndarray, env: Environment, lidar: Lidar, *, body_length: float = 0.40,
                               body_width: float = 0.20, iters: int = 14) -> Tuple[Optional[int], Optional[float]]:
    """Trova primo contatto via LiDAR lungo la storia: ritorna (k, alpha) con k il primo indice in cui c'è contatto
    e alpha la frazione in (k-1,k] in cui la misura di clearance attraversa 0 (bisezione su pose interpolate).
    Se contatto a frame 0: (0, 0.0). Se nessun contatto: (None, None)."""
    n = len(history)
    if n <= 0:
        return None, None
    # Misura iniziale
    m0 = _lidar_clearance_measure(history[0], lidar, env, body_length, body_width)
    if m0 <= 0.0:
        return 0, 0.0
    # Cerca primo frame con misura <= 0
    k_hit = None
    for k in range(1, n):
        mk = _lidar_clearance_measure(history[k], lidar, env, body_length, body_width)
        if mk <= 0.0:
            k_hit = k
            break
    if k_hit is None:
        return None, None
    # Bisezione tra (k-1, k]
    lo, hi = 0.0, 1.0
    p0 = history[k_hit - 1]
    p1 = history[k_hit]
    for _ in range(max(1, int(iters))):
        mid = 0.5 * (lo + hi)
        pose_mid = _interp_pose_local(p0, p1, mid)
        mm = _lidar_clearance_measure(pose_mid, lidar, env, body_length, body_width)
        if mm <= 0.0:
            hi = mid
        else:
            lo = mid
    return int(k_hit), float(hi)


# ------------------ Fine collisione via LiDAR ------------------


def _env_bounds_diag(env: Environment) -> float:
    try:
        x0, y0, x1, y1 = env.bounds.bounds  # type: ignore[union-attr]
        w = float(x1 - x0)
        h = float(y1 - y0)
        return float((w * w + h * h) ** 0.5)
    except (AttributeError, TypeError, ValueError):
        return 10.0


def apply_loop_closure_correction(trajectory: np.ndarray, is_circular: bool = False,
                                  closure_threshold: float = 0.3) -> np.ndarray:
    """
    Applica correzione di loop closure per traiettorie circolari.
    Se il punto finale è vicino all'inizio, distribuisce l'errore lungo tutta la traiettoria.

    Args:
        trajectory: Array Nx3 [x, y, theta]
        is_circular: Se True, forza la chiusura del loop
        closure_threshold: Distanza massima per considerare il loop chiuso (metri)

    Returns:
        Traiettoria corretta
    """
    if len(trajectory) < 10 or not is_circular:
        return trajectory

    # Controlla se il loop è quasi chiuso
    start = trajectory[0, :2]
    end = trajectory[-1, :2]
    distance = np.linalg.norm(end - start)

    if distance > closure_threshold:
        # Loop non abbastanza vicino, non correggere
        return trajectory

    # Calcola errore totale
    error_xy = start - end
    error_theta = trajectory[0, 2] - trajectory[-1, 2]

    # Normalizza errore angolare a [-pi, pi]
    while error_theta > np.pi:
        error_theta -= 2 * np.pi
    while error_theta < -np.pi:
        error_theta += 2 * np.pi

    # Distribuisci la correzione linearmente lungo la traiettoria
    corrected = trajectory.copy()
    n = len(trajectory)

    for i in range(1, n):
        # Frazione del percorso completato
        alpha = float(i) / float(n - 1)

        # Applica correzione proporzionale
        corrected[i, 0] += alpha * error_xy[0]
        corrected[i, 1] += alpha * error_xy[1]
        corrected[i, 2] += alpha * error_theta

    return corrected


def compute_odometry_trajectory(history: np.ndarray, dt: float,
                                noise_pos: float = 0.01,
                                noise_angle: float = 0.005) -> np.ndarray:
    """
    Simula una traiettoria odometrica con drift accumulato a partire dal ground truth.

    Questa funzione prende la traiettoria reale del robot e simula come sarebbe
    la traiettoria ricostruita usando solo odometria, aggiungendo rumore alle
    velocità lineari e angolari per simulare il drift caratteristico dei sensori
    odometrici.

    Args:
        history: Traiettoria ground truth [N, 3] con formato (x, y, theta)
        dt: Passo temporale tra frame consecutivi (secondi)
        noise_pos: Deviazione standard del rumore sulla velocità lineare (m/s)
        noise_angle: Deviazione standard del rumore sulla velocità angolare (rad/s)

    Returns:
        Traiettoria odometrica con drift accumulato [N, 3]
    """
    odom_traj = np.zeros_like(history)
    odom_traj[0] = history[0].copy()  # Stessa posizione di partenza

    for k in range(1, len(history)):
        # Calcola il movimento reale tra i frame k-1 e k
        dx_real = history[k, 0] - history[k - 1, 0]
        dy_real = history[k, 1] - history[k - 1, 1]
        dtheta_real = history[k, 2] - history[k - 1, 2]

        # Calcola velocità reali dal movimento ground truth
        v_real = np.sqrt(dx_real ** 2 + dy_real ** 2) / dt
        omega_real = dtheta_real / dt

        # Aggiungi rumore gaussiano alle velocità per simulare errori odometrici
        v_noisy = v_real + np.random.normal(0, noise_pos)
        omega_noisy = omega_real + np.random.normal(0, noise_angle)

        # Integra con Eulero usando le velocità rumorose
        theta_prev = odom_traj[k - 1, 2]
        odom_traj[k, 0] = odom_traj[k - 1, 0] + v_noisy * np.cos(theta_prev) * dt
        odom_traj[k, 1] = odom_traj[k - 1, 1] + v_noisy * np.sin(theta_prev) * dt
        odom_traj[k, 2] = odom_traj[k - 1, 2] + omega_noisy * dt

        # Normalizza angolo nell'intervallo [-π, π]
        odom_traj[k, 2] = _wrap_angle(float(odom_traj[k, 2]))

    return odom_traj

def compute_odometry_from_commands(
    commands: np.ndarray,
    dt: float,
    start_pose: np.ndarray,
    noise_v: float = 0.01,      # deviazione standard "per secondo" (m/s)
    noise_omega: float = 0.01,  # deviazione standard "per secondo" (rad/s)
) -> np.ndarray:
    """
        Simula una traiettoria odometrica integrando i comandi di velocità lineare e angolare con integrazione di Eulero,
        aggiungendo rumore e bias per simulare il drift.

         Args:
            commands: Array [N, 2] con comandi (v, omega) per ogni frame
            dt: Passo temporale tra frame consecutivi (secondi)
            start_pose: Posa iniziale (x, y, theta)
            noise_v: deviazione standard del rumore su v.
            noise_omega: deviazione standard del rumore su omega.

        Returns:
            Traiettoria odometrica integrata [N+1, 3] con formato (x, y, theta),
            includendo la posa iniziale come primo elemento.
        """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt deve essere > 0")

    cmds = np.asarray(commands, dtype=float)  # genero array numpy
    n = len(cmds)  # calcolo numero di comandi

    out = np.zeros((n + 1, 3), dtype=float)  # preallocazione array per traiettoria odometrica
    out[0] = np.asarray(start_pose[:3], dtype=float)  # posa iniziale (x,y,theta)

    x, y, th = map(float, out[0])  # inizializzo variabili per integrazione

    # bias dovuto a slittamento, raggio ruota stimato male, ecc.
    bias_v = float(np.random.normal(0.0, 0.01))  # bias moltiplicativo su v (σ ≈ 1%)
    bias_w = float(np.random.normal(0.0, 0.02))  # bias additivo su ω (rad/s)

    sdt = math.sqrt(dt)

    for k in range(n):
        v, w = float(cmds[k, 0]), float(cmds[k, 1])

        # rumore bianco discreto: noise_* è scalato con sqrt(dt)
        if noise_v > 0.0:
            v += float(np.random.normal(0.0, noise_v * sdt))
        if noise_omega > 0.0:
            w += float(np.random.normal(0.0, noise_omega * sdt))

        # applico bias (drift sistematico)

        v = v * (1.0 + bias_v)
        w = w + bias_w

        # integrazione di Eulero
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th = _wrap_angle(th + w * dt)

        out[k + 1] = [x, y, th]

    return out

def _build_lidars_for_cases(envs: List[Environment], titles: List[str]) -> List[Lidar]:
    """Crea una lista di Lidar per singolo caso con r_max adattivo per non coprire sempre tutti gli ostacoli.
    Strategia: r_max = fattore * diagonale dei bounds, con fattori più piccoli per i casi rettilinei."""
    lidars: List[Lidar] = []
    for idx, (env, _unused_title) in enumerate(zip(envs, titles)):
        diag = _env_bounds_diag(env)
        # Fattori per caso: più conservativi sui rettilinei
        if idx == 0:  # Rettilinea v costante: aumenta r_max e n_rays per avere più hit
            factor = 0.55
        elif idx == 1:  # Rettilinea v variabile
            factor = 0.40
        elif idx in (2, 3):  # circolari
            factor = 0.50
        elif idx == 4:  # otto
            factor = 0.45
        else:  # random walk
            factor = 0.55
        # Micro-ritocchi: più copertura e densità raggi per casi 4 (idx==3) e 5 (idx==4)
        if idx in (3, 4):
            factor = 0.60
        r_max = max(1.0, factor * diag)
        # Numero raggi: più densi per i casi 3 e 4
        if idx == 0:
            n_rays = 240
        elif idx == 1:
            n_rays = 180
        elif idx in (3, 4):
            n_rays = 300
        else:
            n_rays = 240

        # Aggiunta rumore Lidar

        LIDAR_RANGE_STD = 0.015 # deviazione standard del rumore sulle misure di distanza (metri)

        lidar = Lidar(n_rays=n_rays,
                      angle_span=2 * math.pi,
                      r_max=r_max,
                      angle_offset=0.0,
                      add_noise=True,
                      noise_std=LIDAR_RANGE_STD)
        lidars.append(lidar)
    return lidars


def _interp_pose_local(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolazione lineare (x,y,theta) con wrapping di theta in [-pi,pi)."""
    a = float(max(0.0, min(1.0, alpha)))
    x0, y0, t0 = map(float, p0)
    x1, y1, t1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0
    dth = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi
    x = x0 + a * dx
    y = y0 + a * dy
    th = t0 + a * dth
    th = (th + math.pi) % (2.0 * math.pi) - math.pi
    return np.array([x, y, th], dtype=float)


# ===== Parser ICP da log (usa ESATTAMENTE i valori stampati) =====
_re_case_hdr = re.compile(r"^\s*CASO\s+(\d+):\s*(.+)$")
_re_icp_pose = re.compile(
    r"^\s*ICP:\s*Δx=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*m,\s*Δy=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*m,\s*α=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*deg\s*$")
_ansi_re = re.compile(r"\x1b\[[0-9;]*m")

# Nuovo: regex generica per tre etichette (Reali, ICP, RAW)
_re_pose_labeled = re.compile(
    r"^\s*(Reali:|ICP:|RAW:)\s*Δx=([+\-]?\d+(?:\.\d+)?)\s*m,\s*Δy=([+\-]?\d+(?:\.\d+)?)\s*m,\s*α=([+\-]?\d+(?:\.\d+)?)\s*deg\s*$"
)


def _accumulate_icp_deltas_to_traj(deltas: List[Tuple[float, float, float]]) -> np.ndarray:
    """Dati Δ pose locali (dx [m], dy [m], alpha_deg [deg]) nel frame k-1,
    integra in una traiettoria globale partendo da (0,0,0) senza trasformazioni extra."""
    n = len(deltas)
    hist = np.zeros((n + 1, 3), dtype=float)
    x = 0.0
    y = 0.0
    th = 0.0
    hist[0] = [x, y, th]
    for i, (dx, dy, a_deg) in enumerate(deltas, start=1):
        a = math.radians(float(a_deg))
        # Trasforma l'incremento locale (dx,dy) nel mondo ruotandolo dell'orientamento corrente
        c, s = math.cos(th), math.sin(th)
        gx = c * dx - s * dy
        gy = s * dx + c * dy
        x += gx
        y += gy
        th = (th + a + math.pi) % (2.0 * math.pi) - math.pi
        hist[i] = [x, y, th]
    return hist


def _parse_icp_trajectories_from_log(log_path: str, n_cases: int) -> List[Optional[np.ndarray]]:
    """[DEPRECATO] Mantiene la vecchia API: ritorna solo ICP filtrato.
    Usata per compatibilità, delega al parser completo e prende la serie 'icp'."""
    triplets = parse_icp_triplets_from_log(log_path, n_cases)
    out: List[Optional[np.ndarray]] = []
    for case in triplets:
        out.append(case.get('icp'))
    return out


def parse_icp_triplets_from_log(log_path: str, n_cases: int) -> List[Dict[str, Optional[np.ndarray]]]:
    """Parsa il file di log corrente e ricostruisce per ciascun CASO le traiettorie
    usando ESATTAMENTE i Δ stampati per: Reali, ICP filtrato ("ICP:"), RAW ("RAW:").
    Ritorna una lista per-caso di dizionari: {'real': np.ndarray|None, 'icp': np.ndarray|None, 'raw': np.ndarray|None}.
    Ogni traiettoria parte da (0,0,0)."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return [{"real": None, "icp": None, "raw": None} for _ in range(int(n_cases))]

    # Accumula Δ per caso e per etichetta
    per_case: List[Dict[str, List[Tuple[float, float, float]]]] = [
        {"real": [], "icp": [], "raw": []} for _ in range(int(n_cases))
    ]
    cur_case_idx: Optional[int] = None

    for raw_line in lines:
        clean = _ansi_re.sub('', raw_line)
        line = clean.rstrip('\n')
        m_hdr = _re_case_hdr.match(line)
        if m_hdr:
            try:
                idx = int(m_hdr.group(1))
                cur_case_idx = idx - 1 if 1 <= idx <= n_cases else None
            except (ValueError, IndexError):
                # Gestisce errori di conversione o accesso al gruppo
                cur_case_idx = None
            continue
        if cur_case_idx is None:
            continue
        m_pose = _re_pose_labeled.match(line)
        if not m_pose:
            continue
        label = m_pose.group(1)
        try:
            dx = float(m_pose.group(2))
            dy = float(m_pose.group(3))
            a_deg = float(m_pose.group(4))
        except (ValueError, IndexError):
            # Gestisce errori di conversione float o accesso ai gruppi
            continue
        if label.startswith('Reali'):
            per_case[cur_case_idx]['real'].append((dx, dy, a_deg))
        elif label.startswith('ICP'):
            per_case[cur_case_idx]['icp'].append((dx, dy, a_deg))
        elif label.startswith('RAW'):
            per_case[cur_case_idx]['raw'].append((dx, dy, a_deg))

    # Costruisci traiettorie
    out: List[Dict[str, Optional[np.ndarray]]] = []
    for cs in per_case:
        item: Dict[str, Optional[np.ndarray]] = {}
        for k in ('real', 'icp', 'raw'):
            deltas = cs.get(k, [])
            item[k] = _accumulate_icp_deltas_to_traj(deltas) if deltas else None
        out.append(item)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Simulatore traiettorie + salvatore immagini")
    parser.add_argument("--skip-collision", action="store_true",
                        help="Salta il calcolo collisioni per avvio piu' rapido")
    parser.add_argument("--skip-viewer", action="store_true", help="Non aprire il viewer interattivo")
    parser.add_argument("--scan-interval", type=float, default=1.0, help="Intervallo tra scansioni LiDAR salvate [s]")
    parser.add_argument("--viewer-lidar-every", type=int, default=4,
                        help="Aggiorna LiDAR nel viewer ogni N frame (default 4)")
    parser.add_argument("--run-icp", action="store_true",
                        help="Esegui ICP su coppie (k-1,k) in frame locale e stampa confronto ICP filtrato vs RAW")
    parser.add_argument("--viewer-icp-grid", action="store_true",
                        help="[DEPRECATO] Usa --viewer-mode grid al posto di questo flag")
    parser.add_argument("--skip-icp", action="store_true", help="Non eseguire l'ICP prima dell'apertura del viewer")
    parser.add_argument("--viewer-mode", choices=["grid", "carousel"], default="carousel",
                        help="Seleziona viewer: 'grid' (ICP a 5 pannelli) o 'carousel' (standard) - default: carousel")
    parser.add_argument("--quiet", action="store_true",
                        help="Se presente sopprime stampe durante salvataggio immagini (default: stampe attive)")
    parser.add_argument("--no-icp-verbose", dest="icp_verbose", action="store_false",
                        help="Disabilita stampe dettagliate ICP durante l'esecuzione principale")
    parser.add_argument("--viewer-log-align-world", action="store_true",
                        help="Allinea le traiettorie ricostruite dal LOG (Reali/RAW/ICP) al mondo usando la prima posa reale del caso")
    parser.add_argument("--run-scan2map", action="store_true",
                        help="Esegue scan-to-map e mostra la traiettoria nel viewer (al posto di run_icp_pair)")

    parser.set_defaults(icp_verbose=True)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        args.run_scan2map = True
        args.skip_viewer = False
        args.viewer_mode = "carousel"
        args.viewer_log_align_world = True
        args.skip_icp = True

    return args


def build_cases_and_envs(dt: float):
    # Parametri base di riferimento - AUMENTATI per migliorare ICP
    v_ref = 1.0
    radius_ref = 1.5
    v_min_ref = 0.4
    v_max_ref = 1.6
    omega_std_ref = 0.8

    tg = TrajectoryGenerator()  # Generatore delle traiettorie
    sim = build_simulator()  # Simulatore con robot iniziale di default

    histories = []  # Lista delle storie [x,y,theta] per ogni traiettoria (complete)
    titles = []  # Titoli da mostrare nel carosello
    commands_list = []  # Lista parallela dei comandi (v, omega) per ogni traiettoria (complete)

    def _run_case(title: str, vs, omegas):
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append(title)

    # 1) Rettilinea (v costante)
    t_straight = 20.0
    v = v_ref
    vs, omegas = tg.straight(v=v, T=t_straight, dt=dt)
    _run_case("Rettilinea (v costante)", vs, omegas)

    # 2) Rettilinea (v variabile)
    t_straight_var = 20.0
    v_min, v_max = v_min_ref, v_max_ref
    vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=t_straight_var, dt=dt, phase=0.0)
    _run_case("Rettilinea (v variabile)", vs, omegas)

    # 3) Circolare (v costante) — 1 giro intero
    v = v_ref
    r_ref = radius_ref
    period = (2.0 * math.pi * r_ref) / max(v, 1e-9)
    n_steps = max(1, int(round(period / dt)))
    t_circle = n_steps * dt
    vs, omegas = tg.circle(v=v, radius=r_ref, T=t_circle, dt=dt)
    _run_case("Circolare (v costante)", vs, omegas)

    # 4) Circolare (v variabile) — 1 giro intero
    v_min, v_max = v_min_ref, v_max_ref
    v_mid = 0.5 * (v_min + v_max)
    period_var = (2.0 * math.pi * r_ref) / max(v_mid, 1e-9)
    n_steps_var = max(1, int(round(period_var / dt)))
    t_circle_var = n_steps_var * dt
    vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=r_ref, T=t_circle_var, dt=dt, phase=0.0)
    _run_case("Circolare (v variabile)", vs, omegas)

    # 5) Traiettoria a 8 — ciclo completo con raggio maggiore per separare i lobi
    v = v_ref
    r_eight = 1.8
    period_eight = (4.0 * math.pi * r_eight) / max(v, 1e-9)
    n_steps_eight = max(2, int(round(period_eight / dt)))
    if n_steps_eight % 2 == 1:
        n_steps_eight += 1
    t_eight = (n_steps_eight - 1e-9) * dt
    vs, omegas = tg.eight(v=v, radius=r_eight, T=t_eight, dt=dt)
    _run_case("Traiettoria a 8", vs, omegas)

    # 6) Random walk
    t_rw = 40.0
    v_mean = v_ref
    omega_std = omega_std_ref
    vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=t_rw, dt=dt, seed=456)
    _run_case("Random walk", vs, omegas)

    # Costruisci ambienti specifici per ciascuna traiettoria (usando le storie complete)
    envs = setup_environments_per_trajectory(histories, titles)

    # Istanzia LiDAR per-caso con portata adattiva
    lidars = _build_lidars_for_cases(envs, titles)

    return histories, titles, commands_list, envs, lidars

def voxel_downsample_2d(points, voxel_size=0.05):
    """
    Downsample di punti 2d usando Voxel Grid Filter di Open3D.
    Coverte i punti 2D in 3D ponendo z=0, applica i filtri e ritorna in 2 dimensioni.

    Args:
        points:
        voxel_size:  dimensione del voxel in metri (voxel è un quadrato di lato voxel_size)

    Returns:

    """

    if len(points) == 0:
        return points

    # Converte Nx2 → Nx3 aggiungendo z=0
    pts3d = np.hstack([points, np.zeros((len(points), 1))])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)

    pcd_down = pcd.voxel_down_sample(voxel_size)

    down = np.asarray(pcd_down.points)
    return down[:, :2]  # Torna in 2D


def main():
    args = parse_args()

    # Attiva tee su file per duplicare l'output della console in un .txt della sessione
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    _log_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    try:
        os.makedirs(_log_dir, exist_ok=True)
    except OSError:
        # fallback alla root del progetto se non riesce a creare logs/
        _log_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    _ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    _log_path = os.path.join(_log_dir, f"run_output_{_ts}.txt")
    _log_file = open(_log_path, 'w', encoding='utf-8', newline='', buffering=1)
    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)

    try:
        # Compatibilità: se viene passato il flag deprecato, forza viewer_mode a 'grid'
        if getattr(args, "viewer_icp_grid", False):
            print("[AVVISO] --viewer-icp-grid è deprecato; usa --viewer-mode grid. Forzo viewer_mode=grid.")
            args.viewer_mode = "grid"

        # Pulisci vecchie immagini per evitare accumulo: trajectories, scans, scans_polar
        try:
            visualizer.cleanup_output_images(subfolders=("trajectories", "scans", "scans_polar", "icp"),
                                             remove_root=False)
        except OSError as e:
            print(f"[main] Avviso: impossibile pulire cartelle immagini: {e}")

        # --- Generazione traiettorie e ambienti per i casi di test ---

        dt = 0.05  # Passo temporale di integrazione (Eulero)
        histories, titles, commands_list, envs, lidars = build_cases_and_envs(dt)

        # ----------------
        #CALCOLO ODOMETRIA PURA
        # ----------------

        # Parametri rumore odometrico (deviazioni standard)
        ODOM_NOISE_V = 0.05
        ODOM_NOISE_OMEGA = 0.05

        odom_histories = []

        for hist, cmds in zip(histories, commands_list):
            start = np.asarray(hist[0], dtype=float)  # stessa posa iniziale del GT
            odom = compute_odometry_from_commands(
                cmds,
                dt=dt,
                start_pose=start,
                noise_v=ODOM_NOISE_V,
                noise_omega=ODOM_NOISE_OMEGA
            )
            odom_histories.append(odom)

        # ----------------
        # SCAN-TO-MAP: se il flag è presente, esegue scan-to-map per ogni caso usando RAW e INIT come stime iniziali,
        # -----------------

        # --- Variabili per risultati scan-to-map (RAW e INIT) ---

        scan2map_histories_world_raw = None
        scan2map_histories_world_init = None
        scan2map_ncases = 0  # per viewer

        if getattr(args, "run_scan2map", False):
            print("\n[SCAN-TO-MAP] Avvio calcolo scan-to-map (RAW + INIT)...", flush=True)

            # --- Parametri scan-to-map (globali per tutti i casi) ---

            _max_iter = 20  # massimo numero di iterazioni ICP
            _tolerance = 1e-5  # soglia di convergenza ICP (cambio minimo in RMSE)
            _maxcorr = 0.30  # distanza massima per considerare una corrispondenza valida (metri)
            icp_scan_interval = 0.1  # intervallo in cui aggiorno la mappa con nuove scansioni (secondi)

            # --- Criteri di accettazione (gating) ---
            '''
            _corr_* # Numero minimo di corrispondenze accettabili nell'ultima iterazione ICP per considerare l'allineamento valido.
            _thr_* # RMSE massimo accettabile per considerare l'allineamento ICP valido. 
            '''

            _corr_raw = 30
            _thr_raw = 0.25
            _corr_init = 40
            _thr_init = 0.20

            # liste per memorizzare le traiettrie

            scan2map_histories_world_raw = []
            scan2map_histories_world_init = []

            cases_iter = list(zip(histories, titles, envs, lidars))

            scan2map_ncases = len(cases_iter)

            try:
                visualizer.ensure_icp_dirs('error_over_time', 'trajectories',
                                           'scan2map_overlays')  # creo cartelle per immagini scan-to-map (RAW e INIT) e grafico error_over_time
            except OSError:
                pass

            # loop sui casi
            for idx, (case_hist, case_title, case_env, case_lid) in enumerate(cases_iter):
                base_slug = _slugify_local(case_title)

                _step = max(1, int(round(icp_scan_interval / max(1e-9, float(
                    dt)))))  # messo 1e-9 per evitare divisione per zero se dt piccolo

                # 1) prima scansione
                scan0 = case_lid.scan_hits(case_hist[0], case_env, frame="local")

                odom_traj = odom_histories[idx] # odometria ottenuta integrando i comandi, con rumore
                odom_samples = [odom_traj[0].copy()] # salvo prima posa odometrica per confronto error_over_time (posizione iniziale)

                if len(scan0) < 10:  # se scansione troppo vuota, inizializzo traiettorie vuote
                    traj_map_raw = np.zeros((1, 3), dtype=float)
                    traj_map_init = np.zeros((1, 3), dtype=float)
                    real_samples = np.asarray(case_hist[:1], dtype=float)
                    odom_samples = np.asarray(odom_samples, dtype=float)
                else:
                    map_raw = scan0.copy()
                    map_init = scan0.copy()

                    # in ogni ciclo inizializzo per traiettoria successiva
                    R_raw_stim = np.eye(2)
                    t_raw_stim = np.zeros(2)

                    R_init_stim = np.eye(2)
                    t_init_stim = np.zeros(2)

                    traj_map_raw = [np.array([0.0, 0.0, 0.0], dtype=float)]
                    traj_map_init = [np.array([0.0, 0.0, 0.0], dtype=float)]
                    real_samples = [case_hist[0].copy()]

                    # 2) loop scan2map: allinea ogni _step frame usando RAW e INIT, con stima iniziale di INIT basata su odometria
                    for k in range(_step, len(case_hist),
                                   _step):

                        scan_k = case_lid.scan_hits(case_hist[k], case_env,
                                                    frame="local")
                        if len(scan_k) < 10:
                            continue

                        # ---------------- pipeline RAW ----------------
                        out_raw = run_icp_scan_to_map_pair(
                            map_world=map_raw,  # mappa aggiornata fino a k-1
                            curr_scan_local=scan_k,
                            init_R=None, init_t=None,
                            raw_init_R=R_raw_stim, raw_init_t=t_raw_stim,
                            # stime iniziali RAW basate su stime precedenti
                            max_iterations=_max_iter,
                            tolerance=_tolerance,  #
                            max_correspondence_distance=_maxcorr
                        )

                        # gating RAW
                        rmse_r = float(out_raw["raw"]["rmse"])
                        nc_r = int(out_raw["raw"]["n_corr_last"])
                        ok_raw = (nc_r >= _corr_raw) and (rmse_r <= _thr_raw)

                        if ok_raw:  # se ok aggiorno
                            map_raw = out_raw["map_new_raw"]
                            if len(map_raw) > 10000: # se la mappa è troppo grande, downsample per mantenere velocità ICP
                                map_raw = voxel_downsample_2d(map_raw, voxel_size=0.03)
                            R_raw_stim = out_raw["raw"]["R"]
                            t_raw_stim = out_raw["raw"]["t"]

                        th_raw = float(np.arctan2(R_raw_stim[1, 0], R_raw_stim[
                            0, 0]))  # conversione da matrice a angolo (rotazione)
                        traj_map_raw.append(
                            np.array([float(t_raw_stim[0]), float(t_raw_stim[1]), float(th_raw)], dtype=float))

                        # ---------------- INIT pipeline (map_init) ----------------
                        prev_o = odom_traj[k - _step]
                        curr_o = odom_traj[k]
                        R_delta, t_delta = compute_relative_transform_from_odometry(prev_o, curr_o)

                        # calcolo predizione della posa iniziale di ICP
                        R_pred = R_init_stim @ R_delta  # moltiplico la stima precedente per il delta odometrico
                        t_pred = t_init_stim + (
                                    R_init_stim @ t_delta)  # traslo la stima precedente del delta odometrico ruotato dalla stima precedente

                        # ---------------- pipeline INIT ----------------
                        out_init = run_icp_scan_to_map_pair(
                            map_world=map_init,
                            curr_scan_local=scan_k,
                            init_R=R_pred, init_t=t_pred,  # a differenza di RAW, uso la predizione basata su odometria
                            max_iterations=_max_iter,
                            tolerance=_tolerance,
                            max_correspondence_distance=_maxcorr
                        )

                        rmse_i = float(out_init["init"]["rmse"])
                        nc_i = int(out_init["init"]["n_corr_last"])

                        ok_init = (nc_i >= _corr_init) and (rmse_i <= _thr_init)  # gating INIT

                        if ok_init:
                            map_init = out_init["map_new_init"]
                            if len(map_init) > 10000:
                                map_init = voxel_downsample_2d(map_init, voxel_size=0.03)
                            R_init_stim = out_init["init"]["R"]
                            t_init_stim = out_init["init"]["t"]
                            th_init: float = _wrap_angle(float(out_init["init"]["alpha_rad"]))  # angolo di rotazione
                        else:
                            # se ICP non va bene, mantengo la stima basata su odometria (non aggiornata con ICP)
                            R_init_stim = R_pred
                            t_init_stim = t_pred
                            th_init = _wrap_angle(float(np.arctan2(R_init_stim[1, 0], R_init_stim[0, 0])))

                        # salvo le pose stimata INIT
                        traj_map_init.append(
                            np.array([float(t_init_stim[0]), float(t_init_stim[1]), float(th_init)], dtype=float))

                        real_samples.append(case_hist[k].copy())  # salva posa reale per confronto error_over_time
                        odom_samples.append(odom_traj[k].copy())  # salva posa odometrica per confronto error_over_time

                        if (len(traj_map_raw) % 10) == 0:  # ogni 10 iterazioni salvo overlay scan-to-map per RAW e INIT
                            try:
                                visualizer.ensure_icp_dirs('scan2map_overlays')
                            except OSError:
                                pass

                            save_scan2map_overlay(
                                map_raw,
                                out_raw["raw"]["src_transformed"],
                                f"{case_title} – RAW (k={k})",
                                visualizer.icp_out_path('scan2map_overlays', f"{base_slug}_raw_k{k}.png")
                            )
                            save_scan2map_overlay(
                                map_init,
                                out_init["init"]["src_transformed"],
                                f"{case_title} – INIT (k={k})",
                                visualizer.icp_out_path('scan2map_overlays', f"{base_slug}_init_k{k}.png")
                            )

                    # a fine del ciclo, converto le liste di pose in array numpy per facilitare manipolazioni successive
                    traj_map_raw = np.vstack(traj_map_raw)
                    traj_map_init = np.vstack(traj_map_init)
                    real_samples = np.vstack(real_samples)
                    odom_samples = np.vstack(odom_samples)

                # 3) allinea le traiettorie da frame mappa a frame mondo usando la prima scansione (prima posa reale)
                traj_world_raw = _apply_world_transform(traj_map_raw, case_hist[0])
                traj_world_init = _apply_world_transform(traj_map_init, case_hist[0])

                scan2map_histories_world_raw.append(traj_world_raw)
                scan2map_histories_world_init.append(traj_world_init)

                save_scan2map_trajectories(
                    traj_gt=case_hist,  # GT world
                    traj_init=traj_world_init,  # ICP con odomtria
                    traj_raw=traj_world_raw,  # ICP RAW
                    title=f"{case_title} – Scan-to-map",
                    out_path=visualizer.icp_out_path(
                        'trajectories',
                        f"{base_slug}_scan2map_xy.png"
                    ),
                    traj_odom = odom_histories[idx][:len(case_hist)] # traiettoria odometrica da sovrapporre con lunghezza GT
                )

                save_error_over_time(
                    real_samples,  # campioni reali (world)
                    traj_world_init,  # INIT (filtrata)
                    traj_world_raw,  # RAW
                    case_title,
                    visualizer.icp_out_path('error_over_time', f"{base_slug}_scan2map_error_over_time.png"),
                    dt=icp_scan_interval,
                    odom_traj= odom_samples # odom campionata agli stessi k dello scan-to-map
                )

            print("[SCAN-TO-MAP] Completato (RAW + INIT).", flush=True)

        # Passi per disegnare la posa del robot (in ordine dei casi)
        show_steps = [80, 80, 40, 40, 120, 120]

        # --------- Barra di progresso unica (tqdm) per tutti i salvataggi ---------
        # Calcola numero totale di immagini da salvare: traiettorie + (scans + polari) per ciascun caso
        step_idx = max(1, int(round(float(args.scan_interval) / max(1e-9, float(dt)))))
        total_steps = len(histories)  # una per traiettoria
        for hist in histories:
            n = int(len(hist))
            scans_count = (0 if n <= 0 else ((n - 1) // step_idx + 1))
            total_steps += 2 * scans_count  # scans punti + scans polari

        def _run_all_saves(progress_cb_fn):
            # Salva immagini di traiettoria (usa progress globale)
            visualizer.save_trajectories_images(
                histories, titles,
                show_orient_every=show_steps,
                environment=envs,
                fit_to='environment',
                progress_cb=progress_cb_fn,
                quiet=args.quiet,
            )
            # Salva scansioni (punti) e polari per ciascun caso (usa progress globale)
            for save_hist, save_title, save_env, save_lid in zip(histories, titles, envs, lidars):
                visualizer.save_lidar_scans_images(
                    save_hist, save_title, save_lid, save_env, dt,
                    interval_s=float(args.scan_interval),
                    progress_cb=progress_cb_fn,
                    quiet=args.quiet,
                )
                visualizer.save_lidar_polar_images(
                    save_hist, save_title, save_lid, save_env, dt,
                    interval_s=float(args.scan_interval),
                    include_misses=True,
                    progress_cb=progress_cb_fn,
                    quiet=args.quiet,
                )

        if _tqdm is not None:
            with _tqdm(total=total_steps, desc="Salvataggio immagini", unit="img", ncols=90) as pbar:
                progress_cb = lambda _cur, _tot: pbar.update(1)
                _run_all_saves(progress_cb)
        else:
            start_t = time.time()
            state = {"done": 0}
            width = 36

            def _eta(sec: float) -> str:
                m, s = divmod(int(round(max(0.0, sec))), 60)
                return f"{m:02d}:{s:02d}"

            def _ascii_cb(_c, _t):
                state["done"] += 1
                done = min(state["done"], total_steps)
                progress_fraction = done / max(1, total_steps)
                filled = int(round(width * progress_fraction))
                bar = '#' * filled + '-' * (width - filled)
                elapsed = time.time() - start_t
                per_step = elapsed / max(1, done)
                remain = per_step * max(0, total_steps - done)
                print(f"\rSalvataggio immagini [{bar}] {done}/{total_steps}  ETA {_eta(remain)}", end='', flush=True)
                if done >= total_steps:
                    print()

            _run_all_saves(_ascii_cb)
        # --------- Fine barra di progresso unica ---------

        # Calcola collisioni via LiDAR solo se richiesto
        stop_indices = [None] * len(histories)
        stop_fractions = [None] * len(histories)
        if not args.skip_collision:
            stop_indices = []
            stop_fractions = []
            for hist, env, lid in zip(histories, envs, lidars):
                kcol, frac = _first_collision_via_lidar(hist, env, lid, body_length=0.40, body_width=0.20)
                stop_indices.append(kcol)
                stop_fractions.append(frac)

        # ===== Ricostruzione traiettorie da LOG corrente (se richiesto) =====
        icp_histories_from_log: Optional[List[np.ndarray]] = None
        icp_raw_from_log: Optional[List[np.ndarray]] = None
        icp_filt_from_log: Optional[List[np.ndarray]] = None
        if not args.skip_icp:
            # Assicura che le stampe ICP siano state flushate su file prima di leggere
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except (IOError, OSError):
                # Gestisce errori di flush degli stream
                pass
            triplets = parse_icp_triplets_from_log(_log_path, n_cases=len(titles))
            icp_histories_from_log = []
            icp_raw_from_log = []
            icp_filt_from_log = []
            for idx, item in enumerate(triplets):
                real = item.get('real')
                icp = item.get('icp')
                raw = item.get('raw')
                # ognuno può mancare: inserisci fallback minimale
                real_f = real if isinstance(real, np.ndarray) and real.size > 0 else np.zeros((1, 3), dtype=float)
                raw_f = raw if isinstance(raw, np.ndarray) and raw.size > 0 else np.zeros((1, 3), dtype=float)
                icp_f = icp if isinstance(icp, np.ndarray) and icp.size > 0 else np.zeros((1, 3), dtype=float)

                # Applica loop closure correction SOLO per traiettorie circolari (idx 2, 3)
                # NON applicare al caso 4 (otto) perché ha una forma diversa
                # is_circular = idx in (2, 3)
                # if is_circular and len(icp_f) > 10:
                #   icp_f = apply_loop_closure_correction(icp_f, is_circular=True, closure_threshold=0.5)
                #  raw_f = apply_loop_closure_correction(raw_f, is_circular=True, closure_threshold=0.5)

                # Allineamento opzionale al mondo: usa la prima posa reale del caso simulato
                if getattr(args, 'viewer_log_align_world', False) and idx < len(histories) and len(histories[idx]) > 0:
                    base = np.asarray(histories[idx][0], dtype=float)
                    real_f = _apply_world_transform(real_f, base)
                    raw_f = _apply_world_transform(raw_f, base)
                    icp_f = _apply_world_transform(icp_f, base)
                icp_histories_from_log.append(real_f)
                icp_raw_from_log.append(raw_f)
                icp_filt_from_log.append(icp_f)

            # Salva grafici degli errori nel tempo per ogni caso
            print("[Grafici Errori] Salvataggio grafici errori ICP nel tempo...")
            for idx, (real_traj, icp_traj, raw_traj, title) in enumerate(
                    zip(icp_histories_from_log, icp_filt_from_log, icp_raw_from_log, titles)):
                if real_traj is not None and icp_traj is not None and raw_traj is not None:
                    base_slug = _slugify_local(title)
                    save_error_over_time(
                        real_traj, icp_traj, raw_traj,
                        title,
                        visualizer.icp_out_path('error_over_time', f"{base_slug}_error_over_time.png"),
                        dt=dt,
                        odom_traj = odom_histories[idx]
                    )

        # ===== Fine ricostruzione =====

        # Mostra viewer dopo tutti i salvataggi e (opzionale) ICP
        if not args.skip_viewer:
            # Scegli set di traiettorie per il viewer
            # Se abbiamo ricostruito dal log: usa real(icp_histories_from_log) come pannello "Reale – ... (ICP da log)"
            # e passa raw/filtrato al viewer per i due pannelli ICP
            if icp_histories_from_log is not None:
                viewer_histories = icp_histories_from_log
                viewer_titles = titles
                viewer_cmds = None
                viewer_raw = icp_raw_from_log
                viewer_filt = icp_filt_from_log
            else:
                viewer_histories = histories
                viewer_titles = titles
                viewer_cmds = commands_list
                viewer_raw = None
                viewer_filt = None

            if scan2map_histories_world_raw is not None and scan2map_histories_world_init is not None:
                viewer_raw = scan2map_histories_world_raw
                viewer_filt = scan2map_histories_world_init

            if args.viewer_mode == "grid":
                visualizer.show_trajectories_icp_grid(
                    viewer_histories,
                    viewer_titles,
                    environment=envs,
                    fit_to='environment',
                )
            else:
                visualizer.show_trajectories_carousel(
                    viewer_histories,
                    viewer_titles,
                    show_orient_every=show_steps,
                    _save_each=False,
                    _commands_list=viewer_cmds,
                    _dts=dt,
                    _show_info=True,
                    environment=envs,
                    _fit_to='environment',
                    _stop_indices=stop_indices,
                    _stop_fractions=stop_fractions,
                    lidar=lidars,
                    _show_lidar=True,
                    _lidar_every=int(max(1, args.viewer_lidar_every)),
                    icp_raw_histories=viewer_raw,
                    icp_filt_histories=viewer_filt,
                )


    finally:
        # Ripristina stream originali e chiudi il file di log
        try:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        finally:
            try:
                _log_file.close()
            except (IOError, OSError):
                # Gestisce errori durante la chiusura del file
                pass


if __name__ == "__main__":
    main()
