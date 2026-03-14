import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import open3d as o3d


def to_o3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converte un insieme di punti 2D in una point cloud Open3D,
    aggiungendo una coordinata z nulla a ogni punto.

    Args:
        points: array Nx2 contenente punti 2D nel formato (x, y)

    Returns:
        Oggetto o3d.geometry.PointCloud contenente gli stessi punti
        rappresentati in 3D come (x, y, z=0).
        Se l'array è vuoto, ritorna una point cloud vuota.
    """
    points = np.asarray(points, dtype=float)
    pc = o3d.geometry.PointCloud()

    if len(points) == 0:
        pc.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        return pc

    pts3 = np.hstack([points, np.zeros((len(points), 1), dtype=float)])
    pc.points = o3d.utility.Vector3dVector(pts3)
    return pc


def wrap_angle(angle: float) -> float:
    """
    Normalizza un angolo espresso in radianti nell'intervallo [-pi, pi].

    Args:
        angle: angolo in radianti da normalizzare

    Returns:
        Angolo equivalente normalizzato nell'intervallo [-pi, pi].
    """
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    """
    Calcola la differenza angolare assoluta minima tra due angoli.

    Args:
        a: primo angolo in radianti
        b: secondo angolo in radianti

    Returns:
        Differenza angolare assoluta minima tra a e b, nell'intervallo [0, pi].
    """
    return abs(wrap_angle(float(a) - float(b)))


def pose_distance_xy(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calcola la distanza euclidea nel piano tra due pose,
    considerando solo le componenti x e y e ignorando l'orientamento.

    Args:
        p1: prima posa nel formato (x, y, theta)
        p2: seconda posa nel formato (x, y, theta)

    Returns:
        Distanza euclidea tra le componenti planari delle due pose.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def pose_to_rt(pose: np.ndarray):
    """
    Converte una posa 2D nel formato (x, y, theta) in una
    matrice di rotazione 2x2 e in un vettore di traslazione 2D.

    Args:
        pose: array contenente la posa del robot nel formato (x, y, theta)

    Returns:
        R: matrice di rotazione 2x2 associata all'angolo theta
        t: vettore di traslazione 2D corrispondente a (x, y)
    """
    x, y, th = map(float, np.asarray(pose, dtype=float)[:3])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s],
                  [s, c]], dtype=float)
    t = np.array([x, y], dtype=float)
    return R, t


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Applica una rotazione e una traslazione a un insieme di punti 2D.

    Ogni punto viene prima ruotato usando la matrice R e poi
    traslato usando il vettore t.

    Args:
        points: array Nx2 di punti 2D (x, y) da trasformare
        R: matrice di rotazione 2x2
        t: vettore di traslazione (x, y)

    Returns:
        Array Nx2 contenente i punti trasformati.
        Se l'array di input è vuoto, viene restituito un array vuoto.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.copy()
    return (R @ pts.T).T + t


@dataclass
class Keyframe:
    """
    Struttura dati che rappresenta un keyframe usato per la loop closure.

    Attributi:
        k: indice temporale del keyframe
        pose: posa associata al keyframe nel formato (x, y, theta)
        scan_local: scansione LiDAR locale acquisita in quel keyframe
    """
    k: int
    pose: np.ndarray
    scan_local: np.ndarray


def add_keyframe(
        keyframes: List[Keyframe],
        k: int,
        pose: np.ndarray,
        scan_local: np.ndarray,
) -> Keyframe:
    """
    Crea un nuovo keyframe e lo aggiunge alla lista dei keyframe esistenti.

    Args:
        keyframes: lista dei keyframe già presenti
        k: indice temporale associato al nuovo keyframe
        pose: posa del robot nel formato (x, y, theta)
        scan_local: scansione locale associata al keyframe come array Nx2

    Returns:
        Il nuovo keyframe creato e aggiunto alla lista.
    """
    kf = Keyframe(
        k=int(k),
        pose=np.asarray(pose, dtype=float).copy(),
        scan_local=np.asarray(scan_local, dtype=float).copy(),
    )
    keyframes.append(kf)
    return kf


def keyframe_scan_in_map(kf: Keyframe) -> np.ndarray:
    """
    Trasforma la scansione locale di un keyframe nel sistema di riferimento mappa.

    Args:
        kf: keyframe di cui trasformare la scansione locale

    Returns:
        Array Nx2 contenente la scansione del keyframe espressa nel frame mappa.
    """
    R, t = pose_to_rt(kf.pose)
    return transform_points(kf.scan_local, R, t)


def should_create_keyframe(
        curr_pose: np.ndarray,
        last_kf_pose: Optional[np.ndarray],
        dist_thresh: float = 0.5,
        angle_thresh: float = np.deg2rad(15.0),
) -> bool:
    """
    Determina se la posa corrente giustifica la creazione di un nuovo keyframe.

    La decisione viene presa confrontando la posa corrente con quella
    dell'ultimo keyframe creato. Un nuovo keyframe viene generato se:
    - la distanza planare supera una certa soglia, oppure
    - la differenza angolare supera una certa soglia.

    Args:
        curr_pose: posa corrente del robot nel formato (x, y, theta)
        last_kf_pose: posa dell'ultimo keyframe nel formato (x, y, theta),
            oppure None se non esiste ancora alcun keyframe
        dist_thresh: soglia minima sulla distanza planare
        angle_thresh: soglia minima sulla differenza angolare

    Returns:
        True se deve essere creato un nuovo keyframe, False altrimenti.
        Se non esiste ancora un keyframe precedente, ritorna True.
    """
    if last_kf_pose is None:
        return True

    curr_pose = np.asarray(curr_pose, dtype=float)
    last_kf_pose = np.asarray(last_kf_pose, dtype=float)

    dist = pose_distance_xy(curr_pose, last_kf_pose)
    dtheta = angle_diff(curr_pose[2], last_kf_pose[2])

    return (dist >= dist_thresh) or (dtheta >= angle_thresh)


def find_loop_candidate(
        curr_pose: np.ndarray,
        keyframes: List[Keyframe],
        curr_k: int,
        min_separation: int = 40,
        search_radius: float = 1.0,
) -> Optional[Keyframe]:
    """
    Cerca un keyframe candidato per la loop closure.

    Vengono considerati solo i keyframe sufficientemente lontani nel tempo
    rispetto all'indice corrente. Tra quelli entro `search_radius`, viene
    selezionato il più vicino in distanza planare alla posa corrente.

    Args:
        curr_pose: posa corrente del robot nel formato (x, y, theta)
        keyframes: lista dei keyframe disponibili
        curr_k: indice temporale corrente
        min_separation: distanza minima in step temporali tra il frame corrente
            e il keyframe candidato
        search_radius: raggio massimo di ricerca nel piano

    Returns:
        Il keyframe candidato migliore, oppure None se non viene trovato
        alcun candidato valido.
    """
    curr_pose = np.asarray(curr_pose, dtype=float)
    best_kf = None
    best_dist = float("inf")

    for kf in keyframes:
        if (curr_k - kf.k) < min_separation:
            continue

        d = float(np.linalg.norm(curr_pose[:2] - kf.pose[:2]))
        if d < search_radius and d < best_dist:
            best_dist = d
            best_kf = kf

    return best_kf


def try_loop_closure(
        curr_scan_local: np.ndarray,
        curr_pose_pred: np.ndarray,
        candidate_kf: Keyframe,
        max_corr_dist: float = 0.3,
        max_rmse: float = 0.15,
        min_fitness: float = 0.3,
):
    """
    Tenta una loop closure tra la scansione corrente e la scansione
    di un keyframe candidato usando ICP point-to-point di Open3D.

    La scansione corrente viene allineata alla scansione del keyframe
    espressa nel frame mappa, partendo da una stima iniziale della posa corrente.

    Il risultato viene accettato solo se:
    - l'RMSE finale è sufficientemente basso,
    - la fitness finale è sufficientemente alta.

    Args:
        curr_scan_local: scansione locale corrente come array Nx2
        curr_pose_pred: stima iniziale della posa corrente nel formato (x, y, theta)
        candidate_kf: keyframe candidato con cui tentare la loop closure
        max_corr_dist: distanza massima per accettare corrispondenze ICP
        max_rmse: soglia massima di RMSE per accettare il risultato
        min_fitness: soglia minima di fitness per accettare il risultato

    Returns:
        Un dizionario con:
            "pose_corrected": posa corretta stimata da ICP
            "rmse": errore quadratico medio finale
            "fitness": fitness finale dell'allineamento

        Ritorna None se il risultato non soddisfa i criteri di accettazione.
    """
    src = to_o3d_cloud(curr_scan_local)
    tgt = to_o3d_cloud(keyframe_scan_in_map(candidate_kf))

    x, y, th = map(float, curr_pose_pred)
    c, s = np.cos(th), np.sin(th)

    init = np.eye(4, dtype=float)
    init[0, 0] = c
    init[0, 1] = -s
    init[1, 0] = s
    init[1, 1] = c
    init[0, 3] = x
    init[1, 3] = y

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr_dist,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    rmse = float(result.inlier_rmse)
    fitness = float(result.fitness)

    if rmse > max_rmse or fitness < min_fitness:
        return None

    T = np.asarray(result.transformation, dtype=float)
    corrected_pose = np.array(
        [T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])],
        dtype=float
    )

    return {
        "pose_corrected": corrected_pose,
        "rmse": rmse,
        "fitness": fitness,
    }
