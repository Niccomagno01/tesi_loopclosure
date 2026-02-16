"""Visualizer di traiettorie del robot

Funzionalità principali:
- Disegno della traiettoria con simboli del robot a intervalli regolari.
- Salvataggio immagini statiche in PNG nella cartella img/ con nomi basati sul titolo e timestamp.
- Viewer interattivo “carousel” con pulsanti (precedente/play/successivo) e
  pannello informazioni opzionale. Nel viewer i simboli del robot lungo la traiettoria compaiono
  progressivamente quando il robot mobile raggiunge quelle posizioni; le immagini salvate invece
  li mostrano tutti, come in precedenza.

Note implementative:
- Le routine di disegno del robot sono centralizzate (draw_robot), e funzioni di supporto calcolano
  dimensioni coerenti con l’estensione della traiettoria.
- Il viewer usa un timer di Matplotlib per far avanzare i frame; gli “artisti” grafici creati
  per il robot mobile vengono rimossi e ricreati ogni frame per un aggiornamento pulito.
- Per rendere snello il codice, le icone dei pulsanti usano simboli Unicode (compatibili su Windows)
  al posto di patch disegnate manualmente.
"""

# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt  # API principale per creare figure/assi e tracciare linee/frecce
from matplotlib.patches import Circle, Rectangle  # Primitive grafiche 2D per centro/ruote e corpo del robot
import numpy as np  # Calcolo numerico: array, trigonometria, differenze, range
from pathlib import Path  # Percorsi portabili per cartella img/ e file PNG
import re  # Normalizzazione del titolo in un nome file sicuro (slugify minimo)
from datetime import datetime  # Timestamp per nomi univoci ed evitare sovrascritture
from matplotlib.widgets import Button  # Pulsanti UI per navigazione e Play/Pausa
from matplotlib import transforms as mtransforms  # Trasformazioni affini: rotazione/traslazione delle patch
from typing import Optional, List, Sequence, Union  # Annotazioni di tipo per migliori suggerimenti e linting
from matplotlib.text import Text  # Artista di testo (pannello info, legenda)
from contextlib import suppress  # Ignora eccezioni non critiche in operazioni best-effort
from matplotlib.artist import Artist  # Tipo base di tutti gli elementi disegnabili (patch, arrow, ecc.)
from environment import Environment  # Per disegnare confini e ostacoli
from lidar import Lidar  # Tipo del sensore per visualizzazione raggi
import shutil  # Per pulire cartelle di output delle immagini
# Nuovi import per ICP grid
from typing import Dict
from icp import run_icp_over_history  # Import della funzione ICP

# Eccezioni comuni gestite in modo sicuro (piu' strette di Exception)
COMMON_EXC = (AttributeError, ValueError, TypeError, RuntimeError)


# Helper per rimuovere in sicurezza un artista matplotlib (gestisce None ed eccezioni)
def _safe_remove_artist(artist: Optional[object]):
    """Prova a rimuovere un artista (patch/annotazione ecc.) ignorando errori e None."""
    if artist is not None and hasattr(artist, 'remove'):
        with suppress(*COMMON_EXC):
            artist.remove()  # type: ignore[attr-defined]


def _clear_artists(artists: Optional[List[Artist]]):
    """Rimuove tutti gli artisti in lista e la svuota in-place (ignora errori)."""
    if not artists:
        return
    try:
        for a in list(artists):
            _safe_remove_artist(a)
    finally:
        try:
            artists.clear()
        except COMMON_EXC:
            pass


def _rect_dims_from_radius(robot_radius: float):
    """Deriva dimensioni del rettangolo dal parametro di scala robot_radius.
    - width (lato corto, fronte): ~2× robot_radius
    - length (lato lungo, direzione di marcia): ~4× robot_radius
    Ritorna (width, length)."""
    width = 2.0 * robot_radius
    length = 4.0 * robot_radius
    return width, length


def _wheel_params(robot_radius: float):
    """Parametri delle rotelle a partire dalla scala del robot.
    Ritorna (wheel_radius, offset_out) dove offset_out è l'offset del centro ruota
    verso l'esterno rispetto alla fiancata (in coordinate locali)."""
    wheel_radius = 0.22 * robot_radius
    offset_out = 0.15 * robot_radius
    return wheel_radius, offset_out


def draw_robot(ax, state, robot_radius=0.1, color='tab:blue', dir_len=None, arrow_color='orange', center_color='orange',
               wheel_facecolor='white', wheel_edgecolor='k') -> List[Artist]:
    """Disegna il robot come rettangolo orientato con freccia e ruote.

    Parametri principali:
    - ax: axes Matplotlib su cui disegnare
    - state: [x, y, theta] posa del robot nel mondo
    - robot_radius: scala complessiva (controlla dimensioni corpo/ruote/freccia)
    - color, arrow_color, center_color: colori per corpo, freccia, pallino centrale

    Ritorna: lista degli artisti creati (utile per rimuoverli al frame successivo).
    """
    x, y, th = state
    artists: List[Artist] = []

    # Corpo rettangolare: lato lungo allineato con l'orientamento (theta)
    width, length = _rect_dims_from_radius(robot_radius)

    # Definisco il rettangolo nel frame locale (centro = 0) e applico rotazione+traslazione
    rect = Rectangle((-length / 2.0, -width / 2.0), length, width, linewidth=1.0, facecolor=color, alpha=0.3,
                     edgecolor='k', zorder=3)
    trans = mtransforms.Affine2D().rotate(th).translate(x, y) + ax.transData
    rect.set_transform(trans)
    ax.add_patch(rect)
    artists.append(rect)

    # Rotelle: quattro cerchi vicino alle estremità dei lati lunghi (sempre disegnate)
    w_r, w_off = _wheel_params(robot_radius)
    wheel_long_frac = 0.8  # posizione lungo il lato lungo (80% della semi-lunghezza)
    x_off = wheel_long_frac * (length / 2.0)
    corners = [
        (+x_off, +width / 2.0 + w_off),  # lato superiore, estremità destra
        (-x_off, +width / 2.0 + w_off),  # lato superiore, estremità sinistra
        (+x_off, -width / 2.0 - w_off),  # lato inferiore, estremità destra
        (-x_off, -width / 2.0 - w_off),  # lato inferiore, estremità sinistra
    ]
    for cx, cy in corners:
        wheel = Circle((cx, cy), w_r, facecolor=wheel_facecolor, edgecolor=wheel_edgecolor, linewidth=1.0, zorder=4)
        wheel.set_transform(trans)
        ax.add_patch(wheel)
        artists.append(wheel)

    # Pallino centrale (rende evidente il centro del corpo)
    center_r = 0.25 * robot_radius
    center = Circle((0.0, 0.0), center_r, fill=True, color=center_color, ec='none', zorder=4)
    center.set_transform(trans)
    ax.add_patch(center)
    artists.append(center)

    # Freccia di orientamento (punta nella direzione di marcia)
    if dir_len is None:
        dir_len = 3.0 * robot_radius  # lunghezza default della freccia
    dx = dir_len * np.cos(th)
    dy = dir_len * np.sin(th)
    arr = ax.arrow(
        x,  # punto di partenza (posizione del robot)
        y,
        dx,  # componente x della freccia
        dy,  # componente y della freccia
        head_width=0.3 * robot_radius,
        head_length=0.4 * robot_radius,
        fc=arrow_color,
        ec=arrow_color,
        length_includes_head=True,
        zorder=4,
    )
    # ax.arrow ritorna un artista (FancyArrow) che posso rimuovere in seguito
    if isinstance(arr, Artist):
        artists.append(arr)

    return artists


def _default_save_path(title: str, *, subfolder: Optional[str] = None) -> Path:
    """Costruisce il percorso di salvataggio in img/ (o sotto-cartella) con titolo normalizzato + timestamp.
    - subfolder: percorso relativo dentro img/ (es. 'trajectories' o 'scans/rettilinea_v_costante')
    """
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    if subfolder:
        img_dir = img_dir / subfolder
    img_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify(title)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return img_dir / f"{base}_{stamp}.png"


def _slugify(text: str) -> str:
    """Normalizza un testo per uso in nomi file/cartelle: minuscole, _ e - consentiti."""
    base = (text or '').lower().strip() or 'traiettoria'
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'[^a-z0-9_\-]', '', base)
    return base


# ----------------------- Pulizia output immagini -----------------------

def cleanup_output_images(*, subfolders: Sequence[str] = ("trajectories", "scans", "scans_polar"),
                          remove_root: bool = False) -> None:
    """Elimina le immagini generate in precedenza sotto img/ per avere solo gli output dell'ultimo run.

    - subfolders: sottocartelle di img/ da pulire. Di default: trajectories, scans, scans_polar.
    - remove_root: se True, elimina l'intera cartella img/ e la ricrea vuota.
    """
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    if not img_dir.exists():
        return
    try:
        if remove_root:
            shutil.rmtree(img_dir, ignore_errors=True)
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            for sub in subfolders:
                target = img_dir / sub
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    target.mkdir(parents=True, exist_ok=True)
        print(f"Pulizia immagini completata in: {img_dir}")
    except OSError as e:
        # Non bloccare l'esecuzione in caso di problemi di file system
        print(f"[cleanup_output_images] Avviso: non sono riuscito a pulire completamente {img_dir}: {e}")


# ----------------------- Fine pulizia output immagini -----------------------


def _robot_scale_from_history(history):
    """Deriva una scala per robot/freccia dall'estensione della traiettoria.
    Ritorna (robot_radius, dir_len)."""
    x_range = float(np.ptp(history[:, 0]))  # ampiezza su x
    y_range = float(np.ptp(history[:, 1]))  # ampiezza su y
    ref = max(x_range, y_range, 1.0)  # evita raggio nullo
    robot_radius = max(0.02, 0.012 * ref)  # raggio proporzionale all'estensione
    dir_len = 2.5 * robot_radius  # lunghezza freccia proporzionale al raggio
    return robot_radius, dir_len


def _compute_axes_limits_with_glyphs(history, step, r_robot, d_arrow, env: Optional[Environment] = None, *,
                                     fit_to: str = 'trajectory'):
    """Calcola i limiti degli assi includendo corpo, ruote, punte freccia.

    fit_to:
    - 'trajectory' (default): adatta i limiti alla traiettoria (più stretto, niente dezoom).
    - 'environment': estende i limiti per includere anche i bounds dell'ambiente.
    """
    xs = history[:, 0]
    ys = history[:, 1]
    # Estensione base della traiettoria
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    # Margine legato al corpo + ruote esterne
    width, length = _rect_dims_from_radius(r_robot)
    w_r, w_off = _wheel_params(r_robot)
    body_half_diag = 0.5 * float(np.hypot(length, width))
    wheels_extra = float(w_off + w_r)
    extent_radius = body_half_diag + wheels_extra

    x_min -= extent_radius
    x_max += extent_radius
    y_min -= extent_radius
    y_max += extent_radius

    # Includi punte delle frecce valutate a intervalli e sempre quella finale
    n = len(history)
    step = max(1, int(step))
    indices = list(range(0, n, step))
    if (n - 1) not in indices and n > 0:
        indices.append(n - 1)
    for i in indices:
        x, y, th = map(float, history[i])
        tip_x = float(x + d_arrow * np.cos(th))
        tip_y = float(y + d_arrow * np.sin(th))
        x_min = min(x_min, tip_x)
        x_max = max(x_max, tip_x)
        y_min = min(y_min, tip_y)
        y_max = max(y_max, tip_y)

    # Opzionale: includi i bounds dell'ambiente solo se richiesto
    if fit_to == 'environment' and env is not None and getattr(env, 'bounds', None) is not None:
        try:
            bx, by = env.bounds.exterior.xy  # type: ignore[attr-defined]
            bx_min, bx_max = float(np.min(bx)), float(np.max(bx))
            by_min, by_max = float(np.min(by)), float(np.max(by))
            x_min = min(x_min, bx_min)
            x_max = max(x_max, bx_max)
            y_min = min(y_min, by_min)
            y_max = max(y_max, by_max)
        except (AttributeError, TypeError, ValueError):
            pass

    # Piccolo margine finale per aria attorno al disegno
    pad = 0.02 * max(x_max - x_min, y_max - y_min, 1.0)
    return x_min - pad, x_max + pad, y_min - pad, y_max + pad


# Helper privato per disegnare una singola traiettoria statica sugli axes
# Centralizza la logica ripetuta in plot_trajectory, show_trajectories_carousel e save_trajectories_images
# Restituisce (r_robot, d_arrow) calcolati per la traiettoria

def _plot_static_trajectory_on_axes(
        ax,
        hist: np.ndarray,
        step: int,
        title: Optional[str] = None,
        include_title: bool = True,
        include_axis_labels: bool = True,
        *,
        draw_glyphs: bool = True,
        environment: Optional[Environment] = None,
        fit_to: str = 'trajectory',
):
    """Disegna lo sfondo dell'ambiente (opzionale), la linea della traiettoria e (opzionalmente) i robot statici sparsi.

    - draw_glyphs=False è usato nel viewer interattivo per non mostrare i robot statici
      finché non vengono “rivelati” durante la riproduzione.
    - fit_to controlla se i limiti assi si adattano alla sola traiettoria (default) o includono i bounds dell'ambiente.
    Ritorna (r_robot, d_arrow).
    """
    n = len(hist)
    step = max(1, int(step))

    # Disegna l'ambiente in background, se fornito (bounds e ostacoli)
    if environment is not None:
        environment.plot(ax=ax)

    # Traccia la traiettoria (linea nera) sopra lo sfondo
    ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.5, color='k', zorder=2)
    # Scala robot/freccia coerente con l’estensione
    r_robot, d_arrow = _robot_scale_from_history(hist)

    if draw_glyphs:
        # Disegna i simboli del robot a intervalli regolari
        for i in range(0, n, step):
            if i == 0:
                body_col, arr_col, ctr_col = 'green', 'orange', 'green'  # partenza
            elif i == n - 1:
                body_col, arr_col, ctr_col = 'red', 'orange', 'red'  # arrivo
            else:
                body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'  # punti intermedi
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col,
                       center_color=ctr_col)
        # Assicura il disegno della posa finale anche se non multipla di step
        if n > 0 and ((n - 1) % step != 0 or n == 1):
            draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange',
                       center_color='red')

    # Limiti assi calcolati in base alla scelta di fit
    x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step, r_robot, d_arrow, env=environment, fit_to=fit_to)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Aspetto e labeling
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    if include_axis_labels:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    if include_title and title is not None:
        ax.set_title(title)

    return r_robot, d_arrow


def _build_info_text(
        hist: np.ndarray,
        k_pose: int,
        dt: float,
        commands: Optional[np.ndarray] = None,
        *,
        use_cmd_of_prev: bool = True,
        show_next_pose: bool = False,
) -> str:
    """Crea il testo del pannello info (tempo, velocità, posa).

    - Se sono forniti comandi [v, w], vengono usati; altrimenti v e w sono stimati da differenze finite.
    - show_next_pose permette di mostrare la posa successiva (utile dopo un ridisegno statico).
    """
    n_total = int(len(hist) if hist is not None else 0)  # rinominato da N
    dt = float(max(dt, 1e-9))  # evita divisioni per zero
    k_pose = int(max(0, min(k_pose, max(n_total - 1, 0))))

    # v, w: da comandi se disponibili, altrimenti stimati dal moto tra due pose
    if commands is not None and len(commands) > 0:
        cmd_idx = (k_pose - 1) if use_cmd_of_prev else k_pose
        cmd_idx = int(max(0, min(cmd_idx, len(commands) - 1)))
        v_k = float(commands[cmd_idx][0])
        w_k = float(commands[cmd_idx][1])
    else:
        if n_total >= 2:
            k2 = int(max(1, min(k_pose, n_total - 1)))
            k1 = k2 - 1
            dx = float(hist[k2][0] - hist[k1][0])
            dy = float(hist[k2][1] - hist[k1][1])
            dth = float(hist[k2][2] - hist[k1][2])
            v_k = (dx ** 2 + dy ** 2) ** 0.5 / dt
            dth = (dth + np.pi) % (2 * np.pi) - np.pi  # normalizza in [-π, π)
            w_k = dth / dt
        else:
            v_k = 0.0
            w_k = 0.0

    # Tempo e posa (corrente o successiva)
    t_k = float(k_pose) * dt
    if show_next_pose and n_total > 0:
        pose_idx = int(min(k_pose + 1, n_total - 1))
    else:
        pose_idx = int(k_pose)

    if n_total > 0:
        x_k, y_k, th_k = map(float, hist[pose_idx])
    else:
        x_k = y_k = th_k = 0.0

    info_text = (
        f"t={t_k:.2f} s\n"
        f"v={v_k:.2f} m/s,  ω={w_k:.2f} rad/s\n"
        f"x={x_k:.2f} m,  y={y_k:.2f} m,  α={th_k:.2f} rad"
    )
    return info_text


def _update_info_artist(fig, info_artist: Optional[Text], info_text: str) -> Text:
    """Aggiorna il box info (rimuove il precedente se esiste e crea un nuovo fig.text)."""
    if info_artist is not None:
        _safe_remove_artist(info_artist)
    return fig.text(
        0.98,  # allineato a destra
        0.96,  # alto
        info_text,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
    )


def _update_error_artist(fig, err_artist: Optional[Text], msg: Optional[str]) -> Optional[Text]:
    """Mostra un messaggio di errore in alto al centro; se msg è None rimuove l'artista."""
    if err_artist is not None:
        _safe_remove_artist(err_artist)
        err_artist = None
    if msg:
        err_artist = fig.text(
            0.5, 0.98, msg,
            ha='center', va='top', fontsize=11, color='crimson', fontweight='bold',
        )
    return err_artist


def _draw_lidar_rays(ax, origin_xy, lidar_points: np.ndarray, *, ray_color: str = 'tab:red',
                     hit_marker_color: str = 'tab:red', alpha: float = 0.35) -> List[Artist]:
    """Disegna i raggi LiDAR come segmenti dall'origine ai punti misurati; ritorna gli artisti per pulizia."""
    x0, y0 = map(float, origin_xy[:2])
    arts: List[Artist] = []
    # Linee dei raggi
    for px, py in lidar_points:
        ln = ax.plot([x0, float(px)], [y0, float(py)], color=ray_color, alpha=alpha, linewidth=0.8, zorder=1.5)[0]
        arts.append(ln)
    # Marker sui punti di impatto (leggeri)
    scat = ax.scatter(lidar_points[:, 0], lidar_points[:, 1], s=5, c=hit_marker_color, alpha=min(1.0, alpha + 0.20),
                      zorder=2)
    if isinstance(scat, Artist):
        arts.append(scat)
    return arts


def plot_trajectory(history, show_orient_every=20, title="Traiettoria del robot", save_path=None, *,
                    environment: Optional[Environment] = None, fit_to: str = 'trajectory',
                    error_message: Optional[str] = None):
    """Plotta una singola traiettoria e (opzionalmente) salva l'immagine PNG.

    Nota: l'overlay d'errore non viene mostrato nelle immagini statiche; il messaggio appare solo nel viewer al momento della collisione.
    """
    # Consuma error_message per evitare warning (API compatibile)
    if error_message is not None:
        _ = error_message
    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, int(show_orient_every))
    _plot_static_trajectory_on_axes(ax, history, step=step, title=title, include_title=True, include_axis_labels=True,
                                    environment=environment, fit_to=fit_to)
    out_path = Path(save_path) if save_path else _default_save_path(title, subfolder='trajectories')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()


def _interp_pose(p0, p1, alpha: float):
    """Interpolazione lineare di posa (x,y,theta) con wrapping angolare.
    alpha in [0,1]."""
    alpha = float(max(0.0, min(1.0, alpha)))
    x0, y0, t0 = map(float, p0)
    x1, y1, t1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0
    # differenza angolare normalizzata in [-pi, pi)
    dth = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
    x = x0 + alpha * dx
    y = y0 + alpha * dy
    th = t0 + alpha * dth
    # normalizza
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, th], dtype=float)


def show_trajectories_carousel(
        histories,
        titles,
        show_orient_every=20,
        _save_each=False,  # Unused: placeholder per compatibilità API
        _commands_list=None,  # Placeholder per compatibilità API
        _dts=None,  # Placeholder per compatibilità API
        _show_info=False,  # Placeholder per compatibilità API
        _show_legend=True,  # Unused: placeholder per compatibilità API
        *,
        environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None,
        _fit_to: str = 'trajectory',  # Placeholder per compatibilità API
        _error_messages: Optional[Sequence[Optional[str]]] = None,  # Placeholder per compatibilità API
        _stop_indices: Optional[Sequence[Optional[int]]] = None,  # Placeholder per compatibilità API
        _stop_fractions: Optional[Sequence[Optional[float]]] = None,  # Placeholder per compatibilità API
        lidar: Optional[Union[Lidar, Sequence[Optional[Lidar]]]] = None,
        _show_lidar: bool = True,  # Unused: placeholder per compatibilità API
        _lidar_every: int = 1,  # Unused: placeholder per compatibilità API
        # Nuovo: traiettorie ICP già ricostruite dal LOG (stessa lunghezza di histories)
        icp_raw_histories: Optional[Sequence[Optional[np.ndarray]]] = None,
        icp_filt_histories: Optional[Sequence[Optional[np.ndarray]]] = None,
):
    """Viewer interattivo per più traiettorie con pulsanti e Play/Pausa.

    - error_messages: messaggi opzionali da mostrare SOLO quando si raggiunge la collisione.
    - stop_indices: indice (per-traiettoria) a cui fermare il player (collisione); None => nessun blocco.
    - stop_fractions: frazione temporale tra stop_indices-1 e stop_indices dove avviene l'impatto (0..1].
    - lidar: singolo sensore o lista per-traiettoria; se presente, disegna i raggi del frame corrente.
    - lidar_every: aggiorna la visualizzazione LiDAR ogni N frame (default 1 = ogni frame).
    - icp_raw_histories/icp_filt_histories: se forniti, i pannelli RAW/Filtrato useranno ESATTAMENTE queste
      traiettorie (derivate dal log), senza ricalcolare l'ICP o leggere JSON.
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(
            histories), "show_orient_every deve avere stessa lunghezza di delle traiettorie"
    if _commands_list is not None:
        assert len(_commands_list) == len(histories), "commands_list deve avere stessa lunghezza di histories"
    if _error_messages is not None:
        assert len(_error_messages) == len(histories), "error_messages deve avere stessa lunghezza di histories"
    if _stop_indices is not None:
        assert len(_stop_indices) == len(histories), "stop_indices deve avere stessa lunghezza di histories"
    if _stop_fractions is not None:
        assert len(_stop_fractions) == len(histories), "stop_fractions deve avere stessa lunghezza di histories"
    if isinstance(lidar, (list, tuple)):
        assert len(lidar) == len(histories), "lidar (lista) deve avere stessa lunghezza di histories"
    if icp_raw_histories is not None:
        assert len(icp_raw_histories) == len(histories), "icp_raw_histories deve avere stessa lunghezza di histories"
    if icp_filt_histories is not None:
        assert len(icp_filt_histories) == len(histories), "icp_filt_histories deve avere stessa lunghezza di histories"

    # Normalizza _dts a lista per uso uniforme
    if _dts is None:
        dts_resolved = [1.0] * len(histories)
    elif isinstance(_dts, (list, tuple)):
        assert len(_dts) == len(histories), "dts deve avere stessa lunghezza di histories"
        dts_resolved = [float(x) for x in _dts]
    else:
        dts_resolved = [float(_dts)] * len(histories)

    def _resolve_env(idx: int) -> Optional[Environment]:
        """Ritorna l'Environment per la traiettoria idx (singolo o per-traiettoria)."""
        if environment is None:
            return None
        if isinstance(environment, (list, tuple)):
            assert len(environment) == len(histories), "environment (lista) deve avere stessa lunghezza di histories"
            return environment[idx]
        return environment

    def _resolve_lidar(idx: int) -> Optional[Lidar]:
        if lidar is None:
            return None
        if isinstance(lidar, (list, tuple)):
            return lidar[idx]
        return lidar

    # Layout personalizzato: SINISTRA = Reale (grande), DESTRA = RAW sopra + Filtrato sotto
    # Figsize aumentato per adattarsi meglio allo schermo intero senza zoom eccessivo
    fig = plt.figure(figsize=(18, 12))

    # SINISTRA: Reale occupa tutta l'altezza (50% larghezza)
    ax_real = fig.add_subplot(1, 2, 1)  # 1 riga, 2 colonne, posizione 1

    # DESTRA: RAW sopra + Filtrato sotto (50% larghezza, diviso in 2 righe)
    ax_raw_none = fig.add_subplot(2, 2, 2)    # 2 righe, 2 colonne, posizione 2 (alto-destra)
    ax_filt_none = fig.add_subplot(2, 2, 4)   # 2 righe, 2 colonne, posizione 4 (basso-destra)

    # Margini aggiustati: più spazio in basso a sinistra per i pulsanti sotto il grafico reale
    # hspace aumentato per evitare sovrapposizione tra label x grafico sopra e titolo grafico sotto
    plt.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.12, wspace=0.15, hspace=0.35)

    state = {"idx": 0, "playing": False, "frame": 0}
    cache: Dict[int, Dict[str, np.ndarray]] = {}
    info_artist: Optional[Text] = None
    err_artist: Optional[Text] = None

    timer = fig.canvas.new_timer(interval=100)

    def _set_timer_interval_for_current():
        cur_dt = max(1e-6, float(dts_resolved[state["idx"]]))
        interval_ms = int(round(cur_dt * 1000))
        try:
            timer.interval = interval_ms
            if hasattr(timer, 'set_interval'):
                timer.set_interval(interval_ms)
        except COMMON_EXC:
            pass

    # Elementi dinamici (solo varianti RAW e Filtrato)
    artists_real: List[Artist] = []
    line_raw_none = None
    line_filt_none = None
    icp_robot_artists = {"raw_none": [], "none": []}

    def _clear_icp_artists():
        nonlocal line_raw_none, line_filt_none
        for ln in [line_raw_none, line_filt_none]:
            if ln is not None and hasattr(ln, 'remove'):
                with suppress(*COMMON_EXC):
                    ln.remove()
        line_raw_none = line_filt_none = None
        for k in list(icp_robot_artists.keys()):
            _clear_artists(icp_robot_artists[k])
            icp_robot_artists[k] = []

    def _clear_real_artists():
        nonlocal artists_real
        _clear_artists(artists_real)

    def _set_common_limits(ax_list, hist: np.ndarray, env: Optional[Environment]):
        r_robot, d_arrow = _robot_scale_from_history(hist)
        x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step=max(1, len(hist)), r_robot=r_robot,
                                                          d_arrow=d_arrow, env=env, fit_to=_fit_to)
        for a in ax_list:
            a.set_xlim(x0, x1)
            a.set_ylim(y0, y1)
            a.set_aspect('equal', 'box')
            a.grid(False)
            with suppress(*COMMON_EXC):
                a.set_xlabel('x [m]')
                a.set_ylabel('y [m]')

    def _draw_background(ax, hist: np.ndarray, title: str, env: Optional[Environment], draw_line: bool):
        if env is not None:
            with suppress(*COMMON_EXC):
                env.plot(ax=ax)
        if draw_line:
            ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.2, color='k', zorder=2)
        ax.set_title(title)

    def _draw_real_robot(k: int):
        nonlocal artists_real
        _clear_real_artists()
        idxc = state["idx"]
        hist = histories[idxc]
        r_robot, d_arrow = _robot_scale_from_history(hist)
        is_first = (k == 0)
        # Clippa k al range
        k = int(max(0, min(k, len(hist) - 1)))
        is_last = (k == len(hist) - 1) if len(hist) > 0 else False
        body_col = 'green' if is_first else ('red' if is_last else 'tab:blue')
        center_col = 'green' if is_first else ('red' if is_last else 'orange')
        artists_real = draw_robot(ax_real, hist[k], robot_radius=r_robot, dir_len=d_arrow, color=body_col,
                                  arrow_color='orange', center_color=center_col)

        # AGGIUNGI RAGGI LIDAR ANIMATI (solo hit, no raggi a vuoto)
        lidar_current = _resolve_lidar(idxc)
        env_cur = _resolve_env(idxc)
        if lidar_current is not None and env_cur is not None:
            robot_pose = hist[k]  # (x, y, theta)
            try:
                # Esegui scan LIDAR con ranges
                points, ranges = lidar_current.scan(robot_pose, env_cur, return_ranges=True)
                x, y = float(robot_pose[0]), float(robot_pose[1])

                # Disegna solo i raggi con hit reali (distanza < r_max)
                for i, (pt, r) in enumerate(zip(points, ranges)):
                    if r < lidar_current.r_max - 1e-9:  # Solo hit reali
                        # Linea dal robot al punto di hit
                        line, = ax_real.plot([x, pt[0]], [y, pt[1]],
                                            color='red', alpha=0.3, linewidth=0.5, zorder=1)
                        artists_real.append(line)
            except (ValueError, TypeError, IndexError):
                # Ignora errori di conversione o accesso agli indici
                pass

    def _draw_icp_at(k: int, trajs: Dict[str, np.ndarray]):
        nonlocal line_raw_none, line_filt_none

        def _upd(ax, line, traj, color, label=None, style='-'):
            # Clippa k per non superare la lunghezza di traj
            k_use = int(max(0, min(k, len(traj) - 1)))
            xs = traj[:k_use + 1, 0]
            ys = traj[:k_use + 1, 1]
            if line is None:
                ln, = ax.plot(xs, ys, style, color=color, linewidth=1.5, label=label)
                return ln
            else:
                with suppress(*COMMON_EXC):
                    line.set_data(xs, ys)
                return line

        line_raw_none = _upd(ax_raw_none, line_raw_none, trajs['raw_none'], 'tab:blue', label='RAW')
        line_filt_none = _upd(ax_filt_none, line_filt_none, trajs['none'], 'tab:green', label='Filtrato')
        # Rimosso: nessuna traiettoria globale
        for key, axp, line_col in [
            ('raw_none', ax_raw_none, 'tab:blue'),
            ('none', ax_filt_none, 'tab:green'),
        ]:
            _clear_artists(icp_robot_artists[key])
            hist_k = trajs[key]
            r_robot, d_arrow = _robot_scale_from_history(hist_k)
            # Clippa k
            kk = int(max(0, min(k, len(hist_k) - 1)))
            is_first = (kk == 0)
            is_last = (kk == len(hist_k) - 1)
            body_col = 'green' if is_first else ('red' if is_last else line_col)
            center_col = 'green' if is_first else ('red' if is_last else 'orange')
            icp_robot_artists[key] = draw_robot(axp, hist_k[kk], robot_radius=r_robot, dir_len=d_arrow, color=body_col,
                                                arrow_color='orange', center_color=center_col)

    def draw_current():
        nonlocal info_artist, err_artist
        with suppress(*COMMON_EXC):
            timer.stop()
        state["playing"] = False
        state["frame"] = 0
        _clear_real_artists()
        _clear_icp_artists()
        for a in [ax_real, ax_raw_none, ax_filt_none]:
            a.clear()
        if info_artist is not None:
            _safe_remove_artist(info_artist)
            info_artist = None
        err_artist = _update_error_artist(fig, err_artist, None)
        idxc = state["idx"]
        hist = np.asarray(histories[idxc], dtype=float)
        title = titles[idxc]
        # Rimuovi "(ICP da log)" dal titolo per il pannello reale
        title_clean = title.replace(" (ICP da log)", "").replace("(ICP da log)", "")
        env_cur = _resolve_env(idxc)
        # lid_cur non più necessario qui (lidar gestito in _draw_lidar_at)
        _draw_background(ax_real, hist, f"Reale – {title_clean}", env_cur, draw_line=True)
        _draw_background(ax_raw_none, hist, "ICP RAW", env_cur, draw_line=False)
        _draw_background(ax_filt_none, hist, "ICP Filtrato", env_cur, draw_line=False)
        _set_common_limits([ax_real, ax_raw_none, ax_filt_none], hist, env_cur)
        if idxc not in cache:
            info_artist = fig.text(0.5, 0.02, f"Calcolo ICP per: {title}...", ha='center', va='bottom', fontsize=10)
            fig.canvas.draw_idle()
            # Usa traiettorie fornite dal log
            icp_res_trajs = None
            if icp_raw_histories is not None or icp_filt_histories is not None:
                raw_arr = None if icp_raw_histories is None else icp_raw_histories[idxc]
                filt_arr = None if icp_filt_histories is None else icp_filt_histories[idxc]
                if isinstance(raw_arr, np.ndarray) and raw_arr.size > 0 and isinstance(filt_arr, np.ndarray) and filt_arr.size > 0:
                    icp_res_trajs = {
                        'raw_none': np.asarray(raw_arr, dtype=float),
                        'none': np.asarray(filt_arr, dtype=float),
                    }

            # Fallback: traiettoria reale se ICP non disponibile
            if icp_res_trajs is None:
                hist_arr = np.asarray(hist, dtype=float)
                icp_res_trajs = {'raw_none': hist_arr.copy(), 'none': hist_arr.copy()}

            cache[idxc] = icp_res_trajs
            _safe_remove_artist(info_artist)
            info_artist = None
        trajs = cache[idxc]
        _draw_real_robot(0)
        _draw_icp_at(0, trajs)
        if _show_info:
            dt_cur = float(dts_resolved[idxc])
            cmds = _commands_list[idxc] if _commands_list is not None else None
            info_text = _build_info_text(hist, k_pose=0, dt=dt_cur, commands=cmds, use_cmd_of_prev=False,
                                         show_next_pose=False)
            info_artist = fig.text(0.98, 0.96, info_text, ha='right', va='top', fontsize=9,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'))
        _set_timer_interval_for_current()
        fig.canvas.draw_idle()

    def _stop_if_collision_reached(next_k: int) -> bool:
        nonlocal err_artist
        if _stop_indices is None:
            return False
        idxc = state["idx"]
        stop_k = _stop_indices[idxc] if idxc < len(histories) else None
        if stop_k is None:
            return False
        if next_k >= int(stop_k):
            hist = histories[idxc]
            kcol = int(stop_k)
            frac = 1.0
            if _stop_fractions is not None and idxc < len(_stop_fractions) and _stop_fractions[idxc] is not None:
                frac = float(_stop_fractions[idxc])
            if kcol >= 1:
                pose = _interp_pose(hist[kcol - 1], hist[kcol], max(0.0, min(1.0, frac - 1e-3)))
            else:
                pose = np.asarray(hist[0], dtype=float)
            _clear_artists(artists_real)
            r_robot, d_arrow = _robot_scale_from_history(hist)
            body_col = 'tab:blue'
            center_col = 'orange'
            artists_real.extend(
                draw_robot(ax_real, pose, robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color='orange',
                           center_color=center_col))
            trajs = cache.get(idxc)
            if trajs is not None:
                kcut = int(min(kcol, len(hist) - 1))
                _draw_icp_at(kcut, trajs)
            msg = None
            if _error_messages is not None and idxc < len(_error_messages) and _error_messages[idxc]:
                msg = _error_messages[idxc]
            if not msg:
                msg = "Ostacolo lungo la traiettoria"
            err_artist = _update_error_artist(fig, err_artist, msg)
            state["playing"] = False
            with suppress(*COMMON_EXC):
                timer.stop()
            _set_play_label('▶')
            fig.canvas.draw_idle()
            return True
        return False

    def _on_timer():
        nonlocal info_artist
        idxc = state["idx"]
        hist = histories[idxc]
        n = len(hist)
        k_next = state["frame"] + 1
        if _stop_if_collision_reached(k_next):
            return
        if k_next >= n:
            state["playing"] = False
            with suppress(*COMMON_EXC):
                timer.stop()
            _set_play_label('▶')
            return
        state["frame"] = k_next
        _draw_real_robot(k_next)
        trajs = cache[idxc]
        _draw_icp_at(k_next, trajs)
        if _show_info:
            dt_cur = float(dts_resolved[idxc])
            cmds = _commands_list[idxc] if _commands_list is not None else None
            info_text = _build_info_text(np.asarray(hist), k_pose=int(k_next), dt=dt_cur, commands=cmds,
                                         use_cmd_of_prev=True, show_next_pose=False)
            if info_artist is not None:
                info_artist.set_text(info_text)
            else:
                info_artist = fig.text(0.98, 0.96, info_text, ha='right', va='top', fontsize=9,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'))
        fig.canvas.draw_idle()

    timer.add_callback(_on_timer)

    # Pulsanti piccoli in basso a sinistra sotto il grafico reale
    # Posizione: left, bottom, width, height (in frazione della figura)
    btn_width = 0.08   # Larghezza ridotta (era 0.18)
    btn_height = 0.04  # Altezza ridotta (era 0.07)
    btn_spacing = 0.01 # Spazio tra pulsanti
    btn_bottom = 0.02  # Posizione verticale (più in basso)
    btn_left_start = 0.05  # Inizio a sinistra

    # Pulsante Precedente
    ax_prev = fig.add_axes((btn_left_start, btn_bottom, btn_width, btn_height))
    btn_prev = Button(ax_prev, '◀◀')

    # Pulsante Play
    ax_play = fig.add_axes((btn_left_start + btn_width + btn_spacing, btn_bottom, btn_width, btn_height))
    btn_play = Button(ax_play, '▶')

    # Pulsante Successivo
    ax_next = fig.add_axes((btn_left_start + 2*(btn_width + btn_spacing), btn_bottom, btn_width, btn_height))
    btn_next = Button(ax_next, '▶▶')

    def _set_play_label(text: str):
        with suppress(*COMMON_EXC):
            btn_play.label.set_text(text)

    def _navigate(delta: int):
        state["idx"] = (state["idx"] + int(delta)) % len(histories)
        state["playing"] = False
        with suppress(*COMMON_EXC):
            timer.stop()
        _set_play_label('▶')  # Solo simbolo Play
        draw_current()

    def on_play(_event):
        if not state["playing"]:
            state["playing"] = True
            _set_timer_interval_for_current()
            with suppress(*COMMON_EXC):
                timer.start()
            _set_play_label('▮▮')  # Solo simbolo Pausa
        else:
            state["playing"] = False
            with suppress(*COMMON_EXC):
                timer.stop()
            _set_play_label('▶')  # Solo simbolo Play

    btn_prev.on_clicked(lambda _e: _navigate(-1))
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(lambda _e: _navigate(+1))

    # Primo disegno e show
    draw_current()


    plt.show()


# ----------------------- API di salvataggio immagini -----------------------

def save_trajectories_images(
        histories,
        titles,
        show_orient_every=20,
        *,
        environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None,
        fit_to: str = 'trajectory',
        error_messages: Optional[Sequence[Optional[str]]] = None,
        progress_cb: Optional[callable] = None,
        quiet: bool = True,
):
    """Salva PNG per ciascuna traiettoria con simboli del robot sovrapposti."""
    assert len(histories) == len(titles) and len(histories) > 0
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories)
    if error_messages is not None:
        _ = error_messages

    def _resolve_show_every(idx: int) -> int:
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    def _resolve_env(idx: int) -> Optional[Environment]:
        if environment is None:
            return None
        if isinstance(environment, (list, tuple)):
            assert len(environment) == len(histories)
            return environment[idx]
        return environment

    total = len(histories)
    for i, (hist, title_str) in enumerate(zip(histories, titles), start=1):
        fig, ax = plt.subplots(figsize=(7, 7))
        step = _resolve_show_every(i - 1)
        env_cur = _resolve_env(i - 1)
        _plot_static_trajectory_on_axes(ax, hist, step=step, title=title_str, include_title=True,
                                        include_axis_labels=True, environment=env_cur, fit_to=fit_to)
        out_path = _default_save_path(title_str, subfolder='trajectories')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if not quiet and progress_cb is None:
            print(f"[save_trajectories_images] Salvato: {out_path}")
        if callable(progress_cb):
            try:
                progress_cb(i, total)
            except TypeError:
                # Ignora errori nella chiamata al callback
                pass
        plt.close(fig)


def save_lidar_scans_images(
        history: np.ndarray,
        title: str,
        lidar: Lidar,
        environment: Optional[Environment],
        dt: float,
        *,
        interval_s: float = 1.0,
        progress_cb: Optional[callable] = None,
        quiet: bool = True,
) -> None:
    """Salva immagini dei punti di impatto LiDAR (hit) a intervalli regolari lungo history."""
    if history is None or len(history) == 0:
        return
    step_idx = max(1, int(round(float(interval_s) / max(1e-9, float(dt)))))
    n_len = len(history)
    total = len(range(0, n_len, step_idx))

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / 'img' / f"scans/{_slugify(title)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx_k, k in enumerate(range(0, n_len, step_idx), start=1):
        pose = history[k]
        try:
            scan_pts, ranges = lidar.scan(pose, environment, return_ranges=True)
        except (ValueError, AttributeError, TypeError):
            # Gestisce errori durante la scansione lidar
            continue
        scan_pts = np.asarray(scan_pts)
        ranges = np.asarray(ranges)
        mask_hit = ranges < float(lidar.r_max) - 1e-12
        hits = scan_pts[mask_hit]

        fig, ax = plt.subplots(figsize=(7, 7))
        if environment is not None:
            try:
                environment.plot(ax=ax)
            except (AttributeError, ValueError, TypeError):
                # Ignora errori durante il plot dell'ambiente
                pass
        # Limiti: preferisci bounds ambiente
        if environment is not None and getattr(environment, 'bounds', None) is not None:
            try:
                bx, by = environment.bounds.exterior.xy  # type: ignore[attr-defined]
                x_min, x_max = float(np.min(bx)), float(np.max(bx))
                y_min, y_max = float(np.min(by)), float(np.max(by))
            except (AttributeError, ValueError, TypeError):
                # Gestisce errori di accesso attributi o conversione
                x_min = y_min = -1.0
                x_max = y_max = 1.0
        else:
            if hits.size > 0:
                x_min, x_max = float(np.min(hits[:, 0])), float(np.max(hits[:, 0]))
                y_min, y_max = float(np.min(hits[:, 1])), float(np.max(hits[:, 1]))
            else:
                x_min = y_min = -1.0
                x_max = y_max = 1.0
        pad = 0.04 * max(x_max - x_min, y_max - y_min, 1.0)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')

        # Disegna raggi e hit
        x0, y0, _ = map(float, pose)
        if hits.size > 0:
            for px, py in hits:
                ax.plot([x0, float(px)], [y0, float(py)], color='tab:red', alpha=0.35, linewidth=0.8)
            ax.scatter(hits[:, 0], hits[:, 1], s=6, c='tab:red', alpha=0.65)

        # Aggiungi rappresentazione del robot nella sua posa corrente
        # Calcola scala del robot basata sull'estensione dell'ambiente
        span = max(x_max - x_min, y_max - y_min, 1.0)
        robot_radius = max(0.02, 0.015 * span)  # Scala proporzionale all'ambiente
        dir_len = 2.5 * robot_radius  # Lunghezza freccia

        # Disegna il robot nella posa corrente (blu per distinguerlo dalla traiettoria)
        draw_robot(ax, pose, robot_radius=robot_radius, dir_len=dir_len,
                  color='tab:blue', arrow_color='orange', center_color='orange')

        out_path = out_dir / f"{_slugify(title)}_t{float(k) * float(dt):.2f}s_points_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if not quiet and progress_cb is None:
            print(f"[save_lidar_scans_images] Salvato: {out_path}")
        if callable(progress_cb):
            try:
                progress_cb(idx_k, total)
            except TypeError:
                pass
        plt.close(fig)


def save_lidar_polar_images(
        history: np.ndarray,
        title: str,
        lidar: Lidar,
        environment: Optional[Environment],
        dt: float,
        *,
        interval_s: float = 1.0,
        include_misses: bool = True,
        progress_cb: Optional[callable] = None,
        quiet: bool = True,
) -> None:
    """Salva grafici r(θ) in gradi delle scansioni LiDAR a intervalli regolari."""
    if history is None or len(history) == 0:
        return
    step_idx = max(1, int(round(float(interval_s) / max(1e-9, float(dt)))))
    n_len = len(history)
    total = len(range(0, n_len, step_idx))

    # Angoli relativi come in Lidar.scan
    half = 0.5 * float(lidar.angle_span)
    rel = np.linspace(-half, half, num=lidar.n_rays, endpoint=True)
    rel_deg = (np.degrees(rel) + 360.0) % 360.0

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / 'img' / f"scans_polar/{_slugify(title)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx_k, k in enumerate(range(0, n_len, step_idx), start=1):
        pose = history[k]
        try:
            _pts, ranges = lidar.scan(pose, environment, return_ranges=True)
        except (ValueError, AttributeError, TypeError):
            # Gestisce errori durante la scansione
            continue
        ranges = np.asarray(ranges)
        mask_hit = ranges < float(lidar.r_max) - 1e-12
        th_hit = rel_deg[mask_hit]
        rr_hit = ranges[mask_hit]
        th_miss = rel_deg[~mask_hit]
        rr_miss = ranges[~mask_hit]

        fig, ax = plt.subplots(figsize=(7, 4))
        if th_hit.size > 0:
            ax.scatter(th_hit, rr_hit, s=10, c='tab:blue', alpha=0.95, label='hit')
        if include_misses and th_miss.size > 0:
            ax.scatter(th_miss, rr_miss, s=8, c='tab:gray', alpha=0.6, label='miss (r_max)')
        ax.set_xlabel('θ [°]')
        ax.set_ylabel('r [m]')
        ax.grid(True, alpha=0.25)
        ax.set_title(f"r(θ) – {title} – t={float(k) * float(dt):.2f} s")
        ax.set_ylim(-0.02 * float(lidar.r_max), 1.02 * float(lidar.r_max))
        ax.set_xlim(-1e-3, 360.0 + 1e-3)
        try:
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        except (ValueError, TypeError):
            pass
        if (th_hit.size > 0) or (include_misses and th_miss.size > 0):
            ax.legend(loc='upper right', framealpha=0.85, fontsize=8)
        out_path = out_dir / f"{_slugify(title)}_polar_t{float(k) * float(dt):.2f}s_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if not quiet and progress_cb is None:
            print(f"[save_lidar_polar_images] Salvato: {out_path}")
        if callable(progress_cb):
            try:
                progress_cb(idx_k, total)
            except TypeError:
                pass
        plt.close(fig)


# Path/dir per output ICP

def ensure_icp_dirs(*subfolders: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / 'img' / 'icp'
    base.mkdir(parents=True, exist_ok=True)
    for s in subfolders:
        (base / s).mkdir(parents=True, exist_ok=True)


def icp_out_path(*parts: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / 'img' / 'icp'
    return str(base.joinpath(*parts))


# ---- Supporto per costruire traiettorie ICP accumulate ----

def _angle_from_r(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _accumulate_icp_trajectory(history: np.ndarray, icp_results: List[Dict], key: str) -> np.ndarray:
    n = int(len(history))
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    out = np.zeros((n, 3), dtype=float)
    out[0, :] = np.asarray(history[0], dtype=float)
    th0 = float(out[0, 2])
    r_w_prev = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]], dtype=float)
    by_k: Dict[int, Dict] = {int(r.get('k')): r for r in icp_results}
    for k in range(1, n):
        x_prev, y_prev, _ = map(float, out[k - 1])
        res = by_k.get(k)
        if res is None or not res.get('ok', False):
            out[k, :] = out[k - 1]
            continue
        block = res.get(key)
        if not isinstance(block, dict):
            out[k, :] = out[k - 1]
            continue
        r_rel = np.asarray(block.get('R'))
        t_rel = np.asarray(block.get('t')).reshape(2)
        if r_rel.shape != (2, 2) or t_rel.shape != (2,):
            out[k, :] = out[k - 1]
            continue
        # Composizione (versione precedente): usa direttamente R_rel e t_rel come incremento nel frame k-1
        t_world = r_w_prev @ t_rel
        x_k = x_prev + float(t_world[0])
        y_k = y_prev + float(t_world[1])
        r_w_k = r_w_prev @ r_rel
        try:
            u, _s, vt = np.linalg.svd(r_w_k)
            r_w_k = u @ vt
        except (np.linalg.LinAlgError, ValueError):
            # Gestisce errori SVD
            pass
        th_k = float(np.arctan2(r_w_k[1, 0], r_w_k[0, 0]))
        out[k, 0] = x_k
        out[k, 1] = y_k
        out[k, 2] = th_k
        r_w_prev = r_w_k
    return out


def _compute_icp_trajectories_for_case(history: np.ndarray, lidar: Lidar, env: Optional[Environment], *,
                                       _max_correspondence_distance: float = 0.40,  # Unused: legacy parameter
                                       _trim_fraction: float = 0.6,  # Unused: legacy parameter
                                       _damping_enabled: bool = True,  # Unused: legacy parameter
                                       _angle_thresh_deg: float = 10.0,  # Unused: legacy parameter
                                       _struct_ratio_thresh: float = 0.02,  # Unused: legacy parameter
                                       _damp_factor: float = 0.75,  # Unused: legacy parameter
                                       _sliding_filter_enabled: bool = True,  # Unused: legacy parameter
                                       _sliding_cos_threshold: float = 0.985,  # Unused: legacy parameter
                                       _angle_balance_enabled: bool = True,  # Unused: legacy parameter
                                       _angle_bin_deg: float = 8.0,  # Unused: legacy parameter
                                       _angle_max_per_bin: int = 18,  # Unused: legacy parameter
                                       _angle_prefer_far: bool = True,  # Unused: legacy parameter
                                       _use_scipy: bool = True,  # Unused: legacy parameter
                                       _progress_cb: Optional[callable] = None) -> Dict[str, np.ndarray]:  # Unused: legacy parameter
    if env is None or lidar is None:
        zero = np.asarray(history, dtype=float).copy()
        for k in range(1, len(zero)):
            zero[k, :] = zero[0, :]
        return {'raw_none': zero.copy(), 'none': zero.copy()}

    # Nota: run_icp_over_history è una funzione stub che restituisce traiettorie vuote
    # I parametri avanzati (max_iterations, tolerance, ecc.) sono stati rimossi
    # perché non più supportati nella nuova implementazione semplificata
    icp_results = run_icp_over_history(
        np.asarray(history, dtype=float),
        lidar,
        env,
        step=1
    )
    return {
        'raw_none': _accumulate_icp_trajectory(history, icp_results, 'raw_none'),
        'none': _accumulate_icp_trajectory(history, icp_results, 'none'),
    }


def show_trajectories_icp_grid(
        histories,
        titles,
        *,
        environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None,
        fit_to: str = 'environment',
):
    """Visualizzazione statica a griglia delle traiettorie ICP.

    Args:
        histories: Lista di array numpy con le traiettorie
        titles: Titoli per ogni traiettoria
        environment: Ambiente singolo o lista di ambienti per ogni traiettoria
        fit_to: Come adattare i limiti degli assi ('trajectory' o 'environment')
    """
    assert len(histories) == len(titles) and len(histories) > 0

    def _resolve_env(idx: int) -> Optional[Environment]:
        if environment is None:
            return None
        if isinstance(environment, (list, tuple)):
            assert len(environment) == len(histories)
            return environment[idx]
        return environment

    # Nota: _resolve_lidar rimossa - non utilizzata in questa funzione
    # Il lidar non viene più usato per questa visualizzazione statica

    for traj_idx, (hist, title) in enumerate(zip(histories, titles)):
        env_cur = _resolve_env(traj_idx)
        # lid_cur rimosso: non utilizzato in questa funzione
        title_clean = title
        # Figsize aumentato per evitare zoom eccessivo a schermo intero
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.20, hspace=0.55)
        ax_real = axes[0, 0]; ax_raw_none = axes[0, 1]; ax_filt_none = axes[1, 0]
        axes[0, 2].axis('off'); axes[1, 1].axis('off'); axes[1, 2].axis('off')
        hist_np = np.asarray(hist, dtype=float)
        # Reale: disegno traiettoria e ambiente
        _plot_static_trajectory_on_axes(
            ax_real, hist_np, step=max(1, int(len(hist_np))),
            title=f"Reale – {title_clean}", include_title=True, include_axis_labels=True,
            environment=env_cur, fit_to='environment'
        )
        # Sfondo ambiente sugli altri pannelli
        if env_cur is not None:
            try:
                env_cur.plot(ax=ax_raw_none)
            except (AttributeError, ValueError, TypeError):
                pass
            try:
                env_cur.plot(ax=ax_filt_none)
            except (AttributeError, ValueError, TypeError):
                pass
        ax_raw_none.set_title('ICP RAW')
        ax_filt_none.set_title('ICP Filtrato')
        # Limiti coerenti su tutti i pannelli
        r_robot, d_arrow = _robot_scale_from_history(hist_np)
        x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist_np, step=max(1, len(hist_np)), r_robot=r_robot,
                                                          d_arrow=d_arrow, env=env_cur, fit_to=fit_to)
        for a in (ax_real, ax_raw_none, ax_filt_none):
            a.set_xlim(x0, x1)
            a.set_ylim(y0, y1)
            a.set_aspect('equal', 'box')
            a.grid(False)
            with suppress(*COMMON_EXC):
                a.set_xlabel('x [m]')
                a.set_ylabel('y [m]')
        # Calcolo e disegno traiettorie ICP (fallback: usa traiettoria reale)
        # Nota: in produzione le traiettorie ICP vengono sempre fornite dal log
        trajs = {'raw_none': hist_np.copy(), 'none': hist_np.copy()}
        raw_tr = trajs.get('raw_none'); filt_tr = trajs.get('none')
        if isinstance(raw_tr, np.ndarray) and raw_tr.size > 0:
            ax_raw_none.plot(raw_tr[:, 0], raw_tr[:, 1], '-', color='tab:blue', linewidth=1.5, label='RAW')
        if isinstance(filt_tr, np.ndarray) and filt_tr.size > 0:
            ax_filt_none.plot(filt_tr[:, 0], filt_tr[:, 1], '-', color='tab:green', linewidth=1.5, label='Filtrato')
            with suppress(*COMMON_EXC):
                ax_raw_none.legend(loc='best', framealpha=0.85, fontsize=8)
                ax_filt_none.legend(loc='best', framealpha=0.85, fontsize=8)
        else:
            with suppress(*COMMON_EXC):
                ax_raw_none.text(0.5, 0.5, 'LiDAR/Ambiente non disponibili', ha='center', va='center', transform=ax_raw_none.transAxes)
                ax_filt_none.text(0.5, 0.5, 'LiDAR/Ambiente non disponibili', ha='center', va='center', transform=ax_filt_none.transAxes)
        fig.suptitle(f"Traiettorie – {title_clean}", fontsize=12)


    plt.show()
