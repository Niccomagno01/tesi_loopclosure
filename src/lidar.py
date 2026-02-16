# Classe che ha il compito di simulare un sensore LiDAR, realizzato con Shapely e ha il compito di fare scansioni

import numpy as np
from shapely.geometry import LineString

class Lidar:
    def __init__(self, n_rays=360, angle_span=2*np.pi, r_max=6.0, angle_offset=0.0, add_noise=False, noise_std=0.01):
        """
        :param n_rays: numero di raggi per scansione
        :param angle_span: ampiezza angolare totale(rad)
        :param r_max: portata massima dei raggi
        :param angle_offset: orientamento iniziale relativo al robot (rad)
        :param add_noise: aggiunge rumore gaussiano alle distanze misurate
        :param noise_std: deviazione standard del rumore (metri)
        """
        self.n_rays = int(n_rays)
        self.angle_span = float(angle_span)
        self.r_max = float(r_max)
        self.angle_offset = float(angle_offset)
        self.add_noise = bool(add_noise)
        self.noise_std = float(noise_std)

    def scan(self, robot_state, env, return_ranges=False):
        """
        Esegue una scansione dal robot_state = (x, y, theta) dal centro del robot.
        env: Environment (deve esporre first_intersection_with_line)
        return_ranges: se True restituisce (points, ranges), altrimenti solo points
        Ritorna: np.ndarray di forma (n_rays, 2) con coordinate mondo dei punti misurati (o r_max se nulla)
        """
        x, y, theta = robot_state
        # Angoli in frame mondo: heading + offset + angolo relativo
        half = 0.5 * self.angle_span
        angles = np.linspace(-half, half, num=self.n_rays, endpoint=True) + theta + self.angle_offset
        points = np.zeros((self.n_rays, 2), dtype=float)
        ranges = np.full((self.n_rays,), self.r_max, dtype=float)

        for i, ang in enumerate(angles):
            end_x = float(x + self.r_max * np.cos(ang))
            end_y = float(y + self.r_max * np.sin(ang))
            ray = LineString([(float(x), float(y)), (end_x, end_y)])
            inter = env.first_intersection_with_line(ray)
            if inter is not None:
                px, py = inter
                # distanza
                r = float(np.hypot(px - x, py - y))
                ranges[i] = r
                points[i, :] = [px, py]
            else:
                # nessuna intersezione: punto a r_max
                points[i, :] = [end_x, end_y]
                ranges[i] = self.r_max

            # optional noise sulla distanza: si applica e rimappa punto
            if self.add_noise and ranges[i] < self.r_max:
                noisy_r = max(0.0, float(ranges[i]) + float(np.random.normal(0.0, self.noise_std)))
                ranges[i] = noisy_r
                points[i, 0] = float(x + noisy_r * np.cos(ang))
                points[i, 1] = float(y + noisy_r * np.sin(ang))

        if return_ranges:
            return points, ranges
        return points

    def scan_hits(self, robot_state, env, frame: str = 'world'):
        """Ritorna SOLO i punti di impatto reali (escludendo i raggi a r_max) come array Nx2.

        - frame='world': punti in coordinate mondo (default, come scan())
          frame='local': punti nel frame locale del LiDAR (origine al sensore, asse x in avanti),
                        ottenuti ruotando di -(theta+angle_offset) e traslando di -(x,y).
        """
        pts, ranges = self.scan(robot_state, env, return_ranges=True)
        # Filtra solo gli hit reali (distanze < r_max)
        mask_hits = np.asarray(ranges) < float(self.r_max) - 1e-12
        hit_pts = np.asarray(pts)[mask_hits]
        if frame == 'world':
            return hit_pts
        if frame == 'local':
            # Trasforma da mondo a frame sensore locale
            x, y, theta = map(float, robot_state)
            ca = np.cos(-(theta + float(self.angle_offset)))
            sa = np.sin(-(theta + float(self.angle_offset)))
            dx = hit_pts[:, 0] - x
            dy = hit_pts[:, 1] - y
            x_local = ca * dx - sa * dy
            y_local = sa * dx + ca * dy
            return np.stack([x_local, y_local], axis=1)
        raise ValueError("frame deve essere 'world' o 'local'")

    def scan_hits_indexed(self, robot_state, env, frame: str = 'world'):
        """Ritorna gli hit con indice di raggio.

        Restituisce (idx, pts) dove:
        - idx: array di indici di raggio (int)
        - pts: array Nx2 di punti nel frame richiesto ('world' o 'local')
        I raggi senza impatto (a r_max) sono esclusi.
        """
        pts_w, ranges = self.scan(robot_state, env, return_ranges=True)
        mask_hits = np.asarray(ranges) < float(self.r_max) - 1e-12
        idx = np.nonzero(mask_hits)[0].astype(int)
        pts_sel = np.asarray(pts_w)[mask_hits]
        if frame == 'world':
            return idx, pts_sel
        if frame == 'local':
            x, y, theta = map(float, robot_state)
            ca = np.cos(-(theta + float(self.angle_offset)))
            sa = np.sin(-(theta + float(self.angle_offset)))
            dx = pts_sel[:, 0] - x
            dy = pts_sel[:, 1] - y
            x_local = ca * dx - sa * dy
            y_local = sa * dx + ca * dy
            return idx, np.stack([x_local, y_local], axis=1)
        raise ValueError("frame deve essere 'world' o 'local'")
