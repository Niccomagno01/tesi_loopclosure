# Classe che rappresenta lo stato (x, y, theta) del robot, applica l'integrazione discreta (Euler),
# normalizza l'angolo e imposta i comandi v e omega.

import numpy as np

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta) # Angolo in radianti
        self.v = 0.0  # Velocità lineare
        self.omega = 0.0  # Velocità angolare

    def state(self):
        """Restituisce [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])

    def set_command(self, v, omega):
        """Imposta i comandi di velocità lineare e angolare"""
        self.v = float(v)
        self.omega = float(omega)

    def step(self, dt):
        """Aggiorna lo stato del robot con schema di Eulero esplicito fornito:
        x_{k+1} = x_k + v_k * cos(theta_k) * dt
        y_{k+1} = y_k + v_k * sin(theta_k) * dt
        theta_{k+1} = theta_k + omega_k * dt
        Poi normalizza l'angolo."""
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        self._normalize_angle()

    def _normalize_angle(self):
        """Normalizza l'angolo theta nell'intervallo [-pi, pi]"""
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi