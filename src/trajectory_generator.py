# Classe che fornisce traiettorie prefissate (circle, eight, line) e casuali per il robot

import numpy as np

class TrajectoryGenerator:
    @staticmethod
    def straight(v, T, dt):
        """Genera una traiettoria lineare con velocità v per un tempo T con passo dt
        Ritorna due array: (velocità lineari, velocità angolari)
        - v: velocità lineare costante
        - omega: sempre 0 (moto rettilineo)
        """
        n = int(np.ceil(T/dt))  # Numero di step discreti (arrotondato per eccesso)
        return np.full(n, v), np.zeros(n)  # v costante per tutti gli step, omega=0 ⇒ linea retta

    @staticmethod
    def straight_var_speed(v_min, v_max, T, dt, phase=0.0):
        """Moto rettilineo con velocità lineare variabile (profilo sinusoidale tra v_min e v_max).
        - v(t) = v_mid + v_amp * sin(2π t / T + phase)
        - omega(t) = 0 ⇒ traiettoria in linea retta
        Note: supponiamo v_min ≤ v_max e velocità non negative (tipico per uniciclo)."""
        n = int(np.ceil(T/dt))  # Numero di step discreti
        t = np.linspace(0, T, n)  # Asse temporale uniformemente campionato
        v_mid = 0.5 * (v_max + v_min)  # Valore medio del profilo di velocità
        v_amp = 0.5 * (v_max - v_min)  # Ampiezza dell'oscillazione
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)  # Profilo v(t) sinusoidale
        omegas = np.zeros(n)  # Nessuna rotazione: moto rettilineo
        return vs, omegas

    @staticmethod
    def circle(v, radius, T, dt):
        """Genera una traiettoria circolare con velocità v e raggio radius per un tempo T con passo dt
        - omega = v / R: velocità angolare costante (rad/s) per descrivere un cerchio di raggio "radius"
        """
        omega = v/float(radius)  # Relazione cinematica cerchio: omega = v / R
        n = int(np.ceil(T/dt))  # Numero di campioni temporali
        return np.full(n, v), np.full(n, omega)  # v e omega costanti ⇒ traiettoria circolare

    @staticmethod
    def circle_var_speed(v_min, v_max, radius, T, dt, phase=0.0):
        """Traiettoria circolare a raggio costante con velocità lineare variabile (sinusoidale).
        - v(t) sinusoidale tra v_min e v_max
        - omega(t) = v(t) / R, così il raggio rimane costante mentre varia la velocità (e la velocità angolare)"""
        n = int(np.ceil(T/dt))  # Numero di campioni
        t = np.linspace(0, T, n)  # Asse temporale
        v_mid = 0.5 * (v_max + v_min)  # Media del profilo
        v_amp = 0.5 * (v_max - v_min)  # Ampiezza del profilo
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)  # v(t) sinusoidale
        omegas = vs / float(radius)  # Mantiene il raggio costante imponendo omega(t) coerente
        return vs, omegas

    @staticmethod
    def eight(v, radius, T, dt):
        """Traiettoria "otto" migliorata con transizione smooth tra i due lobi.
        Prima metà: cerchio orario, seconda metà: cerchio antiorario.
        Transizione graduale molto smooth per evitare discontinuità."""
        n = int(np.ceil(T / dt))              # Numero di step discreti totali
        mid = n // 2                          # Punto di transizione tra i due lobi
        vs = np.full(n, v)                    # Velocità lineare costante
        omegas = np.zeros(n)                  # Pre-allocazione velocità angolare

        # Zona di transizione: 15% del tempo totale (più ampia per smooth migliore)
        transition_width = max(3, int(0.15 * n))
        transition_start = mid - transition_width // 2
        transition_end = mid + transition_width // 2

        omega_val = v / float(radius)

        # Prima metà: curvatura positiva
        omegas[:transition_start] = omega_val

        # Zona di transizione: interpolazione smooth con coseno
        for i in range(transition_start, min(transition_end, n)):
            # alpha va da 0 a 1 nella zona di transizione
            alpha = (i - transition_start) / float(transition_end - transition_start)
            # Interpolazione smooth usando coseno (smooth in derivata)
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            omegas[i] = omega_val * (1 - 2 * smooth_alpha)  # Da +omega a -omega

        # Seconda metà: curvatura negativa
        omegas[transition_end:] = -omega_val

        return vs, omegas

    @staticmethod
    def random_walk(v_mean, omega_std, T, dt, seed=None):
        """Genera una traiettoria randomica con velocità media v_mean e deviazione standard omega_std per un tempo T con passo dt
        - v_mean: velocità lineare costante (media del moto)
        - omega ~ N(0, omega_std^2): rumore gaussiano per esplorazione angolare
        - seed: rende il risultato riproducibile se specificato
        """
        rng = np.random.default_rng(seed)  # Generatore di numeri casuali (riproducibile tramite seed)
        n = int(np.ceil(T/dt))  # Numero di step
        vs = np.full(n, v_mean)  # Velocità lineare costante per tutta la durata
        omegas = rng.normal(0.0, omega_std, size=n)  # Campiona omega i.i.d. da N(0, sigma^2)
        return vs, omegas  # Ritorna le sequenze (v(t), omega(t))
