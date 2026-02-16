# Classe che ha il compito di orchestrare la simulazione: avanzare il robot nel tempo, applicare comandi e salvare
# lo stato e le scansioni

import numpy as np
from robot import Robot


class Simulator:
    def __init__(self, robot=None):
        self.robot = robot or Robot()  # Usa il robot passato o crea una nuova istanza di Robot
        self.history = None  # Conterrà la storia degli stati (x, y, theta) del robot
        self.commands = None  # Conterrà la sequenza di comandi (v, omega) applicati

    def run_from_sequence(self, vs, omegas, dt):
        """Esegue la simulazione per una sequenza di comandi senza alcun controllo di collisione.

        - Salva history (N+1,3) e commands (N,2), partendo dallo stato corrente del robot.
        """
        n = len(vs)  # Numero di passi temporali (lunghezza della sequenza di velocità)
        self.history = np.zeros((n+1, 3))  # Array per salvare gli stati: n+1 perché include lo stato iniziale
        self.commands = np.zeros((n, 2))  # Array per salvare i comandi (v, omega) applicati ad ogni step
        self.history[0] = self.robot.state()  # Salva lo stato iniziale del robot come prima riga della history

        for k in range(n):  # Itera su ogni passo temporale
            self.robot.set_command(vs[k], omegas[k])  # Imposta i comandi di velocità lineare e angolare per questo step
            self.robot.step(dt)  # Avanza la dinamica del robot di dt secondi applicando il comando appena impostato
            self.history[k+1] = self.robot.state()  # Registra il nuovo stato dopo l'avanzamento
            self.commands[k] = [vs[k], omegas[k]]  # Memorizza il comando applicato in questo step

        return self.history  # Ritorna l'intera traiettoria degli stati

    def reset_robot(self, x=0.0, y=0.0, theta=0.0):
        """Reimposta la posizione del robot"""
        self.robot = Robot(x=x, y=y, theta=theta)  # Crea un nuovo robot nelle coordinate specificate
        self.history = None  # Azzera la storia perché parte una nuova simulazione
        self.commands = None  # Azzera anche i comandi precedenti
