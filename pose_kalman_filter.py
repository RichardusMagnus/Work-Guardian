from __future__ import annotations

# Modulo standard utilizzato per ottenere timestamp monotoni, cioè tempi
# crescenti indipendenti da eventuali modifiche dell'orologio di sistema.
import time

# Tipi usati per rendere più esplicita l'interfaccia delle funzioni e chiarire
# quali strutture dati sono attese in ingresso.
from typing import Any, Mapping, Optional

# NumPy viene impiegato per rappresentare vettori e matrici del filtro di Kalman
# e per svolgere le operazioni lineari necessarie a predizione e correzione.
import numpy as np


class PositionKalmanFilter:
    """
    Filtro di Kalman lineare per stabilizzare una posizione 3D.

    Stato:
        [x, y, z, vx, vy, vz]^T

    Misura:
        [x, y, z]^T

    Il filtro è pensato per essere applicato alla posa stimata del drone/camera
    dopo la fusione AprilTag, non alle posizioni note dei tag nel mondo.
    """

    def __init__(
        self,
        process_noise: float = 0.15,
        measurement_noise: float = 0.08,
        initial_covariance: float = 1.0,
    ):
        # I parametri numerici vengono convertiti esplicitamente in float per
        # garantire coerenza nei calcoli matriciali e prevenire ambiguità di tipo.
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_covariance = float(initial_covariance)

        # Vettore di stato del filtro: le prime tre componenti rappresentano
        # la posizione, mentre le ultime tre rappresentano la velocità stimata.
        self.x = np.zeros((6, 1), dtype=np.float32)

        # Matrice di covarianza dello stato: quantifica l'incertezza iniziale
        # associata alla stima di posizione e velocità.
        self.P = np.eye(6, dtype=np.float32) * self.initial_covariance

        # Matrice di osservazione H: seleziona dal vettore di stato solo le
        # componenti misurabili direttamente, cioè x, y e z.
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Matrice R del rumore di misura: descrive l'incertezza associata alla
        # posizione osservata in ingresso al filtro.
        self.R = np.eye(3, dtype=np.float32) * (self.measurement_noise ** 2)

        # Flag che indica se il filtro è già stato inizializzato con una prima
        # misura valida. Prima di tale misura non è possibile eseguire una vera
        # predizione temporale.
        self.initialized = False

        # Timestamp dell'ultimo aggiornamento, necessario per calcolare il passo
        # temporale dt tra due misure successive.
        self.last_timestamp: Optional[float] = None

    @staticmethod
    def _normalize_position(position_world: Any) -> np.ndarray:
        """
        Converte una posizione in un vettore colonna NumPy 3x1.

        Sono accettati:
        - dict con chiavi x, y, z;
        - lista/tupla/array con 3 elementi;
        - array di forma 3x1.
        """
        # Se la posizione è fornita come Mapping, ad esempio un dizionario,
        # vengono estratte esplicitamente le tre coordinate spaziali richieste.
        if isinstance(position_world, Mapping):
            # Verifica la presenza delle chiavi obbligatorie e controlla che i
            # valori associati non siano nulli.
            missing_keys = [
                key
                for key in ("x", "y", "z")
                if key not in position_world or position_world.get(key) is None
            ]
            if missing_keys:
                raise ValueError(
                    "La posizione in formato Mapping deve contenere le chiavi x, y, z "
                    f"con valori numerici non nulli. Chiavi mancanti/non valide: {missing_keys}"
                )

            # Conversione del Mapping in una sequenza ordinata di coordinate,
            # coerente con la convenzione [x, y, z].
            position_world = [
                position_world.get("x"),
                position_world.get("y"),
                position_world.get("z"),
            ]

        # Conversione dell'ingresso in array NumPy monodimensionale. La reshape(-1)
        # consente di trattare in modo uniforme liste, tuple, array riga e array colonna.
        position = np.asarray(position_world, dtype=np.float32).reshape(-1)

        # Il filtro è definito per uno spazio tridimensionale, quindi la misura
        # deve contenere esattamente tre coordinate.
        if position.size != 3:
            raise ValueError(
                "La posizione deve contenere esattamente tre coordinate: x, y, z."
            )

        # Controllo di validità numerica: valori NaN o infiniti renderebbero
        # instabile o non significativo l'aggiornamento del filtro.
        if not np.all(np.isfinite(position)):
            raise ValueError(
                "La posizione deve contenere solo valori numerici finiti."
            )

        # La posizione viene restituita come vettore colonna 3x1, formato
        # coerente con la notazione matriciale del filtro di Kalman.
        return position.reshape(3, 1)

    def reset(self):
        """
        Reinizializza il filtro.
        """
        # Riporta lo stato alla configurazione iniziale, eliminando le stime
        # precedenti di posizione e velocità.
        self.x = np.zeros((6, 1), dtype=np.float32)

        # Ripristina la covarianza iniziale, cioè l'incertezza associata allo
        # stato prima della ricezione di nuove misure.
        self.P = np.eye(6, dtype=np.float32) * self.initial_covariance

        # Il filtro torna nello stato non inizializzato e attenderà una nuova
        # prima misura per impostare la posizione iniziale.
        self.initialized = False
        self.last_timestamp = None

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Costruisce la matrice di transizione dello stato per il modello
        cinematico a velocità costante.
        """
        # La matrice parte dall'identità, poiché in assenza di evoluzione
        # temporale ogni componente dello stato rimarrebbe invariata.
        F = np.eye(6, dtype=np.float32)

        # Nel modello a velocità costante, la posizione futura è ottenuta come:
        # posizione_attuale + velocità_attuale * dt.
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        return F

    def _build_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Costruisce la matrice Q del rumore di processo.

        Il rumore di processo rappresenta quanto ci si aspetta che il moto reale
        possa deviare dal modello a velocità costante.
        """
        # La varianza del rumore di processo è ottenuta elevando al quadrato
        # il parametro configurabile process_noise.
        q = self.process_noise ** 2

        # Potenze del passo temporale usate nella discretizzazione del rumore
        # per un modello cinematico con posizione e velocità.
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        # Matrice Q 6x6: contiene i termini di covarianza relativi a posizione,
        # velocità e correlazione tra posizione e velocità.
        Q = np.zeros((6, 6), dtype=np.float32)

        # La stessa struttura di rumore viene applicata indipendentemente alle
        # tre dimensioni spaziali x, y e z.
        for i in range(3):
            Q[i, i] = dt4 / 4.0 * q
            Q[i, i + 3] = dt3 / 2.0 * q
            Q[i + 3, i] = dt3 / 2.0 * q
            Q[i + 3, i + 3] = dt2 * q

        return Q

    def update(self, position_world: Any, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Aggiorna il filtro con una nuova misura di posizione.

        Restituisce la posizione filtrata come array NumPy 3x1.
        """
        # Se il chiamante non fornisce un timestamp, viene usato un tempo
        # monotono per calcolare in modo robusto l'intervallo tra aggiornamenti.
        if timestamp is None:
            timestamp = time.monotonic()

        # La misura in ingresso viene normalizzata nel formato vettoriale 3x1
        # richiesto dalle equazioni del filtro.
        z = self._normalize_position(position_world)

        # Alla prima misura non si effettua ancora una predizione: la posizione
        # osservata inizializza direttamente lo stato, mentre la velocità viene
        # posta a zero per mancanza di informazione dinamica precedente.
        if not self.initialized:
            self.x[0:3, :] = z
            self.x[3:6, :] = 0.0
            self.last_timestamp = float(timestamp)
            self.initialized = True
            return self.x[0:3, :].copy()

        # Calcolo del tempo trascorso dall'ultimo aggiornamento, necessario per
        # costruire il modello di evoluzione temporale dello stato.
        dt = float(timestamp) - float(self.last_timestamp)
        self.last_timestamp = float(timestamp)

        # In caso di timestamp non crescente, viene imposto un piccolo dt
        # positivo per evitare degenerazioni numeriche nella predizione.
        if dt <= 0.0:
            dt = 1e-3

        # Costruzione delle matrici dipendenti dal passo temporale corrente.
        F = self._build_transition_matrix(dt)
        Q = self._build_process_noise_matrix(dt)

        # Predizione
        # Stima a priori dello stato e della sua covarianza secondo il modello
        # cinematico a velocità costante.
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Correzione
        # Innovazione: differenza tra misura osservata e misura prevista dal
        # modello a partire dallo stato predetto.
        y = z - self.H @ self.x

        # Covarianza dell'innovazione: combina l'incertezza della predizione
        # con l'incertezza della misura.
        S = self.H @ self.P @ self.H.T + self.R

        # Guadagno di Kalman: stabilisce quanto la nuova misura deve correggere
        # la predizione in funzione delle incertezze relative.
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Aggiornamento dello stato: la stima predetta viene corretta usando
        # l'innovazione pesata dal guadagno di Kalman.
        self.x = self.x + K @ y

        # Aggiornamento della covarianza: dopo la correzione, l'incertezza dello
        # stato viene ridotta in base all'informazione fornita dalla misura.
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # Viene restituita solo la componente di posizione filtrata, lasciando
        # interna al filtro la stima della velocità.
        return self.x[0:3, :].copy()

    def filter_pose_estimate(
        self,
        pose_estimate: Optional[Mapping[str, Any]],
        timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Applica il filtro a una struttura pose_estimate compatibile con
        quelle prodotte da CameraPoseEstimator.

        Restituisce una nuova struttura, senza modificare direttamente l'originale.
        """
        # Se la struttura è assente o vuota, non è disponibile alcuna posa da
        # filtrare; la funzione segnala quindi l'assenza di risultato.
        if not pose_estimate:
            return None

        # La chiave position_world è il dato minimo necessario per applicare il
        # filtro, poiché contiene la posizione tridimensionale stimata.
        if "position_world" not in pose_estimate:
            return None

        # Applicazione del filtro alla posizione contenuta nella stima di posa.
        filtered_position = self.update(
            pose_estimate["position_world"],
            timestamp=timestamp,
        )

        # Si crea una copia superficiale della struttura originale per evitare
        # effetti collaterali sul dizionario ricevuto in ingresso.
        filtered_pose = dict(pose_estimate)

        # La posizione originale viene sostituita con la posizione filtrata.
        filtered_pose["position_world"] = filtered_position

        # Il campo source viene aggiornato per mantenere traccia del fatto che
        # la stima è stata elaborata dal filtro di Kalman.
        filtered_pose["source"] = f"kalman({pose_estimate.get('source', 'raw')})"

        return filtered_pose