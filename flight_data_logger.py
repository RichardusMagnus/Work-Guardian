import logging
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class FlightDataLogger:
    """
    Logger per raccogliere dati di volo del drone in tempo reale.
    Registra posizione, yaw e ID dei tag usati per la stima.
    Produce un file di testo con i dati accumulati.
    """

    def __init__(self, filename: str | Path = "flight_data.txt"):
        """
        Inizializza il logger con il nome del file di output.
        """
        self.filename = Path(filename)
        self.data_entries = []  # Lista di dizionari con i dati
        LOGGER.info("FlightDataLogger inizializzato | file=%s", self.filename)

    @staticmethod
    def _normalize_position(position_world: Any) -> np.ndarray:
        """
        Normalizza una posizione 3D in un vettore colonna NumPy di forma (3, 1).

        Sono accettati input come:
        - array/lista/tupla con 3 elementi;
        - array di forma (3, 1) o equivalente;
        - dizionario con chiavi x, y, z.
        """
        if isinstance(position_world, Mapping):
            position_world = [
                position_world.get("x"),
                position_world.get("y"),
                position_world.get("z"),
            ]

        position_array = np.asarray(position_world, dtype=np.float32).reshape(-1)
        if position_array.size != 3:
            raise ValueError(
                "position_world deve contenere esattamente 3 coordinate. "
                f"Ricevuto shape/contenuto incompatibile: {position_world!r}"
            )
        return position_array.reshape(3, 1)

    def has_data(self) -> bool:
        """
        Indica se il logger contiene almeno una misura registrata.
        """
        return len(self.data_entries) > 0

    def log_position(
        self,
        position_world: np.ndarray,
        yaw_world_deg: float,
        tag_ids: List[int],
        timestamp: Optional[float] = None,
    ):
        """
        Registra una nuova posizione del drone.
        - position_world: vettore 3x1 con posizione [x, y, z] in metri
        - yaw_world_deg: angolo di yaw in gradi
        - tag_ids: lista degli ID dei tag usati per la stima
        - timestamp: timestamp in secondi (se None, usa time.time())
        """
        if timestamp is None:
            timestamp = time.time()

        position_world = self._normalize_position(position_world)
        x, y, z = position_world.flatten()
        entry = {
            "timestamp": float(timestamp),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw_deg": float(yaw_world_deg),
            "tag_ids": [int(tag_id) for tag_id in (tag_ids or [])],
        }
        self.data_entries.append(entry)
        LOGGER.debug(
            "Posizione registrata | t=%.3f | pos=(%.2f,%.2f,%.2f) | yaw=%.1f | tags=%s",
            timestamp,
            x,
            y,
            z,
            yaw_world_deg,
            tag_ids,
        )

    def log_pose_estimate(self, pose_estimate: Optional[Mapping[str, Any]], timestamp: Optional[float] = None) -> bool:
        """
        Registra una posa proveniente direttamente dalla struttura restituita
        dallo stimatore AprilTag.

        La funzione si aspetta un dizionario con almeno le chiavi:
        - position_world
        - yaw_world_deg
        - source_tag_ids

        Restituisce True se la registrazione è avvenuta, False altrimenti.
        """
        if not pose_estimate:
            return False

        try:
            position_world = pose_estimate["position_world"]
            yaw_world_deg = pose_estimate["yaw_world_deg"]
            tag_ids = pose_estimate.get("source_tag_ids", [])
        except Exception:
            LOGGER.exception("Struttura pose_estimate non valida per il logging.")
            return False

        try:
            self.log_position(
                position_world=position_world,
                yaw_world_deg=float(yaw_world_deg),
                tag_ids=list(tag_ids),
                timestamp=timestamp,
            )
        except Exception:
            LOGGER.exception("Errore durante la registrazione della posa stimata.")
            return False

        return True

    def save_to_file(self) -> Optional[Path]:
        """
        Salva tutti i dati accumulati nel file di testo.
        Formato: timestamp,x,y,z,yaw_deg,tag_ids
        """
        try:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            with self.filename.open("w", encoding="utf-8") as f:
                # Header
                f.write("# Flight Data Logger\n")
                f.write("# Timestamp (s), X (m), Y (m), Z (m), Yaw (deg), Tag IDs\n")
                f.write("# Generated at: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("\n")

                # Dati
                for entry in self.data_entries:
                    tag_ids_str = ",".join(str(tid) for tid in entry["tag_ids"])
                    line = "{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{}\n".format(
                        entry["timestamp"],
                        entry["x"],
                        entry["y"],
                        entry["z"],
                        entry["yaw_deg"],
                        tag_ids_str,
                    )
                    f.write(line)

            LOGGER.info("Dati salvati in %s | %d entries", self.filename, len(self.data_entries))
            return self.filename
        except Exception:
            LOGGER.exception("Errore nel salvataggio del file di log di volo.")
            return None

    def clear_data(self):
        """
        Cancella tutti i dati accumulati (utile per reset).
        """
        self.data_entries.clear()
        LOGGER.info("Dati cancellati")

    def get_summary(self) -> str:
        """
        Restituisce un riassunto dei dati accumulati.
        """
        if not self.data_entries:
            return "Nessun dato registrato"

        start_time = self.data_entries[0]["timestamp"]
        end_time = self.data_entries[-1]["timestamp"]
        duration = end_time - start_time
        total_entries = len(self.data_entries)

        return (
            f"Dati volo: {total_entries} posizioni registrate\n"
            f"Durata: {duration:.1f} secondi\n"
            f"Da {time.strftime('%H:%M:%S', time.localtime(start_time))} "
            f"a {time.strftime('%H:%M:%S', time.localtime(end_time))}"
        )


# Esempio di utilizzo (commentato, per documentazione)
"""
from flight_data_logger import FlightDataLogger
import numpy as np

logger = FlightDataLogger("my_flight.txt")

# Nel loop principale, dopo ogni stima posa:
position = np.array([[1.2], [3.4], [0.5]])  # Esempio posizione
yaw = 45.0
tag_ids = [0, 1]
logger.log_position(position, yaw, tag_ids)

# Oppure direttamente dalla posa fusa restituita dallo stimatore:
pose_estimate = {
    "position_world": np.array([[1.2], [3.4], [0.5]]),
    "yaw_world_deg": 45.0,
    "source_tag_ids": [0, 1],
}
logger.log_pose_estimate(pose_estimate)

# Alla fine del volo:
logger.save_to_file()
print(logger.get_summary())
"""
