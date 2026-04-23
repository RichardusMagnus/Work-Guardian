import logging
import time
from typing import List, Optional
import numpy as np

LOGGER = logging.getLogger(__name__)


class FlightDataLogger:
    """
    Logger per raccogliere dati di volo del drone in tempo reale.
    Registra posizione, yaw e ID dei tag usati per la stima.
    Produce un file di testo con i dati accumulati.
    """

    def __init__(self, filename: str = "flight_data.txt"):
        """
        Inizializza il logger con il nome del file di output.
        """
        self.filename = filename
        self.data_entries = []  # Lista di dizionari con i dati
        LOGGER.info("FlightDataLogger inizializzato | file=%s", filename)

    def log_position(
        self,
        position_world: np.ndarray,
        yaw_world_deg: float,
        tag_ids: List[int],
        timestamp: Optional[float] = None
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

        x, y, z = position_world.flatten()
        entry = {
            "timestamp": timestamp,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw_deg": float(yaw_world_deg),
            "tag_ids": tag_ids.copy() if tag_ids else [],
        }
        self.data_entries.append(entry)
        LOGGER.debug("Posizione registrata | t=%.3f | pos=(%.2f,%.2f,%.2f) | yaw=%.1f | tags=%s",
                     timestamp, x, y, z, yaw_world_deg, tag_ids)

    def save_to_file(self):
        """
        Salva tutti i dati accumulati nel file di testo.
        Formato: timestamp,x,y,z,yaw_deg,tag_ids
        """
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
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
                        tag_ids_str
                    )
                    f.write(line)

            LOGGER.info("Dati salvati in %s | %d entries", self.filename, len(self.data_entries))
        except Exception as e:
            LOGGER.error("Errore nel salvataggio del file: %s", e)

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

# Alla fine del volo:
logger.save_to_file()
print(logger.get_summary())
"""