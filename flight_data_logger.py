from __future__ import annotations

# Modulo standard per la gestione dei messaggi di log.
# Viene utilizzato per tracciare inizializzazioni, errori, salvataggi
# e operazioni diagnostiche senza interrompere il flusso del programma.
import logging

# Modulo standard per la gestione del tempo:
# consente di acquisire timestamp e generare nomi di sessione basati su data e ora.
import time

# Pathlib fornisce un'interfaccia orientata agli oggetti per la gestione dei percorsi
# di file e cartelle, più robusta e leggibile rispetto alle stringhe semplici.
from pathlib import Path

# Tipi usati per annotare in modo esplicito input flessibili,
# strutture dati generiche e parametri opzionali.
from typing import Any, List, Mapping, Optional

# NumPy è impiegato per normalizzare, validare e manipolare vettori numerici
# relativi alla posizione del drone nello spazio tridimensionale.
import numpy as np


# Logger associato al modulo corrente.
# L'uso di __name__ permette di integrare questo logger nella gerarchia
# di logging dell'applicazione principale.
LOGGER = logging.getLogger(__name__)


class FlightDataLogger:
    """
    Logger per raccogliere dati di volo del drone in tempo reale.

    Il logger supporta due modalità:
    1. logging semplice di una sola posa;
    2. logging comparativo tra posa grezza AprilTag e posa filtrata Kalman.

    La seconda modalità è utile per analizzare l'effetto del filtro di Kalman
    sulle posizioni stimate durante il volo.
    """

    # Separatore testuale utilizzato per salvare più ID di tag nella stessa colonna
    # del file CSV-like generato dal logger.
    TAG_IDS_SEPARATOR = "|"

    def __init__(self, filename: str | Path = "flight_data.txt"):
        """
        Inizializza il logger con il nome del file di output.
        """
        # Il nome del file viene convertito in Path per uniformare la gestione
        # dei percorsi indipendentemente dal fatto che l'utente passi una stringa
        # o un oggetto Path.
        self.filename = Path(filename)

        # Modalità legacy: singole pose registrate.
        # Ogni elemento della lista è un dizionario contenente timestamp,
        # coordinate cartesiane, yaw e ID dei tag sorgente.
        self.data_entries = []

        # Modalità comparativa: posa grezza AprilTag + posa filtrata Kalman.
        # Questa lista contiene misure accoppiate, utili per analizzare l'effetto
        # del filtro sulle stime di posizione.
        self.comparison_entries = []

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
        # Se la posizione è fornita come Mapping, ad esempio un dizionario,
        # si richiede esplicitamente la presenza delle tre coordinate x, y e z.
        # Questa scelta consente di accettare strutture dati descrittive,
        # mantenendo però un formato numerico uniforme nelle fasi successive.
        if isinstance(position_world, Mapping):
            missing_keys = [
                key
                for key in ("x", "y", "z")
                if key not in position_world or position_world.get(key) is None
            ]

            # In assenza di una o più coordinate, il dato non può rappresentare
            # una posizione tridimensionale valida e viene quindi rifiutato.
            if missing_keys:
                raise ValueError(
                    "position_world in formato Mapping deve contenere le chiavi x, y, z "
                    f"con valori numerici non nulli. Chiavi mancanti/non valide: {missing_keys}"
                )

            # Il Mapping viene convertito in una lista ordinata di coordinate,
            # così da poter essere trattato allo stesso modo di array, liste o tuple.
            position_world = [
                position_world.get("x"),
                position_world.get("y"),
                position_world.get("z"),
            ]

        # Conversione dell'input in array NumPy a precisione float32.
        # Il reshape(-1) appiattisce eventuali forme equivalenti, ad esempio
        # (3, 1), (1, 3) o una semplice lista di tre elementi.
        position_array = np.asarray(position_world, dtype=np.float32).reshape(-1)

        # La posizione del drone nel mondo deve essere espressa da esattamente
        # tre coordinate cartesiane: x, y e z.
        if position_array.size != 3:
            raise ValueError(
                "position_world deve contenere esattamente 3 coordinate. "
                f"Ricevuto shape/contenuto incompatibile: {position_world!r}"
            )

        # Si verifica che tutte le coordinate siano finite.
        # Valori NaN o infiniti renderebbero non affidabili sia il salvataggio
        # sia i grafici di confronto.
        if not np.all(np.isfinite(position_array)):
            raise ValueError(
                "position_world deve contenere solo valori numerici finiti. "
                f"Ricevuto: {position_world!r}"
            )

        # La posizione viene restituita come vettore colonna (3, 1), formato
        # comodo e coerente con molte rappresentazioni matematiche di pose e stati.
        return position_array.reshape(3, 1)

    @classmethod
    def _serialize_tag_ids(cls, tag_ids: list[int]) -> str:
        """
        Serializza gli ID dei tag in una singola colonna testuale.
        """
        # Se non sono presenti ID di tag, viene restituita una stringa vuota:
        # in questo modo il file mantiene comunque il numero atteso di colonne.
        if not tag_ids:
            return ""

        # Gli ID vengono convertiti a interi e poi a stringhe, quindi concatenati
        # usando il separatore definito a livello di classe.
        return cls.TAG_IDS_SEPARATOR.join(str(int(tag_id)) for tag_id in tag_ids)

    def has_data(self) -> bool:
        """
        Indica se il logger contiene almeno una misura registrata.
        """
        # Il logger viene considerato non vuoto se contiene dati in almeno una
        # delle due modalità supportate: legacy o comparativa.
        return len(self.data_entries) > 0 or len(self.comparison_entries) > 0

    def log_position(
        self,
        position_world: np.ndarray,
        yaw_world_deg: float,
        tag_ids: List[int],
        timestamp: Optional[float] = None,
    ):
        """
        Registra una nuova posizione del drone.

        Questa funzione mantiene la compatibilità con la versione precedente
        del logger.
        """
        # Se il chiamante non fornisce un timestamp, viene acquisito il tempo
        # corrente in secondi Unix. Questo permette di associare ogni misura
        # all'istante effettivo di registrazione.
        if timestamp is None:
            timestamp = time.time()

        # La posizione viene validata e convertita in una forma numerica standard.
        position_world = self._normalize_position(position_world)

        # Si estraggono le tre coordinate dal vettore colonna per memorizzarle
        # separatamente nel dizionario della misura.
        x, y, z = position_world.flatten()

        # Ogni misura legacy è rappresentata come dizionario.
        # La conversione esplicita a float e int rende il dato più stabile
        # per la serializzazione su file.
        entry = {
            "timestamp": float(timestamp),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw_deg": float(yaw_world_deg),
            "tag_ids": [int(tag_id) for tag_id in (tag_ids or [])],
        }

        # La misura viene accumulata in memoria; il salvataggio su file avviene
        # successivamente, quando viene invocato save_to_file o export_session.
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

    def log_pose_estimate(
        self,
        pose_estimate: Optional[Mapping[str, Any]],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Registra una posa proveniente direttamente dalla struttura restituita
        dallo stimatore AprilTag.

        Questa funzione mantiene la compatibilità con la versione precedente.
        """
        # Se la stima è assente o vuota, non vi sono dati da registrare.
        # La funzione restituisce False per segnalare che nessun log è stato aggiunto.
        if not pose_estimate:
            return False

        try:
            # La struttura attesa contiene almeno:
            # - position_world: posizione 3D nel sistema di riferimento globale;
            # - yaw_world_deg: orientamento yaw espresso in gradi;
            # - source_tag_ids: eventuali ID degli AprilTag usati per la stima.
            position_world = pose_estimate["position_world"]
            yaw_world_deg = pose_estimate["yaw_world_deg"]
            tag_ids = pose_estimate.get("source_tag_ids", [])
        except Exception:
            # Qualsiasi struttura non conforme viene intercettata e tracciata
            # tramite logging, evitando che l'errore interrompa il programma principale.
            LOGGER.exception("Struttura pose_estimate non valida per il logging.")
            return False

        try:
            # La registrazione effettiva viene delegata a log_position,
            # così da riutilizzare la stessa validazione e lo stesso formato legacy.
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

    def log_pose_pair(
        self,
        raw_pose_estimate: Optional[Mapping[str, Any]],
        filtered_pose_estimate: Optional[Mapping[str, Any]],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Registra una coppia di pose:

        - raw_pose_estimate      : posa grezza/fusa ottenuta da AprilTag;
        - filtered_pose_estimate : posa filtrata tramite Kalman.

        Questa funzione permette di produrre grafici comparativi.
        """
        # La posa grezza è indispensabile perché rappresenta il riferimento
        # rispetto al quale confrontare la stima filtrata.
        if not raw_pose_estimate:
            return False

        # In assenza di timestamp esplicito, si usa il tempo corrente.
        if timestamp is None:
            timestamp = time.time()

        # Se la posa filtrata non è disponibile, si assume coincidente con quella grezza.
        # Questo evita di perdere il campione e consente di mantenere una struttura
        # uniforme nei dati comparativi.
        if filtered_pose_estimate is None:
            filtered_pose_estimate = raw_pose_estimate

        try:
            # Entrambe le posizioni vengono normalizzate nel medesimo formato.
            # Ciò consente un confronto numerico diretto tra stima raw e stima filtrata.
            raw_position = self._normalize_position(raw_pose_estimate["position_world"])
            filtered_position = self._normalize_position(filtered_pose_estimate["position_world"])

            # Estrazione delle coordinate cartesiane delle due pose.
            raw_x, raw_y, raw_z = raw_position.flatten()
            filtered_x, filtered_y, filtered_z = filtered_position.flatten()

            # Lo yaw viene letto se disponibile; in caso contrario viene assunto nullo
            # per la posa grezza e pari a quello grezzo per la posa filtrata.
            raw_yaw = float(raw_pose_estimate.get("yaw_world_deg", 0.0))
            filtered_yaw = float(filtered_pose_estimate.get("yaw_world_deg", raw_yaw))

            # Gli ID dei tag sono ricavati preferibilmente dalla posa grezza.
            # Se non disponibili, si tenta di recuperarli dalla posa filtrata.
            tag_ids = raw_pose_estimate.get(
                "source_tag_ids",
                filtered_pose_estimate.get("source_tag_ids", []),
            )

            # Dizionario contenente tutte le grandezze necessarie al confronto:
            # coordinate raw, coordinate filtrate, differenze componente per componente
            # e informazioni ausiliarie sulle sorgenti.
            entry = {
                "timestamp": float(timestamp),

                "raw_x": float(raw_x),
                "raw_y": float(raw_y),
                "raw_z": float(raw_z),
                "raw_yaw_deg": raw_yaw,

                "filtered_x": float(filtered_x),
                "filtered_y": float(filtered_y),
                "filtered_z": float(filtered_z),
                "filtered_yaw_deg": filtered_yaw,

                "error_x": float(filtered_x - raw_x),
                "error_y": float(filtered_y - raw_y),
                "error_z": float(filtered_z - raw_z),

                "tag_ids": [int(tag_id) for tag_id in (tag_ids or [])],
                "raw_source": str(raw_pose_estimate.get("source", "raw")),
                "filtered_source": str(filtered_pose_estimate.get("source", "filtered")),
            }

            # La norma euclidea dell'errore quantifica con un singolo valore
            # l'entità complessiva dello scostamento tra posa filtrata e posa raw.
            error_norm = np.linalg.norm(
                [
                    entry["error_x"],
                    entry["error_y"],
                    entry["error_z"],
                ]
            )
            entry["error_norm"] = float(error_norm)

            # La misura comparativa viene accumulata per il successivo salvataggio
            # e per la generazione dei grafici.
            self.comparison_entries.append(entry)

            LOGGER.debug(
                "Posa raw/filtered registrata | t=%.3f | raw=(%.2f,%.2f,%.2f) | filtered=(%.2f,%.2f,%.2f) | tags=%s",
                timestamp,
                raw_x,
                raw_y,
                raw_z,
                filtered_x,
                filtered_y,
                filtered_z,
                tag_ids,
            )

            return True

        except Exception:
            # Eventuali errori di formato, conversione o accesso ai campi
            # vengono registrati senza propagare l'eccezione al chiamante.
            LOGGER.exception("Errore durante la registrazione della coppia raw/filtered.")
            return False

    def _save_legacy_data_to_file(self, output_path: Path) -> Path:
        """
        Salva il log nella vecchia modalità: una sola posa per riga.
        """
        # Il file viene aperto in scrittura con codifica UTF-8.
        # L'uso del context manager garantisce la chiusura corretta del file.
        with output_path.open("w", encoding="utf-8") as f:
            # Intestazione descrittiva del file, utile per interpretare
            # il significato delle colonne in fase di analisi successiva.
            f.write("# Flight Data Logger\n")
            f.write("# Timestamp (s), X (m), Y (m), Z (m), Yaw (deg), Tag IDs\n")
            f.write("# Generated at: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("\n")

            # Ogni entry viene convertita in una riga testuale separata da virgole.
            for entry in self.data_entries:
                tag_ids_str = self._serialize_tag_ids(entry["tag_ids"])
                line = "{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{}\n".format(
                    entry["timestamp"],
                    entry["x"],
                    entry["y"],
                    entry["z"],
                    entry["yaw_deg"],
                    tag_ids_str,
                )
                f.write(line)

        return output_path

    def _save_comparison_data_to_file(self, output_path: Path) -> Path:
        """
        Salva il log comparativo raw AprilTag / filtered Kalman.
        """
        with output_path.open("w", encoding="utf-8") as f:
            # Intestazione del file comparativo.
            # Oltre alla data di generazione, viene indicato anche il separatore
            # usato per rappresentare più ID di tag nella stessa cella.
            f.write("# Flight Data Logger - Raw AprilTag vs Kalman Filter\n")
            f.write("# Generated at: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("# Separator: comma\n")
            f.write("# tag_ids separator: '{}'\n".format(self.TAG_IDS_SEPARATOR))
            f.write("\n")

            # Riga di intestazione delle colonne.
            # Il formato è compatibile con strumenti di analisi dati come Excel,
            # pandas o software di plotting.
            f.write(
                "timestamp,"
                "raw_x,raw_y,raw_z,raw_yaw_deg,"
                "filtered_x,filtered_y,filtered_z,filtered_yaw_deg,"
                "error_x,error_y,error_z,error_norm,"
                "tag_ids,raw_source,filtered_source\n"
            )

            # Scrittura di tutte le osservazioni comparative accumulate.
            for entry in self.comparison_entries:
                tag_ids_str = self._serialize_tag_ids(entry["tag_ids"])

                # La formattazione numerica usa più cifre decimali rispetto
                # alla modalità legacy, perché il confronto raw/filtered può
                # richiedere una maggiore precisione.
                line = (
                    "{timestamp:.3f},"
                    "{raw_x:.6f},{raw_y:.6f},{raw_z:.6f},{raw_yaw_deg:.3f},"
                    "{filtered_x:.6f},{filtered_y:.6f},{filtered_z:.6f},{filtered_yaw_deg:.3f},"
                    "{error_x:.6f},{error_y:.6f},{error_z:.6f},{error_norm:.6f},"
                    "{tag_ids},{raw_source},{filtered_source}\n"
                ).format(
                    timestamp=entry["timestamp"],
                    raw_x=entry["raw_x"],
                    raw_y=entry["raw_y"],
                    raw_z=entry["raw_z"],
                    raw_yaw_deg=entry["raw_yaw_deg"],
                    filtered_x=entry["filtered_x"],
                    filtered_y=entry["filtered_y"],
                    filtered_z=entry["filtered_z"],
                    filtered_yaw_deg=entry["filtered_yaw_deg"],
                    error_x=entry["error_x"],
                    error_y=entry["error_y"],
                    error_z=entry["error_z"],
                    error_norm=entry["error_norm"],
                    tag_ids=tag_ids_str,
                    raw_source=entry["raw_source"],
                    filtered_source=entry["filtered_source"],
                )

                f.write(line)

        return output_path

    def save_to_file(self, output_dir: Optional[str | Path] = None) -> Optional[Path]:
        """
        Salva tutti i dati accumulati nel file di testo.

        Se output_dir è specificato, il file viene salvato dentro tale cartella.
        """
        try:
            # Se non viene specificata una cartella di output, si usa direttamente
            # il percorso indicato in fase di inizializzazione del logger.
            if output_dir is None:
                output_path = self.filename
            else:
                # Se invece viene fornita una cartella, questa viene creata
                # se necessario e il file mantiene il nome originario.
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / self.filename.name

            # Si garantisce che la cartella padre del file esista prima
            # di procedere con l'apertura in scrittura.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # La modalità comparativa ha priorità sulla modalità legacy:
            # se sono presenti misure raw/filtered, viene prodotto il file
            # più informativo per l'analisi del filtro di Kalman.
            if self.comparison_entries:
                saved_path = self._save_comparison_data_to_file(output_path)
            else:
                saved_path = self._save_legacy_data_to_file(output_path)

            LOGGER.info(
                "Dati salvati in %s | legacy=%d | comparison=%d",
                saved_path,
                len(self.data_entries),
                len(self.comparison_entries),
            )

            return saved_path

        except Exception:
            # In caso di errore di I/O o di formattazione dei dati,
            # l'eccezione viene registrata e la funzione restituisce None.
            LOGGER.exception("Errore nel salvataggio del file di log di volo.")
            return None

    def save_plots(self, output_dir: str | Path) -> list[Path]:
        """
        Genera grafici PNG per confrontare posa grezza AprilTag e posa filtrata Kalman.

        I grafici vengono generati solo se esistono almeno due campioni comparativi.
        """
        # La cartella di output viene creata preventivamente, così da poter salvare
        # direttamente tutte le immagini prodotte.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per costruire un grafico temporale significativo sono necessari
        # almeno due campioni.
        if len(self.comparison_entries) < 2:
            LOGGER.warning("Campioni insufficienti per generare grafici Kalman.")
            return []

        try:
            # Import locale di matplotlib: la dipendenza è necessaria solo
            # quando si richiede effettivamente la generazione dei grafici.
            import matplotlib

            # Backend non interattivo, adatto all'esportazione di immagini
            # anche in ambienti senza interfaccia grafica.
            matplotlib.use("Agg")

            import matplotlib.pyplot as plt
        except ImportError:
            LOGGER.exception("matplotlib non installato: impossibile generare i grafici.")
            return []

        # I timestamp assoluti vengono convertiti in un asse temporale relativo,
        # più leggibile per l'analisi della sessione di volo.
        timestamps = np.array(
            [entry["timestamp"] for entry in self.comparison_entries],
            dtype=np.float64,
        )
        t = timestamps - timestamps[0]

        # Estrazione delle coordinate grezze AprilTag per ciascun asse.
        raw_x = np.array([entry["raw_x"] for entry in self.comparison_entries], dtype=np.float64)
        raw_y = np.array([entry["raw_y"] for entry in self.comparison_entries], dtype=np.float64)
        raw_z = np.array([entry["raw_z"] for entry in self.comparison_entries], dtype=np.float64)

        # Estrazione delle coordinate filtrate tramite Kalman per ciascun asse.
        filtered_x = np.array([entry["filtered_x"] for entry in self.comparison_entries], dtype=np.float64)
        filtered_y = np.array([entry["filtered_y"] for entry in self.comparison_entries], dtype=np.float64)
        filtered_z = np.array([entry["filtered_z"] for entry in self.comparison_entries], dtype=np.float64)

        # La norma dell'errore consente di osservare nel tempo quanto il filtro
        # modifichi complessivamente la stima raw.
        error_norm = np.array(
            [entry["error_norm"] for entry in self.comparison_entries],
            dtype=np.float64,
        )

        # Lista dei percorsi dei file PNG effettivamente salvati.
        saved_paths = []

        def save_axis_plot(axis_name: str, raw_values, filtered_values, filename: str):
            # Funzione interna di utilità per evitare duplicazione di codice
            # nella generazione dei grafici relativi agli assi X, Y e Z.
            fig = plt.figure(figsize=(10, 5))
            plt.plot(t, raw_values, label=f"{axis_name} raw AprilTag")
            plt.plot(t, filtered_values, label=f"{axis_name} Kalman")
            plt.xlabel("Tempo relativo [s]")
            plt.ylabel(f"{axis_name} [m]")
            plt.title(f"Confronto {axis_name}: raw AprilTag vs Kalman")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            path = output_dir / filename
            fig.savefig(path, dpi=150)

            # La chiusura esplicita della figura evita accumulo di memoria
            # quando vengono generati più grafici nella stessa sessione.
            plt.close(fig)
            saved_paths.append(path)

        # Grafici temporali separati per le tre coordinate cartesiane.
        save_axis_plot("X", raw_x, filtered_x, "x_raw_vs_filtered.png")
        save_axis_plot("Y", raw_y, filtered_y, "y_raw_vs_filtered.png")
        save_axis_plot("Z", raw_z, filtered_z, "z_raw_vs_filtered.png")

        # Grafico della traiettoria nel piano XY.
        # È utile per osservare visivamente la differenza geometrica tra
        # traiettoria stimata da AprilTag e traiettoria filtrata.
        fig = plt.figure(figsize=(8, 8))
        plt.plot(raw_x, raw_y, label="Traiettoria raw AprilTag")
        plt.plot(filtered_x, filtered_y, label="Traiettoria Kalman")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Traiettoria XY: raw AprilTag vs Kalman")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        path = output_dir / "trajectory_xy_raw_vs_filtered.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        # Grafico della norma dell'errore nel tempo.
        # Questo grafico sintetizza l'intensità della correzione applicata dal filtro.
        fig = plt.figure(figsize=(10, 5))
        plt.plot(t, error_norm, label="Norma differenza Kalman - raw")
        plt.xlabel("Tempo relativo [s]")
        plt.ylabel("Errore posizione [m]")
        plt.title("Entità della correzione introdotta dal filtro")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        path = output_dir / "filter_error_norm.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        LOGGER.info("Grafici Kalman salvati in %s", output_dir)

        return saved_paths

    def export_session(self, output_root: Optional[str | Path] = None) -> Optional[Path]:
        """
        Crea una cartella di sessione contenente:
        - flight_data.txt;
        - grafici raw AprilTag vs Kalman, se disponibili.

        Questa funzione è pensata per essere chiamata quando l'utente preme Options
        e il programma termina.
        """
        try:
            # Se non viene specificata una radice di output, la sessione viene
            # creata nella stessa cartella associata al file di log configurato.
            if output_root is None:
                output_root = self.filename.parent

            output_root = Path(output_root)

            # Il nome della sessione include data e ora, così da distinguere
            # automaticamente esecuzioni diverse del programma.
            session_name = time.strftime("flight_session_%Y%m%d_%H%M%S")
            session_dir = output_root / session_name
            session_dir.mkdir(parents=True, exist_ok=True)

            # Salvataggio del file testuale principale nella cartella di sessione.
            saved_log = self.save_to_file(output_dir=session_dir)
            if saved_log is None:
                return None

            # Generazione dei grafici, se sono disponibili dati comparativi sufficienti.
            self.save_plots(session_dir)

            return session_dir

        except Exception:
            LOGGER.exception("Errore durante l'esportazione della sessione di volo.")
            return None

    def clear_data(self):
        """
        Cancella tutti i dati accumulati.
        """
        # La cancellazione interessa entrambe le modalità di logging,
        # così da riportare l'oggetto a uno stato privo di misure memorizzate.
        self.data_entries.clear()
        self.comparison_entries.clear()
        LOGGER.info("Dati cancellati")

    def get_summary(self) -> str:
        """
        Restituisce un riassunto dei dati accumulati.
        """
        # Se sono presenti dati comparativi, questi hanno priorità nel riepilogo,
        # perché rappresentano la modalità più informativa del logger.
        if self.comparison_entries:
            entries = self.comparison_entries
            start_time = entries[0]["timestamp"]
            end_time = entries[-1]["timestamp"]
            duration = end_time - start_time
            total_entries = len(entries)

            return (
                f"Dati volo: {total_entries} pose raw/filtered registrate\n"
                f"Durata: {duration:.1f} secondi\n"
                f"Da {time.strftime('%H:%M:%S', time.localtime(start_time))} "
                f"a {time.strftime('%H:%M:%S', time.localtime(end_time))}"
            )

        # Se non sono presenti dati comparativi ma esistono dati legacy,
        # il riepilogo viene costruito sulle posizioni singole registrate.
        if self.data_entries:
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

        # Caso in cui il logger non contiene ancora alcuna misura.
        return "Nessun dato registrato"