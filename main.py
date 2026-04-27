import logging
import time
from typing import Optional

# OpenCV è utilizzato per la gestione dei frame video, il disegno degli overlay
# grafici e la visualizzazione della finestra contenente lo stream elaborato.
import cv2

# Pygame è impiegato per l'inizializzazione e l'aggiornamento dell'interfaccia
# associata al joystick, mantenendo reattivo il sottosistema di input.
import pygame

# Configurazione centralizzata dell'applicazione: contiene parametri globali
# come velocità, soglie YOLO, percorsi dei modelli, opzioni video e parametri
# per la stima della posa.
from app_config import APP_CONFIG

# Modulo dedicato alla registrazione delle pose stimate durante il volo e,
# se previsto dalla sua implementazione, all'esportazione dei grafici di sessione.
from flight_data_logger import FlightDataLogger

# Funzioni di supporto per inizializzare il joystick, leggere gli eventi,
# convertire gli input in comandi RC e chiudere correttamente il sottosistema.
from joystick_tello import (
    close_joystick,
    get_command,
    init_joystick,
    print_joystick_help,
    read_events,
)

# Filtro di Kalman usato per stabilizzare la stima della posizione assoluta
# ottenuta dal sottosistema di visione basato su AprilTag.
from pose_kalman_filter import PositionKalmanFilter

# Controller concreto per interagire con il drone Tello reale:
# connessione, decollo, atterraggio, stream video e comandi RC.
from real_tello_controller import RealTelloController

# Stimatore di posa della camera/drone basato su rilevamento di AprilTag
# e su parametri di calibrazione della camera.
from tello_pose_detection import CameraPoseEstimator

# Wrapper per i modelli YOLO utilizzati nella object detection.
from vision_detector import ObjectDetector

# Logger di modulo, utile per eventuali messaggi diagnostici coerenti con il nome del file corrente.
LOGGER = logging.getLogger(__name__)

# Titolo costante della finestra OpenCV. Viene usato sia per mostrare il frame
# sia per verificare se l'utente ha chiuso manualmente la finestra video.
VIDEO_WINDOW_TITLE = "Tello Detection"


def _resolve_log_level(default=logging.WARNING):
    # Converte il livello di log definito nella configurazione applicativa
    # in una costante del modulo logging. Se il valore configurato non è valido,
    # viene utilizzato un livello di default prudenziale.
    level_name = str(APP_CONFIG.log_level).upper()
    return getattr(logging, level_name, default)


def configure_logging():
    """
    Configura il sistema di logging dell'applicazione.

    L'obiettivo è limitare la verbosità dei moduli esterni più "rumorosi"
    e mantenere nel terminale un output leggibile, privilegiando i messaggi
    utente stampati esplicitamente tramite print.
    """
    log_level = _resolve_log_level()

    # Inizializza la configurazione globale del logging.
    # force=True garantisce la sovrascrittura di eventuali configurazioni precedenti.
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s | %(message)s",
        force=True,
    )

    # Limita la verbosità di librerie esterne frequentemente molto prolisse.
    # max(logging.WARNING, log_level) evita di impostare tali librerie a un livello
    # più verboso di WARNING quando il livello applicativo è molto dettagliato.
    logging.getLogger("djitellopy").setLevel(max(logging.WARNING, log_level))
    logging.getLogger("ultralytics").setLevel(max(logging.WARNING, log_level))

    # Imposta il livello di log per i moduli interni principali dell'applicazione.
    # In questo modo, i componenti sviluppati nel progetto seguono una politica
    # di logging uniforme e controllata da APP_CONFIG.
    logging.getLogger(__name__).setLevel(log_level)
    logging.getLogger("real_tello_controller").setLevel(log_level)
    logging.getLogger("vision_detector").setLevel(log_level)
    logging.getLogger("tello_pose_detection").setLevel(log_level)
    logging.getLogger("joystick_tello").setLevel(log_level)
    logging.getLogger("flight_data_logger").setLevel(log_level)
    logging.getLogger("pose_kalman_filter").setLevel(log_level)


def print_phase(title: str):
    # Stampa una intestazione testuale per distinguere chiaramente
    # le diverse fasi del ciclo di inizializzazione del sistema.
    print(f"\n--- {title} ---")


def _format_battery_value(status: dict) -> str:
    # Estrae il valore di batteria dal dizionario di stato e lo rende
    # leggibile in output. Se l'informazione non è disponibile, mostra "--".
    battery = status.get("battery")
    return "--" if battery is None else f"{battery}%"


def print_drone_status(controller: RealTelloController, yolo_enabled: bool):
    # Recupera e stampa in forma compatta lo stato corrente del drone,
    # includendo connettività, volo, batteria e stato della detection.
    # Questa funzione è usata sia dopo l'inizializzazione sia periodicamente
    # durante il ciclo principale.
    status = controller.get_status()

    print("\n[STATO DRONE]")
    print(f"  Connesso      : {status.get('connected')}")
    print(f"  In volo       : {status.get('flying')}")
    print(f"  Batteria      : {_format_battery_value(status)}")
    print(f"  YOLO attiva   : {yolo_enabled}")
    print("----------------------------------------")


class DroneControlLoop:
    # Questa classe incapsula la logica di controllo manuale del drone.
    # Si occupa di:
    # - leggere gli eventi provenienti dal joystick;
    # - gestire i comandi ad alto livello (decollo, atterraggio, uscita);
    # - attivare/disattivare la detection;
    # - inviare i comandi RC continui al controller del drone.
    #
    # La separazione tra DroneControlLoop e VisionLoop consente di mantenere
    # distinta la logica di pilotaggio dalla logica di elaborazione video.

    def __init__(self, controller: RealTelloController, speed: int, detection_available: bool = True):
        # Riferimento al controller reale del drone.
        # Attraverso questo oggetto vengono eseguite tutte le operazioni fisiche:
        # decollo, atterraggio e invio dei comandi di velocità.
        self.controller = controller

        # Velocità da utilizzare per convertire gli input del joystick
        # nei comandi di movimento RC.
        self.speed = speed

        # Flag che indica se i moduli di detection sono effettivamente disponibili.
        # Il valore evita che l'utente possa attivare una funzionalità non inizializzata.
        self.detection_available = detection_available

        # Stato interno dell'attivazione della object detection.
        # Il valore viene modificato tramite un comando discreto del joystick.
        self._detection_enabled = False

    def is_detection_enabled(self) -> bool:
        # Restituisce lo stato corrente della detection,
        # usato dal ciclo principale e dal modulo di visione.
        return self._detection_enabled

    def step(self) -> bool:
        # Esegue un singolo passo del ciclo di controllo:
        # 1. legge gli eventi del joystick;
        # 2. gestisce eventuali comandi discreti;
        # 3. invia al drone i comandi RC correnti;
        # 4. restituisce True/False per indicare se il programma deve continuare.
        actions = read_events()

        # Gestione del decollo.
        # Il comando viene inoltrato solo al controller, che decide se l'operazione
        # è coerente con lo stato attuale del drone.
        if actions["takeoff"]:
            if self.controller.takeoff():
                print("\n[EVENTO] Decollo eseguito.")
            else:
                print("\n[AVVISO] Decollo ignorato: controlla connessione o stato del drone.")

        # Gestione dell'atterraggio.
        # Anche in questo caso la verifica effettiva dello stato di volo è delegata
        # al controller, mantenendo qui una logica applicativa di alto livello.
        if actions["land"]:
            if self.controller.land():
                print("\n[EVENTO] Atterraggio eseguito.")
            else:
                print("\n[AVVISO] Atterraggio ignorato: il drone non risulta in volo.")

        # Gestione dell'attivazione/disattivazione della object detection.
        # Il comando "detect" agisce come toggle: ogni pressione inverte lo stato.
        if actions.get("detect", False):
            if not self.detection_available:
                print("\n[AVVISO] Detection YOLO non disponibile: comando ignorato.")
            else:
                self._detection_enabled = not self._detection_enabled
                stato = "attivata" if self._detection_enabled else "disattivata"
                print(f"\n[EVENTO] Detection YOLO {stato}.")

        # Richiesta di uscita: prima si azzerano i comandi RC per sicurezza,
        # poi si segnala al chiamante di terminare il ciclo principale.
        if actions["quit"]:
            self.controller.send_rc_control(0, 0, 0, 0)
            print("\n[EVENTO] Uscita richiesta.")
            return False

        # Calcola il comando di movimento continuo sulla base dello stato corrente
        # del joystick e della velocità configurata.
        # Il dizionario restituito contiene le quattro componenti RC:
        # sinistra/destra, avanti/indietro, su/giù e imbardata.
        command = get_command(self.speed)
        self.controller.send_rc_control(
            command["lr"],
            command["fb"],
            command["ud"],
            command["yaw"],
        )

        return True


class VisionLoop:
    # Questa classe gestisce il sottosistema di visione.
    # In particolare:
    # - acquisisce i frame dal controller del drone;
    # - normalizza il formato colore se necessario;
    # - opzionalmente corregge la distorsione dell'immagine;
    # - esegue object detection multi-modello;
    # - opzionalmente esegue stima di posa;
    # - applica opzionalmente un filtro di Kalman alla posa stimata;
    # - disegna overlay informativi e visualizza il frame finale.
    #
    # La classe mantiene inoltre una cache dello stato del drone, così da evitare
    # interrogazioni troppo frequenti al controller durante l'elaborazione video.

    def __init__(
        self,
        detectors,
        pose_estimator: Optional[CameraPoseEstimator],
        frame_timeout_sec: float,
        frame_from_controller_is_rgb: bool = False,
        status_refresh_sec: float = 1.0,
        flight_data_logger: Optional[FlightDataLogger] = None,
        pose_filter: Optional[PositionKalmanFilter] = None,
    ):
        # Lista dei detector YOLO inizializzati. Viene forzata a lista per
        # garantire una struttura uniforme anche se in ingresso è None o iterabile generico.
        self.detectors = list(detectors or [])

        # Stimatore di posa della camera/drone, eventualmente assente.
        # Quando None, l'applicazione si limita allo stream video e alla detection YOLO.
        self.pose_estimator = pose_estimator

        # Tempo massimo tollerato senza ricezione di frame video.
        # Superata questa soglia, il ciclo di visione segnala un errore di stream.
        self.frame_timeout_sec = frame_timeout_sec

        # Specifica se il frame ricevuto dal controller è in formato RGB;
        # in tal caso sarà convertito in BGR per compatibilità con OpenCV.
        self.frame_from_controller_is_rgb = frame_from_controller_is_rgb

        # Intervallo minimo tra due refresh dello stato del drone,
        # per evitare interrogazioni troppo frequenti.
        self.status_refresh_sec = status_refresh_sec

        # Timestamp dell'ultimo frame ricevuto correttamente.
        # time.monotonic() è adatto a misure di intervalli temporali perché
        # non risente di eventuali modifiche dell'orologio di sistema.
        self.last_frame_received_at = time.monotonic()

        # Timestamp dell'ultimo aggiornamento dello stato del drone.
        self.last_status_refresh_at = 0.0

        # Cache locale dello stato del drone, usata per mostrare le informazioni
        # a video anche quando il refresh non viene eseguito a ogni frame.
        self.cached_status = {
            "connected": False,
            "flying": False,
            "battery": None,
        }

        # Logger opzionale dei dati di volo, alimentato a partire dalla posa fusa
        # stimata dal modulo AprilTag quando disponibile.
        self.flight_data_logger = flight_data_logger

        # Filtro di Kalman opzionale applicato alla posa assoluta stimata.
        # Il filtro non modifica la logica di detection AprilTag: opera solo
        # dopo la fusione della posa camera/drone.
        self.pose_filter = pose_filter

        # Ultima posa filtrata disponibile. Può essere usata per debug,
        # logging o future estensioni del controllo.
        self.last_filtered_pose_estimate = None

    def _normalize_frame_for_opencv(self, frame):
        # OpenCV utilizza tipicamente il formato BGR.
        # Se il controller restituisce frame RGB, si esegue la conversione.
        # La funzione centralizza questa scelta, evitando duplicazioni nel ciclo principale.
        if self.frame_from_controller_is_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _refresh_status(self, controller: RealTelloController):
        # Aggiorna la cache dello stato del drone solo se è trascorso
        # un intervallo sufficiente dall'ultimo refresh.
        now = time.monotonic()
        if (now - self.last_status_refresh_at) < self.status_refresh_sec:
            return

        self.last_status_refresh_at = now
        try:
            # In caso di successo, salva nella cache solo i campi di interesse.
            # Questa scelta riduce il contenuto gestito dall'interfaccia video
            # alle sole informazioni effettivamente visualizzate.
            status = controller.get_status()
            self.cached_status = {
                "connected": status.get("connected", False),
                "flying": status.get("flying", False),
                "battery": status.get("battery"),
            }
        except Exception:
            # In caso di errore nel recupero dello stato, si imposta una condizione
            # sicura e neutra, evitando il blocco dell'interfaccia video.
            self.cached_status = {
                "connected": False,
                "flying": False,
                "battery": None,
            }

    def _draw_status_overlay(self, frame, detection_enabled: bool = False):
        # Disegna sul frame corrente una sovraimpressione testuale contenente
        # informazioni di stato rilevanti per l'utente.
        connected_text = str(self.cached_status.get("connected", False))
        flying_text = str(self.cached_status.get("flying", False))

        battery_value = self.cached_status.get("battery")
        battery_text = "--" if battery_value is None else f"{battery_value}%"

        detection_text = str(detection_enabled)

        # Le stringhe sono organizzate in una lista per poterle disegnare
        # in modo uniforme con un semplice ciclo.
        lines = [
            f"Connesso      : {connected_text}",
            f"In volo       : {flying_text}",
            f"Batteria      : {battery_text}",
            f"YOLO attiva   : {detection_text}",
        ]

        # Coordinate iniziali del riquadro testuale.
        x = 20
        y = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 30

    def _log_latest_pose_estimate(self):
        # Registra nel logger la posa assoluta più informativa disponibile.
        # Si privilegia la posa del drone/body; in assenza di essa si usa,
        # come fallback, la posa assoluta della camera.
        if self.flight_data_logger is None or self.pose_estimator is None:
            return

        raw_pose = self.pose_estimator.last_fused_body_pose
        if raw_pose is None:
            raw_pose = self.pose_estimator.last_fused_camera_pose

        if raw_pose is None:
            return

        # All'inizio la posa filtrata coincide con la posa grezza.
        # Se il filtro è disponibile, viene poi sostituita dalla posa Kalman.
        filtered_pose = raw_pose

        if self.pose_filter is not None:
            try:
                kalman_pose = self.pose_filter.filter_pose_estimate(raw_pose)
                if kalman_pose is not None:
                    filtered_pose = kalman_pose
                    self.last_filtered_pose_estimate = kalman_pose
            except Exception:
                # L'errore del filtro viene registrato nel logger, ma non interrompe
                # il ciclo di visione: la stima grezza rimane comunque disponibile.
                LOGGER.exception("Errore durante il filtraggio Kalman della posa.")
                self.last_filtered_pose_estimate = None

        # Registrazione comparativa:
        # - raw_pose      = posa stimata/fusa da AprilTag;
        # - filtered_pose = posa stabilizzata dal filtro di Kalman.
        #
        # In questo modo, alla fine del volo, è possibile generare grafici
        # che mostrano l'effetto del filtro.
        if hasattr(self.flight_data_logger, "log_pose_pair"):
            self.flight_data_logger.log_pose_pair(
                raw_pose_estimate=raw_pose,
                filtered_pose_estimate=filtered_pose,
            )
        else:
            # Fallback per compatibilità con una vecchia versione del logger.
            # In assenza del metodo comparativo, viene registrata soltanto
            # la posa filtrata o, se il filtro non è disponibile, la posa grezza.
            self.flight_data_logger.log_pose_estimate(filtered_pose)

    def _draw_multi_model_detections(self, frame, analysis_frame):
        # Esegue la detection per ciascun modello disponibile e disegna
        # bounding box e label sul frame da mostrare a video.
        # Ogni detector può essere associato a un nome e a un colore specifico,
        # così da distinguere visivamente i risultati dei diversi modelli.
        for item in self.detectors:
            detector = item["detector"]
            color = item["color"]
            model_name = item["name"]

            try:
                # Il metodo detect restituisce il frame elaborato e la lista
                # delle detection; qui interessa soprattutto il secondo elemento.
                _, detections = detector.detect(analysis_frame)
            except Exception as exc:
                # Il fallimento di un singolo detector non deve interrompere
                # l'intero ciclo di visione: si segnala il problema e si prosegue.
                print(f"\n[ERRORE] Detector '{model_name}' non disponibile sul frame corrente: {exc}")
                continue

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{model_name}: {det['label']} {det['confidence']:.2f}"

                # Disegno del rettangolo di delimitazione dell'oggetto rilevato.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Disegno dell'etichetta con nome del modello, classe e confidenza.
                # max(20, y1 - 10) evita che il testo venga posizionato fuori
                # dall'immagine quando il bounding box è vicino al bordo superiore.
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    def step(self, controller: RealTelloController, run_detection: bool = False) -> bool:
        # Esegue un singolo passo del ciclo di visione:
        # 1. acquisisce un frame;
        # 2. controlla eventuali timeout del video stream;
        # 3. applica preprocessamenti e analisi;
        # 4. visualizza il risultato;
        # 5. restituisce True/False per indicare se il programma può continuare.
        frame = controller.get_frame()

        # Se il frame non è disponibile, si verifica se il ritardo supera
        # la soglia consentita; in tal caso si considera il video stream in errore.
        if frame is None:
            elapsed = time.monotonic() - self.last_frame_received_at
            if elapsed >= self.frame_timeout_sec:
                print(f"\n[ERRORE] Timeout del video stream: nessun frame ricevuto da {elapsed:.2f} secondi.")
                return False
            return True

        # Aggiorna il timestamp dell'ultimo frame valido e rinfresca,
        # se necessario, le informazioni di stato mostrate in overlay.
        self.last_frame_received_at = time.monotonic()
        self._refresh_status(controller)
        frame = self._normalize_frame_for_opencv(frame)

        # analysis_frame rappresenta il frame sul quale verranno eseguite
        # le analisi di visione; inizialmente coincide con il frame originale.
        analysis_frame = frame
        analysis_frame_is_undistorted = False

        # Se disponibile e abilitato, applica la correzione di distorsione.
        # La correzione migliora la coerenza geometrica della stima di posa,
        # soprattutto quando si usano parametri di calibrazione della camera.
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            try:
                undistorted_frame = self.pose_estimator.undistort_frame(frame)
                if undistorted_frame is not None:
                    analysis_frame = undistorted_frame
                    analysis_frame_is_undistorted = True
            except Exception:
                print("\n[ERRORE] Correzione della distorsione fallita sul frame corrente.")

        # Il frame mostrato a video è una copia del frame di analisi,
        # così da poter disegnare overlay senza alterare i dati di base.
        display_frame = analysis_frame.copy()

        # La detection è attiva solo se richiesta dal controllo
        # e se almeno un detector è stato caricato correttamente.
        detection_is_active = run_detection and len(self.detectors) > 0

        if detection_is_active:
            try:
                self._draw_multi_model_detections(display_frame, analysis_frame)
            except Exception:
                print("\n[ERRORE] Object detection multi-modello fallita sul frame corrente.")

        # Se lo stimatore di posa è disponibile, esegue l'elaborazione aggiuntiva
        # e disegna i relativi elementi sul frame da visualizzare.
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            try:
                display_frame, _ = self.pose_estimator.process_frame(
                    analysis_frame,
                    drawing_frame=display_frame,
                    frame_is_undistorted=analysis_frame_is_undistorted,
                )
                self._log_latest_pose_estimate()
            except Exception:
                print("\n[ERRORE] Stima posa AprilTag fallita sul frame corrente.")

        # Disegna informazioni di stato generali e mostra il risultato finale.
        self._draw_status_overlay(display_frame, detection_enabled=detection_is_active)
        cv2.imshow(VIDEO_WINDOW_TITLE, display_frame)

        # Se l'utente chiude manualmente la finestra OpenCV, il ciclo viene terminato
        # in modo esplicito invece di proseguire silenziosamente senza interfaccia video.
        try:
            if cv2.getWindowProperty(VIDEO_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[EVENTO] Finestra video chiusa dall'utente.")
                return False
        except cv2.error:
            # Alcune piattaforme possono sollevare un errore se la finestra non è
            # ancora completamente inizializzata o è già stata distrutta.
            # In tal caso si evita di interrompere il programma per un dettaglio GUI.
            pass

        return True


def create_controller():
    # Factory minimale per creare l'oggetto di controllo del drone.
    # L'uso di una funzione dedicata favorisce la separazione tra logica applicativa
    # e istanziazione concreta dei componenti.
    return RealTelloController()


def create_detectors():
    # Inizializza i detector YOLO definiti nella configurazione applicativa.
    # La funzione:
    # - verifica la disponibilità di torch;
    # - seleziona automaticamente GPU o CPU;
    # - tenta il caricamento di tutti i modelli configurati;
    # - restituisce i detector caricati correttamente.
    try:
        import torch
    except ImportError as exc:
        # Il caricamento dei detector richiede torch e il relativo ecosistema.
        # L'eccezione viene rilanciata con un messaggio più esplicativo per l'utente.
        raise ImportError(
            "Per usare il detector devi installare torch e le dipendenze richieste da ultralytics."
        ) from exc

    # Se CUDA è disponibile, usa il device 0 (prima GPU); altrimenti ripiega sulla CPU.
    detector_device = 0 if torch.cuda.is_available() else "cpu"
    detectors = []
    load_errors = []

    # Caricamento iterativo dei modelli definiti in configurazione.
    # Ogni elemento della lista contiene metadati utili alla visualizzazione
    # e l'oggetto detector vero e proprio.
    for model_cfg in APP_CONFIG.yolo_models:
        try:
            detectors.append(
                {
                    "name": model_cfg.name,
                    "color": model_cfg.color,
                    "detector": ObjectDetector(
                        model_path=str(model_cfg.path),
                        conf=APP_CONFIG.confidence_threshold,
                        imgsz=APP_CONFIG.image_size,
                        device=detector_device,
                    ),
                }
            )
        except Exception as exc:
            # Gli errori vengono raccolti per consentire un report aggregato,
            # senza interrompere il caricamento degli altri modelli.
            load_errors.append(f"{model_cfg.name} ({model_cfg.path}): {exc}")

    # Se nessun detector è stato inizializzato e sono presenti errori,
    # si solleva un'eccezione bloccante.
    if not detectors and load_errors:
        raise RuntimeError(
            "Nessun modello YOLO inizializzato correttamente. Errori: "
            + " | ".join(load_errors)
        )

    # Se almeno un modello è stato caricato, ma alcuni hanno fallito,
    # si stampa un avviso non bloccante.
    if load_errors:
        print("\n[AVVISO] Alcuni modelli YOLO non sono stati caricati:")
        for error_message in load_errors:
            print(f"  - {error_message}")

    return detectors


def create_flight_data_logger():
    # Crea il logger dei dati di volo usando il percorso centralizzato
    # definito nella configurazione applicativa.
    return FlightDataLogger(filename=APP_CONFIG.flight_data_log_path)


def create_pose_estimator():
    # Crea e configura il modulo di stima posa sulla base dei parametri
    # definiti in APP_CONFIG.camera_pose.
    pose_cfg = APP_CONFIG.camera_pose

    # Se la stima posa è disabilitata in configurazione, non viene creato alcun oggetto.
    # Il resto dell'applicazione è progettato per funzionare anche senza questo modulo.
    if not pose_cfg.enabled:
        return None

    # Costruzione dello stimatore con tutti i parametri geometrici e algoritmici necessari.
    # I parametri includono calibrazione intrinseca della camera, distorsione,
    # famiglia di tag, dimensione fisica del tag e informazioni di posa nel mondo.
    return CameraPoseEstimator(
        camera_matrix=pose_cfg.camera_matrix,
        dist_coeffs=pose_cfg.dist_coeffs,
        tag_family=pose_cfg.tag_family,
        threads=pose_cfg.threads,
        decimate=pose_cfg.decimate,
        tag_size_m=pose_cfg.tag_size_m,
        tag_position=pose_cfg.tag_position,
        tag_orientation_rpy_deg=pose_cfg.tag_orientation_rpy_deg,
        world_tags=pose_cfg.world_tags,
        fusion_mode=pose_cfg.fusion_mode,
        drone_extrinsics=pose_cfg.drone_extrinsics,
        enabled=pose_cfg.enabled,
    )


def create_pose_filter():
    # Crea il filtro di Kalman per stabilizzare la posizione assoluta stimata
    # tramite AprilTag. Il filtro viene applicato alla posa già fusa, non alle
    # posizioni note dei tag nel mondo.
    #
    # I parametri numerici regolano il compromesso tra fiducia nel modello dinamico
    # e fiducia nelle misure: valori più alti di measurement_noise rendono il filtro
    # meno sensibile alle misure istantanee.
    return PositionKalmanFilter(
        process_noise=0.15,
        measurement_noise=0.08,
        initial_covariance=1.0,
    )


def main():
    # Funzione principale dell'applicazione.
    # Coordina l'inizializzazione dei sottosistemi, l'esecuzione del ciclo
    # di controllo/visione e le operazioni di chiusura sicura delle risorse.
    configure_logging()

    # Inizializzazione preventiva dei riferimenti, così da poterli gestire
    # in modo sicuro nel blocco finally anche in caso di errore parziale.
    screen = None
    clock = None
    controller = None
    flight_data_logger = None

    try:
        print_phase("1. Inizializzazione input utente")
        screen = init_joystick()
        print("Joystick inizializzato correttamente.")
        print_joystick_help()
        clock = pygame.time.Clock()

        print_phase("2. Connessione al drone")
        controller = create_controller()
        controller.connect()
        print("Connessione al drone completata.")

        print_phase("3. Avvio video stream")
        controller.start_video_stream()
        print("Video stream attivo.")

        print_phase("4. Inizializzazione moduli di visione")
        detectors = []
        try:
            detectors = create_detectors()
            print(f"Detector YOLO inizializzati: {len(detectors)}")
            for item in detectors:
                print(f"  - {item['name']}")
        except Exception as exc:
            # La mancata inizializzazione dei detector non blocca l'applicazione:
            # il sistema può comunque funzionare come interfaccia di pilotaggio e streaming.
            print(f"Detector YOLO non disponibili: {exc}")

        pose_estimator = None
        try:
            pose_estimator = create_pose_estimator()
            if pose_estimator is not None:
                print("Stima posa AprilTag inizializzata.")
            else:
                print("Stima posa AprilTag disattivata.")
        except Exception as exc:
            # Anche la stima posa è opzionale: il sistema prosegue senza tale funzionalità.
            print(f"Stima posa AprilTag non disponibile: {exc}")

        pose_filter = None
        if pose_estimator is not None:
            try:
                pose_filter = create_pose_filter()
                print("Filtro di Kalman per la posizione inizializzato.")
            except Exception as exc:
                # Il filtro è opzionale: se non viene creato, si continua con la posa grezza/fusa.
                print(f"Filtro di Kalman non disponibile: {exc}")

        flight_data_logger = create_flight_data_logger()
        print(f"Flight data logger attivo: {APP_CONFIG.flight_data_log_path}")

        # Costruzione dei due cicli principali dell'applicazione:
        # uno dedicato al controllo del drone, l'altro alla visione.
        control_loop = DroneControlLoop(
            controller=controller,
            speed=APP_CONFIG.speed,
            detection_available=len(detectors) > 0,
        )

        vision_loop = VisionLoop(
            detectors=detectors,
            pose_estimator=pose_estimator,
            frame_timeout_sec=APP_CONFIG.frame_timeout_sec,
            frame_from_controller_is_rgb=APP_CONFIG.frame_from_controller_is_rgb,
            flight_data_logger=flight_data_logger,
            pose_filter=pose_filter,
        )

        print_drone_status(controller, yolo_enabled=control_loop.is_detection_enabled())

        print("\nSistema pronto.")
        print("Avvia il decollo con il controller quando vuoi.")
        print("Premi Options per uscire e generare cartella sessione con log e grafici.")

        # Gestione della periodicità di stampa dello stato nel terminale.
        # La stampa non viene effettuata a ogni iterazione per evitare output eccessivo.
        last_status_print_at = time.monotonic()
        status_print_interval_sec = 10.0

        running = True
        while running:
            # Passo di controllo: legge input e invia comandi di pilotaggio.
            running = control_loop.step()
            if not running:
                continue

            # Passo di visione: acquisisce ed elabora il frame corrente.
            # Lo stato della detection viene letto dal ciclo di controllo,
            # poiché l'attivazione è comandata dal joystick.
            running = vision_loop.step(
                controller,
                run_detection=control_loop.is_detection_enabled(),
            )
            if not running:
                continue

            # Consente l'uscita anche dalla finestra OpenCV tramite il tasto "q".
            # Prima della terminazione vengono azzerati i comandi RC per sicurezza.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                controller.send_rc_control(0, 0, 0, 0)
                print("\n[EVENTO] Uscita richiesta dalla finestra video.")
                running = False
                continue

            # Stampa periodica dello stato del drone nel terminale.
            now = time.monotonic()
            if now - last_status_print_at >= status_print_interval_sec:
                print_drone_status(controller, yolo_enabled=control_loop.is_detection_enabled())
                last_status_print_at = now

            # Aggiornamento minimale della finestra pygame, utile per mantenere
            # reattiva l'interfaccia associata all'input.
            if screen is not None:
                screen.fill((30, 30, 30))
                pygame.display.flip()

            # Regolazione del frame rate del ciclo principale.
            # Questa istruzione evita che il ciclo consumi inutilmente CPU.
            if clock is not None:
                clock.tick(APP_CONFIG.fps)

    except KeyboardInterrupt:
        # Gestione esplicita dell'interruzione da tastiera da parte dell'utente.
        print("\n[EVENTO] Interruzione richiesta dall'utente.")
    except Exception as exc:
        # Segnalazione di errore fatale seguita da rilancio dell'eccezione,
        # così da non mascherare il problema durante debug o sviluppo.
        print(f"\n[ERRORE FATALE] {exc}")
        raise
    finally:
        # Blocco di cleanup eseguito sempre, indipendentemente da errori o terminazione normale.
        # È una sezione essenziale nei programmi che interagiscono con hardware reale,
        # perché garantisce il rilascio ordinato delle risorse e riduce il rischio
        # di lasciare il drone in uno stato non controllato.
        if controller is not None:
            try:
                # Azzeramento dei comandi RC per evitare movimenti residui.
                controller.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

            try:
                # Se il controller segnala che il drone è in volo, prova un atterraggio di sicurezza.
                if getattr(controller, "is_flying", False):
                    controller.land()
            except Exception:
                pass

            try:
                # Chiusura della sessione con il drone e rilascio delle risorse associate.
                controller.end()
            except Exception:
                pass

        if flight_data_logger is not None:
            try:
                if flight_data_logger.has_data():
                    if hasattr(flight_data_logger, "export_session"):
                        session_dir = flight_data_logger.export_session()

                        if session_dir is not None:
                            print(f"\n[LOG VOLO] Cartella sessione generata: {session_dir}")
                            print("[LOG VOLO] Contenuto previsto:")
                            print("  - flight_data.txt")
                            print("  - x_raw_vs_filtered.png")
                            print("  - y_raw_vs_filtered.png")
                            print("  - z_raw_vs_filtered.png")
                            print("  - trajectory_xy_raw_vs_filtered.png")
                            print("  - filter_error_norm.png")
                        else:
                            print("\n[LOG VOLO] Errore durante la generazione della cartella sessione.")
                    else:
                        # Fallback per compatibilità con una vecchia versione del logger.
                        saved_path = flight_data_logger.save_to_file()
                        if saved_path is not None:
                            print(f"\n[LOG VOLO] Dati salvati in: {saved_path}")

                    print(flight_data_logger.get_summary())
                else:
                    print("\n[LOG VOLO] Nessuna posa valida registrata: file non generato.")
            except Exception:
                LOGGER.exception("Errore durante il salvataggio finale dei dati di volo.")

        # Chiusura delle finestre e dei sottosistemi grafici/input.
        # L'ordine consente di rilasciare sia la parte OpenCV sia quella pygame
        # dopo la chiusura della connessione con il drone.
        cv2.destroyAllWindows()
        close_joystick()
        pygame.quit()


if __name__ == "__main__":
    # Punto di ingresso del programma quando il file viene eseguito come script principale.
    main()