import logging
import time
from typing import Optional

# OpenCV è utilizzato per la gestione del flusso video, la visualizzazione
# dei frame e l'interazione con la finestra video principale.
import cv2

# pygame viene impiegato per la gestione del controller, degli eventi utente
# e del timing del ciclo principale dell'applicazione.
import pygame

# Configurazione centralizzata dell'applicazione: parametri video, logging,
# velocità di comando, configurazioni della detection e della stima di posa.
from app_config import APP_CONFIG

# Modulo dedicato alla gestione del joystick e alla traduzione degli input
# del controller in comandi elementari per il drone.
from joystick_tello import (
    close_joystick,
    get_command,
    init_joystick,
    print_joystick_help,
    read_events,
)

# Controller ad alto livello per l'interazione con il drone reale Tello.
from real_tello_controller import RealTelloController

# Stimatore della posa della camera basato su marker/tag visivi.
from tello_pose_detection import CameraPoseEstimator

# Rilevatore di oggetti, verosimilmente basato su YOLO.
from vision_detector import ObjectDetector

# Logger di modulo usato per eventuali messaggi diagnostici e di debug.
LOGGER = logging.getLogger(__name__)


def _resolve_log_level(default=logging.WARNING):
    # Converte il livello di log configurato in APP_CONFIG nel corrispondente
    # valore numerico previsto dal modulo logging.
    # Se il nome specificato non corrisponde a un attributo valido di logging,
    # viene usato il valore di default.
    level_name = str(APP_CONFIG.log_level).upper()
    return getattr(logging, level_name, default)


def configure_logging():
    """
    Riduce i log rumorosi nel terminale.
    L'output utente principale viene gestito con print strutturati.
    """
    # Determina il livello di logging desiderato a partire dalla configurazione.
    log_level = _resolve_log_level()

    # Configurazione globale del sistema di logging:
    # - livello minimo dei messaggi da mostrare;
    # - formato uniforme per tutti i logger;
    # - force=True per sovrascrivere eventuali configurazioni precedenti.
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s | %(message)s",
        force=True,
    )

    # Alcune librerie esterne possono produrre messaggi molto verbosi;
    # per questo motivo si impone almeno WARNING per tali logger.
    logging.getLogger("djitellopy").setLevel(max(logging.WARNING, log_level))
    logging.getLogger("ultralytics").setLevel(max(logging.WARNING, log_level))

    # Si allinea il livello dei principali moduli del progetto alla configurazione scelta.
    logging.getLogger(__name__).setLevel(log_level)
    logging.getLogger("real_tello_controller").setLevel(log_level)
    logging.getLogger("vision_detector").setLevel(log_level)
    logging.getLogger("tello_pose_detection").setLevel(log_level)
    logging.getLogger("joystick_tello").setLevel(log_level)


def print_phase(title: str):
    # Stampa una intestazione testuale che segmenta l'esecuzione del programma
    # in fasi logiche, rendendo più leggibile l'output su terminale.
    print(f"\n--- {title} ---")


def _format_battery_value(status: dict) -> str:
    # Estrae il valore di batteria dal dizionario di stato del drone.
    # Se il valore non è disponibile, restituisce un segnaposto testuale.
    battery = status.get("battery")
    return "--" if battery is None else f"{battery}%"


def print_drone_status(controller: RealTelloController, yolo_enabled: bool):
    # Recupera lo stato corrente del drone tramite il controller
    # e lo presenta in forma leggibile all'utente.
    status = controller.get_status()

    print("\n[STATO DRONE]")
    print(f"  Connesso      : {status.get('connected')}")
    print(f"  In volo       : {status.get('flying')}")
    print(f"  Batteria      : {_format_battery_value(status)}")
    print(f"  YOLO attiva   : {yolo_enabled}")
    print("----------------------------------------")


class DroneControlLoop:
    # Questa classe incapsula il ciclo logico di controllo del drone
    # basato sugli eventi generati dal joystick e sui comandi analogici continui.
    # Il suo compito principale è interpretare l'input utente e tradurlo
    # in richieste di decollo, atterraggio, attivazione detection e comandi RC.
    def __init__(self, controller: RealTelloController, speed: int, detection_available: bool = True):
        # controller: interfaccia verso il drone reale.
        # speed: valore massimo di velocità da applicare ai comandi analogici.
        # detection_available: indica se il modulo di object detection è stato
        # inizializzato correttamente ed è quindi utilizzabile.
        self.controller = controller
        self.speed = speed
        self.detection_available = detection_available

        # Stato interno che memorizza se la detection YOLO è attiva o meno.
        self._detection_enabled = False

    def is_detection_enabled(self) -> bool:
        # Restituisce lo stato corrente della detection, utile per sincronizzare
        # il comportamento del ciclo di visione con quello del controllo.
        return self._detection_enabled

    def step(self) -> bool:
        # Esegue un singolo passo del ciclo di controllo.
        # Restituisce:
        # - True se il programma può continuare;
        # - False se è stata richiesta l'uscita.
        actions = read_events()

        # Gestione degli eventi discreti provenienti dal controller.
        if actions["takeoff"]:
            self.controller.takeoff()
            print("\n[EVENTO] Decollo richiesto.")

        if actions["land"]:
            self.controller.land()
            print("\n[EVENTO] Atterraggio richiesto.")

        # Gestione del comando di attivazione/disattivazione della detection.
        if actions.get("detect", False):
            if not self.detection_available:
                print("\n[AVVISO] Detection YOLO non disponibile: comando ignorato.")
            else:
                self._detection_enabled = not self._detection_enabled
                stato = "attivata" if self._detection_enabled else "disattivata"
                print(f"\n[EVENTO] Detection YOLO {stato}.")

        # Gestione della richiesta di uscita.
        if actions["quit"]:
            # Prima di terminare si invia un comando nullo per arrestare eventuali
            # movimenti residui del drone.
            self.controller.send_rc_control(0, 0, 0, 0)
            print("\n[EVENTO] Uscita richiesta.")
            return False

        # Lettura del comando continuo dagli assi del joystick.
        command = get_command(self.speed)

        # Invio del comando RC al drone.
        self.controller.send_rc_control(
            command["lr"],
            command["fb"],
            command["ud"],
            command["yaw"],
        )

        return True


class VisionLoop:
    # Questa classe gestisce il ciclo di elaborazione video.
    # Essa si occupa di:
    # - acquisire i frame dal controller del drone;
    # - applicare, se disponibili, correzione geometrica e stima di posa;
    # - eseguire la object detection, se richiesta;
    # - visualizzare il frame finale con sovraimpressioni informative.
    def __init__(
        self,
        detector: Optional[ObjectDetector],
        pose_estimator: Optional[CameraPoseEstimator],
        frame_timeout_sec: float,
        frame_from_controller_is_rgb: bool = False,
        status_refresh_sec: float = 1.0,
    ):
        # detector: modulo di object detection opzionale.
        # pose_estimator: modulo di stima posa opzionale.
        # frame_timeout_sec: timeout massimo ammesso senza ricevere frame.
        # frame_from_controller_is_rgb: specifica il formato colore del frame
        # restituito dal controller.
        # status_refresh_sec: intervallo minimo tra due aggiornamenti dello stato.
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.frame_timeout_sec = frame_timeout_sec
        self.frame_from_controller_is_rgb = frame_from_controller_is_rgb
        self.status_refresh_sec = status_refresh_sec

        # Timestamp dell'ultimo frame ricevuto, utile per rilevare timeout video.
        self.last_frame_received_at = time.monotonic()

        # Timestamp dell'ultimo aggiornamento dello stato del drone.
        self.last_status_refresh_at = 0.0

        # Stato cache del drone, usato per evitare interrogazioni troppo frequenti
        # e per disegnare informazioni a video anche in caso di errore temporaneo.
        self.cached_status = {
            "connected": False,
            "flying": False,
            "battery": None,
        }

    def _normalize_frame_for_opencv(self, frame):
        # Normalizza il frame nel formato colore atteso da OpenCV.
        # Se il frame arriva in RGB, viene convertito in BGR;
        # altrimenti viene restituito invariato.
        if self.frame_from_controller_is_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _refresh_status(self, controller: RealTelloController):
        # Aggiorna periodicamente lo stato del drone.
        # La frequenza di refresh è limitata per ridurre chiamate ripetute
        # al controller durante il ciclo video.
        now = time.monotonic()
        if (now - self.last_status_refresh_at) < self.status_refresh_sec:
            return

        self.last_status_refresh_at = now
        try:
            status = controller.get_status()
            self.cached_status = {
                "connected": status.get("connected", False),
                "flying": status.get("flying", False),
                "battery": status.get("battery"),
            }
        except Exception:
            # In caso di errore, si imposta uno stato conservativo di fallback.
            self.cached_status = {
                "connected": False,
                "flying": False,
                "battery": None,
            }

    def _draw_status_overlay(self, frame, detection_enabled: bool = False):
        # Disegna sul frame un pannello testuale con le principali informazioni
        # sullo stato del drone e della detection.
        connected_text = str(self.cached_status.get("connected", False))
        flying_text = str(self.cached_status.get("flying", False))

        battery_value = self.cached_status.get("battery")
        battery_text = "--" if battery_value is None else f"{battery_value}%"

        detection_text = str(detection_enabled)

        lines = [
            f"Connesso      : {connected_text}",
            f"In volo       : {flying_text}",
            f"Batteria      : {battery_text}",
            f"YOLO attiva   : {detection_text}",
        ]

        # Coordinate iniziali della scrittura a video.
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

    def step(self, controller: RealTelloController, run_detection: bool = False) -> bool:
        # Esegue un singolo passo del ciclo di visione.
        # Restituisce:
        # - True se il sistema può continuare;
        # - False in caso di errore grave, ad esempio timeout prolungato del video stream.
        frame = controller.get_frame()

        # Se il frame non è disponibile, si controlla da quanto tempo
        # il flusso video è assente.
        if frame is None:
            elapsed = time.monotonic() - self.last_frame_received_at
            if elapsed >= self.frame_timeout_sec:
                print(f"\n[ERRORE] Timeout del video stream: nessun frame ricevuto da {elapsed:.2f} secondi.")
                return False
            return True

        # Aggiornamento del timestamp di ultimo frame ricevuto.
        self.last_frame_received_at = time.monotonic()

        # Aggiornamento periodico dello stato del drone.
        self._refresh_status(controller)

        # Uniformazione del frame al formato atteso dalle successive elaborazioni.
        frame = self._normalize_frame_for_opencv(frame)

        # analysis_frame rappresenta il frame usato per le elaborazioni numeriche.
        analysis_frame = frame

        # Se disponibile, si applica la correzione della distorsione ottica.
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            try:
                undistorted_frame = self.pose_estimator.undistort_frame(frame)
                if undistorted_frame is not None:
                    analysis_frame = undistorted_frame
            except Exception:
                print("\n[ERRORE] Correzione della distorsione fallita sul frame corrente.")

        # display_frame è il frame effettivamente mostrato a schermo,
        # su cui possono essere sovrapposti bounding box, pose o testi.
        display_frame = analysis_frame.copy()

        # La detection è considerata attiva solo se richiesta esplicitamente
        # e se il detector è stato inizializzato correttamente.
        detection_is_active = run_detection and self.detector is not None

        # Esecuzione opzionale della object detection.
        if detection_is_active:
            try:
                yolo_frame, _ = self.detector.detect(analysis_frame)
                if yolo_frame is not None:
                    display_frame = yolo_frame
            except Exception:
                print("\n[ERRORE] Object detection fallita sul frame corrente.")

        # Esecuzione opzionale della stima di posa e del relativo disegno.
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            try:
                display_frame, _ = self.pose_estimator.process_frame(
                    analysis_frame,
                    drawing_frame=display_frame,
                )
            except Exception:
                print("\n[ERRORE] Stima posa AprilTag fallita sul frame corrente.")

        # Sovrapposizione delle informazioni di stato sul frame finale.
        self._draw_status_overlay(display_frame, detection_enabled=detection_is_active)

        # Visualizzazione del frame finale nella finestra OpenCV.
        cv2.imshow("Tello Detection", display_frame)
        return True


def create_controller():
    # Factory minimale per creare il controller del drone.
    # L'isolamento in una funzione dedicata rende più chiara la struttura
    # del main e facilita eventuali future estensioni.
    return RealTelloController()


def create_detector():
    # Crea il detector di oggetti.
    # La funzione verifica innanzitutto la disponibilità di torch,
    # da cui dipende l'esecuzione del modello.
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Per usare il detector devi installare torch e le dipendenze richieste da ultralytics."
        ) from exc

    # Se è disponibile una GPU CUDA, il detector utilizza il dispositivo 0;
    # altrimenti ricade sull'esecuzione su CPU.
    detector_device = 0 if torch.cuda.is_available() else "cpu"

    return ObjectDetector(
        model_path=str(APP_CONFIG.model_path),
        conf=APP_CONFIG.confidence_threshold,
        imgsz=APP_CONFIG.image_size,
        device=detector_device,
    )


def create_pose_estimator():
    # Crea lo stimatore di posa a partire dalla configurazione dell'applicazione.
    # Se il modulo è disabilitato in configurazione, restituisce None.
    pose_cfg = APP_CONFIG.camera_pose
    if not pose_cfg.enabled:
        return None

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
        enabled=pose_cfg.enabled,
    )


def main():
    # Funzione principale del programma.
    # Coordina:
    # 1. inizializzazione del logging;
    # 2. inizializzazione degli input utente;
    # 3. connessione al drone;
    # 4. avvio del video stream;
    # 5. inizializzazione dei moduli di visione;
    # 6. esecuzione del ciclo principale di controllo e visualizzazione.
    configure_logging()

    # Riferimenti inizializzati a None per consentire una pulizia robusta
    # delle risorse anche in presenza di eccezioni durante l'avvio.
    screen = None
    clock = None
    controller = None

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
        detector = None
        try:
            detector = create_detector()
            print("Detector YOLO inizializzato.")
        except Exception as exc:
            # La mancata inizializzazione del detector non impedisce l'esecuzione
            # del programma: il sistema continuerà senza object detection.
            print(f"Detector YOLO non disponibile: {exc}")

        pose_estimator = None
        try:
            pose_estimator = create_pose_estimator()
            if pose_estimator is not None:
                print("Stima posa AprilTag inizializzata.")
            else:
                print("Stima posa AprilTag disattivata.")
        except Exception as exc:
            # Anche la stima di posa è opzionale: in caso di errore
            # il sistema continua senza questo modulo.
            print(f"Stima posa AprilTag non disponibile: {exc}")

        # Creazione del ciclo di controllo del drone.
        control_loop = DroneControlLoop(
            controller=controller,
            speed=APP_CONFIG.speed,
            detection_available=detector is not None,
        )

        # Creazione del ciclo di visione.
        vision_loop = VisionLoop(
            detector=detector,
            pose_estimator=pose_estimator,
            frame_timeout_sec=APP_CONFIG.frame_timeout_sec,
            frame_from_controller_is_rgb=APP_CONFIG.frame_from_controller_is_rgb,
        )

        # Stampa dello stato iniziale del drone.
        print_drone_status(controller, yolo_enabled=control_loop.is_detection_enabled())

        print("\nSistema pronto.")
        print("Avvia il decollo con il controller quando vuoi.")

        # Variabili per la stampa periodica dello stato del drone.
        last_status_print_at = time.monotonic()
        status_print_interval_sec = 5.0

        # Ciclo principale dell'applicazione.
        running = True
        while running:
            # Prima si esegue il passo di controllo, responsabile della gestione
            # degli input utente e dell'invio dei comandi RC al drone.
            running = control_loop.step()
            if not running:
                continue

            # Successivamente si esegue il passo di visione,
            # che elabora e visualizza il frame corrente.
            running = vision_loop.step(
                controller,
                run_detection=control_loop.is_detection_enabled(),
            )
            if not running:
                continue

            # La finestra video OpenCV prevede anch'essa una scorciatoia per uscire.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                controller.send_rc_control(0, 0, 0, 0)
                print("\n[EVENTO] Uscita richiesta dalla finestra video.")
                running = False
                continue

            # Stampa periodica dello stato sintetico del drone.
            now = time.monotonic()
            if now - last_status_print_at >= status_print_interval_sec:
                print_drone_status(controller, yolo_enabled=control_loop.is_detection_enabled())
                last_status_print_at = now

            # Aggiornamento della finestra pygame, usata principalmente
            # per mantenere attivo il contesto degli eventi del joystick.
            if screen is not None:
                screen.fill((30, 30, 30))
                pygame.display.flip()

            # Regolazione della frequenza del ciclo principale.
            if clock is not None:
                clock.tick(APP_CONFIG.fps)

    except KeyboardInterrupt:
        # Gestione dell'interruzione manuale da tastiera.
        print("\n[EVENTO] Interruzione richiesta dall'utente.")
    except Exception as exc:
        # In caso di errore non gestito, il programma informa l'utente
        # e rilancia l'eccezione per non mascherare la causa originaria.
        print(f"\n[ERRORE FATALE] {exc}")
        raise
    finally:
        # Blocco di pulizia finale eseguito indipendentemente dall'esito.
        if controller is not None:
            try:
                # Invio di un comando nullo per arrestare i motori comandati via RC.
                controller.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass

            try:
                # Se il drone risulta ancora in volo, si tenta un atterraggio controllato.
                if getattr(controller, "is_flying", False):
                    controller.land()
            except Exception:
                pass

            try:
                # Chiusura della sessione di comunicazione con il drone.
                controller.end()
            except Exception:
                pass

        # Rilascio delle risorse grafiche e di input.
        cv2.destroyAllWindows()
        close_joystick()
        pygame.quit()


# Punto di ingresso standard dello script.
# main() viene eseguita solo se questo file è lanciato direttamente.
if __name__ == "__main__":
    main()