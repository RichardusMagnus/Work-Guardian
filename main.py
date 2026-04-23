import logging
import time
from typing import Optional

import cv2
import pygame

from app_config import APP_CONFIG
from joystick_tello import (
    close_joystick,
    get_command,
    init_joystick,
    print_joystick_help,
    read_events,
)
from real_tello_controller import RealTelloController
from tello_pose_detection import CameraPoseEstimator
from vision_detector import ObjectDetector
from flight_data_logger import FlightDataLogger

logger = FlightDataLogger("flight_log.txt")  # Inizializza una volta

# Logger di modulo, utile per eventuali messaggi diagnostici coerenti con il nome del file corrente.
LOGGER = logging.getLogger(__name__)


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
    logging.getLogger("djitellopy").setLevel(max(logging.WARNING, log_level))
    logging.getLogger("ultralytics").setLevel(max(logging.WARNING, log_level))

    # Imposta il livello di log per i moduli interni principali dell'applicazione.
    logging.getLogger(__name__).setLevel(log_level)
    logging.getLogger("real_tello_controller").setLevel(log_level)
    logging.getLogger("vision_detector").setLevel(log_level)
    logging.getLogger("tello_pose_detection").setLevel(log_level)
    logging.getLogger("joystick_tello").setLevel(log_level)


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

    def __init__(self, controller: RealTelloController, speed: int, detection_available: bool = True):
        # Riferimento al controller reale del drone.
        self.controller = controller

        # Velocità da utilizzare per convertire gli input del joystick
        # nei comandi di movimento RC.
        self.speed = speed

        # Flag che indica se i moduli di detection sono effettivamente disponibili.
        self.detection_available = detection_available

        # Stato interno dell'attivazione della object detection.
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
        if actions["takeoff"]:
            if self.controller.takeoff():
                print("\n[EVENTO] Decollo eseguito.")
            else:
                print("\n[AVVISO] Decollo ignorato: controlla connessione o stato del drone.")

        # Gestione dell'atterraggio.
        if actions["land"]:
            if self.controller.land():
                print("\n[EVENTO] Atterraggio eseguito.")
            else:
                print("\n[AVVISO] Atterraggio ignorato: il drone non risulta in volo.")

        # Gestione dell'attivazione/disattivazione della object detection.
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
    # - disegna overlay informativi e visualizza il frame finale.

    def __init__(
        self,
        detectors,
        pose_estimator: Optional[CameraPoseEstimator],
        frame_timeout_sec: float,
        frame_from_controller_is_rgb: bool = False,
        status_refresh_sec: float = 1.0,
    ):
        # Lista dei detector YOLO inizializzati. Viene forzata a lista per
        # garantire una struttura uniforme anche se in ingresso è None o iterabile generico.
        self.detectors = list(detectors or [])

        # Stimatore di posa della camera/drone, eventualmente assente.
        self.pose_estimator = pose_estimator

        # Tempo massimo tollerato senza ricezione di frame video.
        self.frame_timeout_sec = frame_timeout_sec

        # Specifica se il frame ricevuto dal controller è in formato RGB;
        # in tal caso sarà convertito in BGR per compatibilità con OpenCV.
        self.frame_from_controller_is_rgb = frame_from_controller_is_rgb

        # Intervallo minimo tra due refresh dello stato del drone,
        # per evitare interrogazioni troppo frequenti.
        self.status_refresh_sec = status_refresh_sec

        # Timestamp dell'ultimo frame ricevuto correttamente.
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

    def _normalize_frame_for_opencv(self, frame):
        # OpenCV utilizza tipicamente il formato BGR.
        # Se il controller restituisce frame RGB, si esegue la conversione.
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

    def _draw_multi_model_detections(self, frame, analysis_frame):
        # Esegue la detection per ciascun modello disponibile e disegna
        # bounding box e label sul frame da mostrare a video.
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

        self.last_frame_received_at = time.monotonic()
        self._refresh_status(controller)
        frame = self._normalize_frame_for_opencv(frame)

        # analysis_frame rappresenta il frame sul quale verranno eseguite
        # le analisi di visione; inizialmente coincide con il frame originale.
        analysis_frame = frame
        analysis_frame_is_undistorted = False

        # Se disponibile e abilitato, applica la correzione di distorsione.
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
                display_frame, pose_results = self.pose_estimator.process_frame(
                    analysis_frame,
                    drawing_frame=display_frame,
                    frame_is_undistorted=analysis_frame_is_undistorted,
                )
                if pose_results and any(r.get("type") == "fused_drone_pose_world" for r in pose_results):
                    fused = next(r for r in pose_results if r["type"] == "fused_drone_pose_world")
                    position = np.array([[fused["position_world"]["x"]], 
                                        [fused["position_world"]["y"]], 
                                        [fused["position_world"]["z"]]])
                    yaw = fused["yaw_world_deg"]
                    tag_ids = fused["source_tag_ids"]
                    logger.log_position(position, yaw, tag_ids)
            except Exception:
                print("\n[ERRORE] Stima posa AprilTag fallita sul frame corrente.")

        # Disegna informazioni di stato generali e mostra il risultato finale.
        self._draw_status_overlay(display_frame, detection_enabled=detection_is_active)
        cv2.imshow("Tello Detection", display_frame)
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
        raise ImportError(
            "Per usare il detector devi installare torch e le dipendenze richieste da ultralytics."
        ) from exc

    # Se CUDA è disponibile, usa il device 0 (prima GPU); altrimenti ripiega sulla CPU.
    detector_device = 0 if torch.cuda.is_available() else "cpu"
    detectors = []
    load_errors = []

    # Caricamento iterativo dei modelli definiti in configurazione.
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


def create_pose_estimator():
    # Crea e configura il modulo di stima posa sulla base dei parametri
    # definiti in APP_CONFIG.camera_pose.
    pose_cfg = APP_CONFIG.camera_pose

    # Se la stima posa è disabilitata in configurazione, non viene creato alcun oggetto.
    if not pose_cfg.enabled:
        return None

    # Costruzione dello stimatore con tutti i parametri geometrici e algoritmici necessari.
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
        )

        print_drone_status(controller, yolo_enabled=control_loop.is_detection_enabled())

        print("\nSistema pronto.")
        print("Avvia il decollo con il controller quando vuoi.")

        # Gestione della periodicità di stampa dello stato nel terminale.
        last_status_print_at = time.monotonic()
        status_print_interval_sec = 10.0

        running = True
        while running:
            # Passo di controllo: legge input e invia comandi di pilotaggio.
            running = control_loop.step()
            if not running:
                continue

            # Passo di visione: acquisisce ed elabora il frame corrente.
            running = vision_loop.step(
                controller,
                run_detection=control_loop.is_detection_enabled(),
            )
            if not running:
                continue

            # Consente l'uscita anche dalla finestra OpenCV tramite il tasto "q".
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

        logger.save_to_file()
        print(logger.get_summary())

        # Chiusura delle finestre e dei sottosistemi grafici/input.
        cv2.destroyAllWindows()
        close_joystick()
        pygame.quit()


if __name__ == "__main__":
    # Punto di ingresso del programma quando il file viene eseguito come script principale.
    main()