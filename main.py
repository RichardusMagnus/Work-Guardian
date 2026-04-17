import logging
import time
from typing import Optional

import cv2
import pygame

# Configurazione centralizzata dell'applicazione:
# raccoglie parametri relativi a logging, joystick, velocità,
# detector, gestione video e stima della posa.
from app_config import APP_CONFIG

# Modulo dedicato alla gestione del joystick:
# - init_joystick(): inizializza pygame e il dispositivo di input;
# - read_events(): legge gli eventi generati dall'utente;
# - get_command(): traduce lo stato del joystick in comandi RC.
from joystick_tello import get_command, init_joystick, read_events

# Interfaccia di controllo del drone reale DJI Tello.
from real_tello_controller import RealTelloController

# Modulo per la stima della posa della camera tramite AprilTag
# e per l'eventuale correzione della distorsione ottica.
from tello_pose_detection import CameraPoseEstimator

# Modulo di object detection, tipicamente basato su YOLO.
from vision_detector import ObjectDetector


# Logger di modulo: consente di tracciare messaggi diagnostici
# riferiti specificamente a questo file.
LOGGER = logging.getLogger(__name__)


def configure_logging(level: str):
    """Configura il logging applicativo."""
    # Converte il livello di logging espresso come stringa
    # (ad esempio "INFO", "DEBUG", "WARNING") nel corrispondente
    # valore numerico previsto dal modulo logging.
    # Se il livello non è riconosciuto, viene usato INFO come default.
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # basicConfig inizializza la configurazione globale del logging.
    # force=True assicura che eventuali configurazioni precedenti
    # vengano sovrascritte.
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def print_joystick_help():
    """Stampa a terminale i comandi del joystick."""
    # Recupera dalla configurazione la mappatura semantica
    # dei pulsanti e degli assi del joystick.
    mapping = APP_CONFIG.joystick

    # Stampa una guida sintetica per l'utente.
    # L'uso di specificatori di formato (<25) allinea i testi
    # e rende la visualizzazione più leggibile a terminale.
    print("\n============================================================")
    print("COMANDI JOYSTICK")
    print("============================================================")
    print(f"{mapping.label_takeoff:<25} -> Decollo")
    print(f"{mapping.label_land:<25} -> Atterraggio")
    print(f"{mapping.label_quit:<25} -> Uscita dal programma")
    print(f"{mapping.label_axis_lr:<25} -> Sinistra / Destra")
    print(f"{mapping.label_axis_fb:<25} -> Avanti / Indietro")
    print(f"{mapping.label_axis_ud:<25} -> Su / Giù")
    print(f"{mapping.label_axis_yaw:<25} -> Rotazione (Yaw)")
    print("ESC da tastiera          -> Uscita dal programma")
    print("q nella finestra video   -> Uscita dal programma")
    print("============================================================\n")


class DroneControlLoop:
    """
    Gestisce il controllo del drone tramite joystick:
    decollo, atterraggio, quit e comandi RC continui.
    """

    def __init__(self, controller: RealTelloController, speed: int, require_focus: bool = False):
        # Riferimento all'oggetto incaricato di inviare i comandi
        # al drone reale.
        self.controller = controller

        # Velocità scalare usata per convertire gli input del joystick
        # in valori RC compatibili con il drone.
        self.speed = speed

        # Se True, i comandi vengono accettati solo quando la finestra
        # pygame associata al joystick è attiva/in focus.
        self.require_focus = require_focus

    def _focus_allows_control(self, actions: dict) -> bool:
        """
        Restituisce True se i comandi joystick devono essere accettati.

        Quando require_focus è attivo, blocchiamo sia i pulsanti
        sia i comandi continui se la finestra pygame non è attiva.
        """
        # Se require_focus è False, il controllo è sempre consentito.
        # In caso contrario, si controlla il flag "focused" restituito
        # dal sistema di lettura eventi.
        return (not self.require_focus) or actions.get("focused", True)

    def step(self) -> bool:
        # Legge gli eventi correnti generati da joystick/tastiera.
        # Il dizionario risultante contiene tipicamente flag booleani
        # per azioni discrete (takeoff, land, quit) e informazioni
        # contestuali, come il focus della finestra.
        actions = read_events()

        # Determina se, in questo istante, il sistema deve accettare
        # o ignorare i comandi dell'utente.
        focus_allows_control = self._focus_allows_control(actions)

        # Le azioni discrete di decollo e atterraggio vengono accettate
        # solo se il controllo è abilitato dal focus.
        if focus_allows_control:
            if actions["takeoff"]:
                self.controller.takeoff()

            if actions["land"]:
                self.controller.land()

        # Il comando di uscita è sempre considerato prioritario.
        if actions["quit"]:
            # Prima di uscire, azzeriamo i comandi RC per evitare che
            # il drone continui a ricevere un ultimo comando non nullo.
            self.controller.send_rc_control(0, 0, 0, 0)
            return False

        # Se il controllo è consentito, il comando continuo viene ricavato
        # leggendo lo stato corrente del joystick; altrimenti si impone
        # un vettore nullo, così da congelare il moto.
        if focus_allows_control:
            command = get_command(self.speed, require_focus=self.require_focus)
        else:
            command = {
                "lr": 0,
                "fb": 0,
                "ud": 0,
                "yaw": 0,
            }

        # Invio del comando RC al drone.
        # Le quattro componenti rappresentano:
        # - lr: left/right
        # - fb: forward/backward
        # - ud: up/down
        # - yaw: rotazione attorno all'asse verticale
        self.controller.send_rc_control(
            command["lr"],
            command["fb"],
            command["ud"],
            command["yaw"],
        )

        # True indica che il loop principale può proseguire.
        return True


class VisionLoop:
    """
    Gestisce il flusso video:
    - acquisizione frame dal Tello
    - normalizzazione eventuale RGB/BGR
    - undistortion per la posa camera
    - YOLO detection
    - AprilTag pose estimation
    - visualizzazione finale
    """

    def __init__(
        self,
        detector: ObjectDetector,
        pose_estimator: Optional[CameraPoseEstimator],
        frame_timeout_sec: float,
        frame_from_controller_is_rgb: bool = False,
    ):
        # Detector usato per l'analisi semantica del frame.
        self.detector = detector

        # Stimatore della posa della camera. È opzionale:
        # se None, il programma eseguirà solo l'object detection.
        self.pose_estimator = pose_estimator

        # Timeout massimo ammesso senza ricezione di frame video.
        # Oltre tale soglia il flusso viene considerato guasto.
        self.frame_timeout_sec = frame_timeout_sec

        # Flag che specifica il formato cromatico dei frame ricevuti.
        # OpenCV utilizza convenzionalmente BGR, mentre altri sistemi
        # possono produrre frame in RGB.
        self.frame_from_controller_is_rgb = frame_from_controller_is_rgb

        # Timestamp dell'ultimo frame valido ricevuto.
        # Viene inizializzato all'istante corrente per avviare il conteggio
        # del timeout fin dalla creazione dell'oggetto.
        self.last_frame_received_at = time.monotonic()

    def _normalize_frame_for_opencv(self, frame):
        """
        Se il controller restituisce frame RGB, li convertiamo in BGR
        perché OpenCV lavora in BGR.
        """
        # Conversione necessaria per mantenere compatibilità con le
        # funzioni OpenCV e con eventuali moduli che assumono BGR in input.
        if self.frame_from_controller_is_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def step(self, controller: RealTelloController) -> bool:
        # Acquisisce il frame corrente dal controller del drone.
        frame = controller.get_frame()

        # Se non è disponibile alcun frame, non si interrompe
        # immediatamente il programma: si verifica prima da quanto tempo
        # manca il segnale video.
        if frame is None:
            elapsed = time.monotonic() - self.last_frame_received_at
            if elapsed >= self.frame_timeout_sec:
                LOGGER.error(
                    "Timeout del video stream: nessun frame ricevuto da %.2f secondi.",
                    elapsed,
                )
                return False
            return True

        # Aggiorna il timestamp dell'ultimo frame ricevuto correttamente.
        self.last_frame_received_at = time.monotonic()

        # 1. Normalizzazione per OpenCV
        frame = self._normalize_frame_for_opencv(frame)

        # 2. Se la posa camera è attiva, lavoriamo su frame undistorto
        # analysis_frame rappresenta il frame effettivamente usato
        # per le analisi di visione.
        analysis_frame = frame
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            undistorted_frame = self.pose_estimator.undistort_frame(frame)

            # Se la correzione della distorsione riesce, il frame corretto
            # viene usato nelle fasi successive; altrimenti si continua
            # sul frame originale, evitando di interrompere l'elaborazione.
            if undistorted_frame is not None:
                analysis_frame = undistorted_frame
            else:
                LOGGER.warning("Undistortion non riuscita: uso il frame originale per l'analisi.")

        # 3. Object detection YOLO
        try:
            # Il detector restituisce tipicamente:
            # - un frame annotato, pronto per la visualizzazione;
            # - informazioni ausiliarie sulle detection.
            yolo_frame, _ = self.detector.detect(analysis_frame)
        except Exception:
            # Un errore nel detector non deve necessariamente bloccare
            # l'intera applicazione: il frame originale di analisi viene
            # comunque mantenuto per la visualizzazione.
            LOGGER.exception("Object detection fallita sul frame corrente.")
            yolo_frame = analysis_frame

        # Se YOLO non restituisce frame annotato, usiamo comunque il frame di analisi
        display_frame = yolo_frame if yolo_frame is not None else analysis_frame

        # 4. Stima posa camera / AprilTag
        if self.pose_estimator is not None and self.pose_estimator.enabled:
            try:
                # La stima della posa utilizza analysis_frame come base
                # di analisi e può annotare il risultato direttamente
                # sul frame già elaborato da YOLO.
                display_frame, _ = self.pose_estimator.process_frame(
                    analysis_frame,
                    drawing_frame=display_frame,
                )
            except Exception:
                # Anche in questo caso l'errore viene registrato ma non
                # interrompe necessariamente il flusso dell'applicazione.
                LOGGER.exception("Stima posa AprilTag fallita sul frame corrente.")

        # Visualizzazione finale del frame, se disponibile.
        if display_frame is not None:
            cv2.imshow("Tello Detection", display_frame)

        return True


def create_controller():
    """
    Crea e restituisce il controller del drone reale.

    Returns:
        RealTelloController: controller per il drone DJI Tello reale.
    """
    # La funzione isola la creazione del controller in un punto dedicato,
    # favorendo chiarezza, centralizzazione e possibile estendibilità.
    return RealTelloController()


def create_detector():
    """Crea il detector YOLO usando GPU se disponibile."""
    # L'import di torch viene eseguito localmente per intercettare
    # in modo esplicito l'assenza della dipendenza soltanto quando
    # il detector viene effettivamente richiesto.
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Per usare il detector devi installare torch e le dipendenze richieste da ultralytics."
        ) from exc

    # Se CUDA è disponibile, si utilizza la GPU (device 0);
    # in caso contrario, l'elaborazione viene eseguita su CPU.
    detector_device = 0 if torch.cuda.is_available() else "cpu"
    LOGGER.info("Dispositivo YOLO: %s", detector_device)

    # I parametri del detector sono letti dalla configurazione applicativa
    # per mantenere separati codice e impostazioni sperimentali.
    return ObjectDetector(
        model_path=str(APP_CONFIG.model_path),
        conf=APP_CONFIG.confidence_threshold,
        imgsz=APP_CONFIG.image_size,
        device=detector_device,
    )


def create_pose_estimator():
    """Crea il modulo di stima posa camera, se abilitato."""
    # Sezione di configurazione dedicata alla camera pose estimation.
    pose_cfg = APP_CONFIG.camera_pose

    # Se il modulo è disabilitato da configurazione, la funzione
    # restituisce None e il resto del programma adatterà di conseguenza
    # il flusso operativo.
    if not pose_cfg.enabled:
        LOGGER.info("Stima posa camera disattivata da configurazione.")
        return None

    # Costruzione dell'estimatore con parametri intrinseci della camera,
    # coefficienti di distorsione e opzioni relative al rilevamento dei tag.
    return CameraPoseEstimator(
        camera_matrix=pose_cfg.camera_matrix,
        dist_coeffs=pose_cfg.dist_coeffs,
        tag_family=pose_cfg.tag_family,
        threads=pose_cfg.threads,
        decimate=pose_cfg.decimate,
        tag_size_m=pose_cfg.tag_size_m,
        tag_position=pose_cfg.tag_position,
        enabled=pose_cfg.enabled,
    )


def main():
    """Avvia il controllo del Tello con joystick, YOLO e stima posa camera."""
    # Inizializzazione del sistema di logging secondo la configurazione.
    configure_logging(APP_CONFIG.log_level)

    # Inizializzazione preventiva delle variabili che saranno usate
    # nel blocco finally. Questo evita errori dovuti a riferimenti
    # a variabili non definite in caso di eccezioni precoci.
    screen = None
    clock = None
    controller = None

    try:
        # 1. Inizializzazione input utente
        # Viene predisposta l'interfaccia pygame per la lettura del joystick
        # e stampata a terminale la guida ai comandi disponibili.
        screen = init_joystick()
        print_joystick_help()
        clock = pygame.time.Clock()

        # 2. Inizializzazione drone
        # Creazione del controller, connessione al drone e attivazione
        # dello stream video.
        controller = create_controller()
        controller.connect()
        controller.start_video_stream()

        # 3. Inizializzazione moduli di visione
        # Si costruiscono i moduli deputati all'analisi del video:
        # detector YOLO e, opzionalmente, estimatore della posa camera.
        detector = create_detector()
        pose_estimator = create_pose_estimator()

        # 4. Creazione loop di controllo e visione
        # I due loop sono separati per mantenere distinta la logica
        # di controllo del drone dalla logica di elaborazione video.
        control_loop = DroneControlLoop(
            controller=controller,
            speed=APP_CONFIG.speed,
            require_focus=APP_CONFIG.require_focus_for_control,
        )

        vision_loop = VisionLoop(
            detector=detector,
            pose_estimator=pose_estimator,
            frame_timeout_sec=APP_CONFIG.frame_timeout_sec,
            frame_from_controller_is_rgb=APP_CONFIG.frame_from_controller_is_rgb,
        )

        # 5. Loop principale
        # Il programma esegue ciclicamente:
        # - lettura e invio dei comandi di controllo;
        # - acquisizione ed elaborazione del frame video;
        # - gestione dell'interfaccia grafica e del frame rate.
        running = True
        while running:
            # Aggiorna la parte di controllo del drone.
            running = control_loop.step()
            if not running:
                continue

            # Aggiorna la parte di visione artificiale.
            running = vision_loop.step(controller)
            if not running:
                continue

            # Uscita da finestra OpenCV
            # Consente la chiusura del programma anche dalla finestra video.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                controller.send_rc_control(0, 0, 0, 0)
                running = False
                continue

            # Aggiornamento finestra pygame
            # La finestra viene ridisegnata a ogni iterazione per mantenere
            # attivo il contesto grafico usato da pygame.
            if screen is not None:
                screen.fill((30, 30, 30))
                pygame.display.flip()

            # Limitazione FPS
            # Regola la frequenza del ciclo principale per evitare
            # uso eccessivo della CPU e mantenere un comportamento stabile.
            if clock is not None:
                clock.tick(APP_CONFIG.fps)

    except KeyboardInterrupt:
        # Gestione esplicita dell'interruzione da tastiera (Ctrl+C).
        LOGGER.info("Interruzione richiesta dall'utente.")
    except Exception:
        # Qualsiasi altra eccezione viene registrata come errore fatale
        # e poi rilanciata per non nascondere il problema.
        LOGGER.exception("Errore fatale durante l'esecuzione del programma.")
        raise
    finally:
        # Cleanup sicuro
        # Questa sezione viene eseguita sempre, sia in caso di terminazione
        # normale sia in presenza di eccezioni.
        if controller is not None:
            try:
                # Azzeramento finale dei comandi RC per impedire che il drone
                # mantenga un comando residuo dopo la chiusura del programma.
                controller.send_rc_control(0, 0, 0, 0)
            except Exception:
                LOGGER.exception("Errore durante l'azzeramento dei comandi RC.")

            try:
                # Se il drone risulta ancora in volo, si tenta un atterraggio
                # di sicurezza prima della chiusura completa del controller.
                if getattr(controller, "is_flying", False):
                    controller.land()
            except Exception:
                LOGGER.exception("Errore durante l'atterraggio di sicurezza.")

            try:
                # Chiusura della connessione e rilascio delle risorse
                # associate al controller del drone.
                controller.end()
            except Exception:
                LOGGER.exception("Errore durante la chiusura del controller.")

        # Chiusura delle finestre OpenCV e terminazione di pygame.
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    # Punto di ingresso del programma quando il file viene eseguito
    # direttamente come script principale.
    main()