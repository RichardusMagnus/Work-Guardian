from dataclasses import dataclass, field
from pathlib import Path

# BASE_DIR rappresenta la directory che contiene il file corrente.
# Tale riferimento viene utilizzato come punto di partenza robusto per
# costruire percorsi relativi ad altre risorse del progetto, ad esempio
# modelli di deep learning o file di configurazione.
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class JoystickMapping:
    # Configurazione immutabile della mappatura tra pulsanti/assi del joystick
    # e comandi logici dell'applicazione.
    #
    # L'uso di una dataclass frozen=True consente di trattare questa struttura
    # come un insieme di parametri di configurazione non modificabili durante
    # l'esecuzione, riducendo il rischio di alterazioni accidentali.

    # Indici dei pulsanti associati alle principali azioni di controllo.
    button_takeoff: int = 0
    button_land: int = 1
    button_detection: int = 2
    button_quit: int = 9

    # Indici degli assi analogici impiegati per il controllo del drone
    # lungo le diverse direzioni di movimento.
    axis_lr: int = 0
    axis_fb: int = 1
    axis_yaw: int = 2
    axis_ud: int = 3

    # Soglia minima oltre la quale il valore analogico di un asse viene
    # considerato significativo. Serve a ignorare piccole oscillazioni
    # indesiderate del joystick attorno alla posizione neutra.
    deadzone: float = 0.15

    # Etichette testuali dei pulsanti, utili per mostrare all'utente una
    # corrispondenza leggibile tra input fisici e comandi applicativi.
    label_takeoff: str = "X / Cross"
    label_land: str = "O / Circle"
    label_detection: str = "[] / Square"
    label_quit: str = "Options"

    # Etichette descrittive degli assi analogici.
    label_axis_lr: str = "Stick sinistro orizzontale"
    label_axis_fb: str = "Stick sinistro verticale"
    label_axis_ud: str = "Stick destro verticale"
    label_axis_yaw: str = "Stick destro orizzontale"


@dataclass(frozen=True)
class AprilTagWorldPose:
    # Posa nota di un AprilTag nel sistema di riferimento mondo.
    #
    # Convenzione scelta:
    # - posizione_m = (x, y, z) in metri;
    # - orientation_rpy_deg = (roll, pitch, yaw) in gradi;
    # - il frame mondo ha asse Z verso l'alto;
    # - yaw positivo antiorario attorno a Z.
    position_m: tuple[float, float, float]
    orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraPoseConfig:
    # Configurazione dei parametri utilizzati per la stima di posa della camera
    # rispetto a un marker fiduciale, verosimilmente tramite AprilTag.

    # Abilita o disabilita il sottosistema di stima della posa.
    enabled: bool = True

    # Famiglia di tag da rilevare.
    tag_family: str = "tag25h9"

    # Numero di thread utilizzati dal processo di rilevamento.
    # Un valore maggiore può migliorare le prestazioni, se l'hardware lo consente.
    threads: int = 4

    # Fattore di decimazione dell'immagine per velocizzare la detection.
    # Valori più alti riducono il dettaglio, ma possono aumentare la rapidità.
    decimate: float = 2.0

    # Dimensione reale del tag nel mondo fisico, espressa in metri.
    # Questo parametro è necessario per ricostruire correttamente la posa 3D.
    tag_size_m: float = 0.2

    # Posizione del tag nel sistema di riferimento scelto.
    # La tupla contiene tipicamente coordinate (x, y, z).
    #
    # Campo mantenuto per retrocompatibilità con il vecchio caso a singolo tag.
    tag_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Orientamento del tag singolo espresso come roll, pitch, yaw in gradi.
    # Usato solo nel caso legacy a singolo tag.
    tag_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mappa dei tag noti nel mondo.
    # Chiave: ID del tag AprilTag.
    # Valore: posa assoluta nota del tag nel frame mondo.
    #
    # Esempio da personalizzare con i vostri 4 tag:
    # {
    #     0: AprilTagWorldPose(position_m=(0.0, 0.0, 0.0), orientation_rpy_deg=(0.0, 0.0, 0.0)),
    #     1: AprilTagWorldPose(position_m=(2.0, 0.0, 0.0), orientation_rpy_deg=(0.0, 0.0, 180.0)),
    #     2: AprilTagWorldPose(position_m=(2.0, 2.0, 0.0), orientation_rpy_deg=(0.0, 0.0, -90.0)),
    #     3: AprilTagWorldPose(position_m=(0.0, 2.0, 0.0), orientation_rpy_deg=(0.0, 0.0, 90.0)),
    # }
    world_tags: dict[int, AprilTagWorldPose] = field(
        default_factory=lambda: {
            0: AprilTagWorldPose(
                position_m=(0.0, 0.0, 0.0),
                orientation_rpy_deg=(0.0, 0.0, 0.0),
            ),
            1: AprilTagWorldPose(
                position_m=(2.0, 0.0, 0.0),
                orientation_rpy_deg=(0.0, 0.0, 180.0),
            ),
            2: AprilTagWorldPose(
                position_m=(2.0, 2.0, 0.0),
                orientation_rpy_deg=(0.0, 0.0, -90.0),
            ),
            3: AprilTagWorldPose(
                position_m=(0.0, 2.0, 0.0),
                orientation_rpy_deg=(0.0, 0.0, 90.0),
            ),
        }
    )

    # Modalità di fusione delle ipotesi di posa assoluta quando sono visibili
    # più tag contemporaneamente:
    # - "weighted_average": media pesata di posizione e yaw
    # - "best_tag": usa solo il tag con peso migliore
    fusion_mode: str = "weighted_average"

    # Matrice intrinseca della camera, necessaria per la proiezione prospettica
    # e per la stima di posa. I parametri sono generalmente ottenuti da una
    # procedura di calibrazione della camera.
    camera_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (939.9451542597831, 0.0, 486.5451569111854),
        (0.0, 940.8447164292168, 329.0528996241572),
        (0.0, 0.0, 1.0),
    )

    # Coefficienti di distorsione dell'obiettivo.
    # Anch'essi derivano dalla calibrazione e permettono di correggere
    # le deformazioni introdotte dalla lente.
    dist_coeffs: tuple[float, float, float, float, float] = (
        0.02411954416409598,
        -0.18722784189053135,
        -0.011223295008542767,
        -0.0019630367896265257,
        0.5027508681156333,
    )


@dataclass(frozen=True)
class AppConfig:
    # Configurazione generale dell'applicazione.
    # Questa classe raccoglie in un unico oggetto tutti i parametri principali
    # necessari all'esecuzione del programma.

    # Dimensione della finestra grafica principale.
    window_size: tuple[int, int] = (960, 720)

    # Titolo mostrato nella finestra dell'applicazione.
    window_title: str = "Tello Detection - PS4 Controller"

    # Velocità di comando da inviare al drone.
    speed: int = 30

    # Frequenza di aggiornamento desiderata dell'applicazione.
    fps: int = 20

    # Timeout massimo per la ricezione dei frame video.
    # Se non arrivano frame entro questo intervallo, il programma può
    # considerare la sorgente video non aggiornata o temporaneamente assente.
    frame_timeout_sec: float = 2.0

    # Soglia minima di confidenza per accettare una detection come valida.
    confidence_threshold: float = 0.5

    # Dimensione dell'immagine da fornire al modello di visione.
    image_size: int = 640

    # I frame restituiti dal controller sono già adatti a OpenCV; un'ulteriore
    # conversione RGB->BGR altererebbe i colori e può peggiorare la detection.
    frame_from_controller_is_rgb: bool = True

    # Livello di verbosità del logging applicativo.
    log_level: str = "INFO"

    # Percorso del file contenente i pesi del modello.
    # Il percorso è costruito in modo relativo rispetto alla directory del file,
    # così da rendere il progetto più portabile tra ambienti diversi.
    model_path: Path = (
        BASE_DIR / "models" / "ver2clean_n300_extra_nets_s" / "weights" / "best.pt"
    )

    # Configurazione della mappatura del joystick.
    # default_factory viene usato per creare una nuova istanza di default
    # senza condividere oggetti mutabili tra eventuali istanze della classe.
    joystick: JoystickMapping = field(default_factory=JoystickMapping)

    # Configurazione dei parametri di camera e stima della posa.
    camera_pose: CameraPoseConfig = field(default_factory=CameraPoseConfig)


# Istanza globale della configurazione applicativa.
# In questo modo il resto del progetto può importare APP_CONFIG e accedere
# in maniera centralizzata ai parametri di configurazione.
APP_CONFIG = AppConfig()