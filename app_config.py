# Importazione del decoratore @dataclass e della funzione field.
# @dataclass consente di definire classi orientate principalmente alla memorizzazione
# di dati in modo sintetico e leggibile; field permette di specificare modalità
# particolari di inizializzazione degli attributi.
from dataclasses import dataclass, field

# Path fornisce un'interfaccia ad oggetti per la gestione dei percorsi di file e cartelle.
from pathlib import Path


# Directory base del file corrente.
# Viene calcolata a partire dal percorso assoluto del file Python in esecuzione
# e costituisce il riferimento per costruire percorsi relativi, ad esempio
# verso modelli o risorse applicative.
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class JoystickMapping:
    # Classe di configurazione immutabile che raccoglie la mappatura tra
    # i controlli fisici del joystick e i comandi logici utilizzati dall'applicazione.
    # L'opzione frozen=True rende gli oggetti non modificabili dopo la creazione,
    # così da preservare la coerenza della configurazione durante l'esecuzione.

    # Indici reali usati da pygame per identificare i pulsanti del controller.
    # Tali valori dipendono dalla convenzione adottata dal dispositivo e dalla libreria.
    button_takeoff: int = 0
    button_land: int = 1
    button_quit: int = 9

    # Indici degli assi analogici associati ai diversi gradi di libertà del drone:
    # left-right, forward-backward, yaw e up-down.
    axis_lr: int = 0
    axis_fb: int = 1
    axis_yaw: int = 2
    axis_ud: int = 3

    # Soglia di deadzone degli stick analogici.
    # Valori assoluti inferiori a questa soglia vengono normalmente ignorati
    # per evitare movimenti indesiderati dovuti a piccole oscillazioni o rumore.
    deadzone: float = 0.15

    # Etichette descrittive da mostrare all'utente nell'interfaccia o nei messaggi.
    # Questi campi non influiscono sulla logica di controllo, ma migliorano la leggibilità.
    label_takeoff: str = "X / Cross"
    label_land: str = "O / Circle"
    label_quit: str = "Options"

    label_axis_lr: str = "Stick sinistro orizzontale"
    label_axis_fb: str = "Stick sinistro verticale"
    label_axis_ud: str = "Stick destro verticale"
    label_axis_yaw: str = "Stick destro orizzontale"


@dataclass(frozen=True)
class CameraPoseConfig:
    # Classe di configurazione immutabile dedicata alla stima della posa della camera
    # tramite rilevazione di marker AprilTag. Essa raccoglie parametri di attivazione,
    # calibrazione e rilevazione utilizzati dal sottosistema di visione.

    # Abilita/disabilita la stima della posa tramite AprilTag.
    # Questo flag consente di escludere l'intero blocco di stima senza dover
    # modificare il resto del codice applicativo.
    enabled: bool = True

    # Parametri del detector AprilTag.
    # tag_family identifica la famiglia di marker cercata;
    # threads specifica il numero di thread di elaborazione;
    # decimate regola un eventuale ridimensionamento dell'immagine per accelerare il detector.
    tag_family: str = "tag25h9"
    threads: int = 4
    decimate: float = 2.0

    # Dimensione reale del tag espressa in metri.
    # Questo parametro è fondamentale per convertire la stima geometrica
    # da grandezze puramente pixel a una posa metrica nello spazio.
    tag_size_m: float = 0.1167

    # Posizione del tag nel mondo.
    # Per ora viene mantenuta come riferimento configurabile;
    # la posa mostrata a video resta quella della camera nel frame del tag.
    # La tupla rappresenta una posizione tridimensionale (x, y, z).
    tag_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Matrice intrinseca della camera.
    # Questa matrice 3x3 contiene i parametri di calibrazione interna:
    # lunghezze focali e coordinate del punto principale.
    # È necessaria per la ricostruzione geometrica e per la stima della posa.
    camera_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (939.9451542597831, 0.0, 486.5451569111854),
        (0.0, 940.8447164292168, 329.0528996241572),
        (0.0, 0.0, 1.0),
    )

    # Coefficienti di distorsione dell'ottica.
    # Tali valori permettono di correggere le deformazioni introdotte dalla lente,
    # migliorando l'accuratezza della stima visiva.
    dist_coeffs: tuple[float, float, float, float, float] = (
        0.02411954416409598,
        -0.18722784189053135,
        -0.011223295008542767,
        -0.0019630367896265257,
        0.5027508681156333,
    )


@dataclass(frozen=True)
class AppConfig:
    # Classe principale di configurazione dell'applicazione.
    # Riunisce in un unico oggetto i parametri relativi all'interfaccia grafica,
    # al controllo del drone, alla gestione del flusso video, al modello di detection
    # e alle sottoconfigurazioni di joystick e stima della posa.

    # Dimensione della finestra grafica espressa in pixel: (larghezza, altezza).
    window_size: tuple[int, int] = (960, 720)

    # Titolo della finestra applicativa.
    window_title: str = "Tello Detection - PS4 Controller"

    # Velocità base dei comandi RC inviati al drone.
    # Il valore rappresenta il livello di intensità dei comandi di movimento.
    speed: int = 30

    # FPS del loop principale.
    # Determina la frequenza di aggiornamento dell'applicazione e quindi
    # il ritmo con cui vengono gestiti eventi, rendering e comandi.
    fps: int = 20

    # Se True, il joystick controlla il drone solo quando la finestra pygame è attiva.
    # Questa scelta riduce il rischio di inviare comandi involontari quando
    # l'utente sta interagendo con altre applicazioni.
    require_focus_for_control: bool = True

    # Timeout massimo senza ricevere frame video.
    # Può essere usato per rilevare eventuali interruzioni del flusso della camera.
    frame_timeout_sec: float = 2.0

    # Parametri YOLO.
    # confidence_threshold rappresenta la soglia minima di confidenza per considerare valida una detection;
    # image_size definisce la dimensione dell'immagine in input al modello.
    confidence_threshold: float = 0.5
    image_size: int = 640

    # Se il frame ottenuto dal controller fosse RGB, viene convertito in BGR per OpenCV.
    # Questo parametro serve a gestire correttamente la convenzione dei canali colore
    # attesa dalla libreria di elaborazione delle immagini.
    frame_from_controller_is_rgb: bool = False

    # Livello di verbosità del logging applicativo.
    log_level: str = "INFO"

    # Percorso del modello YOLO.
    # Il path viene costruito in modo robusto a partire dalla directory del file corrente,
    # così da rendere l'applicazione meno dipendente dalla cartella di esecuzione.
    model_path: Path = BASE_DIR / "models" / "ver2clean_n300_extra_nets_s" / "weights" / "best.pt"

    # Sottoconfigurazione relativa al joystick.
    # field(default_factory=...) viene usato per creare una nuova istanza
    # per ogni oggetto AppConfig, evitando problematiche legate a valori mutabili condivisi.
    joystick: JoystickMapping = field(default_factory=JoystickMapping)

    # Sottoconfigurazione relativa alla stima della posa della camera.
    camera_pose: CameraPoseConfig = field(default_factory=CameraPoseConfig)


# Istanza globale della configurazione applicativa.
# In questo modo il resto del programma può accedere a un unico punto centralizzato
# contenente tutti i parametri necessari all'esecuzione.
APP_CONFIG = AppConfig()