from __future__ import annotations

# Importa il decoratore dataclass e la funzione field, utilizzati per
# definire classi di sola configurazione in modo compatto, leggibile e tipizzato.
# In particolare, field permette di specificare valori di default costruiti
# dinamicamente, evitando problemi con oggetti mutabili condivisi.
from dataclasses import dataclass, field

# Path fornisce una gestione robusta e indipendente dal sistema operativo
# dei percorsi di file e directory. È preferibile rispetto alle stringhe semplici
# quando il codice deve costruire percorsi relativi a risorse del progetto.
from pathlib import Path

# BASE_DIR rappresenta la directory che contiene il file corrente.
# Tale riferimento viene utilizzato come punto di partenza robusto per
# costruire percorsi relativi ad altre risorse del progetto, ad esempio
# modelli di deep learning o file di configurazione.
#
# L'uso di Path(__file__).resolve().parent consente di ottenere un percorso
# assoluto affidabile, indipendente dalla directory di esecuzione del programma.
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class JoystickMapping:
    # Configurazione immutabile della mappatura tra pulsanti/assi del joystick
    # e comandi logici dell'applicazione.
    #
    # L'uso di una dataclass frozen=True consente di trattare questa struttura
    # come un insieme di parametri di configurazione non modificabili durante
    # l'esecuzione, riducendo il rischio di alterazioni accidentali.
    #
    # Questa classe centralizza la corrispondenza tra input hardware del controller
    # e azioni software, favorendo chiarezza, manutenzione e riuso.

    # Indici dei pulsanti associati alle principali azioni di controllo.
    # Ogni indice identifica la posizione del pulsante nel sistema di input
    # fornito dalla libreria che gestisce il joystick.
    button_takeoff: int = 0
    button_land: int = 1
    button_detection: int = 2
    button_quit: int = 6

    # Indici degli assi analogici impiegati per il controllo del drone
    # lungo le diverse direzioni di movimento.
    #
    # Convenzione tipica:
    # - lr   = left-right  -> movimento laterale;
    # - fb   = forward-backward -> avanzamento/arretramento;
    # - yaw  = rotazione attorno all'asse verticale;
    # - ud   = up-down -> quota verticale.
    axis_lr: int = 0
    axis_fb: int = 1
    axis_yaw: int = 2
    axis_ud: int = 3

    # Soglia minima oltre la quale il valore analogico di un asse viene
    # considerato significativo. Serve a ignorare piccole oscillazioni
    # indesiderate del joystick attorno alla posizione neutra.
    #
    # Questo parametro è particolarmente utile per compensare il rumore
    # o le imprecisioni meccaniche tipiche degli stick analogici.
    deadzone: float = 0.15

    # Etichette testuali dei pulsanti, utili per mostrare all'utente una
    # corrispondenza leggibile tra input fisici e comandi applicativi.
    # Tali stringhe possono essere impiegate, ad esempio, in interfacce grafiche,
    # schermate informative o messaggi di supporto all'utente.
    label_takeoff: str = "X / Cross"
    label_land: str = "O / Circle"
    label_detection: str = "[] / Square"
    label_quit: str = "Options"

    # Etichette descrittive degli assi analogici.
    # Anche queste informazioni hanno una funzione documentativa e di supporto
    # all'interazione uomo-macchina.
    label_axis_lr: str = "Stick sinistro orizzontale"
    label_axis_fb: str = "Stick sinistro verticale"
    label_axis_ud: str = "Stick destro verticale"
    label_axis_yaw: str = "Stick destro orizzontale"


@dataclass(frozen=True)
class YoloModelConfig:
    # Configurazione di un singolo modello YOLO.
    # Ogni voce descrive:
    # - nome logico del modello;
    # - percorso del file dei pesi;
    # - colore usato per le annotazioni video.
    #
    # Questa struttura permette di gestire più modelli di rilevamento in modo
    # uniforme, senza disperdere i relativi parametri nel resto del codice.
    name: str
    path: Path
    color: tuple[int, int, int] = (0, 255, 0)


@dataclass(frozen=True)
class AprilTagWorldPose:
    # Posa nota di un AprilTag nel sistema di riferimento mondo.
    #
    # Convenzione scelta:
    # - posizione_m = (x, y, z) in metri;
    # - orientation_rpy_deg = (roll, pitch, yaw) in gradi;
    # - il frame mondo ha asse Z verso l'alto;
    # - yaw positivo antiorario attorno a Z.
    #
    # Questa informazione è necessaria quando si vuole stimare una posa assoluta
    # della camera o del drone a partire dall'osservazione di marker fiduciali
    # con posizione/orientamento noti nell'ambiente.
    position_m: tuple[float, float, float]
    orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DroneExtrinsicsConfig:
    # Trasformazione rigida nota tra frame drone/body e frame camera.
    #
    # Convenzione adottata:
    # - camera_position_in_drone_frame_m = posizione dell'origine camera
    #   espressa nel frame drone/body;
    # - camera_orientation_rpy_deg = orientamento del frame camera rispetto
    #   al frame drone/body, espresso come roll, pitch, yaw in gradi.
    #
    # Con valori nulli si assume, per retrocompatibilità, che la posa della
    # camera coincida con la posa del drone/body.
    #
    # Questa configurazione è rilevante quando la camera non è perfettamente
    # coincidente con il centro di riferimento del drone e si desidera convertire
    # una stima di posa della camera in una stima di posa del veicolo.
    camera_position_in_drone_frame_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraPoseConfig:
    # Configurazione dei parametri utilizzati per la stima di posa della camera
    # rispetto a un marker fiduciale, verosimilmente tramite AprilTag.
    #
    # Questa classe raccoglie tutti i parametri necessari al sottosistema
    # di localizzazione basato su marker: parametri del detector, geometria
    # dei tag, conoscenza del mondo e parametri di calibrazione della camera.

    # Abilita o disabilita il sottosistema di stima della posa.
    # In questo modo la funzionalità può essere esclusa senza intervenire
    # sul codice applicativo principale.
    enabled: bool = True

    # Famiglia di tag da rilevare.
    # Il valore specifica il dizionario AprilTag atteso dal detector.
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
    # In una versione estesa del sistema, la localizzazione è gestita tramite
    # la mappa world_tags definita più avanti.
    tag_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Orientamento del tag singolo espresso come roll, pitch, yaw in gradi.
    # Usato solo nel caso legacy a singolo tag.
    tag_orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mappa dei tag noti nel mondo.
    # Chiave: ID del tag AprilTag.
    # Valore: posa assoluta nota del tag nel frame mondo.
    #
    # L'uso di field(default_factory=...) è necessario per evitare di condividere
    # accidentalmente lo stesso oggetto mutabile tra istanze diverse della classe.
    world_tags: dict[int, AprilTagWorldPose] = field(
        default_factory=lambda: {
            # Le voci commentate mantengono traccia di una possibile precedente
            # configurazione sperimentale dei marker, senza interferire con la
            # configurazione attualmente attiva.
            #0: AprilTagWorldPose(
            #    position_m=(1.7, 0.9, 1.4),
            #    orientation_rpy_deg=(-90.0, 0.0, 90.0),
            #),
            #1: AprilTagWorldPose(
            #    position_m=(-1.7, -0.15, 1.4),
            #    orientation_rpy_deg=(-90.0, 0.0, -90.0),
            #),
            #2: AprilTagWorldPose(
            #    position_m=(0.0, 0.0, 0.0),
            #    orientation_rpy_deg=(0.0, 0.0, 0.0),
            #),
            #3: AprilTagWorldPose(
            #    position_m=(-0.60, -4.35, 1.83),
            #    orientation_rpy_deg=(-90.0, 0.0, 0.0),
            #),

            # Configurazione attiva della mappa dei marker.
            # Ogni identificativo numerico corrisponde a un AprilTag fisicamente
            # posizionato nell'ambiente di prova. Le coordinate e gli orientamenti
            # sono impiegati per trasformare una misura relativa tag-camera in
            # una stima assoluta rispetto al frame mondo.
            0: AprilTagWorldPose(
                position_m=(0.0, 0.0, 0.0),
                orientation_rpy_deg=(0.0, 0.0, 0.0),
            ),
            1: AprilTagWorldPose(
                position_m=(0.0, -2.11, 0.0),
                orientation_rpy_deg=(0.0, 0.0, 0.0),
            ),
            2: AprilTagWorldPose(
                position_m=(0.0, 2.96, 1.25),
                orientation_rpy_deg=(-90.0, 180.0, 0.0),
            ),
            3: AprilTagWorldPose(
                position_m=(0.0, -5.07, 1.25),
                orientation_rpy_deg=(-90.0, 0.0, 0.0),
            ),
            4: AprilTagWorldPose(
                position_m=(1.86, 0.0, 1.4),
                orientation_rpy_deg=(-90.0, 0.0, 90.0),
            ),
            5: AprilTagWorldPose(
                position_m=(1.86, -2.11, 1.4),
                orientation_rpy_deg=(-90.0, 0.0, 90.0),
            ),
            6: AprilTagWorldPose(
                position_m=(1.86, -3.91, 1.4),
                orientation_rpy_deg=(-90.0, 0.0, 90.0),
            ),
            7: AprilTagWorldPose(
                position_m=(1.86, 1.8, 1.4),
                orientation_rpy_deg=(-90.0, 0.0, 90.0),
            ),
            8: AprilTagWorldPose(
                position_m=(-2.04, 0.0, 1.5),
                orientation_rpy_deg=(-90.0, 0.0, -90.0),
            ),
            9: AprilTagWorldPose(
                position_m=(-2.04, -1.35, 1.5),
                orientation_rpy_deg=(-90.0, 0.0, -90.0),
            ),
            10: AprilTagWorldPose(
                position_m=(-1.18, 2.1, 1.25),
                orientation_rpy_deg=(-90.0, 0.0, -90.0),
            ),
            11: AprilTagWorldPose(
                position_m=(-1.31, -1.35, 1.25),
                orientation_rpy_deg=(-90.0, 0.0, -90.0),
            ),
        }
    )

    # Extrinseca rigida camera -> drone/body.
    # Il default_factory crea una nuova configurazione di default per ogni
    # istanza di CameraPoseConfig, mantenendo la struttura estendibile.
    drone_extrinsics: DroneExtrinsicsConfig = field(default_factory=DroneExtrinsicsConfig)

    # Modalità di fusione delle ipotesi di posa assoluta quando sono visibili
    # più tag contemporaneamente.
    #
    # Il valore "weighted_average" suggerisce che le stime ottenute da più marker
    # vengano combinate tramite una media pesata, presumibilmente in funzione
    # dell'affidabilità geometrica o della qualità della detection.
    fusion_mode: str = "weighted_average"

    # Matrice intrinseca della camera.
    #
    # La matrice contiene i parametri di calibrazione interni:
    # - lunghezze focali espresse in pixel;
    # - coordinate del centro ottico;
    # - termine omogeneo finale.
    #
    # Tali valori sono necessari per stimare correttamente la posa 3D a partire
    # dalle coordinate 2D osservate nell'immagine.
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
    #
    # Questi parametri descrivono le deformazioni introdotte dalla lente
    # e vengono usati dagli algoritmi di visione per correggere o modellare
    # la proiezione dell'immagine.
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
    #
    # Questa classe raccoglie in un'unica struttura i principali parametri
    # dell'applicativo: interfaccia grafica, elaborazione video, modelli YOLO,
    # logging, controllo tramite joystick e localizzazione tramite marker.
    #
    # L'approccio permette di separare i parametri configurabili dalla logica
    # esecutiva del programma, rendendo il codice principale più leggibile
    # e riducendo la duplicazione di costanti.

    # Dimensioni della finestra grafica di visualizzazione espresse in pixel.
    window_size: tuple[int, int] = (960, 720)

    # Titolo mostrato nella finestra dell'applicazione.
    window_title: str = "Tello Detection - PS4 Controller"

    # Velocità di comando del drone.
    # Il valore viene verosimilmente usato per scalare i comandi inviati
    # al drone lungo le diverse direzioni di movimento.
    speed: int = 40

    # Frequenza di aggiornamento desiderata del ciclo video/elaborativo.
    # Questo parametro fornisce un riferimento temporale per il ciclo principale
    # di acquisizione, elaborazione e visualizzazione dei frame.
    fps: int = 20

    # Tempo massimo di attesa per un frame prima di considerare anomala
    # la ricezione del flusso video.
    frame_timeout_sec: float = 2.0

    # Soglia minima di confidenza per accettare una detection YOLO.
    # Le predizioni con confidenza inferiore a questo valore vengono
    # presumibilmente scartate per limitare falsi positivi.
    confidence_threshold: float = 0.5

    # Dimensione dell'immagine di input per il modello YOLO.
    # Il ridimensionamento a una misura standard consente di mantenere
    # coerenza con le aspettative del modello di deep learning.
    image_size: int = 640

    # Indica se i frame provenienti dal controller video sono già nel formato RGB.
    # Questa informazione è importante perché alcune librerie, come OpenCV,
    # utilizzano convenzionalmente il formato BGR; un'impostazione errata può
    # causare colori alterati nelle visualizzazioni o nelle annotazioni.
    frame_from_controller_is_rgb: bool = True

    # Livello di verbosità del logging applicativo.
    # Valori come "INFO" consentono di controllare la quantità di messaggi
    # diagnostici prodotti durante l'esecuzione.
    log_level: str = "INFO"

    # Elenco dei modelli YOLO caricati dal programma principale.
    #
    # Ogni elemento della tupla contiene il nome logico del detector, il percorso
    # dei pesi addestrati e il colore delle annotazioni. L'uso di una tupla
    # riflette l'intenzione di trattare la lista dei modelli come configurazione
    # statica dell'applicazione.
    yolo_models: tuple[YoloModelConfig, ...] = (
        YoloModelConfig(
            name="PPE_Detector",
            path=BASE_DIR / "YOLO" / "ModelTester" / "models" / "ver2clean_n300_extra_nets_s" / "weights" / "best.pt",
            color=(0, 0, 255),
        ),
        YoloModelConfig(
            name="Fall_Detector",
            path=BASE_DIR / "YOLO" / "ModelTester" / "models" / "Fall_Detector_DEFHJ1" / "weights" / "best.pt",
            color=(255, 0, 0),
        ),
    )

    # Percorso del file di log dei dati di volo generato durante l'esecuzione.
    # L'uso di BASE_DIR rende il salvataggio indipendente dalla cartella da cui
    # viene lanciato il programma principale.
    flight_data_log_path: Path = BASE_DIR / "flight_data.txt"

    # Sottoconfigurazione relativa alla mappatura del joystick.
    # Il default_factory evita di riutilizzare implicitamente la stessa istanza
    # e mantiene coerente la costruzione della configurazione complessiva.
    joystick: JoystickMapping = field(default_factory=JoystickMapping)

    # Sottoconfigurazione relativa alla stima di posa della camera.
    # Questa voce incapsula tutti i parametri del sottosistema AprilTag/camera,
    # separandoli dai parametri di controllo e di visualizzazione.
    camera_pose: CameraPoseConfig = field(default_factory=CameraPoseConfig)


# Istanza globale della configurazione applicativa.
# Gli altri moduli del progetto possono importare APP_CONFIG per accedere
# in modo centralizzato ai parametri definiti in questo file, evitando
# duplicazioni e incoerenze tra componenti diversi del software.
APP_CONFIG = AppConfig()