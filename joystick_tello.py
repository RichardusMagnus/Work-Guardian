import logging
import os

# pygame fornisce il supporto all'acquisizione degli eventi provenienti
# dal controller e alla gestione della finestra applicativa associata.
import pygame

# Import della configurazione centralizzata dell'applicazione.
# In particolare, da APP_CONFIG vengono recuperati:
# - la dimensione e il titolo della finestra pygame;
# - la mappatura tra pulsanti/assi fisici e comandi logici del drone.
from app_config import APP_CONFIG

# Logger di modulo utilizzato per tracciare eventi rilevanti, come
# connessioni/disconnessioni del joystick ed eventuali errori di rilascio risorse.
LOGGER = logging.getLogger(__name__)

# Riferimento globale al joystick attualmente connesso e inizializzato.
# Il valore None indica che non è presente alcun controller attivo.
JOYSTICK = None


def _zero_command():
    # Restituisce un comando nullo su tutti gli assi di controllo.
    # Questa funzione costituisce un valore di fallback sicuro da usare
    # quando il joystick non è disponibile oppure quando la lettura
    # dell'input non può essere effettuata correttamente.
    return {
        "lr": 0,
        "fb": 0,
        "ud": 0,
        "yaw": 0,
    }


def _disconnect_current_joystick():
    # Disconnette in modo sicuro il joystick attualmente associato alla variabile globale.
    # La funzione tenta di chiudere il dispositivo e, indipendentemente dall'esito,
    # azzera il riferimento globale per evitare l'uso successivo di un oggetto
    # non più valido o non più sincronizzato con lo stato reale dell'hardware.
    global JOYSTICK

    # Se non è presente alcun joystick attivo, non è necessario eseguire alcuna operazione.
    if JOYSTICK is None:
        return

    try:
        # Richiede a pygame il rilascio del dispositivo.
        JOYSTICK.quit()
    except Exception:
        # L'eccezione viene registrata nel log per consentire diagnosi successive,
        # senza però interrompere il flusso del programma.
        LOGGER.exception("Errore durante la chiusura del joystick.")
    finally:
        # In ogni caso il riferimento globale viene invalidato,
        # così da mantenere coerente lo stato interno del modulo.
        JOYSTICK = None


def close_joystick():
    """Rilascia in modo sicuro il joystick corrente e il sottosistema joystick di pygame."""
    # Prima viene rilasciato l'eventuale joystick attivo.
    _disconnect_current_joystick()

    try:
        # Se il sottosistema joystick di pygame è inizializzato, viene chiuso.
        # Ciò consente di liberare correttamente le risorse native associate.
        if pygame.joystick.get_init():
            pygame.joystick.quit()
    except pygame.error:
        # Eventuali errori di pygame in fase di chiusura vengono ignorati
        # per privilegiare una terminazione robusta del programma.
        pass


def _enable_background_joystick_events():
    """
    Consente a SDL/pygame di continuare a ricevere eventi del joystick anche
    quando la finestra pygame non ha il focus, ad esempio se l'utente clicca
    sulla finestra OpenCV del video.
    """
    # La variabile d'ambiente viene impostata solo se non già definita.
    # In questo modo si preserva un'eventuale configurazione esterna esplicita.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")


def _get_joystick_instance_id(joystick):
    # Restituisce l'identificativo univoco dell'istanza di joystick.
    # Tale identificativo è utile per distinguere gli eventi relativi
    # al controller attualmente attivo da quelli provenienti da altri dispositivi.
    if joystick is None:
        return None

    try:
        return joystick.get_instance_id()
    except (AttributeError, OSError, pygame.error):
        # Alcune versioni/API potrebbero non supportare tale metodo,
        # oppure il dispositivo potrebbe trovarsi in uno stato non valido.
        return None


def _get_event_instance_id(event):
    # Estrae l'identificativo del joystick associato all'evento.
    # Alcune versioni di pygame espongono l'attributo `instance_id`,
    # mentre altre usano `which`: la funzione uniforma quindi l'accesso
    # a questa informazione per rendere il codice più robusto rispetto
    # alle differenze tra versioni dell'API.
    event_instance_id = getattr(event, "instance_id", None)
    if event_instance_id is not None:
        return event_instance_id

    return getattr(event, "which", None)


def _connect_first_joystick(device_index=0):
    # Collega e inizializza il joystick individuato da device_index.
    # Se l'indice fornito non è valido, viene selezionato il primo dispositivo disponibile.
    global JOYSTICK

    # Numero complessivo di joystick attualmente rilevati da pygame.
    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        # In assenza di dispositivi disponibili, viene sollevata un'eccezione esplicativa.
        raise RuntimeError(
            "Nessun joystick rilevato. Collega il controller PS4 prima di avviare il programma."
        )

    # Validazione dell'indice del dispositivo richiesto.
    # Se l'indice non appartiene all'intervallo valido, si forza l'uso del primo joystick.
    if not 0 <= device_index < joystick_count:
        device_index = 0

    # Prima di collegare un nuovo controller si rilascia quello eventualmente attivo,
    # così da mantenere un solo joystick gestito dal programma.
    _disconnect_current_joystick()

    # Creazione dell'oggetto joystick e inizializzazione del dispositivo.
    joystick = pygame.joystick.Joystick(device_index)
    joystick.init()
    JOYSTICK = joystick

    # Registrazione di alcune informazioni descrittive utili in fase di avvio e debug.
    LOGGER.info("Joystick collegato: %s", JOYSTICK.get_name())
    LOGGER.info(
        "Assi: %s | Pulsanti: %s",
        JOYSTICK.get_numaxes(),
        JOYSTICK.get_numbuttons(),
    )


def init_joystick():
    # Abilita la ricezione degli eventi del joystick anche in background,
    # quindi inizializza pygame e il relativo sottosistema joystick.
    _enable_background_joystick_events()
    pygame.init()
    pygame.joystick.init()

    # Creazione della finestra applicativa in accordo con la configurazione globale.
    # La funzione restituisce il riferimento alla superficie grafica,
    # che potrà essere usata da altri moduli dell'applicazione.
    screen = pygame.display.set_mode(APP_CONFIG.window_size)
    pygame.display.set_caption(APP_CONFIG.window_title)

    # Se un joystick è già presente all'avvio, viene collegato subito.
    # In caso contrario l'applicazione continua comunque a funzionare,
    # lasciando alla gestione degli eventi la possibilità di collegare
    # automaticamente il controller quando verrà inserito.
    if pygame.joystick.get_count() > 0:
        _connect_first_joystick()
    else:
        LOGGER.warning(
            "Nessun joystick rilevato all'avvio: l'app resta attiva in attesa di una connessione."
        )

    # Output della funzione: finestra pygame inizializzata.
    return screen


def print_joystick_help():
    # Stampa a terminale una legenda dei comandi del controller.
    # Le etichette sono ricavate dalla configurazione, così da mantenere coerenza
    # tra documentazione a schermo e mappatura effettivamente usata dal programma.
    mapping = APP_CONFIG.joystick

    print("\n[COMANDI JOYSTICK]")
    print(f"  {mapping.label_takeoff:<28} -> decollo")
    print(f"  {mapping.label_land:<28} -> atterraggio")
    print(f"  {mapping.label_detection:<28} -> attiva/disattiva YOLO")
    print(f"  {mapping.label_quit:<28} -> uscita")
    print()
    print(f"  {mapping.label_axis_lr:<28} -> sinistra / destra")
    print(f"  {mapping.label_axis_fb:<28} -> avanti / indietro")
    print(f"  {mapping.label_axis_ud:<28} -> su / giù")
    print(f"  {mapping.label_axis_yaw:<28} -> rotazione yaw")
    print("----------------------------------------")


def read_events():
    # Legge e interpreta gli eventi generati da pygame, producendo un dizionario
    # di azioni logiche che il resto del programma può utilizzare indipendentemente
    # dai dettagli dell'hardware di input.
    global JOYSTICK

    # Dizionario delle azioni elementari attivabili tramite finestra o controller.
    # Ogni chiave rappresenta un comando logico di alto livello atteso
    # dal resto dell'applicazione.
    actions = {
        "quit": False,
        "takeoff": False,
        "land": False,
        "detect": False,
    }

    # Identificativo dell'istanza di joystick attualmente attiva.
    # Viene usato per filtrare eventi provenienti da dispositivi differenti.
    active_instance_id = _get_joystick_instance_id(JOYSTICK)
    mapping = APP_CONFIG.joystick

    # Recupero di tutti gli eventi attualmente presenti nella coda di pygame.
    # Se la lettura fallisce, per sicurezza si richiede l'uscita dal programma.
    try:
        events = pygame.event.get()
    except pygame.error:
        actions["quit"] = True
        return actions

    # Analisi sequenziale di tutti gli eventi presenti nella coda di pygame.
    for event in events:
        if event.type == pygame.QUIT:
            # Chiusura della finestra grafica tramite interfaccia del sistema operativo.
            actions["quit"] = True

        elif event.type == pygame.KEYDOWN:
            # È prevista anche una scorciatoia da tastiera per uscire dal programma.
            if event.key == pygame.K_ESCAPE:
                actions["quit"] = True

        elif event.type == pygame.JOYBUTTONDOWN:
            # Per gli eventi del joystick si verifica che provengano
            # dal controller attualmente considerato attivo.
            event_instance_id = _get_event_instance_id(event)
            if active_instance_id is not None and event_instance_id not in (None, active_instance_id):
                continue

            # Traduzione dei pulsanti fisici in azioni logiche applicative.
            # Questa mediazione separa il livello hardware dal livello logico.
            if event.button == mapping.button_takeoff:
                actions["takeoff"] = True
            elif event.button == mapping.button_land:
                actions["land"] = True
            elif event.button == mapping.button_detection:
                actions["detect"] = True
            elif event.button == mapping.button_quit:
                actions["quit"] = True

        elif event.type == pygame.JOYDEVICEADDED:
            # Se viene collegato un nuovo joystick e non ce n'è già uno attivo,
            # si prova a inizializzarlo automaticamente.
            if JOYSTICK is None:
                try:
                    _connect_first_joystick(getattr(event, "device_index", 0))
                    active_instance_id = _get_joystick_instance_id(JOYSTICK)
                except RuntimeError:
                    # In caso di problemi di connessione si prosegue senza interrompere il programma.
                    pass

        elif event.type == pygame.JOYDEVICEREMOVED:
            # Gestione della disconnessione del joystick.
            removed_instance_id = _get_event_instance_id(event)

            # Se il dispositivo rimosso coincide con quello attivo,
            # oppure se l'identificativo dell'attivo non è disponibile,
            # si procede al rilascio del controller corrente.
            if active_instance_id is None or removed_instance_id == active_instance_id:
                LOGGER.warning("Joystick disconnesso.")
                _disconnect_current_joystick()
                active_instance_id = None

                # Se sono presenti altri joystick, si tenta una riconnessione automatica.
                if pygame.joystick.get_count() > 0:
                    try:
                        _connect_first_joystick()
                        active_instance_id = _get_joystick_instance_id(JOYSTICK)
                    except RuntimeError:
                        # Se la riconnessione fallisce, il programma viene indirizzato verso l'uscita.
                        actions["quit"] = True
                else:
                    # In assenza di controller disponibili, si richiede la chiusura.
                    actions["quit"] = True

    # Output della funzione: dizionario delle azioni attivate nell'iterazione corrente.
    return actions


def _apply_deadzone(value, deadzone):
    # Applica una zona morta all'input analogico.
    # Valori di piccola ampiezza vengono forzati a zero per eliminare
    # oscillazioni spurie dovute a rumore o imperfezioni meccaniche del joystick.
    if abs(value) < deadzone:
        return 0.0
    return value


def _axis_to_speed(value, speed):
    # Converte il valore analogico di un asse, tipicamente compreso tra -1 e 1,
    # in una velocità intera proporzionale limitata dall'ampiezza massima speed.
    # Il flusso di conversione è il seguente:
    # 1. vincolo della velocità massima nell'intervallo [0, 100];
    # 2. applicazione della deadzone;
    # 3. saturazione del valore analogico nell'intervallo [-1, 1];
    # 4. conversione finale in intero.
    speed = max(0, min(100, int(speed)))
    value = _apply_deadzone(value, APP_CONFIG.joystick.deadzone)
    value = max(-1.0, min(1.0, value))
    return int(value * speed)


def _get_axis_value(index):
    # Restituisce il valore corrente dell'asse indicato.
    # In caso di joystick assente, indice non valido o errore di lettura,
    # viene restituito 0.0 come valore neutro.
    if JOYSTICK is None:
        return 0.0

    try:
        # Verifica preventiva che l'indice richiesto corrisponda
        # a un asse effettivamente disponibile sul controller.
        if index < 0 or index >= JOYSTICK.get_numaxes():
            return 0.0
        return JOYSTICK.get_axis(index)
    except (AttributeError, OSError, pygame.error):
        # Gestione difensiva di possibili anomalie legate al dispositivo o all'API.
        return 0.0


def get_command(speed=50):
    # Produce il comando di movimento corrente leggendo lo stato degli assi analogici
    # del joystick e convertendolo nel formato atteso dal resto dell'applicazione.
    try:
        # Aggiorna internamente lo stato degli eventi di pygame.
        # Questa operazione è necessaria per mantenere aggiornati i valori degli assi.
        pygame.event.pump()
    except pygame.error:
        return _zero_command()

    # Se nessun joystick è attualmente disponibile, si restituisce un comando nullo.
    if JOYSTICK is None:
        return _zero_command()

    mapping = APP_CONFIG.joystick

    # Lettura e conversione degli assi secondo la mappatura configurata.
    # Per alcuni assi viene applicato il segno meno per uniformare
    # la convenzione fisica del controller con quella del sistema di comando.
    lr = _axis_to_speed(_get_axis_value(mapping.axis_lr), speed)
    fb = _axis_to_speed(-_get_axis_value(mapping.axis_fb), speed)
    ud = _axis_to_speed(-_get_axis_value(mapping.axis_ud), speed)
    yaw = _axis_to_speed(_get_axis_value(mapping.axis_yaw), speed)

    # Restituzione del comando completo nelle quattro componenti principali:
    # left-right, forward-backward, up-down e rotazione yaw.
    # Tale struttura dati costituisce l'output logico del modulo verso
    # i componenti che si occupano del controllo del drone.
    return {
        "lr": lr,
        "fb": fb,
        "ud": ud,
        "yaw": yaw,
    }