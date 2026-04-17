import logging

import pygame

# Importa la configurazione applicativa centralizzata.
# Si assume che APP_CONFIG esponga, tra le altre cose:
# - le dimensioni della finestra pygame;
# - il titolo della finestra;
# - la mappatura degli assi e dei pulsanti del joystick;
# - l'ampiezza della deadzone per filtrare piccoli movimenti involontari.
from app_config import APP_CONFIG


# Logger di modulo, utile per tracciare eventi significativi
# come la connessione/disconnessione del controller o eventuali eccezioni.
LOGGER = logging.getLogger(__name__)

# Riferimento globale al joystick attualmente connesso e inizializzato.
# Il valore None indica assenza di un controller disponibile o attivo.
JOYSTICK = None


def _zero_command():
    """Restituisce un comando RC nullo."""
    # Questa funzione produce una struttura dati coerente con il formato
    # dei comandi RC attesi dal resto dell'applicazione, imponendo velocità
    # nulla su tutti i canali di movimento.
    return {
        "lr": 0,
        "fb": 0,
        "ud": 0,
        "yaw": 0,
    }


def _disconnect_current_joystick():
    """Chiude il joystick corrente in modo sicuro."""
    global JOYSTICK

    # Se non esiste alcun joystick attivo, non è necessario eseguire operazioni.
    if JOYSTICK is None:
        return

    try:
        # Rilascia in modo esplicito la risorsa associata al joystick corrente.
        JOYSTICK.quit()
    except Exception:
        # La chiusura del dispositivo non deve compromettere il flusso del programma;
        # l'errore viene quindi registrato nel log insieme allo stack trace.
        LOGGER.exception("Errore durante la chiusura del joystick.")
    finally:
        # In ogni caso si invalida il riferimento globale, evitando che rimanga
        # associato a un oggetto non più affidabile o già disconnesso.
        JOYSTICK = None


def _get_joystick_instance_id(joystick):
    """Restituisce l'instance_id del joystick, se disponibile."""
    # In assenza di joystick, non esiste alcun identificativo da restituire.
    if joystick is None:
        return None

    try:
        # L'instance_id consente di distinguere in modo robusto diversi dispositivi,
        # anche in presenza di riconnessioni o di più joystick collegati.
        return joystick.get_instance_id()
    except (AttributeError, pygame.error):
        # Alcune versioni/configurazioni di pygame o del dispositivo potrebbero
        # non supportare questo metodo: in tal caso si restituisce None.
        return None


def _get_window_focus() -> bool:
    """Restituisce lo stato di focus della finestra pygame in modo robusto."""
    try:
        # Verifica se la finestra pygame possiede il focus di input.
        # L'informazione può essere usata, ad esempio, per inibire i comandi
        # quando la finestra non è in primo piano.
        return bool(pygame.key.get_focused())
    except pygame.error:
        # In caso di errore del sottosistema pygame si adotta una politica prudente:
        # si considera la finestra non focalizzata.
        return False


def _connect_first_joystick(device_index=0):
    """
    Collega il primo joystick disponibile oppure quello indicato.
    Se device_index non è valido, usa il primo disponibile.
    """
    global JOYSTICK

    # Determina quanti joystick risultano attualmente visibili a pygame.
    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        # Se nessun dispositivo è disponibile, il programma non può acquisire input
        # dal controller richiesto.
        raise RuntimeError(
            "Nessun joystick rilevato. Collega il controller PS4 prima di avviare il programma."
        )

    # Se l'indice richiesto non è valido, si effettua un fallback sul primo dispositivo.
    if not 0 <= device_index < joystick_count:
        device_index = 0

    # Prima di aprire un nuovo joystick si chiude in modo sicuro quello eventualmente attivo,
    # così da evitare riferimenti simultanei o stati incoerenti.
    _disconnect_current_joystick()

    # Crea e inizializza l'istanza pygame associata al dispositivo selezionato.
    joystick = pygame.joystick.Joystick(device_index)
    joystick.init()
    JOYSTICK = joystick

    # Log informativi utili in fase di esecuzione e di diagnostica.
    LOGGER.info("Joystick collegato: %s", JOYSTICK.get_name())
    LOGGER.info(
        "Assi: %s | Pulsanti: %s",
        JOYSTICK.get_numaxes(),
        JOYSTICK.get_numbuttons(),
    )


def init_joystick():
    """Inizializza pygame, il sottosistema joystick e la finestra di controllo."""
    # Inizializzazione generale della libreria pygame e del sottosistema joystick.
    pygame.init()
    pygame.joystick.init()

    # Creazione della finestra di controllo secondo i parametri definiti nella configurazione.
    screen = pygame.display.set_mode(APP_CONFIG.window_size)
    pygame.display.set_caption(APP_CONFIG.window_title)

    # Tenta il collegamento del primo joystick disponibile.
    _connect_first_joystick()

    # Restituisce il riferimento alla superficie della finestra,
    # che potrà essere utilizzata dal chiamante per eventuali operazioni grafiche.
    return screen


def read_events():
    """
    Legge gli eventi pygame e restituisce le azioni principali richieste:
    quit, takeoff, land e stato focus finestra.
    """
    global JOYSTICK

    # Dizionario di stato che sintetizza le principali azioni di alto livello
    # richieste dall'utente o dallo stato della finestra.
    actions = {
        "quit": False,
        "takeoff": False,
        "land": False,
        "focused": _get_window_focus(),
    }

    # Identificativo del joystick attualmente attivo; serve per filtrare gli eventi
    # provenienti da dispositivi diversi da quello gestito dal programma.
    active_instance_id = _get_joystick_instance_id(JOYSTICK)

    # Recupera dalla configurazione la corrispondenza tra pulsanti fisici
    # e azioni logiche dell'applicazione.
    mapping = APP_CONFIG.joystick

    # Analizza tutti gli eventi accumulati nella coda di pygame.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Chiusura della finestra da parte dell'utente.
            actions["quit"] = True

        elif event.type == pygame.KEYDOWN:
            # Gestione di input da tastiera: il tasto ESC viene interpretato
            # come richiesta di terminazione del programma.
            if event.key == pygame.K_ESCAPE:
                actions["quit"] = True

        elif event.type == pygame.JOYBUTTONDOWN:
            event_instance_id = getattr(event, "instance_id", None)

            # Se arrivano eventi da un joystick diverso da quello attivo, li ignoriamo
            # per evitare che input estranei generino comandi non desiderati.
            if active_instance_id is not None and event_instance_id not in (None, active_instance_id):
                continue

            # Mappa i pulsanti del controller sulle corrispondenti azioni logiche.
            if event.button == mapping.button_takeoff:
                actions["takeoff"] = True
            elif event.button == mapping.button_land:
                actions["land"] = True
            elif event.button == mapping.button_quit:
                actions["quit"] = True

        elif event.type == pygame.JOYDEVICEADDED:
            # Se non c'è già un joystick attivo, proviamo a collegarne uno
            # usando, se disponibile, l'indice del dispositivo segnalato dall'evento.
            if JOYSTICK is None:
                try:
                    _connect_first_joystick(getattr(event, "device_index", 0))
                    active_instance_id = _get_joystick_instance_id(JOYSTICK)
                except RuntimeError:
                    # In caso di impossibilità di connessione si evita di interrompere
                    # il ciclo eventi; il programma potrà decidere successivamente come reagire.
                    pass

        elif event.type == pygame.JOYDEVICEREMOVED:
            removed_instance_id = getattr(event, "instance_id", None)

            # Se è stato rimosso il joystick attivo, lo chiudiamo e proviamo a ricollegarne uno
            # tra quelli eventualmente ancora disponibili.
            if active_instance_id is None or removed_instance_id == active_instance_id:
                LOGGER.warning("Joystick disconnesso.")
                _disconnect_current_joystick()
                active_instance_id = None

                if pygame.joystick.get_count() > 0:
                    try:
                        _connect_first_joystick()
                        active_instance_id = _get_joystick_instance_id(JOYSTICK)
                    except RuntimeError:
                        # Se esistono dispositivi ma il collegamento fallisce,
                        # si segnala la necessità di uscire.
                        actions["quit"] = True
                else:
                    # Se non è più disponibile alcun joystick, si richiede l'uscita.
                    actions["quit"] = True

        elif event.type == pygame.WINDOWFOCUSLOST:
            # La finestra ha perso il focus: l'informazione viene propagata
            # al chiamante per consentire eventuali blocchi di sicurezza.
            actions["focused"] = False

        elif event.type == pygame.WINDOWFOCUSGAINED:
            # La finestra ha riacquisito il focus.
            actions["focused"] = True

    return actions


def _apply_deadzone(value, deadzone):
    """Azzera piccoli movimenti del joystick."""
    # La deadzone serve a neutralizzare rumore, oscillazioni minime
    # e imprecisioni meccaniche tipiche dei joystick analogici.
    if abs(value) < deadzone:
        return 0.0
    return value


def _axis_to_speed(value, speed):
    """
    Converte il valore normalizzato di un asse joystick in velocità RC intera.
    Lo speed viene limitato a [0, 100].
    """
    # Il parametro speed viene forzato nell'intervallo ammesso,
    # così da garantire coerenza con i limiti del comando RC.
    speed = max(0, min(100, int(speed)))

    # Applica la deadzone definita in configurazione per eliminare piccoli movimenti involontari.
    value = _apply_deadzone(value, APP_CONFIG.joystick.deadzone)

    # Limita il valore dell'asse all'intervallo normalizzato atteso.
    value = max(-1.0, min(1.0, value))

    # Converte il valore continuo in una velocità intera proporzionale.
    return int(value * speed)


def _get_axis_value(index):
    """Legge il valore di un asse, restituendo 0 in caso di errore o joystick assente."""
    # Se non esiste un joystick attivo, il valore dell'asse viene assunto nullo.
    if JOYSTICK is None:
        return 0.0

    try:
        # Verifica preventiva: se l'indice richiesto eccede il numero di assi disponibili,
        # si restituisce 0 per evitare errori di accesso.
        if index >= JOYSTICK.get_numaxes():
            return 0.0
        return JOYSTICK.get_axis(index)
    except (AttributeError, OSError, pygame.error):
        # Eventuali problemi di accesso al dispositivo vengono gestiti in modo robusto
        # restituendo un valore neutro.
        return 0.0


def get_command(speed=50, require_focus=False):
    """
    Restituisce il comando RC corrente letto dal joystick:
    lr, fb, ud, yaw.
    """
    # Aggiorna internamente lo stato degli eventi pygame,
    # garantendo che la lettura degli assi sia coerente con l'input corrente.
    pygame.event.pump()

    # In assenza di joystick si restituisce un comando nullo.
    if JOYSTICK is None:
        return _zero_command()

    # Se richiesto, il comando viene annullato quando la finestra non ha il focus,
    # introducendo una misura di sicurezza operativa.
    if require_focus and not _get_window_focus():
        return _zero_command()

    # Mappatura configurabile tra assi del controller e componenti del comando RC.
    mapping = APP_CONFIG.joystick

    # In pygame gli assi verticali spesso sono invertiti:
    # su = valore negativo, giù = valore positivo.
    # Per questa ragione gli assi forward/backward e up/down vengono negati,
    # così da ottenere una convenzione di comando più intuitiva.
    lr = _axis_to_speed(_get_axis_value(mapping.axis_lr), speed)
    fb = _axis_to_speed(-_get_axis_value(mapping.axis_fb), speed)
    ud = _axis_to_speed(-_get_axis_value(mapping.axis_ud), speed)
    yaw = _axis_to_speed(_get_axis_value(mapping.axis_yaw), speed)

    # Restituisce il comando completo nei quattro gradi di libertà previsti.
    return {
        "lr": lr,
        "fb": fb,
        "ud": ud,
        "yaw": yaw,
    }