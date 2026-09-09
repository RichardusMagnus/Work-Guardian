import logging
import os

# pygame fornisce il supporto all'acquisizione degli eventi provenienti
# dal controller e alla gestione della finestra applicativa associata.
# In questo modulo viene usato sia per leggere gli input del joystick,
# sia per intercettare eventi di finestra o da tastiera.
import pygame

# Import della configurazione centralizzata dell'applicazione.
# In particolare, da APP_CONFIG vengono recuperati:
# - la dimensione e il titolo della finestra pygame;
# - la mappatura tra pulsanti/assi fisici e comandi logici del drone;
# - i parametri di elaborazione dell'input analogico, come la deadzone.
from app_config import APP_CONFIG

# Logger di modulo utilizzato per tracciare eventi rilevanti, come
# connessioni/disconnessioni del joystick ed eventuali errori di rilascio risorse.
# L'uso di un logger dedicato consente di integrare questo modulo nel sistema
# di logging generale dell'applicazione senza ricorrere a stampe non strutturate.
LOGGER = logging.getLogger(__name__)

# Riferimento globale al joystick attualmente connesso e inizializzato.
# Il valore None indica che non è presente alcun controller attivo.
# Questa variabile rappresenta quindi lo stato interno principale del modulo.
JOYSTICK = None


def _zero_command():
    # Restituisce un comando nullo su tutti gli assi di controllo.
    # Questa funzione costituisce un valore di fallback sicuro da usare
    # quando il joystick non è disponibile oppure quando la lettura
    # dell'input non può essere effettuata correttamente.
    #
    # Le chiavi del dizionario rappresentano le quattro componenti fondamentali
    # del comando di movimento del drone:
    # - lr: movimento laterale sinistra/destra;
    # - fb: movimento avanti/indietro;
    # - ud: movimento verticale su/giù;
    # - yaw: rotazione attorno all'asse verticale.
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
        # Questa operazione rende esplicita la chiusura della risorsa hardware
        # prima che venga eventualmente inizializzato un nuovo joystick.
        JOYSTICK.quit()
    except Exception:
        # L'eccezione viene registrata nel log per consentire diagnosi successive,
        # senza però interrompere il flusso del programma. La scelta è coerente
        # con una gestione robusta della disconnessione delle periferiche.
        LOGGER.exception("Errore durante la chiusura del joystick.")
    finally:
        # In ogni caso il riferimento globale viene invalidato,
        # così da mantenere coerente lo stato interno del modulo.
        JOYSTICK = None


def close_joystick():
    """Rilascia in modo sicuro il joystick corrente e il sottosistema joystick di pygame."""
    # Prima viene rilasciato l'eventuale joystick attivo.
    # Questa chiamata gestisce sia la presenza sia l'assenza di un controller.
    _disconnect_current_joystick()

    try:
        # Se il sottosistema joystick di pygame è inizializzato, viene chiuso.
        # Ciò consente di liberare correttamente le risorse native associate.
        if pygame.joystick.get_init():
            pygame.joystick.quit()
    except pygame.error:
        # Eventuali errori di pygame in fase di chiusura vengono ignorati
        # per privilegiare una terminazione robusta del programma.
        # La chiusura del programma non deve infatti fallire a causa
        # di un problema secondario nel rilascio del sottosistema joystick.
        pass


def _enable_background_joystick_events():
    """
    Consente a SDL/pygame di continuare a ricevere eventi del joystick anche
    quando la finestra pygame non ha il focus, ad esempio se l'utente clicca
    sulla finestra OpenCV del video.
    """
    # La variabile d'ambiente viene impostata solo se non già definita.
    # In questo modo si preserva un'eventuale configurazione esterna esplicita.
    # La scelta è particolarmente utile in applicazioni con più finestre grafiche,
    # nelle quali la finestra pygame potrebbe non essere sempre attiva.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")


def _get_joystick_instance_id(joystick):
    # Restituisce l'identificativo univoco dell'istanza di joystick.
    # Tale identificativo è utile per distinguere gli eventi relativi
    # al controller attualmente attivo da quelli provenienti da altri dispositivi.
    # L'instance_id è l'identificatore più adatto nelle versioni moderne di pygame/SDL.
    if joystick is None:
        return None

    try:
        return joystick.get_instance_id()
    except (AttributeError, OSError, pygame.error):
        # Alcune versioni/API potrebbero non supportare tale metodo,
        # oppure il dispositivo potrebbe trovarsi in uno stato non valido.
        # In questi casi si restituisce None e il chiamante userà meccanismi
        # alternativi di compatibilità.
        return None


def _get_joystick_legacy_id(joystick):
    # Restituisce, se disponibile, il vecchio identificativo numerico del joystick.
    # Questo valore corrisponde al "device index" storico di pygame ed è usato
    # come fallback di compatibilità con versioni/API che non espongono ancora
    # in modo coerente l'attributo instance_id sugli eventi.
    if joystick is None:
        return None

    try:
        return joystick.get_id()
    except (AttributeError, OSError, pygame.error):
        # Se il metodo non è disponibile o il joystick non è più valido,
        # il chiamante riceve None e può evitare confronti non affidabili.
        return None


def _get_event_controller_ids(event):
    # Estrae tutti gli identificativi potenzialmente presenti nell'evento.
    # A seconda della versione di pygame/SDL e del backend, un evento joystick
    # può esporre campi diversi, ad esempio:
    # - instance_id (API moderna, raccomandata);
    # - joy        (API legacy documentata come deprecata);
    # - which      (fallback ulteriore osservabile in alcuni contesti).
    #
    # La funzione restituisce una tupla senza duplicati, così da rendere
    # più semplice il confronto con gli identificativi del joystick attivo.
    ids = []
    for attr_name in ("instance_id", "joy", "which"):
        attr_value = getattr(event, attr_name, None)
        if attr_value is None:
            continue
        if attr_value not in ids:
            ids.append(attr_value)
    return tuple(ids)


def _event_matches_active_joystick(event, active_instance_id, active_legacy_id):
    # Determina se l'evento deve essere attribuito al joystick attualmente gestito.
    # La logica è volutamente permissiva in presenza di un solo controller,
    # così da evitare falsi negativi dovuti a differenze tra versioni di pygame.
    event_ids = _get_event_controller_ids(event)

    # Se l'evento non fornisce alcun identificativo, viene accettato.
    # Questa scelta evita di scartare eventi validi prodotti da backend
    # che non valorizzano i campi identificativi.
    if not event_ids:
        return True

    # Corrispondenza diretta con l'identificativo moderno dell'istanza.
    # Questo è il caso preferibile perché l'instance_id è pensato per distinguere
    # in modo affidabile le periferiche anche dopo collegamenti e scollegamenti.
    if active_instance_id is not None and active_instance_id in event_ids:
        return True

    # Corrispondenza con l'identificativo legacy del dispositivo.
    # Questo controllo mantiene compatibilità con versioni o configurazioni
    # in cui gli eventi espongono ancora il vecchio identificatore numerico.
    if active_legacy_id is not None and active_legacy_id in event_ids:
        return True

    # In presenza di un solo joystick collegato, si accetta comunque l'evento.
    # Questo fallback risolve casi pratici in cui il tasto venga effettivamente
    # premuto ma pygame riporti un identificativo differente tra oggetto e evento.
    try:
        if pygame.joystick.get_count() <= 1:
            LOGGER.debug(
                "Evento joystick accettato in fallback compatibilità | event_ids=%s | instance_id=%s | legacy_id=%s",
                event_ids,
                active_instance_id,
                active_legacy_id,
            )
            return True
    except pygame.error:
        # Se anche la consultazione del numero di joystick fallisce,
        # si rinuncia al fallback e l'evento verrà considerato non corrispondente.
        pass

    return False


def _connect_first_joystick(device_index=0):
    # Collega e inizializza il joystick individuato da device_index.
    # Se l'indice fornito non è valido, viene selezionato il primo dispositivo disponibile.
    # La funzione aggiorna la variabile globale JOYSTICK, rendendo disponibile
    # il controller al resto del modulo.
    global JOYSTICK

    # Numero complessivo di joystick attualmente rilevati da pygame.
    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        # In assenza di dispositivi disponibili, viene sollevata un'eccezione esplicativa.
        # Il messaggio è orientato all'utente finale, poiché segnala chiaramente
        # la condizione operativa necessaria per usare il controller.
        raise RuntimeError(
            "Nessun joystick rilevato. Collega il controller PS4 prima di avviare il programma."
        )

    # Validazione dell'indice del dispositivo richiesto.
    # Se l'indice non appartiene all'intervallo valido, si forza l'uso del primo joystick.
    # Questo evita errori di inizializzazione causati da indici obsoleti o non disponibili.
    if not 0 <= device_index < joystick_count:
        device_index = 0

    # Prima di collegare un nuovo controller si rilascia quello eventualmente attivo,
    # così da mantenere un solo joystick gestito dal programma.
    _disconnect_current_joystick()

    # Creazione dell'oggetto joystick e inizializzazione del dispositivo.
    # L'oggetto inizializzato diventa il riferimento globale usato per leggere
    # assi analogici, pulsanti e informazioni descrittive.
    joystick = pygame.joystick.Joystick(device_index)
    joystick.init()
    JOYSTICK = joystick

    # Registrazione di alcune informazioni descrittive utili in fase di avvio e debug.
    # In particolare, numero di assi e pulsanti aiutano a verificare che la mappatura
    # presente in APP_CONFIG sia coerente con il controller effettivamente rilevato.
    LOGGER.info("Joystick collegato: %s", JOYSTICK.get_name())
    LOGGER.info(
        "Assi: %s | Pulsanti: %s | instance_id=%s | legacy_id=%s",
        JOYSTICK.get_numaxes(),
        JOYSTICK.get_numbuttons(),
        _get_joystick_instance_id(JOYSTICK),
        _get_joystick_legacy_id(JOYSTICK),
    )


def init_joystick():
    # Abilita la ricezione degli eventi del joystick anche in background,
    # quindi inizializza pygame e il relativo sottosistema joystick.
    # Questa funzione rappresenta il punto di inizializzazione del modulo
    # e deve essere chiamata prima di leggere eventi o comandi analogici.
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
    # La restituzione della superficie permette al chiamante di integrarla
    # nel ciclo principale dell'applicazione.
    return screen


def print_joystick_help():
    # Stampa a terminale una legenda dei comandi del controller.
    # Le etichette sono ricavate dalla configurazione, così da mantenere coerenza
    # tra documentazione a schermo e mappatura effettivamente usata dal programma.
    mapping = APP_CONFIG.joystick

    # La formattazione con larghezza fissa migliora la leggibilità della legenda,
    # allineando descrizioni e comandi in colonne ordinate.
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
    #
    # La funzione separa quindi il livello fisico degli eventi pygame
    # dal livello applicativo, nel quale interessano solo comandi come:
    # uscita, decollo, atterraggio e attivazione/disattivazione del rilevamento.
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

    # Identificativi del joystick attualmente attivo.
    # Si mantengono sia quello moderno sia quello legacy per compatibilità.
    active_instance_id = _get_joystick_instance_id(JOYSTICK)
    active_legacy_id = _get_joystick_legacy_id(JOYSTICK)
    mapping = APP_CONFIG.joystick

    # Recupero di tutti gli eventi attualmente presenti nella coda di pygame.
    # Se la lettura fallisce, per sicurezza si richiede l'uscita dal programma.
    # Questa scelta evita che il ciclo principale prosegua in uno stato grafico
    # o di input non più affidabile.
    try:
        events = pygame.event.get()
    except pygame.error:
        actions["quit"] = True
        return actions

    # Analisi sequenziale di tutti gli eventi presenti nella coda di pygame.
    # Ogni evento viene classificato in base al suo tipo e tradotto,
    # quando necessario, in un'azione applicativa.
    for event in events:
        if event.type == pygame.QUIT:
            # Chiusura della finestra grafica tramite interfaccia del sistema operativo.
            actions["quit"] = True

        elif event.type == pygame.KEYDOWN:
            # È prevista anche una scorciatoia da tastiera per uscire dal programma.
            # Questo consente la chiusura controllata anche in assenza del joystick.
            if event.key == pygame.K_ESCAPE:
                actions["quit"] = True

        elif event.type == pygame.JOYBUTTONDOWN:
            # Per gli eventi del joystick si verifica che provengano
            # dal controller attualmente considerato attivo.
            # Questa verifica è importante in presenza di più dispositivi
            # o dopo eventi di connessione/disconnessione.
            if not _event_matches_active_joystick(event, active_instance_id, active_legacy_id):
                continue

            LOGGER.debug(
                "JOYBUTTONDOWN ricevuto | button=%s | event_ids=%s",
                getattr(event, "button", None),
                _get_event_controller_ids(event),
            )

            # Traduzione dei pulsanti fisici in azioni logiche applicative.
            # Questa mediazione separa il livello hardware dal livello logico:
            # modificando APP_CONFIG è possibile cambiare mappatura senza alterare
            # il codice che consuma il dizionario actions.
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
            # Questa gestione permette l'inserimento del controller anche
            # dopo l'avvio dell'applicazione.
            if JOYSTICK is None:
                try:
                    device_index = getattr(event, "device_index", getattr(event, "which", 0))
                    _connect_first_joystick(device_index)
                    active_instance_id = _get_joystick_instance_id(JOYSTICK)
                    active_legacy_id = _get_joystick_legacy_id(JOYSTICK)
                except RuntimeError:
                    # In caso di problemi di connessione si prosegue senza interrompere il programma.
                    # Il modulo resterà in attesa di ulteriori eventi o di una successiva lettura valida.
                    pass

        elif event.type == pygame.JOYDEVICEREMOVED:
            # Gestione della disconnessione del joystick.
            # Gli identificativi contenuti nell'evento permettono di capire
            # se il dispositivo rimosso coincide con quello attualmente usato.
            removed_ids = _get_event_controller_ids(event)

            # Se il dispositivo rimosso coincide con quello attivo,
            # oppure se l'identificativo dell'attivo non è disponibile,
            # si procede al rilascio del controller corrente.
            if (
                active_instance_id is None
                or active_instance_id in removed_ids
                or active_legacy_id in removed_ids
            ):
                LOGGER.warning("Joystick disconnesso.")
                _disconnect_current_joystick()
                active_instance_id = None
                active_legacy_id = None

                # Se sono presenti altri joystick, si tenta una riconnessione automatica.
                # In questo modo il programma può continuare a funzionare qualora
                # sia disponibile un controller alternativo.
                if pygame.joystick.get_count() > 0:
                    try:
                        _connect_first_joystick()
                        active_instance_id = _get_joystick_instance_id(JOYSTICK)
                        active_legacy_id = _get_joystick_legacy_id(JOYSTICK)
                    except RuntimeError:
                        # Se la riconnessione fallisce, il programma viene indirizzato verso l'uscita.
                        # Tale scelta evita di continuare in assenza di un input di controllo affidabile.
                        actions["quit"] = True
                else:
                    # In assenza di controller disponibili, si richiede la chiusura.
                    actions["quit"] = True

    # Output della funzione: dizionario delle azioni attivate nell'iterazione corrente.
    # Le azioni non attivate rimangono a False, consentendo al ciclo principale
    # di controllare in modo semplice gli eventi discreti dell'interfaccia.
    return actions


def _apply_deadzone(value, deadzone):
    # Applica una zona morta all'input analogico.
    # Valori di piccola ampiezza vengono forzati a zero per eliminare
    # oscillazioni spurie dovute a rumore o imperfezioni meccaniche del joystick.
    #
    # Questa operazione è essenziale nei controlli analogici, poiché gli assi
    # raramente ritornano esattamente a zero anche quando la leva è rilasciata.
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
    #
    # La saturazione protegge il sistema da valori anomali restituiti dal backend
    # o da configurazioni hardware non perfettamente standard.
    speed = max(0, min(100, int(speed)))
    value = _apply_deadzone(value, APP_CONFIG.joystick.deadzone)
    value = max(-1.0, min(1.0, value))
    return int(value * speed)


def _get_axis_value(index):
    # Restituisce il valore corrente dell'asse indicato.
    # In caso di joystick assente, indice non valido o errore di lettura,
    # viene restituito 0.0 come valore neutro.
    #
    # Il valore neutro è coerente con un comando di assenza di movimento
    # e rende la funzione sicura anche in condizioni hardware non stabili.
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
        # Il chiamante non deve gestire direttamente questi errori: riceverà
        # semplicemente un valore neutro.
        return 0.0


def get_command(speed=50):
    # Produce il comando di movimento corrente leggendo lo stato degli assi analogici
    # del joystick e convertendolo nel formato atteso dal resto dell'applicazione.
    #
    # Il parametro speed stabilisce il modulo massimo della velocità restituita
    # per ciascun asse, dopo la conversione da valore analogico a intero.
    try:
        # Aggiorna internamente lo stato degli eventi di pygame.
        # Questa operazione è necessaria per mantenere aggiornati i valori degli assi.
        pygame.event.pump()
    except pygame.error:
        # In caso di errore del sottosistema eventi, viene restituito un comando nullo.
        # Questa è una scelta prudenziale per evitare invii di comandi non desiderati.
        return _zero_command()

    # Se nessun joystick è attualmente disponibile, si restituisce un comando nullo.
    # Il resto del programma può quindi invocare get_command anche quando
    # il controller non è ancora collegato o è stato disconnesso.
    if JOYSTICK is None:
        return _zero_command()

    mapping = APP_CONFIG.joystick

    # Lettura e conversione degli assi secondo la mappatura configurata.
    # Per alcuni assi viene applicato il segno meno per uniformare
    # la convenzione fisica del controller con quella del sistema di comando.
    # Ad esempio, in molti controller spingere in avanti una leva produce
    # un valore negativo, mentre per il comando del drone può essere preferibile
    # rappresentare l'avanzamento con un valore positivo.
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