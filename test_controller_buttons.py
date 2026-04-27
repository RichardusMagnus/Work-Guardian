# Modulo per l'interazione con il sistema operativo.
# In questo script viene usato per impostare una variabile d'ambiente SDL
# prima dell'inizializzazione di pygame.
import os

# Modulo per l'interazione con l'interprete Python.
# Viene usato alla fine del programma per terminare esplicitamente lo script.
import sys

# Modulo per la gestione del tempo.
# In questo caso consente di limitare la frequenza di stampa dello stato degli assi.
import time

# Libreria utilizzata per acquisire eventi da tastiera, finestra e controller/joystick.
import pygame


# Valore soglia sotto il quale un movimento analogico viene considerato nullo.
# La deadzone è utile per filtrare piccole oscillazioni involontarie degli stick.
DEADZONE = 0.15


def axis_value(v, deadzone=DEADZONE):
    # Restituisce zero quando il valore dell'asse è sufficientemente vicino
    # alla posizione neutra; in caso contrario conserva il valore originale.
    return 0.0 if abs(v) < deadzone else v


def _event_ids(event):
    # Costruisce una rappresentazione testuale degli identificativi associati
    # a un evento pygame. A seconda della versione di pygame e del backend SDL,
    # gli eventi joystick possono esporre attributi diversi.
    ids = []

    # Si controllano più possibili nomi di attributi per rendere la diagnosi
    # compatibile con diverse versioni della libreria.
    for attr_name in ("instance_id", "joy", "which"):
        attr_value = getattr(event, attr_name, None)

        # L'identificativo viene registrato solo se esiste e non è già stato inserito.
        if attr_value is not None and attr_value not in ids:
            ids.append(f"{attr_name}={attr_value}")

    # Se non sono disponibili identificativi, viene restituita una stringa esplicita.
    return ", ".join(ids) if ids else "nessun-id"


def _connect_joystick(device_index=0):
    # Se non sono presenti controller collegati, la funzione segnala l'assenza
    # restituendo una terna di valori nulli.
    if pygame.joystick.get_count() == 0:
        return None, None, None

    # Se l'indice richiesto non è valido, si seleziona per sicurezza
    # il primo controller disponibile.
    if not 0 <= device_index < pygame.joystick.get_count():
        device_index = 0

    # Creazione dell'oggetto joystick associato al dispositivo selezionato.
    joystick = pygame.joystick.Joystick(device_index)

    # Inizializzazione esplicita del controller, necessaria prima di leggerne
    # nome, assi, pulsanti e identificativi.
    joystick.init()

    # L'instance_id è l'identificativo moderno usato da pygame/SDL
    # per distinguere un dispositivo fisico durante la sessione corrente.
    try:
        instance_id = joystick.get_instance_id()
    except Exception:
        instance_id = None

    # Il legacy_id è un identificativo tradizionale, mantenuto per compatibilità
    # con versioni o backend che usano ancora get_id().
    try:
        legacy_id = joystick.get_id()
    except Exception:
        legacy_id = None

    # La funzione restituisce sia l'oggetto joystick sia gli identificativi utili
    # per confrontare gli eventi ricevuti con il controller attivo.
    return joystick, instance_id, legacy_id


def _disconnect_joystick(joystick):
    # Se non esiste alcun joystick attivo, non è richiesta alcuna operazione.
    if joystick is None:
        return

    # La disconnessione viene protetta da try/except perché il dispositivo
    # potrebbe essere già stato rimosso fisicamente o reso non disponibile.
    try:
        joystick.quit()
    except Exception:
        pass


def _print_controller_info(joystick, instance_id, legacy_id):
    # In assenza di controller, la funzione fornisce un messaggio informativo
    # e termina senza tentare l'accesso ai metodi del joystick.
    if joystick is None:
        print("Nessun controller attualmente disponibile.")
        return

    # Stampa delle informazioni principali del controller rilevato.
    # Questi dati sono utili per verificare la configurazione del dispositivo
    # e per interpretare correttamente gli eventi generati da pygame.
    print("=" * 60)
    print("CONTROLLER RILEVATO")
    print(f"Nome       : {joystick.get_name()}")
    print(f"Assi       : {joystick.get_numaxes()}")
    print(f"Pulsanti   : {joystick.get_numbuttons()}")
    print(f"Hat/D-pad  : {joystick.get_numhats()}")
    print(f"instance_id: {instance_id}")
    print(f"legacy_id  : {legacy_id}")
    print("=" * 60)
    print("Premi i tasti per vedere il loro indice.")
    print("Muovi gli stick per vedere gli assi.")
    print("Chiudi la finestra oppure premi ESC per uscire.")
    print("Lo script stampa anche gli identificativi evento per diagnosticare mismatch pygame.")
    print("=" * 60)


def _event_matches_active_joystick(event, active_instance_id, active_legacy_id):
    # Estrae dall'evento gli identificativi disponibili, poiché eventi diversi
    # possono esporre attributi differenti per riferirsi al controller sorgente.
    event_ids = []

    for attr_name in ("instance_id", "joy", "which"):
        attr_value = getattr(event, attr_name, None)

        # Si evita l'inserimento di duplicati per semplificare il confronto.
        if attr_value is not None and attr_value not in event_ids:
            event_ids.append(attr_value)

    # Se l'evento non contiene identificativi, viene accettato.
    # Questa scelta consente di mantenere il programma robusto anche con backend
    # che non popolano tali campi.
    if not event_ids:
        return True

    # Confronto con l'identificativo moderno del controller attivo.
    if active_instance_id is not None and active_instance_id in event_ids:
        return True

    # Confronto con l'identificativo legacy del controller attivo.
    if active_legacy_id is not None and active_legacy_id in event_ids:
        return True

    # Se è presente un solo controller, l'evento viene comunque considerato valido,
    # perché non vi sono ambiguità pratiche sulla sorgente.
    return pygame.joystick.get_count() <= 1


def main():
    # Permette la ricezione degli eventi joystick anche quando la finestra perde il focus.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

    # Inizializzazione generale di pygame e del sottosistema joystick.
    pygame.init()
    pygame.joystick.init()

    # Creazione di una finestra minima: pygame richiede una finestra attiva
    # per gestire correttamente il ciclo degli eventi.
    screen = pygame.display.set_mode((760, 220))
    pygame.display.set_caption("Test controller - pulsanti e assi")

    # Controllo preliminare sulla presenza di almeno un controller collegato.
    # In caso contrario il programma termina in modo ordinato.
    if pygame.joystick.get_count() == 0:
        print("Nessun controller rilevato.")
        print("Collega il controller e rilancia lo script.")
        pygame.quit()
        return

    # Connessione al primo controller disponibile e stampa delle sue informazioni.
    joystick, instance_id, legacy_id = _connect_joystick(0)
    _print_controller_info(joystick, instance_id, legacy_id)

    # Oggetto Clock usato per limitare il ciclo grafico a una frequenza stabile.
    clock = pygame.time.Clock()

    # Timestamp dell'ultima stampa degli assi, usato per evitare un output eccessivo.
    last_axis_print = 0.0

    # Variabile di controllo del ciclo principale dell'applicazione.
    running = True

    while running:
        # Gestione degli eventi accumulati da pygame: chiusura finestra,
        # pressione di tasti, eventi del joystick e connessioni/disconnessioni.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Il tasto ESC consente di uscire dal programma senza chiudere
                # manualmente la finestra.
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.JOYBUTTONDOWN:
                # La pressione di un pulsante viene stampata solo se l'evento
                # appartiene al controller attualmente monitorato.
                if _event_matches_active_joystick(event, instance_id, legacy_id):
                    print(f"[BUTTON DOWN] indice={event.button} | {_event_ids(event)}")

            elif event.type == pygame.JOYBUTTONUP:
                # Il rilascio di un pulsante viene trattato separatamente
                # per distinguere le due fasi dell'interazione.
                if _event_matches_active_joystick(event, instance_id, legacy_id):
                    print(f"[BUTTON UP  ] indice={event.button} | {_event_ids(event)}")

            elif event.type == pygame.JOYHATMOTION:
                # Gli eventi HAT rappresentano in genere il D-pad.
                # Il valore indica la direzione corrente premuta.
                if _event_matches_active_joystick(event, instance_id, legacy_id):
                    print(f"[HAT] indice={event.hat} valore={event.value} | {_event_ids(event)}")

            elif event.type == pygame.JOYDEVICEADDED:
                # Evento generato quando un nuovo controller viene collegato
                # durante l'esecuzione dello script.
                print(
                    f"[INFO] Controller collegato: device_index={getattr(event, 'device_index', None)} | {_event_ids(event)}"
                )

                # Se non c'è già un controller attivo, il nuovo dispositivo
                # viene inizializzato e diventa il controller monitorato.
                if joystick is None:
                    device_index = getattr(event, "device_index", getattr(event, "which", 0))
                    joystick, instance_id, legacy_id = _connect_joystick(device_index)
                    _print_controller_info(joystick, instance_id, legacy_id)

            elif event.type == pygame.JOYDEVICEREMOVED:
                # Evento generato quando un controller viene scollegato.
                print(f"[INFO] Controller scollegato: {_event_ids(event)}")

                # Se il controller scollegato corrisponde a quello attivo,
                # si liberano le risorse e si azzerano gli identificativi.
                if joystick is not None and _event_matches_active_joystick(event, instance_id, legacy_id):
                    _disconnect_joystick(joystick)
                    joystick = None
                    instance_id = None
                    legacy_id = None

                    # Se dopo la disconnessione esistono altri controller,
                    # il programma tenta di agganciarsi automaticamente al primo disponibile.
                    if pygame.joystick.get_count() > 0:
                        joystick, instance_id, legacy_id = _connect_joystick(0)
                        _print_controller_info(joystick, instance_id, legacy_id)

        # Stampa periodica dello stato degli assi.
        # L'intervallo di 0.2 secondi rende l'output leggibile e riduce
        # la quantità di messaggi prodotti durante l'esecuzione.
        now = time.time()
        if now - last_axis_print > 0.2 and joystick is not None:
            try:
                axis_values = []

                # Lettura di tutti gli assi disponibili sul controller.
                # Ogni valore viene filtrato tramite deadzone per eliminare
                # piccole variazioni dovute al rumore degli stick analogici.
                for i in range(joystick.get_numaxes()):
                    v = axis_value(joystick.get_axis(i))
                    axis_values.append(f"asse {i}: {v:+.3f}")

                print("[AXES] " + " | ".join(axis_values))
                last_axis_print = now

            except Exception:
                # La lettura degli assi può fallire se il controller viene scollegato
                # tra un evento e il successivo. In tal caso lo script aggiorna
                # il proprio stato interno senza interrompersi in modo anomalo.
                print("[AVVISO] Lettura assi fallita: controller non più disponibile.")
                _disconnect_joystick(joystick)
                joystick = None
                instance_id = None
                legacy_id = None

        # Riempimento semplice della finestra.
        # La finestra non mostra contenuti complessi: serve principalmente
        # come supporto al ciclo eventi di pygame.
        screen.fill((30, 30, 30))
        pygame.display.flip()

        # Limitazione del ciclo principale a circa 60 iterazioni al secondo.
        clock.tick(60)

    # Fase conclusiva: rilascio ordinato delle risorse del controller,
    # del sottosistema joystick e dell'ambiente pygame.
    _disconnect_joystick(joystick)
    pygame.joystick.quit()
    pygame.quit()
    sys.exit()


# Punto di ingresso standard: consente di eseguire main() solo quando
# il file viene lanciato direttamente, evitando esecuzioni automatiche
# in caso di importazione come modulo.
if __name__ == "__main__":
    main()