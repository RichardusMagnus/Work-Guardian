import logging

# Inizializzazione del logger associato al modulo corrente.
# Tale oggetto viene utilizzato per registrare messaggi informativi,
# avvisi ed errori relativi alla gestione del drone reale.
LOGGER = logging.getLogger(__name__)


class RealTelloController:
    """Controller per il drone DJI Tello reale."""

    def __init__(self):
        # L'import della libreria djitellopy viene eseguito all'interno del costruttore
        # per rendere esplicita la dipendenza solo nel caso in cui si scelga di
        # istanziare effettivamente un controller per hardware reale.
        # In questo modo, moduli che importano questa classe ma non la utilizzano
        # non richiedono necessariamente la presenza immediata della libreria.
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise ImportError(
                "Per usare il drone reale devi installare djitellopy: pip install djitellopy"
            ) from exc

        # Creazione dell'oggetto che incapsula la comunicazione con il drone Tello.
        # Tutte le operazioni operative sul drone fisico vengono demandate a questa istanza.
        self.tello = Tello()

        # Riferimento al gestore del flusso video restituito dalla libreria.
        # Rimane None finché lo stream video non viene avviato correttamente.
        self.frame_reader = None

        # Stato logico locale del controller.
        # Questi attributi non sostituiscono lo stato reale del drone, ma permettono
        # di effettuare controlli preliminari prima di inviare comandi non leciti
        # rispetto al flusso operativo previsto.
        self.is_connected = False
        self.is_flying = False

    @staticmethod
    def _clamp_rc_value(value: int) -> int:
        # Converte il valore ricevuto in intero e lo limita all'intervallo ammesso
        # dai comandi RC del drone, convenzionalmente compreso tra -100 e 100.
        # Questo metodo evita l'invio di valori fuori specifica ai comandi di controllo.
        value = int(value)
        return max(-100, min(100, value))

    def connect(self):
        # Stabilisce la connessione con il drone, a meno che essa non risulti
        # già attiva secondo lo stato interno del controller.
        if self.is_connected:
            LOGGER.info("REAL | Il drone è già connesso.")
            return

        try:
            self.tello.connect()
        except Exception:
            # La connessione rappresenta un prerequisito fondamentale per tutte
            # le operazioni successive; per questo motivo l'errore viene registrato
            # e l'eccezione viene rilanciata al chiamante.
            LOGGER.exception("REAL | Errore durante la connessione al Tello.")
            raise

        # Aggiornamento dello stato locale in seguito al successo della connessione.
        self.is_connected = True

        # Lettura iniziale della batteria come verifica supplementare del corretto
        # scambio di dati con il drone. Tale operazione è utile a confermare che
        # la comunicazione non sia soltanto formalmente aperta, ma anche funzionante.
        try:
            self.tello.get_battery()
        except Exception:
            # Un eventuale errore in questa fase non invalida la connessione già
            # stabilita, ma viene comunque registrato per finalità diagnostiche.
            LOGGER.exception("REAL | Errore durante la lettura iniziale della batteria.")

        LOGGER.info("REAL | Connesso al Tello.")

    def takeoff(self):
        # Esegue il decollo del drone solo se il controller risulta connesso
        # e il drone non è già considerato in volo.
        # Il metodo restituisce True esclusivamente quando il decollo viene eseguito.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile decollare: drone non connesso.")
            return False

        if self.is_flying:
            LOGGER.info("REAL | Il drone è già in volo.")
            return False

        try:
            self.tello.takeoff()
        except Exception:
            # In caso di errore il metodo registra l'eccezione e la propaga,
            # poiché il fallimento del decollo è un evento operativo rilevante.
            LOGGER.exception("REAL | Errore durante il decollo.")
            raise

        # Aggiornamento dello stato locale a seguito del successo del comando.
        self.is_flying = True
        LOGGER.info("REAL | Decollo eseguito.")
        return True

    def land(self):
        # Esegue l'atterraggio del drone solo se esso risulta connesso
        # e attualmente in volo secondo lo stato mantenuto dal controller.
        # Il metodo restituisce True solo quando il comando viene eseguito con successo.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile atterrare: drone non connesso.")
            return False

        if not self.is_flying:
            LOGGER.info("REAL | Il drone è già a terra.")
            return False

        try:
            self.tello.land()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'atterraggio.")
            raise

        # Allineamento dello stato interno dopo l'atterraggio.
        self.is_flying = False
        LOGGER.info("REAL | Atterraggio eseguito.")
        return True

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        # Invia al drone i comandi RC relativi ai quattro assi principali:
        # - lr  : movimento laterale sinistra-destra;
        # - fb  : movimento avanti-indietro;
        # - ud  : movimento verticale alto-basso;
        # - yaw : rotazione attorno all'asse verticale.
        #
        # I valori vengono saturati mediante _clamp_rc_value per garantire
        # il rispetto dei limiti ammessi dall'interfaccia di controllo.
        #
        # Il comando viene inviato solo se il drone è sia connesso sia in volo.
        # In caso contrario il metodo restituisce False senza effettuare alcuna azione.
        if not self.is_connected or not self.is_flying:
            return False

        try:
            self.tello.send_rc_control(
                self._clamp_rc_value(lr),
                self._clamp_rc_value(fb),
                self._clamp_rc_value(ud),
                self._clamp_rc_value(yaw),
            )
        except Exception:
            LOGGER.exception("REAL | Errore durante l'invio dei comandi RC.")
            raise

        return True

    def get_status(self) -> dict:
        # Restituisce un dizionario contenente un riepilogo dello stato del controller.
        # Se possibile, viene inclusa anche la percentuale di batteria letta dal drone.
        battery = None
        if self.is_connected:
            try:
                battery = self.tello.get_battery()
            except Exception:
                # Anche in caso di fallimento della lettura della batteria, il metodo
                # restituisce comunque le altre informazioni disponibili.
                LOGGER.exception("REAL | Errore durante la lettura della batteria.")

        return {
            "mode": "REAL",
            "connected": self.is_connected,
            "flying": self.is_flying,
            "battery": battery,
        }

    def start_video_stream(self):
        # Avvia il flusso video proveniente dal drone.
        # La connessione deve essere già stata stabilita, altrimenti
        # l'operazione non è semanticamente valida.
        if not self.is_connected:
            raise RuntimeError("Connetti prima il drone.")

        # Prima di iniziare un nuovo stream si tenta di chiudere in modo sicuro
        # un eventuale stream precedente, così da evitare la permanenza di reader
        # non più validi o stati incoerenti lato controller.
        self.stop_video_stream()

        # Spegnimento preventivo dello stream lato drone, utile per riallineare
        # lo stato del dispositivo anche nel caso in cui uno stream precedente
        # risultasse ancora attivo ma non più tracciato correttamente dal controller.
        try:
            self.tello.streamoff()
        except Exception:
            pass

        # Variabili di supporto usate per gestire correttamente il rollback
        # in caso di fallimento parziale dell'avvio dello stream.
        frame_reader = None
        stream_started = False

        try:
            # Attivazione dello stream video e acquisizione del relativo frame reader.
            self.tello.streamon()
            stream_started = True
            frame_reader = self.tello.get_frame_read()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'avvio del video stream.")
            if stream_started:
                # Se lo stream è stato acceso ma la procedura non si è completata,
                # si tenta di riportare il drone in uno stato consistente.
                try:
                    self.tello.streamoff()
                except Exception:
                    LOGGER.exception("REAL | Errore durante il rollback dello stream video.")
            raise

        # Verifica di coerenza: dopo l'avvio dello stream, il frame reader deve
        # risultare correttamente inizializzato. In caso contrario si esegue
        # il rollback e si segnala il problema al chiamante.
        if frame_reader is None:
            if stream_started:
                try:
                    self.tello.streamoff()
                except Exception:
                    LOGGER.exception("REAL | Errore durante il rollback dello stream video.")
            raise RuntimeError("Impossibile avviare il reader del video stream.")

        # Memorizzazione del reader nello stato dell'oggetto.
        self.frame_reader = frame_reader
        LOGGER.info("REAL | Video stream avviato.")

    def get_frame(self):
        # Restituisce una copia del frame corrente proveniente dal video stream.
        # Se il flusso non è attivo, il metodo restituisce None.
        if self.frame_reader is None:
            return None

        try:
            frame = self.frame_reader.frame
        except Exception:
            # Eventuali errori di lettura vengono registrati e gestiti restituendo None,
            # così da consentire al chiamante di rilevare l'assenza di un frame valido.
            LOGGER.exception("REAL | Errore durante la lettura del frame video.")
            return None

        if frame is None:
            return None

        # Viene restituita una copia del frame per evitare che modifiche esterne
        # interferiscano con il buffer condiviso gestito dal frame reader.
        return frame.copy()

    def stop_video_stream(self):
        # Arresta in modo sicuro il flusso video e il relativo frame reader.
        # Il metodo è progettato per essere invocato anche in situazioni parzialmente
        # incoerenti, ad esempio quando uno stream è stato acceso ma il reader
        # non è stato inizializzato o viceversa.
        frame_reader = self.frame_reader
        had_reader = frame_reader is not None

        # L'attributo viene azzerato immediatamente per impedire ulteriori accessi
        # a un reader che sta per essere arrestato.
        self.frame_reader = None

        if frame_reader is not None and hasattr(frame_reader, "stop"):
            try:
                frame_reader.stop()
            except Exception:
                LOGGER.exception("REAL | Errore durante lo stop del frame reader.")

        # Se il controller è connesso, si tenta comunque di disattivare lo stream lato drone,
        # così da coprire anche i casi in cui lo stream sia stato attivato correttamente
        # ma il frame reader non sia disponibile o sia già stato invalidato.
        if self.is_connected:
            try:
                self.tello.streamoff()
            except Exception:
                # L'errore viene registrato solo se vi era effettivamente un reader attivo,
                # cioè in una situazione in cui ci si attendeva un arresto regolare del flusso.
                if had_reader:
                    LOGGER.exception("REAL | Errore durante lo stop dello stream video.")

        if had_reader:
            LOGGER.info("REAL | Video stream fermato.")

    def end(self):
        # Termina l'utilizzo del controller:
        # 1. arresta il flusso video, se presente;
        # 2. chiude la sessione con il drone;
        # 3. riallinea lo stato locale dell'oggetto.
        self.stop_video_stream()

        if self.is_connected:
            try:
                self.tello.end()
            except Exception:
                # La chiusura della connessione viene tentata anche in fase finale;
                # eventuali errori vengono soltanto registrati, poiché il metodo
                # deve comunque ripristinare lo stato locale del controller.
                LOGGER.exception("REAL | Errore durante la chiusura della connessione Tello.")

        # Reset dello stato interno, indipendentemente dall'esito della chiusura remota.
        self.is_connected = False
        self.is_flying = False
        LOGGER.info("REAL | Connessione chiusa.")