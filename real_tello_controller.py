import logging


# Inizializzazione del logger di modulo.
# L'uso di __name__ consente di associare i messaggi di log al nome del modulo corrente,
# facilitando il tracciamento dell'esecuzione in applicazioni composte da più file.
LOGGER = logging.getLogger(__name__)


class RealTelloController:
    """Controller per il drone DJI Tello reale."""

    def __init__(self):
        # L'import viene eseguito all'interno del costruttore per rendere il modulo
        # utilizzabile anche in ambienti in cui la libreria djitellopy non è installata,
        # purché non venga istanziato il controller reale.
        try:
            from djitellopy import Tello
        except ImportError as exc:
            # L'eccezione originale viene incapsulata per fornire un messaggio più chiaro
            # all'utente, mantenendo comunque la causa sottostante tramite "from exc".
            raise ImportError(
                "Per usare il drone reale devi installare djitellopy: pip install djitellopy"
            ) from exc

        # Istanza dell'oggetto che rappresenta l'interfaccia verso il drone fisico.
        self.tello = Tello()

        # Reader dei frame video, inizialmente assente finché lo stream non viene avviato.
        self.frame_reader = None

        # Stato logico del controller: connessione al drone e stato di volo.
        # Questi flag evitano operazioni non ammissibili, come il decollo senza connessione.
        self.is_connected = False
        self.is_flying = False

    @staticmethod
    def _clamp_rc_value(value: int) -> int:
        """Limita i comandi RC nell'intervallo accettato dal Tello."""
        # I comandi RC del Tello devono rientrare nell'intervallo [-100, 100].
        # La conversione a int garantisce coerenza del tipo anche se viene passato
        # un valore numerico non intero.
        value = int(value)
        return max(-100, min(100, value))

    def connect(self):
        """Connette il drone e legge il livello batteria."""
        # Se il drone risulta già connesso, il metodo termina senza ripetere la procedura.
        if self.is_connected:
            LOGGER.info("REAL | Il drone è già connesso.")
            return

        try:
            # Apertura della connessione con il drone reale.
            self.tello.connect()
        except Exception:
            # Il log con exception registra automaticamente anche lo stack trace,
            # utile per il debugging di problemi di rete o di inizializzazione.
            LOGGER.exception("REAL | Errore durante la connessione al Tello.")
            raise

        # Il flag viene aggiornato solo dopo una connessione andata a buon fine.
        self.is_connected = True

        # Lettura opzionale del livello di batteria, usata a scopo informativo.
        battery = None
        try:
            battery = self.tello.get_battery()
        except Exception:
            # Un errore nella lettura della batteria non invalida la connessione:
            # per questo viene registrato ma non rilanciato.
            LOGGER.exception("REAL | Errore durante la lettura della batteria.")

        LOGGER.info("REAL | Connesso al Tello. Batteria: %s%%", battery)

    def takeoff(self):
        """Esegue il decollo, se possibile."""
        # Il decollo è consentito solo se il drone è già stato connesso.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile decollare: drone non connesso.")
            return

        # Evita di inviare un comando di decollo ridondante.
        if self.is_flying:
            LOGGER.info("REAL | Il drone è già in volo.")
            return

        try:
            self.tello.takeoff()
        except Exception:
            LOGGER.exception("REAL | Errore durante il decollo.")
            raise

        # Aggiornamento dello stato interno a seguito del decollo riuscito.
        self.is_flying = True
        LOGGER.info("REAL | Decollo eseguito.")

    def land(self):
        """Esegue l'atterraggio, se il drone è in volo."""
        # Non è possibile comandare l'atterraggio senza una connessione attiva.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile atterrare: drone non connesso.")
            return

        # Se il drone non è in volo, l'atterraggio non è necessario.
        if not self.is_flying:
            LOGGER.info("REAL | Il drone è già a terra.")
            return

        try:
            self.tello.land()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'atterraggio.")
            raise

        # Aggiornamento coerente dello stato del controller.
        self.is_flying = False
        LOGGER.info("REAL | Atterraggio eseguito.")

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        """
        Invia i comandi RC al drone.
        Se il drone non è connesso o non è in volo, non invia nulla.
        """
        # I comandi RC vengono trasmessi solo durante il volo effettivo.
        # In caso contrario il metodo termina silenziosamente.
        if not self.is_connected or not self.is_flying:
            return

        try:
            # Ogni componente del comando viene limitata all'intervallo supportato:
            # - lr: movimento laterale sinistra/destra
            # - fb: movimento avanti/indietro
            # - ud: movimento verticale su/giù
            # - yaw: rotazione attorno all'asse verticale
            self.tello.send_rc_control(
                self._clamp_rc_value(lr),
                self._clamp_rc_value(fb),
                self._clamp_rc_value(ud),
                self._clamp_rc_value(yaw),
            )
        except Exception:
            LOGGER.exception("REAL | Errore durante l'invio dei comandi RC.")
            raise

    def get_status(self) -> dict:
        """Restituisce uno stato sintetico del controller/drone."""
        # Il livello della batteria viene richiesto solo se il drone risulta connesso.
        battery = None
        if self.is_connected:
            try:
                battery = self.tello.get_battery()
            except Exception:
                # Anche in questo caso l'errore viene registrato, ma non blocca
                # la costruzione dello stato complessivo del controller.
                LOGGER.exception("REAL | Errore durante la lettura della batteria.")

        # Restituzione di una struttura dati compatta, utile per interfacce utente,
        # debug o monitoraggio dello stato corrente.
        return {
            "mode": "REAL",
            "connected": self.is_connected,
            "flying": self.is_flying,
            "battery": battery,
        }

    def start_video_stream(self):
        """Avvia lo stream video del Tello e inizializza il frame reader."""
        # Lo stream video presuppone che il drone sia già connesso.
        if not self.is_connected:
            raise RuntimeError("Connetti prima il drone.")

        # Proviamo a fermare un eventuale stream precedente
        # Questo passaggio riduce il rischio di stati incoerenti nel caso in cui
        # lo stream fosse già attivo o non fosse stato chiuso correttamente.
        try:
            self.tello.streamoff()
        except Exception:
            # L'errore viene ignorato deliberatamente, poiché l'obiettivo è solo
            # tentare una pulizia preventiva prima di riavviare lo stream.
            pass

        try:
            # Attivazione dello stream e acquisizione dell'oggetto deputato
            # alla lettura asincrona dei frame video.
            self.tello.streamon()
            frame_reader = self.tello.get_frame_read()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'avvio del video stream.")
            raise

        # Controllo di sicurezza: il reader deve essere stato creato correttamente.
        if frame_reader is None:
            raise RuntimeError("Impossibile avviare il reader del video stream.")

        self.frame_reader = frame_reader
        LOGGER.info("REAL | Video stream avviato.")

    def get_frame(self):
        """
        Restituisce una copia del frame corrente.
        Se lo stream non è attivo o ci sono errori, restituisce None.
        """
        # Se il video stream non è stato inizializzato, non esiste alcun frame disponibile.
        if self.frame_reader is None:
            return None

        try:
            # Accesso al frame più recente prodotto dal reader.
            frame = self.frame_reader.frame
        except Exception:
            LOGGER.exception("REAL | Errore durante la lettura del frame video.")
            return None

        # In alcuni casi il reader può esistere ma non avere ancora prodotto un frame valido.
        if frame is None:
            return None

        # Viene restituita una copia del frame per evitare che il chiamante modifichi
        # direttamente il buffer interno gestito dal frame reader.
        return frame.copy()

    def stop_video_stream(self):
        """Ferma lo stream video, se attivo."""
        # Si memorizza temporaneamente il riferimento corrente per poterlo chiudere
        # anche dopo aver azzerato l'attributo di istanza.
        frame_reader = self.frame_reader
        was_streaming = frame_reader is not None
        self.frame_reader = None

        # Se il frame reader espone un metodo stop, si tenta l'arresto esplicito.
        if frame_reader is not None and hasattr(frame_reader, "stop"):
            try:
                frame_reader.stop()
            except Exception:
                LOGGER.exception("REAL | Errore durante lo stop del frame reader.")

        # Lo stream lato drone viene arrestato solo se la connessione è ancora attiva.
        if self.is_connected:
            try:
                self.tello.streamoff()
            except Exception:
                LOGGER.exception("REAL | Errore durante lo stop dello stream video.")

        # Il messaggio informativo viene emesso solo se era effettivamente presente
        # un reader associato allo stream.
        if was_streaming:
            LOGGER.info("REAL | Video stream fermato.")

    def end(self):
        """Chiude stream e connessione col drone."""
        # Prima di terminare la sessione si arresta lo stream video, se presente.
        self.stop_video_stream()

        # Chiusura della connessione con il drone reale.
        if self.is_connected:
            try:
                self.tello.end()
            except Exception:
                LOGGER.exception("REAL | Errore durante la chiusura della connessione Tello.")

        # Ripristino dello stato interno del controller.
        self.is_connected = False
        self.is_flying = False
        LOGGER.info("REAL | Connessione chiusa.")