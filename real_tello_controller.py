import logging

# Logger di modulo utilizzato per registrare eventi informativi,
# avvisi ed errori durante l'interazione con il drone reale.
LOGGER = logging.getLogger(__name__)


class RealTelloController:
    """Controller per il drone DJI Tello reale."""

    def __init__(self):
        # L'import della libreria djitellopy viene effettuato all'interno
        # del costruttore per rendere esplicita la dipendenza solo quando
        # si istanzia effettivamente un controller reale.
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise ImportError(
                "Per usare il drone reale devi installare djitellopy: pip install djitellopy"
            ) from exc

        # Creazione dell'oggetto Tello che incapsula la comunicazione
        # con il drone fisico.
        self.tello = Tello()

        # frame_reader conterrà il gestore del flusso video restituito
        # dalla libreria, oppure None se lo stream non è attivo.
        self.frame_reader = None

        # Stato logico della connessione e del volo, mantenuto localmente
        # per consentire controlli preliminari sui comandi inviati.
        self.is_connected = False
        self.is_flying = False

    @staticmethod
    def _clamp_rc_value(value: int) -> int:
        # Limita ciascun comando RC all'intervallo ammesso dal drone,
        # convenzionalmente compreso tra -100 e 100.
        value = int(value)
        return max(-100, min(100, value))

    def connect(self):
        # Stabilisce la connessione con il drone, se non già attiva.
        if self.is_connected:
            LOGGER.info("REAL | Il drone è già connesso.")
            return

        try:
            self.tello.connect()
        except Exception:
            # In caso di errore la causa viene registrata e l'eccezione rilanciata,
            # poiché la connessione è un prerequisito essenziale.
            LOGGER.exception("REAL | Errore durante la connessione al Tello.")
            raise

        self.is_connected = True

        # Lettura iniziale della batteria come verifica supplementare
        # del corretto scambio con il drone.
        try:
            self.tello.get_battery()
        except Exception:
            # Questo errore non impedisce di considerare il drone connesso,
            # ma viene comunque tracciato nel log.
            LOGGER.exception("REAL | Errore durante la lettura iniziale della batteria.")

        LOGGER.info("REAL | Connesso al Tello.")

    def takeoff(self):
        # Esegue il decollo del drone, previa verifica dello stato.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile decollare: drone non connesso.")
            return

        if self.is_flying:
            LOGGER.info("REAL | Il drone è già in volo.")
            return

        try:
            self.tello.takeoff()
        except Exception:
            LOGGER.exception("REAL | Errore durante il decollo.")
            raise

        # Se il comando ha successo, lo stato interno viene aggiornato.
        self.is_flying = True
        LOGGER.info("REAL | Decollo eseguito.")

    def land(self):
        # Esegue l'atterraggio del drone, previa verifica dello stato.
        if not self.is_connected:
            LOGGER.warning("REAL | Impossibile atterrare: drone non connesso.")
            return

        if not self.is_flying:
            LOGGER.info("REAL | Il drone è già a terra.")
            return

        try:
            self.tello.land()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'atterraggio.")
            raise

        # Aggiornamento dello stato locale dopo l'atterraggio.
        self.is_flying = False
        LOGGER.info("REAL | Atterraggio eseguito.")

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int):
        # Invia al drone i comandi di controllo remoto sui quattro assi principali:
        # - lr: left-right;
        # - fb: forward-backward;
        # - ud: up-down;
        # - yaw: rotazione attorno all'asse verticale.
        #
        # I comandi vengono inviati solo se il drone è connesso e in volo.
        if not self.is_connected or not self.is_flying:
            return

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

    def get_status(self) -> dict:
        # Restituisce un riepilogo dello stato corrente del controller e del drone.
        battery = None
        if self.is_connected:
            try:
                battery = self.tello.get_battery()
            except Exception:
                # Se la lettura della batteria fallisce, il metodo restituisce comunque
                # lo stato disponibile, lasciando battery a None.
                LOGGER.exception("REAL | Errore durante la lettura della batteria.")

        return {
            "mode": "REAL",
            "connected": self.is_connected,
            "flying": self.is_flying,
            "battery": battery,
        }

    def start_video_stream(self):
        # Avvia il flusso video proveniente dal drone.
        if not self.is_connected:
            raise RuntimeError("Connetti prima il drone.")

        # Evita di lasciare attivo un frame reader precedente nel caso in cui
        # il metodo venga richiamato più di una volta.
        self.stop_video_stream()

        # Tentativo preventivo di spegnere lo stream corrente per riallineare
        # lo stato interno del drone, anche se non si ha certezza che sia attivo.
        try:
            self.tello.streamoff()
        except Exception:
            pass

        try:
            self.tello.streamon()
            frame_reader = self.tello.get_frame_read()
        except Exception:
            LOGGER.exception("REAL | Errore durante l'avvio del video stream.")
            raise

        # Verifica di coerenza: il frame reader deve essere stato creato correttamente.
        if frame_reader is None:
            raise RuntimeError("Impossibile avviare il reader del video stream.")

        self.frame_reader = frame_reader
        LOGGER.info("REAL | Video stream avviato.")

    def get_frame(self):
        # Restituisce una copia del frame corrente proveniente dallo stream video.
        if self.frame_reader is None:
            return None

        try:
            frame = self.frame_reader.frame
        except Exception:
            LOGGER.exception("REAL | Errore durante la lettura del frame video.")
            return None

        if frame is None:
            return None

        # La copia viene restituita per evitare effetti collaterali dovuti
        # a modifiche esterne sul buffer condiviso del frame reader.
        return frame.copy()

    def stop_video_stream(self):
        # Arresta in modo sicuro il flusso video e il relativo frame reader.
        frame_reader = self.frame_reader
        was_streaming = frame_reader is not None

        # L'attributo viene azzerato subito, così da evitare accessi successivi
        # a un reader che sta per essere chiuso.
        self.frame_reader = None

        if frame_reader is not None and hasattr(frame_reader, "stop"):
            try:
                frame_reader.stop()
            except Exception:
                LOGGER.exception("REAL | Errore durante lo stop del frame reader.")

        # Lo stream del drone viene spento solo se il controller risulta connesso
        # e se in precedenza era presente un reader attivo.
        if self.is_connected and was_streaming:
            try:
                self.tello.streamoff()
            except Exception:
                LOGGER.exception("REAL | Errore durante lo stop dello stream video.")

        if was_streaming:
            LOGGER.info("REAL | Video stream fermato.")

    def end(self):
        # Chiude il video stream, termina la connessione con il drone
        # e riallinea lo stato interno del controller.
        self.stop_video_stream()

        if self.is_connected:
            try:
                self.tello.end()
            except Exception:
                LOGGER.exception("REAL | Errore durante la chiusura della connessione Tello.")

        self.is_connected = False
        self.is_flying = False
        LOGGER.info("REAL | Connessione chiusa.")