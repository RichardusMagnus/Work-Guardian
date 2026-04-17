import logging
from pathlib import Path


# Logger di modulo utilizzato per tracciare eventi significativi, come il
# caricamento corretto del modello o eventuali errori durante l'inferenza.
LOGGER = logging.getLogger(__name__)


class ObjectDetector:
    """Wrapper del modello YOLO usato per l'object detection."""

    def __init__(self, model_path: str, conf: float = 0.5, imgsz: int = 640, device="cpu"):
        # L'import del modello viene eseguito all'interno del costruttore
        # per rendere il modulo più flessibile: la dipendenza da ultralytics
        # diventa necessaria solo nel momento in cui si istanzia effettivamente
        # il detector.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            # In assenza della libreria richiesta, viene sollevata un'eccezione
            # esplicativa che guida l'utente all'installazione corretta.
            raise ImportError(
                "Per usare il detector devi installare ultralytics: pip install ultralytics"
            ) from exc

        # Il percorso del modello viene normalizzato tramite Path, così da
        # semplificare il controllo di esistenza del file e migliorarne
        # la portabilità tra sistemi operativi.
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            # Il costruttore interrompe l'inizializzazione se il file del modello
            # non è disponibile, evitando errori successivi più difficili da diagnosticare.
            raise FileNotFoundError(f"Modello YOLO non trovato: {model_path}")

        # Caricamento effettivo del modello YOLO a partire dal file specificato.
        self.model = YOLO(str(model_path_obj))

        # Parametri principali dell'inferenza:
        # - conf: soglia minima di confidenza per accettare una detection
        # - imgsz: dimensione dell'immagine usata internamente dal modello
        # - device: dispositivo di esecuzione, ad esempio CPU o GPU
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        # Messaggio informativo utile in fase di esecuzione e debugging.
        LOGGER.info("YOLO caricato da: %s", model_path_obj)

    @staticmethod
    def _tensor_to_python(value):
        """
        Converte tensori/array in oggetti Python standard.
        Utile per serializzare confidence, classi e bbox.
        """
        # Se il valore non è presente, la funzione restituisce direttamente None.
        # Questo consente al chiamante di gestire in modo uniforme l'assenza di dati.
        if value is None:
            return None

        # Se l'oggetto proviene da un framework tensoriale (ad esempio PyTorch),
        # si scollega dal grafo computazionale per evitare dipendenze dal contesto
        # di autograd durante la conversione.
        if hasattr(value, "detach"):
            value = value.detach()

        # Se necessario, si sposta il dato sulla CPU, così da renderne possibile
        # la conversione in strutture Python standard anche quando il modello
        # è eseguito su GPU.
        if hasattr(value, "cpu"):
            value = value.cpu()

        # La conversione finale produce tipicamente liste o scalari Python
        # serializzabili e facilmente manipolabili dal resto del programma.
        return value.tolist()

    def _get_label(self, cls: int) -> str:
        """Restituisce il nome leggibile della classe YOLO."""
        # YOLO può rappresentare i nomi delle classi sia come dizionario
        # sia come lista; il metodo gestisce entrambi i casi.
        names = self.model.names

        if isinstance(names, dict):
            # Se i nomi sono memorizzati in un dizionario, si usa l'indice
            # della classe come chiave; in mancanza della chiave, si restituisce
            # comunque l'identificativo numerico convertito in stringa.
            return str(names.get(cls, cls))

        if names is not None and 0 <= cls < len(names):
            # Se i nomi sono in una sequenza indicizzabile, si verifica prima
            # che l'indice sia valido per evitare accessi fuori limite.
            return str(names[cls])

        # Fallback: se il nome leggibile non è disponibile, si restituisce
        # l'identificativo numerico della classe.
        return str(cls)

    def detect(self, frame):
        """
        Esegue object detection sul frame e restituisce:
        - frame annotato
        - lista di detection strutturate
        """
        # In assenza di un frame valido, la funzione restituisce un risultato
        # neutro: nessun frame annotato e nessuna detection.
        if frame is None:
            return None, []

        try:
            # Esecuzione dell'inferenza sul frame di input.
            # I parametri sono quelli configurati in fase di inizializzazione.
            results = self.model(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        except Exception:
            # Eventuali errori durante l'inferenza vengono registrati nel logger
            # con stack trace completo; l'eccezione viene poi rilanciata per non
            # nascondere il problema al chiamante.
            LOGGER.exception("Errore durante l'inferenza YOLO.")
            raise

        # Se il modello non restituisce risultati, il frame originale viene
        # mantenuto invariato e la lista delle detection resta vuota.
        if not results:
            return frame, []

        # Nel caso più comune si considera il primo risultato, corrispondente
        # al frame elaborato.
        result = results[0]

        # Generazione del frame annotato con bounding box e label disegnate
        # direttamente dal metodo di utilità fornito dalla libreria.
        annotated_frame = result.plot()

        detections = []
        if result.boxes is not None:
            # Ogni elemento di result.boxes rappresenta una detection prodotta
            # dal modello. Per ciascuna di esse si estraggono coordinate,
            # confidenza e classe predetta.
            for box in result.boxes:
                xyxy = self._tensor_to_python(box.xyxy[0])
                conf = self._tensor_to_python(box.conf[0])
                cls = self._tensor_to_python(box.cls[0])

                # Se uno dei campi fondamentali non è disponibile, la detection
                # viene scartata per evitare risultati incompleti o incoerenti.
                if xyxy is None or conf is None or cls is None:
                    continue

                # Le coordinate del bounding box vengono convertite in interi,
                # in quanto rappresentano posizioni di pixel nel frame:
                # (x1, y1) angolo superiore sinistro, (x2, y2) angolo inferiore destro.
                x1, y1, x2, y2 = map(int, xyxy)

                # Confidenza e indice di classe vengono convertiti nei tipi Python
                # attesi per facilitare uso, stampa e serializzazione.
                conf = float(conf)
                cls = int(cls)

                # Ogni detection viene salvata in forma strutturata tramite dizionario,
                # così da rendere l'output semplice da consumare da parte di altri moduli.
                detections.append({
                    "label": self._get_label(cls),
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })

        # Output finale della procedura:
        # - frame con annotazioni grafiche
        # - elenco delle detection estratte in formato strutturato
        return annotated_frame, detections