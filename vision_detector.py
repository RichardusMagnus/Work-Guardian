import logging
from pathlib import Path


# Logger di modulo utilizzato per registrare informazioni operative,
# avvisi ed eventuali errori durante il caricamento del modello e l'inferenza.
LOGGER = logging.getLogger(__name__)


class ObjectDetector:
    """Wrapper del modello YOLO usato per l'object detection."""

    def __init__(self, model_path: str, conf: float = 0.5, imgsz: int = 640, device="cpu"):
        # L'import della libreria viene eseguito all'interno del costruttore
        # per rendere il modulo più robusto: l'errore viene sollevato solo
        # quando si tenta effettivamente di istanziare il detector.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            # In assenza della dipendenza necessaria, viene propagato un messaggio
            # esplicativo che indica come installare il pacchetto richiesto.
            raise ImportError(
                "Per usare il detector devi installare ultralytics: pip install ultralytics"
            ) from exc

        # Il percorso del modello viene convertito in oggetto Path per gestirlo
        # in modo più sicuro e leggibile rispetto a una semplice stringa.
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            # Si verifica preventivamente l'esistenza del file del modello,
            # evitando errori meno chiari nelle fasi successive di caricamento.
            raise FileNotFoundError(f"Modello YOLO non trovato: {model_path}")

        # Caricamento effettivo del modello YOLO a partire dal file specificato.
        self.model = YOLO(str(model_path_obj))

        # Parametri di configurazione dell'inferenza:
        # - conf: soglia minima di confidenza per mantenere una detection;
        # - imgsz: dimensione dell'immagine usata dal modello;
        # - device: dispositivo di esecuzione (ad esempio CPU o GPU).
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        # Registrazione informativa utile per il debugging e la tracciabilità
        # dell'esatto modello caricato a runtime.
        LOGGER.info("YOLO caricato da: %s", model_path_obj)

    @staticmethod
    def _tensor_to_python(value):
        # Metodo di utilità che converte oggetti tensore o valori compatibili
        # in una rappresentazione Python standard.
        if value is None:
            return None

        # Se l'oggetto supporta detach(), viene separato dal grafo computazionale.
        # Questo è tipico dei tensori PyTorch durante o dopo l'inferenza.
        if hasattr(value, "detach"):
            value = value.detach()

        # Se disponibile, il dato viene spostato su CPU per garantirne la
        # conversione in un formato Python serializzabile.
        if hasattr(value, "cpu"):
            value = value.cpu()

        # Restituisce il contenuto come lista Python; il chiamante deciderà poi
        # se interpretarlo come scalare o sequenza.
        return value.tolist()

    def _get_label(self, cls: int) -> str:
        # Recupera la struttura dei nomi delle classi dal modello YOLO.
        names = self.model.names

        # Alcune versioni/configurazioni del modello rappresentano i nomi
        # come dizionario indicizzato dall'identificativo numerico della classe.
        if isinstance(names, dict):
            return str(names.get(cls, cls))

        # In altri casi i nomi sono forniti come lista o sequenza ordinata.
        if names is not None and 0 <= cls < len(names):
            return str(names[cls])

        # Fallback: se il nome non è disponibile, viene restituito l'identificativo
        # numerico convertito in stringa.
        return str(cls)

    def detect(self, frame):
        """
        Esegue object detection sul frame e restituisce:
        - frame annotato
        - lista di detection strutturate

        Nota: nella versione multi-modello il frame annotato non viene usato
        dal main loop; vengono invece usate le detection strutturate per poter
        disegnare box di colori diversi per ciascun modello.
        """
        # Se il frame non è disponibile, il metodo restituisce immediatamente
        # un risultato nullo e una lista vuota di detection.
        if frame is None:
            return None, []

        try:
            # Esegue l'inferenza del modello sul frame corrente utilizzando
            # i parametri di configurazione salvati nell'istanza.
            results = self.model(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        except Exception:
            # In caso di errore durante l'inferenza, l'eccezione viene registrata
            # con stack trace completo e poi rilanciata al chiamante.
            LOGGER.exception("Errore durante l'inferenza YOLO.")
            raise

        # Se il modello non produce alcun risultato, viene restituito il frame
        # originale insieme a una lista vuota di detection.
        if not results:
            return frame, []

        # YOLO restituisce una collezione di risultati; qui si assume di lavorare
        # con un singolo frame, quindi si considera il primo elemento.
        result = results[0]

        # Inizialmente il frame annotato coincide con il frame originale.
        # Verrà eventualmente sostituito con una versione disegnata dal modello.
        annotated_frame = frame

        # Lista che conterrà le detection in forma strutturata, adatta a essere
        # utilizzata dal resto dell'applicazione.
        detections = []

        # Si estraggono altezza e larghezza del frame per validare e limitare
        # le coordinate delle bounding box entro i bordi dell'immagine.
        frame_height, frame_width = frame.shape[:2]

        # Se il risultato contiene bounding box, si procede alla loro elaborazione.
        if result.boxes is not None:
            for box in result.boxes:
                # Estrazione dei dati principali della detection:
                # - xyxy: coordinate del rettangolo delimitatore;
                # - conf: punteggio di confidenza;
                # - cls: indice numerico della classe predetta.
                xyxy = self._tensor_to_python(box.xyxy[0])
                conf = self._tensor_to_python(box.conf[0])
                cls = self._tensor_to_python(box.cls[0])

                # Se una delle componenti fondamentali non è disponibile,
                # la detection viene scartata.
                if xyxy is None or conf is None or cls is None:
                    continue

                # Verifica di consistenza sul formato della bounding box:
                # si richiede una sequenza di esattamente quattro valori.
                if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                    LOGGER.warning("Bounding box YOLO in formato inatteso: %r", xyxy)
                    continue

                # Conversione delle coordinate in interi, come richiesto
                # dalle comuni operazioni di disegno e indicizzazione su immagini.
                x1, y1, x2, y2 = map(int, xyxy)

                # Operazione di clamp delle coordinate per garantire che ogni
                # estremo della bounding box ricada all'interno del frame.
                x1 = max(0, min(frame_width - 1, x1))
                y1 = max(0, min(frame_height - 1, y1))
                x2 = max(0, min(frame_width - 1, x2))
                y2 = max(0, min(frame_height - 1, y2))

                # Dopo il clamp si verifica che la bounding box mantenga
                # area positiva; in caso contrario viene considerata non valida.
                if x2 <= x1 or y2 <= y1:
                    LOGGER.warning(
                        "Bounding box YOLO non valida dopo il clamp: (%s, %s, %s, %s)",
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                    continue

                # Conversione finale dei metadati nei tipi Python desiderati.
                conf = float(conf)
                cls = int(cls)

                # Memorizzazione della detection in forma strutturata:
                # - label: nome leggibile della classe;
                # - confidence: punteggio di affidabilità;
                # - bbox: coordinate finali della bounding box.
                detections.append(
                    {
                        "label": self._get_label(cls),
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        # Se almeno una detection è stata prodotta e validata, si genera anche
        # il frame annotato usando la funzione di plotting fornita da YOLO.
        if detections:
            annotated_frame = result.plot()

        # Output del metodo:
        # - frame annotato (o originale, se non vi sono detection valide);
        # - lista delle detection strutturate.
        return annotated_frame, detections