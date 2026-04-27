import queue
import threading
from pathlib import Path
from threading import Thread

import cv2

from app_config import APP_CONFIG


# BASE_DIR rappresenta la cartella che contiene questo file.
# Viene usata per costruire percorsi ai modelli indipendenti dalla directory
# da cui lo script viene lanciato. In questo file la variabile resta disponibile
# come riferimento generale alla radice locale del modulo.
BASE_DIR = Path(__file__).resolve().parent

# Configurazione dei modelli YOLO da eseguire in parallelo.
# Ogni elemento della lista contiene:
# - il percorso del modello addestrato;
# - il colore da usare per disegnare le relative predizioni;
# - il nome logico del detector, utile per distinguere visivamente i risultati.
#
# La configurazione viene ricavata da APP_CONFIG, così da mantenere coerenza
# con la configurazione centralizzata del progetto ed evitare duplicazioni
# tra script di test e applicazione principale.
MODELS_CFG = [
    {
        "path": model_cfg.path,
        "color": model_cfg.color,
        "name": model_cfg.name,
    }
    for model_cfg in APP_CONFIG.yolo_models
]

# Dimensioni desiderate del frame acquisito dalla webcam.
# Questi valori vengono passati a OpenCV come proprietà del dispositivo video.
IMAGE_WIDTH = 640  # 1280
IMAGE_HEIGHT = 480  # 720


def _tensor_to_python(value):
    """
    Converte tensori PyTorch/Ultralytics in valori Python standard.

    La funzione è utile perché i risultati prodotti da Ultralytics possono essere
    rappresentati come tensori, spesso residenti su GPU. Prima di utilizzarli
    per controlli logici, conversioni numeriche o visualizzazione, vengono quindi
    scollegati dal grafo computazionale, spostati su CPU e convertiti in liste
    o valori Python ordinari.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _safe_bbox_from_box(box, frame_width, frame_height):
    """
    Estrae e valida una bounding box YOLO in formato (x1, y1, x2, y2).

    La funzione svolge tre operazioni principali:
    1. converte le coordinate del box in valori Python standard;
    2. verifica che il formato sia composto da quattro coordinate;
    3. limita le coordinate ai bordi dell'immagine, evitando valori fuori frame.

    Restituisce None se il formato è inatteso o se il box non ha area positiva.
    """
    xyxy = _tensor_to_python(box.xyxy[0])
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        return None

    # Conversione delle coordinate in interi, coerentemente con l'uso richiesto
    # dalle funzioni di disegno di OpenCV.
    x1, y1, x2, y2 = map(int, xyxy)

    # Clipping delle coordinate all'interno dei limiti effettivi del frame.
    # Questo evita errori grafici o accessi non coerenti in caso di predizioni
    # leggermente esterne all'immagine.
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(0, min(frame_width - 1, x2))
    y2 = max(0, min(frame_height - 1, y2))

    # Un box è significativo solo se possiede larghezza e altezza positive.
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _get_model_label(model, cls: int) -> str:
    """
    Restituisce in modo sicuro il nome della classe YOLO.

    Ultralytics può rappresentare i nomi delle classi come dizionario oppure
    come sequenza indicizzata. La funzione gestisce entrambi i casi e, se il nome
    non è disponibile, restituisce l'identificativo numerico della classe.
    """
    names = getattr(model, "names", None)

    if isinstance(names, dict):
        return str(names.get(cls, cls))

    if names is not None and 0 <= cls < len(names):
        return str(names[cls])

    return str(cls)


# Classe che incapsula l'esecuzione di un singolo modello YOLO in un thread dedicato.
# Ogni istanza riceve frame dal thread principale, esegue l'inferenza e conserva
# l'ultimo insieme di risultati disponibili.
class ModelWorker(Thread):
    # Inizializza il thread associato a uno specifico modello.
    def __init__(self, model_path, color, name):
        super().__init__()

        # L'import di Ultralytics viene effettuato qui per intercettare in modo
        # esplicito l'assenza della libreria solo quando il worker viene creato.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Per eseguire il test multi-modello devi installare ultralytics: pip install ultralytics"
            ) from exc

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))        # Modello YOLO caricato dal percorso indicato.
        self.color = color                             # Colore associato al modello per il disegno dei box.
        self.name = name                               # Nome logico del detector visualizzato sull'immagine.
        self.frame_queue = queue.Queue(maxsize=1)      # Buffer contenente al massimo il frame più recente.
        self.results = []                              # Ultimi risultati di inferenza prodotti dal modello.
        self.results_lock = threading.Lock()           # Lock per proteggere l'accesso concorrente ai risultati.
        self.running = True                            # Flag di controllo del ciclo di esecuzione del thread.
        self.daemon = True                             # Consente la chiusura del thread se termina il main.

    # Ciclo principale del worker: attende frame, esegue inferenza e aggiorna i risultati.
    def run(self):
        while self.running:
            try:
                # Prende l'ultimo frame disponibile. Il timeout evita che il thread
                # resti bloccato indefinitamente quando l'applicazione deve terminare.
                frame = self.frame_queue.get(timeout=1)

                # Esegue l'inferenza del modello sul frame ricevuto.
                # verbose=False riduce l'output testuale; stream=False restituisce
                # direttamente la lista dei risultati.
                # https://docs.ultralytics.com/modes/predict/#inference-arguments
                results = self.model(frame, verbose=False, stream=False)

                # I risultati vengono aggiornati in sezione critica per evitare
                # che il thread principale li legga mentre sono in fase di modifica.
                with self.results_lock:
                    self.results = results
            except queue.Empty:
                # Se non arrivano frame entro il timeout, il worker continua a ciclare
                # e verifica nuovamente il flag self.running.
                continue
            except Exception as exc:
                # In caso di errore durante l'inferenza, i risultati vengono svuotati
                # per non mostrare predizioni obsolete o potenzialmente incoerenti.
                with self.results_lock:
                    self.results = []
                print(f"[ERRORE] Inferenza fallita per {self.name}: {exc}")

    # Inserisce un nuovo frame nel buffer del worker.
    def update_frame(self, frame):
        # Se la coda è piena, viene rimosso il frame precedente.
        # Questa scelta privilegia l'elaborazione del frame più recente rispetto
        # all'elaborazione completa di tutti i frame acquisiti, mantenendo il sistema
        # più vicino al comportamento real-time.
        #
        # Si usa un approccio non bloccante per evitare deadlock tra controllo di full()
        # e inserimento effettivo nel buffer.
        while self.running:
            try:
                self.frame_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

    def get_results_snapshot(self):
        # Restituisce una copia stabile dei risultati correnti.
        # La copia evita che il thread principale iteri direttamente su una struttura
        # che potrebbe essere aggiornata contemporaneamente dal worker.
        with self.results_lock:
            return list(self.results)


# Funzione principale: inizializza webcam e modelli, coordina i thread di inferenza
# e visualizza su schermo le predizioni prodotte dai detector.
def main():
    # Verifica preventiva dei percorsi ai modelli.
    # Se un file modello non è presente, l'esecuzione viene interrotta prima
    # dell'avvio della webcam e dei thread.
    for model_cfg in MODELS_CFG:
        if not model_cfg["path"].is_file():
            print(f"Errore: modello non trovato -> {model_cfg['path']}")
            return

    # Inizializza la webcam predefinita, indicata dall'indice 0.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Webcam non disponibile.")
        return

    # Imposta la risoluzione richiesta per i frame acquisiti.
    # L'effettiva risoluzione può dipendere dalle capacità della webcam e dai driver.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    # Crea un worker per ciascun modello configurato.
    # Ogni worker esegue l'inferenza in un thread separato, permettendo di processare
    # più detector senza bloccare direttamente il ciclo di acquisizione video.
    workers = [ModelWorker(m["path"], m["color"], m["name"]) for m in MODELS_CFG]
    for w in workers:
        w.start()

    print("Premi 'q' per uscire.")

    try:
        # Ciclo principale di acquisizione, inferenza e visualizzazione.
        while True:
            # Acquisisce un frame dalla webcam.
            ret, frame = cap.read()
            if not ret:
                break

            # Invia una copia del frame a ciascun worker.
            # La copia evita che modifiche successive al frame principale interferiscano
            # con l'immagine usata dai thread di inferenza.
            for w in workers:
                w.update_frame(frame.copy())

            # Disegna sul frame principale i risultati più recenti salvati nei worker.
            frame_height, frame_width = frame.shape[:2]
            for w in workers:
                for r in w.get_results_snapshot():
                    if r.boxes is None:
                        continue

                    # Ogni box rappresenta una rilevazione prodotta dal modello:
                    # contiene coordinate, confidenza e classe stimata.
                    for box in r.boxes:
                        bbox = _safe_bbox_from_box(box, frame_width, frame_height)
                        conf = _tensor_to_python(box.conf[0])
                        cls = _tensor_to_python(box.cls[0])

                        # Se uno degli elementi essenziali non è valido, la predizione
                        # viene ignorata per evitare errori in fase di disegno.
                        if bbox is None or conf is None or cls is None:
                            continue

                        conf = float(conf)
                        if conf > 0.5:  # Mostra i box solo con confidence > 0.5.
                            x1, y1, x2, y2 = bbox
                            cls = int(cls)
                            label = f"{_get_model_label(w.model, cls)} {conf:.2f}"

                            # Disegna il rettangolo della bounding box e l'etichetta
                            # testuale contenente nome del detector, classe e confidenza.
                            cv2.rectangle(frame, (x1, y1), (x2, y2), w.color, 3)
                            cv2.putText(
                                frame,
                                f"{w.name}: {label}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                w.color,
                                2,
                            )

            # Mostra l'output video annotato con le predizioni dei modelli.
            cv2.imshow("Multi-Model Detection", frame)

            # Esce dal ciclo principale quando l'utente preme il tasto 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Sezione di chiusura eseguita anche in caso di interruzioni o errori.
        # Arresta i worker, rilascia la webcam e chiude le finestre OpenCV.
        for w in workers:
            w.running = False
        for w in workers:
            w.join(timeout=1.5)
        cap.release()
        cv2.destroyAllWindows()


# Punto di ingresso dello script.
# La funzione main viene eseguita solo se il file è lanciato direttamente,
# non quando viene importato come modulo da altri file Python.
if __name__ == "__main__":
    main()