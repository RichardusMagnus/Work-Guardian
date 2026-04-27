from pathlib import Path

# OpenCV viene utilizzato per acquisire i frame dalla webcam,
# visualizzare l'immagine a schermo e disegnare le annotazioni grafiche.
import cv2

# Configurazione centralizzata dell'applicazione, dalla quale vengono letti
# i modelli YOLO disponibili e i relativi percorsi su disco.
from app_config import APP_CONFIG

# BASE_DIR rappresenta la cartella che contiene questo file.
# L'uso di __file__, resolve() e parent consente di costruire percorsi assoluti
# indipendenti dalla directory da cui lo script viene effettivamente eseguito.
BASE_DIR = Path(__file__).resolve().parent

# Nome logico del modello da testare.
# Questo nome deve corrispondere, quando possibile, a uno dei modelli definiti
# nella configurazione centralizzata APP_CONFIG.yolo_models.
MODEL_NAME = "Fall_Detector"

# Risoluzione richiesta per l'acquisizione video.
# I valori indicano rispettivamente larghezza e altezza del frame acquisito
# dalla webcam. I commenti a destra riportano possibili risoluzioni alternative.
IMAGE_WIDTH = 640  # 1280
IMAGE_HEIGHT = 480  # 720


def _resolve_model_path():
    """
    Recupera il path del modello dalla configurazione centralizzata.

    La funzione cerca, tra i modelli YOLO registrati in APP_CONFIG, quello
    avente nome logico uguale a MODEL_NAME. In questo modo il percorso del
    modello non viene duplicato nel codice e resta coerente con il resto
    dell'applicazione.
    """
    # Scansione delle configurazioni dei modelli YOLO definite nel progetto.
    for model_cfg in APP_CONFIG.yolo_models:
        # Quando viene trovato il modello desiderato, se ne restituisce
        # direttamente il percorso associato.
        if model_cfg.name == MODEL_NAME:
            return model_cfg.path

    # Fallback conservativo: mantiene il layout storico del progetto
    # nel caso in cui il modello non sia presente in APP_CONFIG.
    # Questa scelta consente allo script di restare utilizzabile anche
    # in assenza di una configurazione aggiornata.
    return BASE_DIR / "YOLO" / "ModelTester" / "models" / "Fall_Detector_DEFHJ1" / "weights" / "best.pt"


# Path effettivo del modello YOLO da caricare.
# Il valore viene calcolato una sola volta all'avvio del modulo.
MODEL_PATH = _resolve_model_path()


def _tensor_to_python(value):
    """
    Converte tensori PyTorch/Ultralytics in valori Python standard,
    spostandoli prima su CPU quando necessario.

    L'output prodotto da Ultralytics può contenere tensori PyTorch. Per poter
    confrontare, convertire o formattare tali valori in modo sicuro, è utile
    trasformarli in tipi Python ordinari, come float, int, liste o None.
    """
    # Se il valore non è presente, viene propagato None senza ulteriori
    # trasformazioni.
    if value is None:
        return None

    # detach() rimuove il tensore dal grafo computazionale di PyTorch,
    # evitando dipendenze non necessarie dalla fase di calcolo dei gradienti.
    if hasattr(value, "detach"):
        value = value.detach()

    # cpu() garantisce che il dato sia disponibile nella memoria della CPU.
    # Questo è importante quando il modello è eseguito su GPU.
    if hasattr(value, "cpu"):
        value = value.cpu()

    # tolist() converte tensori o array in strutture Python standard.
    if hasattr(value, "tolist"):
        value = value.tolist()

    return value


def _safe_bbox_from_box(box, frame_width, frame_height):
    """
    Estrae e valida una bounding box YOLO in formato (x1, y1, x2, y2).

    La funzione restituisce None se il formato del box non è quello atteso
    oppure se, dopo la correzione rispetto ai limiti del frame, la bounding box
    non possiede area positiva.
    """
    # Ultralytics rappresenta le coordinate della bounding box in formato xyxy:
    # x1, y1 indicano il vertice superiore sinistro; x2, y2 quello inferiore
    # destro. Si estrae il primo elemento perché ogni box contiene una singola
    # predizione.
    xyxy = _tensor_to_python(box.xyxy[0])

    # Controllo difensivo sul formato delle coordinate.
    # Sono accettate solo sequenze di quattro elementi.
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        return None

    # Conversione delle coordinate a interi, necessaria per il disegno
    # della bounding box tramite OpenCV.
    x1, y1, x2, y2 = map(int, xyxy)

    # Limitazione delle coordinate ai bordi dell'immagine.
    # Questo evita errori grafici o accessi fuori dai limiti del frame.
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(0, min(frame_width - 1, x2))
    y2 = max(0, min(frame_height - 1, y2))

    # Una bounding box è significativa solo se ha larghezza e altezza positive.
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _get_model_label(model, cls: int) -> str:
    """
    Restituisce in modo sicuro il nome della classe YOLO.

    La funzione gestisce sia il caso in cui model.names sia un dizionario,
    sia il caso in cui sia una lista o una struttura indicizzabile. Se il nome
    della classe non è disponibile, viene restituito l'identificativo numerico.
    """
    # Recupero dell'attributo names del modello, se presente.
    names = getattr(model, "names", None)

    # Alcuni modelli rappresentano le classi come dizionario:
    # chiave = indice numerico della classe, valore = nome testuale.
    if isinstance(names, dict):
        return str(names.get(cls, cls))

    # In altri casi i nomi sono memorizzati in una lista o sequenza.
    # Il controllo sui limiti evita errori di indice.
    if names is not None and 0 <= cls < len(names):
        return str(names[cls])

    # Fallback: se non è possibile recuperare il nome, si usa l'ID numerico.
    return str(cls)


def main():
    # Import locale per rendere il modulo più robusto:
    # l'errore viene sollevato solo quando si esegue realmente il test.
    # In questo modo è comunque possibile importare il file senza avere
    # necessariamente ultralytics installato nell'ambiente.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Per eseguire il test del modello devi installare ultralytics: pip install ultralytics"
        ) from exc

    # Verifica preventiva del path del modello.
    # Se il file dei pesi non esiste, il programma termina in modo controllato
    # senza tentare il caricamento del modello.
    if not MODEL_PATH.is_file():
        print(f"Modello non trovato: {MODEL_PATH}")
        return

    # Caricamento del modello YOLO a partire dal file dei pesi individuato.
    # La conversione a stringa è richiesta perché YOLO si aspetta comunemente
    # un percorso in formato testuale.
    model = YOLO(str(MODEL_PATH))

    # Apertura del flusso video dalla webcam predefinita.
    # L'indice 0 indica normalmente la prima webcam disponibile nel sistema.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam could not be opened.")
        return

    # Impostazione della risoluzione di acquisizione desiderata.
    # Il driver della webcam potrebbe non rispettare esattamente questi valori,
    # ma OpenCV tenta comunque di configurarli.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    print("Premi 'q' per uscire.")

    try:
        # Loop principale di acquisizione, inferenza e visualizzazione.
        # A ogni iterazione viene letto un frame, analizzato dal modello YOLO
        # e mostrato all'utente con eventuali bounding box sovrapposte.
        while True:
            # Acquisizione di un singolo frame dalla webcam.
            # ret indica se la lettura è avvenuta correttamente.
            ret, frame = cap.read()
            if not ret:
                break

            # Inferenza del modello YOLO sul frame corrente.
            # stream=True restituisce un generatore di risultati e può migliorare
            # l'efficienza, specialmente in scenari di elaborazione continua.
            # https://docs.ultralytics.com/modes/predict/#inference-arguments
            results = model(frame, verbose=False, stream=True)

            # Estrazione delle dimensioni effettive del frame.
            # Questi valori sono usati per validare e correggere le coordinate
            # delle bounding box rispetto ai limiti dell'immagine.
            frame_height, frame_width = frame.shape[:2]

            # Iterazione sui risultati prodotti dal modello per il frame corrente.
            for r in results:
                # Se non sono state rilevate bounding box, si passa al risultato
                # successivo senza effettuare ulteriori elaborazioni.
                if r.boxes is None:
                    continue

                # Ogni elemento di r.boxes rappresenta una predizione del modello,
                # composta da coordinate, confidenza e classe stimata.
                for box in r.boxes:
                    bbox = _safe_bbox_from_box(box, frame_width, frame_height)
                    conf = _tensor_to_python(box.conf[0])
                    cls = _tensor_to_python(box.cls[0])

                    # Se una delle informazioni fondamentali non è valida,
                    # la predizione viene ignorata per evitare errori successivi.
                    if bbox is None or conf is None or cls is None:
                        continue

                    conf = float(conf)                         # Valore di confidenza della predizione.
                    cls = int(cls)                             # Identificativo numerico della classe.
                    x1, y1, x2, y2 = bbox                      # Coordinate della bounding box validata.
                    label = f"{_get_model_label(model, cls)} {conf:.2f}"    # Etichetta testuale da mostrare sul frame.

                    # Soglia minima di confidenza: vengono visualizzate solo
                    # le predizioni considerate sufficientemente affidabili.
                    if conf > 0.5:  # Mostra solo se confidence > 50%
                        # Disegno della bounding box in rosso.
                        # Lo spessore elevato rende il rettangolo ben visibile.
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                        # Disegno dell'etichetta della classe e della confidenza.
                        # La coordinata y viene limitata inferiormente a 20 per
                        # evitare che il testo esca dal bordo superiore del frame.
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3,
                        )

            # Visualizzazione del frame annotato in una finestra OpenCV.
            cv2.imshow("ModelTester", frame)

            # Interruzione del ciclo quando l'utente preme il tasto 'q'.
            # cv2.waitKey(1) attende brevemente un input da tastiera e permette
            # alla finestra grafica di aggiornarsi correttamente.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Rilascio delle risorse hardware e grafiche.
        # Il blocco finally garantisce la chiusura corretta anche in caso
        # di interruzioni o errori durante il ciclo principale.
        cap.release()
        cv2.destroyAllWindows()


# Punto di ingresso dello script.
# Questa condizione assicura che main() venga eseguita solo quando il file
# è lanciato direttamente, e non quando viene importato come modulo.
if __name__ == "__main__":
    main()