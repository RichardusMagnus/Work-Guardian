import queue
import threading
from pathlib import Path
from threading import Thread

import cv2
from ultralytics import YOLO


# BASE_DIR rappresenta la cartella che contiene questo file.
# Viene usata per costruire percorsi ai modelli indipendenti dalla directory
# da cui lo script viene lanciato.
BASE_DIR = Path(__file__).resolve().parent

# Configurazione dei modelli {path, colore, nome}
MODELS_CFG = [
    {
        "path": BASE_DIR / "YOLO" / "ModelTester" / "models" / "ver2clean_n300_extra_nets_s" / "weights" / "best.pt",
        "color": (0, 0, 255),
        "name": "PPE_Detector",
    },
    {
        "path": BASE_DIR / "YOLO" / "ModelTester" / "models" / "Fall_Detector_DEFHJ1" / "weights" / "best.pt",
        "color": (255, 0, 0),
        "name": "Fall_Detector",
    },
]

# Risoluzione
IMAGE_WIDTH = 640  # 1280
IMAGE_HEIGHT = 480  # 720


# Classe del singolo thread
class ModelWorker(Thread):
    # Inizializza il thread
    def __init__(self, model_path, color, name):
        super().__init__()
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))        # Modello
        self.color = color                             # Colore
        self.name = name                               # Nome logico del detector
        self.frame_queue = queue.Queue(maxsize=1)      # Massima lunghezza della lista dei frame
        self.results = []                              # Risultati riportati
        self.results_lock = threading.Lock()           # Lock per accesso thread-safe ai risultati
        self.running = True                            # Partenza
        self.daemon = True                             # Chiude il thread se il main muore

    # Loop di feedback sui frame ricevuti
    def run(self):
        while self.running:
            try:
                # Prende l'ultimo frame disponibile (non blocca per sempre)
                frame = self.frame_queue.get(timeout=1)

                # Esegui inferenza
                # https://docs.ultralytics.com/modes/predict/#inference-arguments
                results = self.model(frame, verbose=False, stream=False)

                with self.results_lock:
                    self.results = results
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[ERRORE] Inferenza fallita per {self.name}: {exc}")

    # Per ricevere frame
    def update_frame(self, frame):
        # Se la coda è piena, rimuovi il vecchio frame (skip) per restare in real-time.
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
        # Restituisce una copia stabile dei risultati correnti, evitando letture
        # concorrenti mentre il thread di inferenza li sta aggiornando.
        with self.results_lock:
            return list(self.results)


# Main thread
def main():
    # Verifica preventiva dei percorsi ai modelli.
    for model_cfg in MODELS_CFG:
        if not model_cfg["path"].is_file():
            print(f"Errore: modello non trovato -> {model_cfg['path']}")
            return

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Webcam non disponibile.")
        return

    # Imposta la risoluzione
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    # Inizializza i worker
    workers = [ModelWorker(m["path"], m["color"], m["name"]) for m in MODELS_CFG]
    for w in workers:
        w.start()

    print("Premi 'q' per uscire.")

    try:
        # Loop di feedback
        while True:
            # Prende un frame dalla webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Invia il frame a entrambi i modelli
            for w in workers:
                w.update_frame(frame.copy())

            # Disegna i risultati salvati nei worker
            for w in workers:
                for r in w.get_results_snapshot():
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.5:  # Mostra i box solo con confidence > 0.5
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            label = f"{w.model.names[cls]} {conf:.2f}"

                            # Disegna sul frame principale
                            cv2.rectangle(frame, (x1, y1), (x2, y2), w.color, 3)
                            cv2.putText(
                                frame,
                                f"{w.name}: {label}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                w.color,
                                2,
                            )

            # Mostra l'output
            cv2.imshow("Multi-Model Detection", frame)

            # Esci premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Stop
        for w in workers:
            w.running = False
        for w in workers:
            w.join(timeout=1.5)
        cap.release()
        cv2.destroyAllWindows()


# Start
if __name__ == "__main__":
    main()