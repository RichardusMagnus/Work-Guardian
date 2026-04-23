from pathlib import Path

import cv2
from ultralytics import YOLO

# BASE_DIR rappresenta la cartella che contiene questo file.
# In questo modo il path del modello resta corretto anche se lo script viene
# lanciato da una directory differente.
BASE_DIR = Path(__file__).resolve().parent

# Path del modello.
# Il percorso è stato allineato a quello usato nel resto del progetto
# (app_config.py e multitest.py), evitando incoerenze tra script di test
# e applicazione principale.
MODEL_PATH = BASE_DIR / "YOLO" / "ModelTester" / "models" / "Fall_Detector_DEFHJ1" / "weights" / "best.pt"

# Risoluzione
IMAGE_WIDTH = 640  # 1280
IMAGE_HEIGHT = 480  # 720


def main():
    # Verifica preventiva del path del modello.
    if not MODEL_PATH.is_file():
        print(f"Modello non trovato: {MODEL_PATH}")
        return

    # Carica il modello
    model = YOLO(str(MODEL_PATH))

    # Capture dalla webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam could not be opened.")
        return

    # Imposta la risoluzione
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    print("Premi 'q' per uscire.")

    try:
        # Loop di feedback
        while True:
            # Prendo un frame dalla webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Inferenza (stream=True migliora la performance)
            # https://docs.ultralytics.com/modes/predict/#inference-arguments
            results = model(frame, verbose=False, stream=True)

            # Disegna le bounding boxes per ciascun frame
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])      # Box coordinates
                    conf = float(box.conf[0])                   # Confidence
                    cls = int(box.cls[0])                       # Class ID
                    label = f"{model.names[cls]} {conf:.2f}"    # Class label

                    if conf > 0.5:  # Mostra solo se confidence > 50%
                        # Rettangolo bold rosso
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        # Label bold rossa
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3,
                        )

            # Mostra l'output
            cv2.imshow("ModelTester", frame)

            # Esci premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Stop
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()