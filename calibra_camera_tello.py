import time

# OpenCV viene impiegato per l'acquisizione, l'elaborazione delle immagini
# e la procedura di calibrazione basata su marker ArUco/ChArUco.
import cv2

# NumPy è utilizzato per la gestione dei vettori e delle matrici numeriche,
# in particolare nella fase di preparazione dei punti per la calibrazione.
import numpy as np


# Verifica preliminare della disponibilità del modulo aruco in OpenCV.
# Tale controllo è necessario perché l'intero processo di calibrazione ChArUco
# dipende dalle funzionalità fornite da questo sottomodulo.
if not hasattr(cv2, "aruco"):
    raise ImportError(
        "Per usare la calibrazione ChArUco devi installare una build di OpenCV con il modulo aruco "
        "(ad esempio opencv-contrib-python)."
    )


# =========================
# CONFIG CALIBRAZIONE CHARUCO
# =========================

# Dizionario ArUco utilizzato per definire i marker presenti sulla board.
# Questo valore deve essere coerente con quello impiegato per generare o stampare la board fisica.
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Numero di quadrati della board stampata lungo gli assi principali.
# Tali parametri descrivono la struttura geometrica della board ChArUco.
SQUARES_X = 7   # numero di quadrati in direzione orizzontale
SQUARES_Y = 5   # numero di quadrati in direzione verticale

# MISURE REALI DELLA TUA STAMPA (in metri)
# Questi parametri devono corrispondere alle dimensioni fisiche reali della board.
# La correttezza della calibrazione dipende direttamente dalla precisione di tali misure.
#
# Esempi:
# se il quadrato misura 24 mm -> 0.024
# se il marker misura 12 mm -> 0.012
SQUARE_LENGTH_M = 0.046
MARKER_LENGTH_M = 0.0235

# Numero minimo consigliato di immagini valide da acquisire prima di eseguire la calibrazione.
# Una quantità insufficiente di viste può compromettere la qualità dei parametri stimati.
MIN_VALID_IMAGES = 15

# Timeout massimo ammesso per l'attesa del primo frame proveniente dal drone.
# Serve a evitare che il programma resti indefinitamente in attesa in caso di problemi nello stream.
FRAME_TIMEOUT_SEC = 5.0


def _create_charuco_board(dictionary):
    """Crea la board ChArUco in modo compatibile con diverse versioni di OpenCV."""
    # Nelle versioni più recenti di OpenCV la board può essere costruita
    # tramite il costruttore CharucoBoard.
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )

    # In versioni precedenti viene invece utilizzata la factory function
    # CharucoBoard_create con firma differente.
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )

    # Se nessuna delle due API è disponibile, l'installazione di OpenCV
    # non supporta la costruzione della board ChArUco.
    raise RuntimeError("OpenCV non supporta ChArUco in questa installazione.")


def _create_charuco_detector(board, dictionary):
    """Crea il detector ChArUco oppure prepara un fallback per API meno recenti."""
    # Nelle versioni più recenti è disponibile un oggetto dedicato che
    # incapsula l'intero processo di rilevazione della board.
    if hasattr(cv2.aruco, "CharucoDetector"):
        return cv2.aruco.CharucoDetector(board)

    # In assenza di CharucoDetector si prepara una struttura di fallback
    # contenente gli elementi necessari per usare le API legacy.
    aruco_detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(dictionary)

    return {
        "board": board,
        "dictionary": dictionary,
        "aruco_detector": aruco_detector,
    }


def _detect_board(detector, board, dictionary, gray):
    """Rileva marker ArUco e corner ChArUco usando API nuove oppure legacy."""
    # Se il detector espone direttamente il metodo detectBoard, si utilizza
    # l'interfaccia moderna di OpenCV, che restituisce simultaneamente
    # corner ChArUco, identificativi e marker rilevati.
    if hasattr(detector, "detectBoard"):
        return detector.detectBoard(gray)

    # Altrimenti si adotta la procedura manuale:
    # 1. rilevazione dei marker ArUco;
    # 2. interpolazione dei corner interni ChArUco.
    aruco_detector = detector["aruco_detector"]
    if aruco_detector is not None:
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    # Inizializzazione dei risultati ChArUco.
    # Se non vengono trovati marker sufficienti, questi valori resteranno nulli.
    charuco_corners = None
    charuco_ids = None

    # L'interpolazione dei corner ChArUco ha senso solo se almeno un marker
    # ArUco è stato rilevato nell'immagine corrente.
    if marker_ids is not None and len(marker_ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def _is_valid_charuco_sample(board, charuco_ids):
    """Verifica che la vista ChArUco sia sufficiente per la calibrazione."""
    # Per essere utile, una vista deve contenere almeno quattro corner identificati.
    # Un numero inferiore non consente una stima affidabile dei parametri geometrici.
    if charuco_ids is None or len(charuco_ids) < 4:
        return False

    # Se i corner risultano collineari, la configurazione geometrica osservata
    # è degenerata e non fornisce informazione sufficiente per la calibrazione.
    if hasattr(board, "checkCharucoCornersCollinear") and board.checkCharucoCornersCollinear(charuco_ids):
        return False

    return True


def _calibrate_charuco(all_charuco_corners, all_charuco_ids, board, image_size):
    """
    Esegue la calibrazione ChArUco con compatibilità verso versioni diverse di OpenCV.

    - Se disponibile, usa cv2.aruco.calibrateCameraCharuco.
    - In caso contrario costruisce object/image points tramite board.matchImagePoints
      e usa cv2.calibrateCamera come fallback.
    """
    # Caso preferenziale: OpenCV mette a disposizione la funzione specializzata
    # per la calibrazione diretta a partire da osservazioni ChArUco.
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        return cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )

    # In assenza della funzione dedicata, si tenta un approccio alternativo
    # basato sulla costruzione esplicita di object points e image points.
    if not hasattr(board, "matchImagePoints"):
        raise RuntimeError(
            "OpenCV non espone calibrateCameraCharuco e la board non supporta matchImagePoints: "
            "fallback di calibrazione non disponibile in questa installazione."
        )

    # Liste che conterranno, per ogni immagine valida, la corrispondenza tra:
    # - punti 3D noti sulla board;
    # - punti 2D osservati nell'immagine.
    all_object_points = []
    all_image_points = []

    # Si elaborano tutte le viste raccolte durante la fase di acquisizione.
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        # Anche in fase di fallback si filtrano le osservazioni non idonee.
        if not _is_valid_charuco_sample(board, ids):
            continue

        obj_points, img_points = board.matchImagePoints(corners, ids)
        if obj_points is None or img_points is None:
            continue

        # Conversione esplicita in array NumPy con forma e tipo numerico compatibili
        # con le aspettative della funzione cv2.calibrateCamera.
        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)

        # Anche in questo caso si richiede un numero minimo di corrispondenze
        # per evitare osservazioni scarsamente informative.
        if len(obj_points) < 4 or len(img_points) < 4:
            continue

        all_object_points.append(obj_points)
        all_image_points.append(img_points)

    # Se dopo il filtraggio non rimane alcuna osservazione valida,
    # la calibrazione non può essere eseguita.
    if not all_object_points:
        raise RuntimeError(
            "OpenCV non espone calibrateCameraCharuco e non sono stati ottenuti abbastanza punti "
            "validi per usare il fallback con calibrateCamera."
        )

    # Calibrazione standard della camera a partire dalle corrispondenze 3D-2D.
    return cv2.calibrateCamera(
        objectPoints=all_object_points,
        imagePoints=all_image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )


def main():
    # L'import della libreria per il drone viene effettuato all'interno della funzione
    # principale per isolare la dipendenza al solo caso d'uso effettivo del programma.
    try:
        from djitellopy import Tello
    except ImportError as exc:
        raise ImportError(
            "Per usare il Tello devi installare djitellopy: pip install djitellopy"
        ) from exc

    # =========================
    # CREAZIONE BOARD E DETECTOR
    # =========================
    # Si costruiscono:
    # - il dizionario dei marker;
    # - la board ChArUco coerente con la configurazione definita;
    # - il detector compatibile con la versione di OpenCV installata.
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = _create_charuco_board(dictionary)
    detector = _create_charuco_detector(board, dictionary)

    # Variabili inizializzate a None per consentire una gestione sicura
    # delle risorse anche in presenza di eccezioni durante l'esecuzione.
    tello = None
    frame_reader = None

    # Liste che accumulano, per ciascuna vista valida acquisita,
    # i corner ChArUco e i relativi identificativi.
    # Questi dati costituiranno l'input della funzione di calibrazione.
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    try:
        # =========================
        # CONNESSIONE TELLO
        # =========================
        # Inizializzazione della connessione al drone e avvio dello stream video.
        tello = Tello()
        tello.connect()
        print(f"Batteria: {tello.get_battery()}%")

        # Si tenta di spegnere un eventuale stream già attivo per evitare
        # conflitti nello stato interno del drone.
        try:
            tello.streamoff()
        except Exception:
            pass

        tello.streamon()
        frame_reader = tello.get_frame_read()

        # Istante di riferimento utilizzato per misurare il tempo trascorso
        # dall'ultima disponibilità di un frame.
        frame_wait_started_at = time.monotonic()

        # Messaggi informativi per l'interazione manuale con l'utente.
        print("\nPremi:")
        print("  c -> salva una vista valida della ChArUco board")
        print("  q -> termina e calibra")
        print()

        # Ciclo principale di acquisizione e visualizzazione dei frame video.
        while True:
            # Recupero del frame corrente dallo stream del drone.
            frame = None if frame_reader is None else frame_reader.frame

            # Se il frame non è ancora disponibile, si attende brevemente
            # per evitare un ciclo di polling troppo aggressivo.
            if frame is None:
                if (time.monotonic() - frame_wait_started_at) >= FRAME_TIMEOUT_SEC:
                    raise TimeoutError(
                        "Nessun frame ricevuto dal Tello entro il timeout iniziale di acquisizione."
                    )
                time.sleep(0.01)
                continue

            # Non appena un frame valido è disponibile, il timer di timeout viene azzerato.
            frame_wait_started_at = time.monotonic()

            # Viene creata una copia del frame per evitare effetti collaterali
            # su eventuali buffer interni gestiti dalla libreria del drone.
            frame = frame.copy()

            # L'immagine display è la versione del frame su cui vengono
            # sovrapposte annotazioni grafiche e informazioni testuali.
            display = frame.copy()

            # Conversione in scala di grigi, necessaria per le operazioni
            # di rilevazione dei marker e dei corner ChArUco.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # image_size è memorizzata nel formato (larghezza, altezza),
            # come richiesto dalle API di calibrazione di OpenCV.
            image_size = gray.shape[::-1]

            # Rilevazione della board ChArUco nel frame corrente.
            # La funzione restituisce:
            # - i corner ChArUco interpolati;
            # - i relativi identificativi;
            # - i corner dei marker ArUco;
            # - gli identificativi dei marker ArUco.
            charuco_corners, charuco_ids, marker_corners, marker_ids = _detect_board(
                detector,
                board,
                dictionary,
                gray,
            )

            # Disegno dei marker ArUco riconosciuti per fornire
            # un feedback visivo immediato all'utente.
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

            # Disegno dei corner ChArUco rilevati.
            # Questi punti sono quelli effettivamente utilizzati nella calibrazione.
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

            # Numero di corner ChArUco validamente rilevati nell'immagine corrente.
            num_corners = 0 if charuco_ids is None else len(charuco_ids)

            # Visualizzazione del numero di viste valide già salvate.
            cv2.putText(
                display,
                f"Viste valide salvate: {len(all_charuco_corners)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # Visualizzazione del numero di corner rilevati nel frame corrente.
            cv2.putText(
                display,
                f"Corner ChArUco rilevati: {num_corners}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Mostra la finestra video con le annotazioni di supporto alla calibrazione.
            cv2.imshow("Calibrazione Tello - ChArUco", display)

            # Lettura del tasto premuto dall'utente.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                # Una vista viene considerata utile per la calibrazione solo
                # se sono stati rilevati almeno 4 corner ChArUco non collineari.
                # Questa soglia minima evita di memorizzare osservazioni troppo povere.
                if _is_valid_charuco_sample(board, charuco_ids):
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    print(
                        f"Vista salvata: {len(all_charuco_corners)} | corner rilevati: {len(charuco_ids)}"
                    )
                else:
                    print("Board non rilevata abbastanza bene: vista non salvata.")

            elif key == ord("q"):
                # Interruzione manuale della fase di acquisizione.
                break

        # Prima di calibrare si verifica che il numero di viste salvate
        # sia almeno pari alla soglia minima raccomandata.
        if len(all_charuco_corners) < MIN_VALID_IMAGES:
            print(f"\nImmagini valide insufficienti: {len(all_charuco_corners)}")
            print(f"Te ne consiglio almeno {MIN_VALID_IMAGES}.")
            return

        # Se non è stato mai acquisito un frame valido, non è possibile
        # determinare la dimensione dell'immagine richiesta dalla calibrazione.
        if image_size is None:
            raise RuntimeError("Nessun frame valido acquisito: calibrazione impossibile.")

        # =========================
        # CALIBRAZIONE
        # =========================
        # Stima dei parametri intrinseci della camera e dei coefficienti di distorsione
        # utilizzando l'insieme delle osservazioni ChArUco raccolte.
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = _calibrate_charuco(
            all_charuco_corners,
            all_charuco_ids,
            board,
            image_size,
        )

        # Stampa dei risultati principali della calibrazione.
        print("\n================ CALIBRAZIONE COMPLETATA ================")
        print(f"RMS reprojection error: {ret}")

        # La camera matrix contiene i parametri intrinseci:
        # focali e coordinate del punto principale.
        print("\nCamera matrix:")
        print(camera_matrix)

        # I coefficienti di distorsione modellano le principali deformazioni ottiche.
        print("\nDistortion coefficients:")
        print(dist_coeffs.ravel())

        # Sezione di output formattata per agevolare il riutilizzo diretto
        # dei parametri all'interno di un file di configurazione Python.
        print("\nDa copiare in app_config.py:\n")

        print(
            "camera_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = ("
        )
        print(f"    ({camera_matrix[0,0]}, {camera_matrix[0,1]}, {camera_matrix[0,2]}),")
        print(f"    ({camera_matrix[1,0]}, {camera_matrix[1,1]}, {camera_matrix[1,2]}),")
        print(f"    ({camera_matrix[2,0]}, {camera_matrix[2,1]}, {camera_matrix[2,2]}),")
        print(")")
        print()

        # I coefficienti vengono linearizzati in un vettore monodimensionale
        # per semplificare la stampa e la copia nel file di configurazione.
        d = dist_coeffs.ravel()

        # Se il modello di distorsione restituisce meno di 5 coefficienti,
        # il vettore viene completato con zeri per mantenere un formato uniforme.
        if len(d) < 5:
            d = np.pad(d, (0, 5 - len(d)))

        print("dist_coeffs: tuple[float, float, float, float, float] = (")
        print(f"    {d[0]},")
        print(f"    {d[1]},")
        print(f"    {d[2]},")
        print(f"    {d[3]},")
        print(f"    {d[4]},")
        print(")")

    finally:
        # Blocco di rilascio risorse eseguito in ogni caso, sia in caso di successo
        # sia in presenza di errori o interruzioni anticipate.
        if frame_reader is not None and hasattr(frame_reader, "stop"):
            try:
                frame_reader.stop()
            except Exception:
                pass

        if tello is not None:
            # Tentativo di arresto dello stream video.
            try:
                tello.streamoff()
            except Exception:
                pass

            # Chiusura della sessione di comunicazione con il drone.
            try:
                tello.end()
            except Exception:
                pass

        # Chiusura di tutte le finestre OpenCV aperte durante l'esecuzione.
        cv2.destroyAllWindows()


# Punto di ingresso standard dello script Python.
# Garantisce che main() venga eseguita solo quando il file è lanciato direttamente
# e non quando è importato come modulo in un altro programma.
if __name__ == "__main__":
    main()