import logging

import cv2
import numpy as np


# Logger di modulo utilizzato per registrare messaggi informativi ed errori
# relativi all'inizializzazione del detector e all'elaborazione dei frame.
LOGGER = logging.getLogger(__name__)


class CameraPoseEstimator:
    """
    Stima la posa della camera rispetto a un AprilTag e disegna il risultato sul frame.

    Versione adattata per usare pyapriltags:
    - non apre una camera propria
    - non contiene un loop while True
    - lavora sui frame già acquisiti dal drone
    """

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        tag_family: str = "tag25h9",
        threads: int = 4,
        decimate: float = 2.0,
        tag_size_m: float = 0.1167,
        tag_position=(0.0, 0.0, 0.0),
        enabled: bool = True,
    ):
        # Flag che abilita o disabilita completamente la pipeline di detection/stima posa.
        # Se posto a False, l'oggetto viene istanziato ma non inizializza il detector.
        self.enabled = enabled

        # Memorizzazione dei parametri intrinseci della camera e dei coefficienti
        # di distorsione in formato NumPy con tipo numerico coerente per OpenCV.
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)

        # Dimensione fisica del tag espressa in metri: informazione necessaria
        # per la stima metrica della posa tramite il detector AprilTag.
        self.tag_size_m = float(tag_size_m)

        # Posizione del tag nel sistema di riferimento globale o applicativo.
        # In questo file viene memorizzata ma non ancora utilizzata nei calcoli successivi;
        # può essere utile per eventuali estensioni del sistema.
        self.tag_position = np.array(tag_position, dtype=np.float32).reshape(3, 1)

        # Verifica di coerenza sul formato della matrice intrinseca:
        # deve essere una matrice 3x3.
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix deve avere forma (3, 3), ricevuto {self.camera_matrix.shape}."
            )

        # La dimensione del tag deve essere strettamente positiva per avere
        # una stima geometrica valida della posa.
        if self.tag_size_m <= 0:
            raise ValueError("tag_size_m deve essere maggiore di zero.")

        # Estrazione dei parametri focali e del centro ottico dalla matrice intrinseca.
        # Tali valori sono richiesti direttamente dall'interfaccia del detector AprilTag.
        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        # Il detector viene inizialmente posto a None e creato solo se il modulo
        # è esplicitamente abilitato.
        self.detector = None

        if self.enabled:
            # Import locale del detector: in questo modo il file può essere importato
            # anche in ambienti in cui pyapriltags non è installato, purché la
            # funzionalità non venga effettivamente attivata.
            try:
                from pyapriltags import Detector
            except ImportError as exc:
                raise ImportError(
                    "Per usare la stima della posa devi installare pyapriltags."
                ) from exc

            # Inizializzazione del detector con i parametri desiderati.
            # - families: famiglia di tag da ricercare
            # - nthreads: numero di thread per accelerare la detection
            # - quad_decimate: fattore di riduzione della risoluzione per velocizzare l'analisi
            try:
                self.detector = Detector(
                    families=tag_family,
                    nthreads=threads,
                    quad_decimate=decimate,
                )
            except Exception as exc:
                raise RuntimeError("Impossibile inizializzare il detector AprilTag.") from exc

            # Messaggio informativo utile in fase di debug o monitoraggio dell'avvio.
            LOGGER.info(
                "AprilTag inizializzato | family=%s | tag_size=%.4f m",
                tag_family,
                self.tag_size_m,
            )

    @staticmethod
    def _get_detection_value(det, key, default=None):
        """
        Accesso robusto agli attributi delle detection pyapriltags.
        """
        # Restituisce l'attributo richiesto dell'oggetto detection, se presente;
        # in caso contrario, restituisce un valore di default.
        # Questo approccio rende il codice più robusto rispetto a possibili differenze
        # tra versioni o configurazioni del package pyapriltags.
        return getattr(det, key, default)

    def undistort_frame(self, frame):
        """
        Corregge la distorsione dell'immagine usando i parametri di calibrazione.
        """
        # In assenza di immagine in ingresso, il metodo non tenta alcuna elaborazione.
        if frame is None:
            return None

        # Restituisce una versione dell'immagine corretta rispetto alla distorsione ottica.
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def _draw_pose_text(self, frame, x, y, z, tag_id=None, line_index=0):
        """Disegna le coordinate della camera nel frame del tag."""
        # Se disponibile, l'identificativo del tag viene incluso nel testo sovraimpresso.
        prefix = f"Tag {tag_id} | " if tag_id is not None else ""

        # Le coordinate vengono formattate con due cifre decimali e unità di misura in metri.
        text = f"{prefix}X={x:.2f} Y={y:.2f} Z={z:.2f} m"

        # Ogni detection viene mostrata su una riga diversa, in modo da evitare sovrapposizioni.
        y_px = 40 + (35 * line_index)

        cv2.putText(
            frame,
            text,
            (20, y_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def _draw_axes(self, frame, R, t):
        """
        Disegna gli assi 3D del tag.
        Il frame qui è già undistorto, quindi nella proiezione usiamo distCoeffs=None.
        """
        # Definizione di quattro punti 3D nel riferimento del tag:
        # - origine
        # - estremo asse X
        # - estremo asse Y
        # - estremo asse Z
        # La lunghezza di ciascun asse è pari a 5 cm.
        axis_3d = np.float32([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, -0.05],
        ])

        # Conversione della matrice di rotazione nella rappresentazione vettore di Rodrigues,
        # richiesta dalla funzione cv2.projectPoints.
        rvec, _ = cv2.Rodrigues(R)

        # Proiezione dei punti 3D sul piano immagine usando i parametri intrinseci della camera
        # e il vettore di traslazione stimato dal detector.
        imgpts, _ = cv2.projectPoints(
            axis_3d,
            rvec,
            t,
            self.camera_matrix,
            None,
        )

        # Conversione in coordinate pixel intere.
        imgpts = imgpts.reshape(-1, 2).astype(int)

        # Associazione semantica dei punti proiettati.
        origin = tuple(imgpts[0])
        x_axis = tuple(imgpts[1])
        y_axis = tuple(imgpts[2])
        z_axis = tuple(imgpts[3])

        # Disegno delle frecce che rappresentano gli assi cartesiani del tag:
        # rosso per X, verde per Y, blu per Z.
        cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)
        cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)
        cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)

    def _draw_tag_outline(self, frame, det):
        """Disegna contorno, centro e ID del tag."""
        # Recupero delle principali informazioni geometriche della detection.
        corners = self._get_detection_value(det, "corners")
        center = self._get_detection_value(det, "center")
        tag_id = self._get_detection_value(det, "tag_id", -1)

        # Se i vertici del tag sono disponibili, viene disegnato il contorno poligonale.
        if corners is not None:
            corners = np.asarray(corners).astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        # Se il centro del tag è disponibile, vengono evidenziati il centro stesso
        # e l'identificativo numerico del tag.
        if center is not None:
            center = tuple(np.asarray(center).astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(tag_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

    def process_frame(self, frame_for_detection, drawing_frame=None):
        """
        Cerca gli AprilTag nel frame e restituisce:
        - frame annotato
        - lista di pose stimate
        """
        # Se il modulo è disabilitato oppure il detector non è stato inizializzato,
        # il metodo restituisce il frame disponibile senza annotazioni e una lista vuota di pose.
        if not self.enabled or self.detector is None:
            if drawing_frame is not None:
                return drawing_frame, []
            return frame_for_detection, []

        # Gestione del caso in cui non sia presente un frame valido da analizzare.
        if frame_for_detection is None:
            if drawing_frame is not None:
                return drawing_frame, []
            return None, []

        # Il frame di output è una copia del frame destinato al disegno.
        # Se drawing_frame non è fornito, si annota direttamente una copia del frame usato per la detection.
        output_frame = drawing_frame.copy() if drawing_frame is not None else frame_for_detection.copy()

        # Conversione in scala di grigi, tipicamente richiesta dagli algoritmi di detection dei marker.
        gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY)

        try:
            # Esecuzione della detection e della stima di posa.
            # I parametri intrinseci della camera e la dimensione fisica del tag
            # consentono di ottenere una posa metrica completa.
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=self.tag_size_m,
            )
        except Exception:
            # In caso di errore durante la detection, si registra il traceback nel log
            # e si restituisce il frame corrente senza risultati.
            LOGGER.exception("Errore durante la detection AprilTag.")
            return output_frame, []

        # Lista dei risultati strutturati da restituire al chiamante.
        pose_results = []

        # Elaborazione di ciascun tag rilevato nel frame.
        for idx, det in enumerate(detections):
            try:
                # Estrazione della traslazione e della rotazione stimate dal detector.
                pose_t = self._get_detection_value(det, "pose_t")
                pose_R = self._get_detection_value(det, "pose_R")
                if pose_t is None or pose_R is None:
                    raise ValueError("Detection AprilTag senza pose_t o pose_R.")

                # Normalizzazione del formato dei dati geometrici.
                t = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                R = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                tag_id = int(self._get_detection_value(det, "tag_id", -1))

                # Calcolo della posizione della camera nel sistema di riferimento del tag.
                # Se R e t descrivono la trasformazione tag -> camera, allora la posizione
                # della camera nel frame del tag è data da -R^T * t.
                camera_pos_tag = -R.T @ t
                x, y, z = camera_pos_tag.flatten()

                # Manteniamo la stessa convenzione del tuo script originale.
                # Questa inversione del segno sull'asse z rispetta la convenzione
                # utilizzata a livello applicativo dall'autore del codice.
                z = -z

                # Annotazione visiva del risultato sul frame di output.
                self._draw_pose_text(output_frame, x, y, z, tag_id=tag_id, line_index=idx)
                self._draw_axes(output_frame, R, t)
                self._draw_tag_outline(output_frame, det)
            except Exception:
                # Eventuali errori relativi a una singola detection non interrompono
                # l'elaborazione delle altre detection presenti nel frame.
                LOGGER.exception("Errore durante l'elaborazione della posa AprilTag.")
                continue

            # Salvataggio del risultato in una struttura dati facilmente riutilizzabile
            # da altri moduli del sistema, ad esempio per navigazione, logging o controllo.
            pose_results.append({
                "tag_id": tag_id,
                "camera_position_in_tag_frame": {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                },
                "translation_vector": tuple(float(v) for v in t.flatten()),
                "rotation_matrix": R.copy(),
            })

        # Restituisce il frame annotato e l'insieme delle pose stimate nel frame corrente.
        return output_frame, pose_results