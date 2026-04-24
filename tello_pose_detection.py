import logging
import math
from dataclasses import asdict, is_dataclass

# OpenCV viene utilizzato per la manipolazione dei frame, la correzione
# della distorsione, la proiezione di punti 3D e il disegno delle annotazioni.
import cv2

# NumPy è impiegato per la rappresentazione e la trasformazione numerica
# di vettori, matrici di rotazione, traslazioni e coordinate dei tag.
import numpy as np


# Logger di modulo utilizzato per registrare messaggi informativi ed errori
# relativi all'inizializzazione del detector e all'elaborazione dei frame.
LOGGER = logging.getLogger(__name__)


class CameraPoseEstimator:
    """
    Stima la posa globale della camera e, opzionalmente, del drone/body
    rispetto a uno o più AprilTag e disegna il risultato sul frame.

    Convenzioni adottate:
    - frame mondo: Z verso l'alto;
    - pose dei tag note a priori nel frame mondo;
    - yaw mostrato come rotazione attorno a Z del frame mondo;
    - se l'extrinseca camera->drone/body è identità, la posa del drone/body
      coincide con quella della camera (retrocompatibilità con il comportamento precedente).
    """

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        tag_family: str,
        threads: int,
        decimate: float,
        tag_size_m: float,
        tag_position=None,
        tag_orientation_rpy_deg=(0.0, 0.0, 0.0),
        world_tags=None,
        fusion_mode: str = "weighted_average",
        drone_extrinsics=None,
        enabled: bool = True,
    ):
        # Flag globale che abilita o disabilita l'intero stimatore.
        # Se False, l'oggetto viene comunque creato ma non esegue detection né stima.
        self.enabled = enabled

        # Parametri intrinseci della camera convertiti in array NumPy con tipo esplicito.
        # La matrice intrinseca deve avere dimensione 3x3.
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)

        # I coefficienti di distorsione vengono memorizzati come vettore colonna
        # per garantire compatibilità con le funzioni OpenCV usate nel seguito.
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)

        # Versione "nulla" dei coefficienti di distorsione, utile quando si lavora
        # con frame già corretti tramite undistortion.
        self.zero_dist_coeffs = np.zeros_like(self.dist_coeffs)

        # Dimensione fisica del lato del tag espressa in metri.
        self.tag_size_m = float(tag_size_m)

        # Verifica formale sui parametri di ingresso principali.
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix deve avere forma (3, 3), ricevuto {self.camera_matrix.shape}."
            )
        if self.tag_size_m <= 0:
            raise ValueError("tag_size_m deve essere maggiore di zero.")

        # Estrazione dei parametri focali e del punto principale.
        # Questi valori sono richiesti dal detector AprilTag per la stima della posa.
        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        # Costruzione della mappa dei tag nel frame mondo.
        # Il metodo supporta sia una configurazione completa multi-tag sia
        # una modalità di fallback retrocompatibile con il caso a singolo tag.
        self.world_tags = self._build_world_tags(
            world_tags=world_tags,
            fallback_tag_position=tag_position,
            fallback_tag_orientation_rpy_deg=tag_orientation_rpy_deg,
        )

        # Modalità di fusione delle ipotesi di posa ottenute dai diversi tag visibili.
        # - weighted_average: media pesata delle stime
        # - best_tag: selezione della stima associata al tag ritenuto più affidabile
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"weighted_average", "best_tag"}:
            raise ValueError(
                f"fusion_mode non valido: {fusion_mode}. Usa 'weighted_average' oppure 'best_tag'."
            )

        # Costruzione della trasformazione rigida tra camera e corpo del drone.
        # Questa informazione consente di derivare la posa del body a partire
        # dalla posa della camera.
        self.drone_extrinsics = self._build_drone_extrinsics(drone_extrinsics)

        # Variabili di stato aggiornate ad ogni frame processato.
        # Memorizzano sia i risultati per singolo tag sia le pose fuse finali.
        self.last_pose_results = []
        self.last_fused_camera_pose = None
        self.last_fused_body_pose = None

        # Il detector viene creato solo se il modulo è abilitato.
        self.detector = None

        if self.enabled:
            try:
                # Import locale: il pacchetto pyapriltags è richiesto solo quando
                # la stima della posa è effettivamente attivata.
                from pyapriltags import Detector
            except ImportError as exc:
                raise ImportError(
                    "Per usare la stima della posa devi installare pyapriltags."
                ) from exc

            try:
                # Inizializzazione del detector AprilTag con i parametri richiesti.
                # quad_decimate controlla il downsampling interno dell'immagine,
                # mentre nthreads determina il parallelismo usato dal detector.
                self.detector = Detector(
                    families=tag_family,
                    nthreads=threads,
                    quad_decimate=decimate,
                )
            except Exception as exc:
                raise RuntimeError("Impossibile inizializzare il detector AprilTag.") from exc

            LOGGER.info(
                "AprilTag inizializzato | family=%s | tag_size=%.4f m | world_tags=%d | fusion=%s | extrinsics_identity=%s",
                tag_family,
                self.tag_size_m,
                len(self.world_tags),
                self.fusion_mode,
                self.drone_extrinsics["is_identity"],
            )

    @staticmethod
    def _get_detection_value(det, key, default=None):
        # Accesso sicuro agli attributi dell'oggetto detection.
        # L'uso di getattr consente di gestire varianti nell'interfaccia
        # del detector senza generare eccezioni immediate.
        return getattr(det, key, default)

    @staticmethod
    def _matrix_to_serializable(matrix):
        # Conversione di una matrice NumPy in una struttura serializzabile
        # (lista di liste), utile per esportare i risultati in JSON o log.
        return np.asarray(matrix, dtype=np.float32).tolist()

    @staticmethod
    def _rotation_matrix_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
        # Conversione degli angoli di Eulero espressi in gradi in radianti.
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        yaw = math.radians(float(yaw_deg))

        # Calcolo dei termini trigonometrici elementari, riutilizzati nella
        # costruzione delle matrici di rotazione attorno agli assi principali.
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # Rotazione elementare attorno all'asse X (roll).
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float32,
        )

        # Rotazione elementare attorno all'asse Y (pitch).
        ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float32,
        )

        # Rotazione elementare attorno all'asse Z (yaw).
        rz = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Convenzione intrinseca roll-pitch-yaw equivalente a Rz(yaw) @ Ry(pitch) @ Rx(roll).
        return rz @ ry @ rx

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        # Riporta un angolo in gradi nell'intervallo (-180, 180].
        # Tale normalizzazione facilita il confronto e la visualizzazione dello yaw.
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        if wrapped == -180.0:
            return 180.0
        return wrapped

    def _normalize_world_tag_entry(self, tag_id, entry):
        # Supporto a configurazioni espresse come dataclass:
        # in tal caso si effettua una conversione preventiva in dizionario.
        if is_dataclass(entry):
            entry = asdict(entry)

        if not isinstance(entry, dict):
            raise ValueError(
                f"Configurazione del tag {tag_id} non valida: atteso dict o dataclass."
            )

        # La posizione del tag nel mondo può essere fornita con più chiavi,
        # per supportare differenti convenzioni di configurazione.
        position = entry.get("position_m", None)
        if position is None:
            position = entry.get("position", None)
        if position is None:
            position = entry.get("tag_position", None)

        # Anche l'orientazione del tag è accettata con chiavi alternative.
        orientation = entry.get("orientation_rpy_deg", None)
        if orientation is None:
            orientation = entry.get("rpy_deg", None)
        if orientation is None:
            orientation = entry.get("orientation_deg", None)

        if position is None:
            raise ValueError(f"Il tag {tag_id} non ha una posizione nel mondo.")
        if orientation is None:
            # Se l'orientazione non è fornita, si assume allineamento con il frame mondo.
            orientation = (0.0, 0.0, 0.0)

        # Normalizzazione numerica degli input.
        position_array = np.asarray(position, dtype=np.float32).reshape(-1)
        orientation_array = np.asarray(orientation, dtype=np.float32).reshape(-1)

        # Controlli dimensionali minimi: posizione 3D e terna roll-pitch-yaw.
        if position_array.size != 3:
            raise ValueError(f"Il tag {tag_id} deve avere 3 coordinate.")
        if orientation_array.size != 3:
            raise ValueError(f"Il tag {tag_id} deve avere 3 angoli roll-pitch-yaw.")

        # Rappresentazione della posizione come vettore colonna 3x1.
        position_vec = position_array.reshape(3, 1)

        # Matrice di rotazione che descrive l'orientazione del tag rispetto al mondo.
        rotation_world_from_tag = self._rotation_matrix_from_rpy_deg(*orientation_array.tolist())

        # Struttura interna standardizzata usata dal resto della classe.
        return {
            "tag_id": int(tag_id),
            "position_world": position_vec,
            "rotation_world_from_tag": rotation_world_from_tag,
            "orientation_rpy_deg": tuple(float(v) for v in orientation_array),
        }

    def _build_world_tags(
        self,
        world_tags,
        fallback_tag_position,
        fallback_tag_orientation_rpy_deg,
    ):
        # Dizionario finale contenente la configurazione normalizzata dei tag.
        normalized = {}

        # Caso principale: configurazione esplicita di uno o più tag nel mondo.
        if isinstance(world_tags, dict) and world_tags:
            for tag_id, entry in world_tags.items():
                normalized[int(tag_id)] = self._normalize_world_tag_entry(tag_id, entry)
            return normalized

        # Fallback: compatibilità con il vecchio caso a singolo tag.
        # Se è stata fornita una sola posizione, essa viene associata convenzionalmente al tag con ID 0.
        if fallback_tag_position is not None:
            normalized[0] = self._normalize_world_tag_entry(
                0,
                {
                    "position_m": fallback_tag_position,
                    "orientation_rpy_deg": fallback_tag_orientation_rpy_deg,
                },
            )

        return normalized

    def _build_drone_extrinsics(self, drone_extrinsics):
        # Se l'utente non fornisce extrinseche, si assume trasformazione identità.
        if drone_extrinsics is None:
            drone_extrinsics = {}

        # Anche in questo caso si supporta l'input come dataclass.
        if is_dataclass(drone_extrinsics):
            drone_extrinsics = asdict(drone_extrinsics)

        if not isinstance(drone_extrinsics, dict):
            raise ValueError("drone_extrinsics deve essere un dict o una dataclass.")

        # Recupero della posizione della camera nel frame del body/drone.
        # Sono gestite due possibili denominazioni della stessa informazione.
        camera_position_in_body = drone_extrinsics.get(
            "camera_position_in_drone_frame_m",
            drone_extrinsics.get("camera_position_in_body_frame_m", (0.0, 0.0, 0.0)),
        )

        # Recupero dell'orientazione della camera rispetto al body/drone.
        camera_orientation_in_body = drone_extrinsics.get(
            "camera_orientation_rpy_deg",
            drone_extrinsics.get("camera_orientation_rpy_deg_in_drone_frame", (0.0, 0.0, 0.0)),
        )

        # Conversione in array NumPy monodimensionali per i controlli successivi.
        p_body_camera = np.asarray(camera_position_in_body, dtype=np.float32).reshape(-1)
        rpy_body_camera = np.asarray(camera_orientation_in_body, dtype=np.float32).reshape(-1)

        if p_body_camera.size != 3:
            raise ValueError("camera_position_in_drone_frame_m deve avere 3 coordinate.")
        if rpy_body_camera.size != 3:
            raise ValueError("camera_orientation_rpy_deg deve avere 3 angoli roll-pitch-yaw.")

        # Posizione della camera nel body come vettore colonna.
        p_body_camera = p_body_camera.reshape(3, 1)

        # Rotazione dal frame camera al frame body.
        R_body_from_camera = self._rotation_matrix_from_rpy_deg(*rpy_body_camera.tolist())

        # Inversione della trasformazione T_body_camera per ottenere T_camera_body.
        # Tale trasformazione è necessaria quando, nota la posa della camera nel mondo,
        # si vuole ricavare la posa del body nel mondo.
        R_camera_from_body = R_body_from_camera.T
        p_camera_body = -R_camera_from_body @ p_body_camera

        # Controllo utile per sapere se la trasformazione è, di fatto, l'identità.
        is_identity = np.allclose(p_body_camera, 0.0) and np.allclose(R_body_from_camera, np.eye(3), atol=1e-6)

        return {
            "camera_position_in_body": p_body_camera,
            "rotation_body_from_camera": R_body_from_camera,
            "body_position_in_camera": p_camera_body,
            "rotation_camera_from_body": R_camera_from_body,
            "is_identity": bool(is_identity),
        }

    def undistort_frame(self, frame):
        # Corregge la distorsione del frame usando i parametri intrinseci della camera.
        # Se il frame è assente, la funzione mantiene un comportamento neutro restituendo None.
        if frame is None:
            return None
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def _draw_axes(self, frame, R, t, frame_is_undistorted: bool = False):
        # Definizione di un piccolo sistema di assi 3D nel frame del tag.
        # Le lunghezze sono espresse in metri e sono scelte unicamente a scopo visuale.
        axis_3d = np.float32(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, -0.05],
            ]
        )

        # Se l'immagine è già stata undistorta, per la proiezione si usano coefficienti nulli.
        dist_coeffs = self.zero_dist_coeffs if frame_is_undistorted else self.dist_coeffs

        # Conversione della matrice di rotazione in vettore di Rodrigues,
        # formato richiesto da cv2.projectPoints.
        rvec, _ = cv2.Rodrigues(R)

        # Proiezione dei punti 3D sul piano immagine.
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, t, self.camera_matrix, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        # Estrazione dei punti proiettati corrispondenti all'origine e agli assi.
        origin = tuple(imgpts[0])
        x_axis = tuple(imgpts[1])
        y_axis = tuple(imgpts[2])
        z_axis = tuple(imgpts[3])

        # Disegno degli assi con convenzione cromatica standard:
        # X rosso, Y verde, Z blu.
        cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)
        cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)
        cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)

    def _draw_tag_outline(self, frame, det):
        # Recupero delle informazioni geometriche di base prodotte dal detector.
        corners = self._get_detection_value(det, "corners")
        center = self._get_detection_value(det, "center")
        tag_id = self._get_detection_value(det, "tag_id", -1)

        if corners is not None:
            # Conversione dei vertici del tag in formato compatibile con OpenCV.
            corners = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            if len(corners) >= 4:
                # Disegno del contorno del tag rilevato.
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        if center is not None:
            # Disegno del centro del tag e del relativo identificativo.
            center_xy = np.asarray(center, dtype=np.int32).reshape(-1)
            if center_xy.size >= 2:
                center_point = (int(center_xy[0]), int(center_xy[1]))
                cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(tag_id),
                    center_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

    @staticmethod
    def _compute_detection_weight(det, t_vec):
        # Più il tag è vicino e più il detector è "sicuro", maggiore è il peso.
        # Il peso combina due elementi:
        # 1) decision_margin del detector, indicatore qualitativo della detection;
        # 2) distanza stimata dal tag, usata per penalizzare osservazioni lontane.
        decision_margin = getattr(det, "decision_margin", None)
        if decision_margin is None:
            decision_margin = 50.0

        try:
            decision_margin = float(decision_margin)
        except (TypeError, ValueError):
            decision_margin = 50.0

        if not math.isfinite(decision_margin):
            decision_margin = 50.0

        # La distanza è ricavata dalla norma della traslazione tag->camera.
        distance = float(np.linalg.norm(t_vec))
        if not math.isfinite(distance):
            distance = 1.0

        # Soglia minima per evitare pesi eccessivamente grandi in caso di distanza quasi nulla.
        distance = max(distance, 0.05)

        return max(decision_margin, 1.0) / distance

    @staticmethod
    def _extract_world_yaw_deg(R_world_from_local, local_forward_axis: str = "x"):
        # Estrae lo yaw come angolo, nel piano XY del mondo, della direzione
        # "forward" del frame locale.
        #
        # Convenzione adottata:
        # - camera: forward = asse +Z (asse ottico);
        # - drone/body: forward = asse +X, convenzione più comune in robotica.
        axis_name = str(local_forward_axis).strip().lower()
        axis_map = {
            "x": np.array([[1.0], [0.0], [0.0]], dtype=np.float32),
            "y": np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
            "z": np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
        }

        if axis_name not in axis_map:
            raise ValueError(
                f"Asse forward non valido: {local_forward_axis}. Usa 'x', 'y' oppure 'z'."
            )

        # Vettore dell'asse "forward" locale espresso nel frame mondo.
        local_forward_world = R_world_from_local @ axis_map[axis_name]

        # Lo yaw è l'angolo del vettore proiettato sul piano XY.
        yaw_rad = math.atan2(
            float(local_forward_world[1, 0]),
            float(local_forward_world[0, 0]),
        )
        return math.degrees(yaw_rad)

    @staticmethod
    def _pose_to_serializable(position_world, yaw_world_deg, source, source_tag_ids, pose_type):
        # Converte una posa in una struttura serializzabile, separando esplicitamente
        # le componenti di posizione, lo yaw e la provenienza della stima.
        return {
            "type": pose_type,
            "position_world": {
                "x": float(position_world[0, 0]),
                "y": float(position_world[1, 0]),
                "z": float(position_world[2, 0]),
            },
            "yaw_world_deg": float(yaw_world_deg),
            "source": str(source),
            "source_tag_ids": [int(tag_id) for tag_id in source_tag_ids],
        }

    def _draw_per_tag_pose_text(self, frame, tag_id, position_world, yaw_world_deg, line_index, label="Drone"):
        # Disegno testuale, per ogni tag rilevato, della posa stimata del soggetto
        # scelto (di default il drone/body) nel frame mondo.
        x, y, z = position_world.flatten()
        text = f"Tag {tag_id} -> {label} X={x:.2f} Y={y:.2f} Z={z:.2f} m | yaw={yaw_world_deg:.1f} deg"
        y_px = 280 + (35 * line_index)

        cv2.putText(
            frame,
            text,
            (20, y_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )

    def _draw_global_pose_text(self, frame, fused_pose, visible_count, label="Drone"):
        # Se non è disponibile alcuna posa assoluta fusa, viene mostrato un messaggio esplicito.
        if fused_pose is None:
            cv2.putText(
                frame,
                f"Posa assoluta {label.lower()}: non disponibile",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 165, 255),
                2,
            )
            return

        # Estrazione dei dati da visualizzare in overlay.
        x, y, z = fused_pose["position_world"].flatten()
        yaw_deg = fused_pose["yaw_world_deg"]
        source = fused_pose["source"]
        tag_ids = ",".join(str(tag_id) for tag_id in fused_pose["source_tag_ids"])

        # Le informazioni sono suddivise in tre righe:
        # posizione, orientazione/fusione, identificativi dei tag usati.
        lines = [
            f"{label} mondo X={x:.2f} Y={y:.2f} Z={z:.2f} m",
            f"{label} yaw={yaw_deg:.1f} deg | fusion={source} | tag visibili={visible_count}",
            f"Tag usati: {tag_ids}",
        ]

        y_px = 170
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y_px),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 0),
                2,
            )
            y_px += 30

    def _fuse_absolute_world_pose(self, hypotheses):
        # Se non esistono ipotesi di posa, non è possibile effettuare alcuna fusione.
        if not hypotheses:
            return None

        # In presenza di una sola ipotesi, oppure se la modalità selezionata è "best_tag",
        # si restituisce direttamente la posa associata al peso massimo.
        if len(hypotheses) == 1 or self.fusion_mode == "best_tag":
            best = max(hypotheses, key=lambda item: item["weight"])
            return {
                "position_world": best["position_world"].copy(),
                "yaw_world_deg": float(best["yaw_world_deg"]),
                "source": "best_tag",
                "source_tag_ids": [int(best["tag_id"])],
            }

        # Somma dei pesi per la media pesata.
        total_weight = sum(item["weight"] for item in hypotheses)
        if total_weight <= 0:
            # Fallback robusto: se i pesi risultano degeneri, si usa una media uniforme.
            total_weight = float(len(hypotheses))
            for item in hypotheses:
                item["weight"] = 1.0

        # Fusione della posizione tramite media pesata vettoriale.
        position_world = (
            sum(item["weight"] * item["position_world"] for item in hypotheses) / total_weight
        )

        # Fusione dell'angolo di yaw tramite media circolare.
        # Si evita così il problema della discontinuità agli estremi ±180°.
        sin_sum = sum(
            item["weight"] * math.sin(math.radians(item["yaw_world_deg"]))
            for item in hypotheses
        )
        cos_sum = sum(
            item["weight"] * math.cos(math.radians(item["yaw_world_deg"]))
            for item in hypotheses
        )
        fused_yaw_world_deg = math.degrees(math.atan2(sin_sum, cos_sum))

        return {
            "position_world": position_world,
            "yaw_world_deg": self._wrap_angle_deg(fused_yaw_world_deg),
            "source": "weighted_average",
            "source_tag_ids": [int(item["tag_id"]) for item in hypotheses],
        }

    def process_frame(self, frame_for_detection, drawing_frame=None, frame_is_undistorted: bool = False):
        # Se lo stimatore è disabilitato o il detector non è disponibile,
        # si restituisce semplicemente il frame in ingresso senza risultati.
        if not self.enabled or self.detector is None:
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            if drawing_frame is not None:
                return drawing_frame, []
            return frame_for_detection, []

        # Gestione del caso limite in cui il frame non sia disponibile.
        if frame_for_detection is None:
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            if drawing_frame is not None:
                return drawing_frame, []
            return None, []

        # Il frame di output è una copia del frame dedicato al disegno, se fornito;
        # altrimenti viene usato il frame destinato alla detection.
        output_frame = (
            drawing_frame.copy()
            if drawing_frame is not None
            else frame_for_detection.copy()
        )

        # Il detector AprilTag opera sull'immagine in scala di grigi.
        gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY)

        try:
            # Esecuzione della detection e stima della posa relativa tag-camera.
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=self.tag_size_m,
            )
            if detections is None:
                detections = []
        except Exception:
            LOGGER.exception("Errore durante la detection AprilTag.")
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            return output_frame, []

        # Collezioni che conterranno:
        # - i risultati completi per singolo tag;
        # - le ipotesi di posa assoluta della camera;
        # - le ipotesi di posa assoluta del body/drone.
        pose_results = []
        absolute_camera_hypotheses = []
        absolute_body_hypotheses = []

        # Elaborazione indipendente di ciascun tag rilevato.
        for idx, det in enumerate(detections):
            try:
                # pose_t e pose_R rappresentano, rispettivamente, traslazione e rotazione
                # del tag nel frame della camera secondo la convenzione del detector.
                pose_t = self._get_detection_value(det, "pose_t")
                pose_R = self._get_detection_value(det, "pose_R")
                if pose_t is None or pose_R is None:
                    raise ValueError("Detection AprilTag senza pose_t o pose_R.")

                t_camera_from_tag = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                R_camera_from_tag = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                tag_id = int(self._get_detection_value(det, "tag_id", -1))

                # Inversione standard della posa per ottenere la camera nel frame del tag.
                R_tag_from_camera_std = R_camera_from_tag.T
                camera_pos_in_tag_std = -R_tag_from_camera_std @ t_camera_from_tag

                # Cambio di convenzione richiesto:
                # manteniamo X e Y come sono, ma invertiamo solo l'asse Z del frame tag.
                # La matrice S implementa tale riflessione rispetto al piano XY.
                S = np.diag([1.0, 1.0, -1.0]).astype(np.float32)

                camera_pos_in_tag = S @ camera_pos_in_tag_std
                R_tag_from_camera = S @ R_tag_from_camera_std @ S

                # Visualizzazione grafica del riferimento del tag e del suo contorno.
                self._draw_axes(
                    output_frame,
                    R_camera_from_tag,
                    t_camera_from_tag,
                    frame_is_undistorted=frame_is_undistorted,
                )
                self._draw_tag_outline(output_frame, det)

                # Struttura base del risultato per singolo tag.
                # In questa fase contiene la posa della camera nel frame locale del tag.
                per_tag_result = {
                    "tag_id": tag_id,
                    "camera_position_in_tag_frame": {
                        "x": float(camera_pos_in_tag[0, 0]),
                        "y": float(camera_pos_in_tag[1, 0]),
                        "z": float(camera_pos_in_tag[2, 0]),
                    },
                    "translation_vector": tuple(float(v) for v in t_camera_from_tag.flatten()),
                    "rotation_matrix": self._matrix_to_serializable(R_tag_from_camera),
                }

                # Se il tag rilevato non è noto nel frame mondo, non è possibile risalire
                # ad una posa assoluta; si conserva quindi solo l'informazione locale.
                world_tag = self.world_tags.get(tag_id)
                if world_tag is None:
                    pose_results.append(per_tag_result)
                    continue

                R_world_from_tag = world_tag["rotation_world_from_tag"]
                p_world_tag = world_tag["position_world"]

                # Composizione nel mondo usando la convenzione corretta del frame tag.
                # Si ottiene la posa assoluta della camera a partire da:
                # - posa del tag nel mondo
                # - posa della camera nel frame del tag
                p_world_camera = R_world_from_tag @ camera_pos_in_tag + p_world_tag
                R_world_from_camera = R_world_from_tag @ R_tag_from_camera

                # Lo yaw della camera è estratto considerando come asse forward l'asse ottico +Z.
                camera_yaw_world_deg = self._wrap_angle_deg(
                    self._extract_world_yaw_deg(R_world_from_camera, local_forward_axis="z")
                )

                # Assegnazione di un peso all'ipotesi, utile nella fase di fusione multi-tag.
                weight = self._compute_detection_weight(det, t_camera_from_tag)

                # Memorizzazione della posa assoluta della camera per il singolo tag.
                per_tag_result["camera_position_in_world_frame"] = {
                    "x": float(p_world_camera[0, 0]),
                    "y": float(p_world_camera[1, 0]),
                    "z": float(p_world_camera[2, 0]),
                }
                per_tag_result["camera_yaw_in_world_deg"] = float(camera_yaw_world_deg)
                per_tag_result["weight"] = float(weight)

                absolute_camera_hypotheses.append(
                    {
                        "tag_id": tag_id,
                        "position_world": p_world_camera,
                        "yaw_world_deg": camera_yaw_world_deg,
                        "weight": weight,
                    }
                )

                # Se disponibile l'extrinseca camera->drone/body, ricaviamo la posa del body nel mondo.
                p_camera_body = self.drone_extrinsics["body_position_in_camera"]
                R_camera_from_body = self.drone_extrinsics["rotation_camera_from_body"]

                p_world_body = R_world_from_camera @ p_camera_body + p_world_camera
                R_world_from_body = R_world_from_camera @ R_camera_from_body

                # Per il body si assume come asse forward l'asse +X, tipico dei frame robotici.
                body_yaw_world_deg = self._wrap_angle_deg(
                    self._extract_world_yaw_deg(R_world_from_body, local_forward_axis="x")
                )

                # Memorizzazione della posa assoluta del drone/body per il singolo tag.
                per_tag_result["drone_position_in_world_frame"] = {
                    "x": float(p_world_body[0, 0]),
                    "y": float(p_world_body[1, 0]),
                    "z": float(p_world_body[2, 0]),
                }
                per_tag_result["drone_yaw_in_world_deg"] = float(body_yaw_world_deg)

                pose_results.append(per_tag_result)
                absolute_body_hypotheses.append(
                    {
                        "tag_id": tag_id,
                        "position_world": p_world_body,
                        "yaw_world_deg": body_yaw_world_deg,
                        "weight": weight,
                    }
                )

                # Overlay testuale per la stima prodotta dal singolo tag.
                self._draw_per_tag_pose_text(
                    output_frame,
                    tag_id=tag_id,
                    position_world=p_world_body,
                    yaw_world_deg=body_yaw_world_deg,
                    line_index=idx,
                    label="Drone",
                )
            except Exception:
                # Un errore relativo ad un singolo tag non interrompe l'elaborazione
                # degli altri tag presenti nel frame.
                LOGGER.exception("Errore durante l'elaborazione della posa AprilTag.")
                continue

        # Fusione finale delle ipotesi assolute ottenute dai diversi tag visibili.
        fused_camera_pose = self._fuse_absolute_world_pose(absolute_camera_hypotheses)
        fused_body_pose = self._fuse_absolute_world_pose(absolute_body_hypotheses)

        # Aggiornamento dello stato interno dell'oggetto.
        self.last_fused_camera_pose = fused_camera_pose
        self.last_fused_body_pose = fused_body_pose

        # Si privilegia la visualizzazione della posa del drone/body.
        # Se non disponibile, si ripiega sulla posa della camera.
        pose_to_draw = fused_body_pose if fused_body_pose is not None else fused_camera_pose
        pose_label = "Drone" if fused_body_pose is not None else "Camera"
        visible_count = len(absolute_body_hypotheses) if fused_body_pose is not None else len(absolute_camera_hypotheses)

        self._draw_global_pose_text(
            output_frame,
            fused_pose=pose_to_draw,
            visible_count=visible_count,
            label=pose_label,
        )

        # Aggiunta della posa fusa della camera all'elenco finale dei risultati.
        if fused_camera_pose is not None:
            pose_results.append(
                self._pose_to_serializable(
                    position_world=fused_camera_pose["position_world"],
                    yaw_world_deg=fused_camera_pose["yaw_world_deg"],
                    source=fused_camera_pose["source"],
                    source_tag_ids=fused_camera_pose["source_tag_ids"],
                    pose_type="fused_camera_pose_world",
                )
            )

        # Aggiunta della posa fusa del drone/body all'elenco finale dei risultati.
        if fused_body_pose is not None:
            pose_results.append(
                self._pose_to_serializable(
                    position_world=fused_body_pose["position_world"],
                    yaw_world_deg=fused_body_pose["yaw_world_deg"],
                    source=fused_body_pose["source"],
                    source_tag_ids=fused_body_pose["source_tag_ids"],
                    pose_type="fused_drone_pose_world",
                )
            )

        # Memorizzazione dell'output completo dell'ultimo frame processato.
        self.last_pose_results = pose_results
        return output_frame, pose_results