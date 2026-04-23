import logging
import math
from dataclasses import asdict, is_dataclass

# Io non sono tuo padre: sono tua madre.

# OpenCV viene utilizzato per la manipolazione dei frame, la correzione
# della distorsione, la proiezione di punti 3D e il disegno delle annotazioni.
import cv2

# NumPy è impiegato per la rappresentazione e la trasformazione numerica
# di vettori, matrici di rotazione, traslazioni e coordinate dei tag.
import numpy as np


# Logger di modulo utilizzato per registrare messaggi informativi ed errori
# relativi all'inizializzazione del detector e all'elaborazione dei frame.
LOGGER = logging.getLogger(__name__)







####################################################################################
## APRIL TAG DETECTOR ##############################################################
####################################################################################
class AprilTagDetector:
    """
    Gestisce la detection e undistortion degli AprilTag.
    """

    def __init__(self, camera_matrix, dist_coeffs, tag_family: str, threads: int, decimate: float, tag_size_m: float):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)
        self.zero_dist_coeffs = np.zeros_like(self.dist_coeffs)
        self.tag_size_m = float(tag_size_m)
        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix deve avere forma (3, 3), ricevuto {self.camera_matrix.shape}.")
        if self.tag_size_m <= 0:
            raise ValueError("tag_size_m deve essere maggiore di zero.")

        try:
            from pyapriltags import Detector
        except ImportError as exc:
            raise ImportError("Per usare il detector devi installare pyapriltags.") from exc

        try:
            self.detector = Detector(families=tag_family, nthreads=threads, quad_decimate=decimate)
        except Exception as exc:
            raise RuntimeError("Impossibile inizializzare il detector AprilTag.") from exc

        LOGGER.info("AprilTag detector inizializzato | family=%s | tag_size=%.4f m", tag_family, self.tag_size_m)

    @staticmethod
    def _get_detection_value(det, key, default=None):
        return getattr(det, key, default)

    def undistort_frame(self, frame):
        if frame is None:
            return None
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def detect(self, frame):
        """
        Esegue la detection degli AprilTag su un frame, restituendo una lista di oggetti detection.
         Ogni oggetto detection contiene almeno i seguenti attributi:
         - tag_id: identificativo del tag rilevato
         - corners: coordinate dei 4 vertici del tag nel piano immagine
         - center: coordinate del centro del tag nel piano immagine
         - pose_t: vettore di traslazione dalla camera al tag (3x1)
         - pose_R: matrice di rotazione dalla camera al tag (3x3)
        Se il frame è assente o se si verifica un errore durante la detection, viene restituita una lista vuota.
        """
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=self.tag_size_m,
            )
            return detections if detections is not None else []
        except Exception:
            LOGGER.warning("Errore durante la detection AprilTag.")
            return []


####################################################################################
## DRONE POSE ESTIMATOR ############################################################
####################################################################################
class DronePoseEstimator:
    """
    Gestisce la stima della posa assoluta del drone a partire da detection AprilTag.
    """

    def __init__(self, world_tags, drone_extrinsics, fusion_mode: str = "weighted_average"):
        self.world_tags = world_tags
        self.drone_extrinsics = drone_extrinsics
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"weighted_average", "best_tag"}:
            raise ValueError(f"fusion_mode non valido: {fusion_mode}. Usa 'weighted_average' oppure 'best_tag'.")

    @staticmethod
    def _compute_detection_weight(det, t_vec):
        """
        Calcola un peso di affidabilità per una singola detection basato sulla distanza stimata.
         La formula usata è: weight = decision_margin / max(distance, 0.05)
         dove decision_margin è un parametro opzionale associato alla detection (default 50.0)
         e distance è la norma del vettore di traslazione t_vec dalla camera al tag.
         """
        decision_margin = getattr(det, "decision_margin", None)
        if decision_margin is None:
            decision_margin = 50.0
        try:
            decision_margin = float(decision_margin)
        except (TypeError, ValueError):
            decision_margin = 50.0
        if not math.isfinite(decision_margin):
            decision_margin = 50.0
        distance = float(np.linalg.norm(t_vec))
        if not math.isfinite(distance):
            distance = 1.0
        distance = max(distance, 0.05)
        return max(decision_margin, 1.0) / distance

    @staticmethod
    def _extract_world_yaw_deg(R_world_from_local, local_forward_axis: str = "x"):
        """
        Estrae lo yaw in gradi da una matrice di rotazione che rappresenta l'orientamento di un oggetto
         rispetto al frame mondo. L'asse locale considerato come "forward" può essere specificato tramite local_forward_axis,
         che accetta i valori 'x', 'y' o 'z'. La funzione restituisce l'angolo di yaw normalizzato nell'intervallo (-180, 180].
        """
        axis_name = str(local_forward_axis).strip().lower()
        axis_map = {
            "x": np.array([[1.0], [0.0], [0.0]], dtype=np.float32),
            "y": np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
            "z": np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
        }
        if axis_name not in axis_map:
            raise ValueError(f"Asse forward non valido: {local_forward_axis}. Usa 'x', 'y' oppure 'z'.")
        local_forward_world = R_world_from_local @ axis_map[axis_name]
        yaw_rad = math.atan2(float(local_forward_world[1, 0]), float(local_forward_world[0, 0]))
        return math.degrees(yaw_rad)

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        if wrapped == -180.0:
            return 180.0
        return wrapped
    




    def _fuse_absolute_world_pose(self, hypotheses):
        """
        Fonde le ipotesi di posa assoluta del drone ottenute da più tag visibili in un'unica stima globale.
         Ogni ipotesi è un dizionario con chiavi:
         - tag_id: identificativo del tag che ha generato l'ipotesi
         - position_world: posizione del drone nel frame mondo
         - yaw_world_deg: yaw del drone nel frame mondo
         - weight: peso di affidabilità dell'ipotesi
         La funzione restituisce un dizionario con la posa globale stimata e le informazioni sui tag usati per la fusione.
         Se non ci sono ipotesi valide, restituisce None.
         La fusione può essere eseguita in due modalità:
         - weighted_average: media pesata delle posizioni e media circolare pesata degli yaw
         - best_tag: selezione dell'ipotesi associata al tag con il peso più alto
         """
        if not hypotheses:
            return None
        if len(hypotheses) == 1 or self.fusion_mode == "best_tag":
            best = max(hypotheses, key=lambda item: item["weight"])
            return {
                "position_world": best["position_world"].copy(),
                "yaw_world_deg": float(best["yaw_world_deg"]),
                "source": "best_tag",
                "source_tag_ids": [int(best["tag_id"])],
            }
        total_weight = sum(item["weight"] for item in hypotheses)
        if total_weight <= 0:
            total_weight = float(len(hypotheses))
            for item in hypotheses:
                item["weight"] = 1.0
        position_world = (
            sum(item["weight"] * item["position_world"] for item in hypotheses) / total_weight
        )
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
    


    

    def estimate_drone_pose(self, detections):
        """
        Stima la posa assoluta del drone nel frame mondo a partire da una lista di detection AprilTag.
         Per ogni detection, se il tag è presente nella configurazione world_tags, viene calcolata un'ipotesi di posa del drone.
         L'ipotesi include la posizione e lo yaw del drone nel frame mondo, insieme a un peso di affidabilità basato sulla detection.
         Infine, tutte le ipotesi vengono fuse in un'unica stima globale usando il metodo specificato da fusion_mode.
         Se non ci sono tag visibili o se nessuna detection è valida, la funzione restituisce None.
         La stima della posa del drone tiene conto dell'extrinseca camera->drone definita in drone_extrinsics.
         Convenzioni:
         - frame mondo: Z verso l'alto;
         - pose dei tag note a priori nel frame mondo;
         - yaw mostrato come rotazione attorno a Z del frame mondo;
         - se l'extrinseca camera->drone è identità, la posa del drone coincide con quella della camera.
         """
        absolute_body_hypotheses = []
        for det in detections:
            try:
                pose_t = self._get_detection_value(det, "pose_t")
                pose_R = self._get_detection_value(det, "pose_R")
                if pose_t is None or pose_R is None:
                    continue
                t_camera_from_tag = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                R_camera_from_tag = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                tag_id = int(self._get_detection_value(det, "tag_id", -1))
                R_tag_from_camera_std = R_camera_from_tag.T
                camera_pos_in_tag_std = -R_tag_from_camera_std @ t_camera_from_tag
                S = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
                camera_pos_in_tag = S @ camera_pos_in_tag_std
                R_tag_from_camera = S @ R_tag_from_camera_std @ S
                world_tag = self.world_tags.get(tag_id)
                if world_tag is None:
                    continue
                R_world_from_tag = world_tag["rotation_world_from_tag"]
                p_world_tag = world_tag["position_world"]
                p_world_camera = R_world_from_tag @ camera_pos_in_tag + p_world_tag
                R_world_from_camera = R_world_from_tag @ R_tag_from_camera
                camera_yaw_world_deg = self._wrap_angle_deg(
                    self._extract_world_yaw_deg(R_world_from_camera, local_forward_axis="z")
                )
                weight = self._compute_detection_weight(det, t_camera_from_tag)
                p_camera_body = self.drone_extrinsics["body_position_in_camera"]
                R_camera_from_body = self.drone_extrinsics["rotation_camera_from_body"]
                p_world_body = R_world_from_camera @ p_camera_body + p_world_camera
                R_world_from_body = R_world_from_camera @ R_camera_from_body
                body_yaw_world_deg = self._wrap_angle_deg(
                    self._extract_world_yaw_deg(R_world_from_body, local_forward_axis="x")
                )
                absolute_body_hypotheses.append(
                    {
                        "tag_id": tag_id,
                        "position_world": p_world_body,
                        "yaw_world_deg": body_yaw_world_deg,
                        "weight": weight,
                    }
                )
            except Exception:
                LOGGER.warning("Errore nell'elaborazione di un singolo tag.")
                continue
        return self._fuse_absolute_world_pose(absolute_body_hypotheses)


####################################################################################
## POSE VISUALIZER #################################################################
####################################################################################
class PoseVisualizer:
    """
    Gestisce il disegno degli overlay sul frame.
    """

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)
        self.zero_dist_coeffs = np.zeros_like(self.dist_coeffs)

    def _draw_axes(self, frame, R, t, frame_is_undistorted: bool = False):
        axis_3d = np.float32([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, -0.05],
        ])
        dist_coeffs = self.zero_dist_coeffs if frame_is_undistorted else self.dist_coeffs
        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, t, self.camera_matrix, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        origin = tuple(imgpts[0])
        x_axis = tuple(imgpts[1])
        y_axis = tuple(imgpts[2])
        z_axis = tuple(imgpts[3])
        cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)
        cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)
        cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)

    def _draw_tag_outline(self, frame, det):
        corners = getattr(det, "corners", None)
        center = getattr(det, "center", None)
        tag_id = getattr(det, "tag_id", -1)
        if corners is not None:
            corners = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            if len(corners) >= 4:
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        if center is not None:
            center_xy = np.asarray(center, dtype=np.int32).reshape(-1)
            if center_xy.size >= 2:
                center_point = (int(center_xy[0]), int(center_xy[1]))
                cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(tag_id), center_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def _draw_per_tag_pose_text(self, frame, tag_id, position_world, yaw_world_deg, line_index, label="Drone"):
        x, y, z = position_world.flatten()
        text = f"Tag {tag_id} -> {label} X={x:.2f} Y={y:.2f} Z={z:.2f} m | yaw={yaw_world_deg:.1f} deg"
        y_px = 280 + (35 * line_index)
        cv2.putText(frame, text, (20, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    def _draw_global_pose_text(self, frame, fused_pose, visible_count, label="Drone"):
        if fused_pose is None:
            cv2.putText(frame, f"Posa assoluta {label.lower()}: non disponibile", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2)
            return
        x, y, z = fused_pose["position_world"].flatten()
        yaw_deg = fused_pose["yaw_world_deg"]
        source = fused_pose["source"]
        tag_ids = ",".join(str(tag_id) for tag_id in fused_pose["source_tag_ids"])
        lines = [
            f"{label} mondo X={x:.2f} Y={y:.2f} Z={z:.2f} m",
            f"{label} yaw={yaw_deg:.1f} deg | fusion={source} | tag visibili={visible_count}",
            f"Tag usati: {tag_ids}",
        ]
        y_px = 170
        for line in lines:
            cv2.putText(frame, line, (20, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
            y_px += 30

    def draw_detections_and_pose(self, frame, detections, drone_pose, frame_is_undistorted=False):
        for idx, det in enumerate(detections):
            pose_t = getattr(det, "pose_t", None)
            pose_R = getattr(det, "pose_R", None)
            if pose_t is not None and pose_R is not None:
                t_camera_from_tag = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                R_camera_from_tag = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                self._draw_axes(frame, R_camera_from_tag, t_camera_from_tag, frame_is_undistorted)
            self._draw_tag_outline(frame, det)
            # Per ora, assumiamo che drone_pose abbia info per tag, ma semplifichiamo
            if drone_pose and "source_tag_ids" in drone_pose:
                tag_id = getattr(det, "tag_id", -1)
                if tag_id in drone_pose["source_tag_ids"]:
                    # Calcola posa per tag (semplificato, usa drone_pose come approx)
                    position_world = drone_pose["position_world"]
                    yaw_world_deg = drone_pose["yaw_world_deg"]
                    self._draw_per_tag_pose_text(frame, tag_id, position_world, yaw_world_deg, idx)
        visible_count = len(detections) if detections else 0
        self._draw_global_pose_text(frame, drone_pose, visible_count)








####################################################################################
## CAMERA POSE ESTIMATOR ###########################################################
####################################################################################
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

        # Costruzione della mappa dei tag nel frame mondo.
        # Il metodo supporta sia una configurazione completa multi-tag sia
        # una modalità di fallback retrocompatibile con il caso a singolo tag.
        world_tags_dict = self._build_world_tags(
            world_tags=world_tags,
            fallback_tag_position=tag_position,
            fallback_tag_orientation_rpy_deg=tag_orientation_rpy_deg,
        )

        # Costruzione della trasformazione rigida tra camera e corpo del drone.
        drone_extrinsics_dict = self._build_drone_extrinsics(drone_extrinsics)

        # Istanza delle classi specializzate
        self.apriltag_detector = AprilTagDetector(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            tag_family=tag_family,
            threads=threads,
            decimate=decimate,
            tag_size_m=tag_size_m,
        ) if enabled else None

        self.pose_estimator = DronePoseEstimator(
            world_tags=world_tags_dict,
            drone_extrinsics=drone_extrinsics_dict,
            fusion_mode=fusion_mode,
        ) if enabled else None

        self.visualizer = PoseVisualizer(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        ) if enabled else None

        # Variabili di stato per backward-compatibility
        self.last_pose_results = []
        self.last_fused_camera_pose = None
        self.last_fused_body_pose = None

        # Flag abilitazione
        self.enabled = enabled

        if enabled:
            LOGGER.info(
                "CameraPoseEstimator inizializzato con classi separate | fusion=%s | extrinsics_identity=%s",
                fusion_mode,
                drone_extrinsics_dict["is_identity"],
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
        # Delega alla classe specializzata se disponibile, altrimenti usa il codice legacy.
        if self.apriltag_detector:
            return self.apriltag_detector.undistort_frame(frame)
        # Corregge la distorsione del frame usando i parametri intrinseci della camera.
        # Se il frame è assente, la funzione mantiene un comportamento neutro restituendo None.
        if frame is None:
            return None
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


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


    def process_frame(self, frame_for_detection, drawing_frame=None, frame_is_undistorted: bool = False):
        """
        Elabora un frame per stimare la posa del drone e disegnare i risultati.
         - frame_for_detection: frame su cui eseguire la detection dei tag (può essere distorto o già corretto).
         - drawing_frame: frame su cui disegnare gli overlay (se None, si usa frame_for_detection).
         - frame_is_undistorted: indica se frame_for_detection è già stato corretto per la distorsione (in tal caso, si usano zero_dist_coeffs per la proiezione).
         La funzione restituisce una tupla (output_frame, pose_results) dove:
         - output_frame: frame con gli overlay disegnati.
         - pose_results: lista dei risultati della stima della posa.
         Se lo stimatore è disabilitato o se frame_for_detection è None, viene restituito un frame vuoto e una lista pose_results vuota."""
        # Se lo stimatore è disabilitato, restituisce frame vuoto.
        if not self.enabled or not self.apriltag_detector or not self.pose_estimator or not self.visualizer:
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            if drawing_frame is not None:
                return drawing_frame, []
            return frame_for_detection, []

        if frame_for_detection is None:
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            if drawing_frame is not None:
                return drawing_frame, []
            return None, []

        output_frame = (
            drawing_frame.copy()
            if drawing_frame is not None
            else frame_for_detection.copy()
        )

        try:
            # Detection
            detections = self.apriltag_detector.detect(frame_for_detection)

            # Stima posa drone
            drone_pose = self.pose_estimator.estimate_drone_pose(detections)

            # Disegno
            self.visualizer.draw_detections_and_pose(output_frame, detections, drone_pose, frame_is_undistorted)

            # Costruzione risultati per backward-compat
            pose_results = []
            for idx, det in enumerate(detections):
                try:
                    pose_t = getattr(det, "pose_t", None)
                    pose_R = getattr(det, "pose_R", None)
                    if pose_t is None or pose_R is None:
                        continue
                    t_camera_from_tag = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                    R_camera_from_tag = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                    tag_id = int(getattr(det, "tag_id", -1))
                    R_tag_from_camera_std = R_camera_from_tag.T
                    camera_pos_in_tag_std = -R_tag_from_camera_std @ t_camera_from_tag
                    S = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
                    camera_pos_in_tag = S @ camera_pos_in_tag_std
                    R_tag_from_camera = S @ R_tag_from_camera_std @ S

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

                    world_tag = self.pose_estimator.world_tags.get(tag_id)
                    if world_tag is not None:
                        R_world_from_tag = world_tag["rotation_world_from_tag"]
                        p_world_tag = world_tag["position_world"]
                        p_world_camera = R_world_from_tag @ camera_pos_in_tag + p_world_tag
                        R_world_from_camera = R_world_from_tag @ R_tag_from_camera
                        camera_yaw_world_deg = self.pose_estimator._wrap_angle_deg(
                            self.pose_estimator._extract_world_yaw_deg(R_world_from_camera, local_forward_axis="z")
                        )
                        per_tag_result["camera_position_in_world_frame"] = {
                            "x": float(p_world_camera[0, 0]),
                            "y": float(p_world_camera[1, 0]),
                            "z": float(p_world_camera[2, 0]),
                        }
                        per_tag_result["camera_yaw_in_world_deg"] = float(camera_yaw_world_deg)

                        p_camera_body = self.pose_estimator.drone_extrinsics["body_position_in_camera"]
                        R_camera_from_body = self.pose_estimator.drone_extrinsics["rotation_camera_from_body"]
                        p_world_body = R_world_from_camera @ p_camera_body + p_world_camera
                        R_world_from_body = R_world_from_camera @ R_camera_from_body
                        body_yaw_world_deg = self.pose_estimator._wrap_angle_deg(
                            self.pose_estimator._extract_world_yaw_deg(R_world_from_body, local_forward_axis="x")
                        )
                        per_tag_result["drone_position_in_world_frame"] = {
                            "x": float(p_world_body[0, 0]),
                            "y": float(p_world_body[1, 0]),
                            "z": float(p_world_body[2, 0]),
                        }
                        per_tag_result["drone_yaw_in_world_deg"] = float(body_yaw_world_deg)

                    pose_results.append(per_tag_result)
                except Exception:
                    LOGGER.warning("Errore nell'elaborazione di un singolo tag.")
                    continue

            # Aggiunta pose fuse
            if drone_pose is not None:
                pose_results.append(
                    self._pose_to_serializable(
                        position_world=drone_pose["position_world"],
                        yaw_world_deg=drone_pose["yaw_world_deg"],
                        source=drone_pose["source"],
                        source_tag_ids=drone_pose["source_tag_ids"],
                        pose_type="fused_drone_pose_world",
                    )
                )

            # Aggiornamento stato
            self.last_pose_results = pose_results
            self.last_fused_body_pose = drone_pose
            self.last_fused_camera_pose = None  # Non calcolato separatamente

            return output_frame, pose_results
        except Exception:
            LOGGER.exception("Errore in process_frame.")
            self.last_pose_results = []
            self.last_fused_camera_pose = None
            self.last_fused_body_pose = None
            return output_frame, []