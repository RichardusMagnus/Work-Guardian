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
    Stima la posa della camera/drone rispetto a uno o più AprilTag e disegna il risultato sul frame.

    Convenzioni adottate:
    - frame mondo: Z verso l'alto;
    - pose dei tag note a priori nel frame mondo;
    - yaw del drone/camera mostrato come rotazione attorno a Z del frame mondo;
    - il drone è assunto allineato alla camera sul piano orizzontale.
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
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32).reshape(-1, 1)
        self.tag_size_m = float(tag_size_m)

        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix deve avere forma (3, 3), ricevuto {self.camera_matrix.shape}."
            )
        if self.tag_size_m <= 0:
            raise ValueError("tag_size_m deve essere maggiore di zero.")

        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        self.world_tags = self._build_world_tags(
            world_tags=world_tags,
            fallback_tag_position=tag_position,
            fallback_tag_orientation_rpy_deg=tag_orientation_rpy_deg,
        )

        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"weighted_average", "best_tag"}:
            raise ValueError(
                f"fusion_mode non valido: {fusion_mode}. Usa 'weighted_average' oppure 'best_tag'."
            )

        self.detector = None

        if self.enabled:
            try:
                from pyapriltags import Detector
            except ImportError as exc:
                raise ImportError(
                    "Per usare la stima della posa devi installare pyapriltags."
                ) from exc

            try:
                self.detector = Detector(
                    families=tag_family,
                    nthreads=threads,
                    quad_decimate=decimate,
                )
            except Exception as exc:
                raise RuntimeError("Impossibile inizializzare il detector AprilTag.") from exc

            LOGGER.info(
                "AprilTag inizializzato | family=%s | tag_size=%.4f m | world_tags=%d | fusion=%s",
                tag_family,
                self.tag_size_m,
                len(self.world_tags),
                self.fusion_mode,
            )

    @staticmethod
    def _get_detection_value(det, key, default=None):
        return getattr(det, key, default)

    @staticmethod
    def _rotation_matrix_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
        roll = math.radians(float(roll_deg))
        pitch = math.radians(float(pitch_deg))
        yaw = math.radians(float(yaw_deg))

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float32,
        )
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
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        if wrapped == -180.0:
            return 180.0
        return wrapped

    def _normalize_world_tag_entry(self, tag_id, entry):
        if is_dataclass(entry):
            entry = asdict(entry)

        if not isinstance(entry, dict):
            raise ValueError(
                f"Configurazione del tag {tag_id} non valida: atteso dict o dataclass."
            )

        position = entry.get("position_m") or entry.get("position") or entry.get("tag_position")
        orientation = (
            entry.get("orientation_rpy_deg")
            or entry.get("rpy_deg")
            or entry.get("orientation_deg")
        )

        if position is None:
            raise ValueError(f"Il tag {tag_id} non ha una posizione nel mondo.")
        if orientation is None:
            orientation = (0.0, 0.0, 0.0)

        position_array = np.asarray(position, dtype=np.float32).reshape(-1)
        orientation_array = np.asarray(orientation, dtype=np.float32).reshape(-1)

        if position_array.size != 3:
            raise ValueError(f"Il tag {tag_id} deve avere 3 coordinate.")
        if orientation_array.size != 3:
            raise ValueError(f"Il tag {tag_id} deve avere 3 angoli roll-pitch-yaw.")

        position_vec = position_array.reshape(3, 1)
        rotation_world_from_tag = self._rotation_matrix_from_rpy_deg(*orientation_array.tolist())

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
        normalized = {}

        if isinstance(world_tags, dict) and world_tags:
            for tag_id, entry in world_tags.items():
                normalized[int(tag_id)] = self._normalize_world_tag_entry(tag_id, entry)
            return normalized

        # Fallback: compatibilità con il vecchio caso a singolo tag.
        if fallback_tag_position is not None:
            normalized[0] = self._normalize_world_tag_entry(
                0,
                {
                    "position_m": fallback_tag_position,
                    "orientation_rpy_deg": fallback_tag_orientation_rpy_deg,
                },
            )

        return normalized

    def undistort_frame(self, frame):
        if frame is None:
            return None
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def _draw_axes(self, frame, R, t):
        axis_3d = np.float32(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, -0.05],
            ]
        )

        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, t, self.camera_matrix, None)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        origin = tuple(imgpts[0])
        x_axis = tuple(imgpts[1])
        y_axis = tuple(imgpts[2])
        z_axis = tuple(imgpts[3])

        cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 2)
        cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 2)
        cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 2)

    def _draw_tag_outline(self, frame, det):
        corners = self._get_detection_value(det, "corners")
        center = self._get_detection_value(det, "center")
        tag_id = self._get_detection_value(det, "tag_id", -1)

        if corners is not None:
            corners = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            if len(corners) >= 4:
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        if center is not None:
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
        decision_margin = getattr(det, "decision_margin", None)
        if decision_margin is None:
            decision_margin = 50.0

        distance = float(np.linalg.norm(t_vec))
        distance = max(distance, 0.05)

        return max(float(decision_margin), 1.0) / distance

    @staticmethod
    def _extract_world_yaw_deg(R_world_from_camera):
        # Assumiamo che la direzione "forward" del drone coincida con l'asse ottico della camera.
        camera_forward_world = R_world_from_camera @ np.array(
            [[0.0], [0.0], [1.0]],
            dtype=np.float32,
        )

        yaw_rad = math.atan2(
            float(camera_forward_world[1, 0]),
            float(camera_forward_world[0, 0]),
        )
        return math.degrees(yaw_rad)

    def _draw_per_tag_pose_text(self, frame, tag_id, position_world, yaw_world_deg, line_index):
        x, y, z = position_world.flatten()
        text = f"Tag {tag_id} -> Mondo X={x:.2f} Y={y:.2f} Z={z:.2f} m | yaw={yaw_world_deg:.1f} deg"
        y_px = 170 + (35 * line_index)

        cv2.putText(
            frame,
            text,
            (20, y_px),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
        )

    def _draw_global_pose_text(self, frame, fused_pose, visible_count):
        if fused_pose is None:
            cv2.putText(
                frame,
                "Posa assoluta drone: non disponibile",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 165, 255),
                2,
            )
            return

        x, y, z = fused_pose["position_world"].flatten()
        yaw_deg = fused_pose["yaw_world_deg"]
        source = fused_pose["source"]
        tag_ids = ",".join(str(tag_id) for tag_id in fused_pose["source_tag_ids"])

        lines = [
            f"Drone mondo X={x:.2f} Y={y:.2f} Z={z:.2f} m",
            f"Drone yaw={yaw_deg:.1f} deg | fusion={source} | tag visibili={visible_count}",
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

    def _fuse_camera_world_pose(self, hypotheses):
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

    def process_frame(self, frame_for_detection, drawing_frame=None):
        if not self.enabled or self.detector is None:
            if drawing_frame is not None:
                return drawing_frame, []
            return frame_for_detection, []

        if frame_for_detection is None:
            if drawing_frame is not None:
                return drawing_frame, []
            return None, []

        output_frame = (
            drawing_frame.copy()
            if drawing_frame is not None
            else frame_for_detection.copy()
        )

        gray = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY)

        try:
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=self.tag_size_m,
            )
        except Exception:
            LOGGER.exception("Errore durante la detection AprilTag.")
            return output_frame, []

        pose_results = []
        absolute_hypotheses = []

        for idx, det in enumerate(detections):
            try:
                pose_t = self._get_detection_value(det, "pose_t")
                pose_R = self._get_detection_value(det, "pose_R")
                if pose_t is None or pose_R is None:
                    raise ValueError("Detection AprilTag senza pose_t o pose_R.")

                t_camera_from_tag = np.asarray(pose_t, dtype=np.float32).reshape(3, 1)
                R_camera_from_tag = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
                tag_id = int(self._get_detection_value(det, "tag_id", -1))

                # Inversione della posa per ottenere camera nel frame del tag.
                R_tag_from_camera = R_camera_from_tag.T
                camera_pos_in_tag = -R_camera_from_tag.T @ t_camera_from_tag

                self._draw_axes(output_frame, R_camera_from_tag, t_camera_from_tag)
                self._draw_tag_outline(output_frame, det)

                per_tag_result = {
                    "tag_id": tag_id,
                    "camera_position_in_tag_frame": {
                        "x": float(camera_pos_in_tag[0, 0]),
                        "y": float(camera_pos_in_tag[1, 0]),
                        "z": float(camera_pos_in_tag[2, 0]),
                    },
                    "translation_vector": tuple(float(v) for v in t_camera_from_tag.flatten()),
                    "rotation_matrix": R_camera_from_tag.copy(),
                }

                world_tag = self.world_tags.get(tag_id)
                if world_tag is None:
                    pose_results.append(per_tag_result)
                    continue

                R_world_from_tag = world_tag["rotation_world_from_tag"]
                p_world_tag = world_tag["position_world"]

                # Composizione: T_world_camera = T_world_tag * T_tag_camera
                p_world_camera = R_world_from_tag @ camera_pos_in_tag + p_world_tag
                R_world_from_camera = R_world_from_tag @ R_tag_from_camera

                yaw_world_deg = self._wrap_angle_deg(
                    self._extract_world_yaw_deg(R_world_from_camera)
                )

                weight = self._compute_detection_weight(det, t_camera_from_tag)

                per_tag_result["camera_position_in_world_frame"] = {
                    "x": float(p_world_camera[0, 0]),
                    "y": float(p_world_camera[1, 0]),
                    "z": float(p_world_camera[2, 0]),
                }
                per_tag_result["camera_yaw_in_world_deg"] = float(yaw_world_deg)
                per_tag_result["weight"] = float(weight)

                pose_results.append(per_tag_result)
                absolute_hypotheses.append(
                    {
                        "tag_id": tag_id,
                        "position_world": p_world_camera,
                        "yaw_world_deg": yaw_world_deg,
                        "weight": weight,
                    }
                )

                self._draw_per_tag_pose_text(
                    output_frame,
                    tag_id=tag_id,
                    position_world=p_world_camera,
                    yaw_world_deg=yaw_world_deg,
                    line_index=idx,
                )
            except Exception:
                LOGGER.exception("Errore durante l'elaborazione della posa AprilTag.")
                continue

        fused_pose = self._fuse_camera_world_pose(absolute_hypotheses)
        self._draw_global_pose_text(
            output_frame,
            fused_pose=fused_pose,
            visible_count=len(absolute_hypotheses),
        )

        if fused_pose is not None:
            pose_results.append(
                {
                    "type": "fused_camera_pose_world",
                    "position_world": {
                        "x": float(fused_pose["position_world"][0, 0]),
                        "y": float(fused_pose["position_world"][1, 0]),
                        "z": float(fused_pose["position_world"][2, 0]),
                    },
                    "yaw_world_deg": float(fused_pose["yaw_world_deg"]),
                    "source": fused_pose["source"],
                    "source_tag_ids": list(fused_pose["source_tag_ids"]),
                }
            )

        return output_frame, pose_results