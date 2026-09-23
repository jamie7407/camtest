"""Turns AprilTag detections from one camera frame into a robot pose + std devs.

- 2+ tags: one multi-tag solvePnP (SQPNP) using every corner of every visible tag.
- 1 tag:   IPPE gives two candidate poses; we keep the better one and compute
           ambiguity = best_error / other_error (same definition as PhotonVision).
- Std devs use 6328's heuristic: coeff * avgDistance^2 / tagCount^2 * cameraFactor.

Coordinate frames:
  OpenCV camera: x right, y down, z forward
  WPILib:        x forward, y left, z up (field is blue-origin)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Rotation3d, Transform3d, Translation3d

# Columns are the WPILib camera axes expressed in OpenCV camera coordinates.
_CV_TO_WPI = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
)


@dataclass
class DetectorSettings:
    family: str = "tag36h11"
    quad_decimate: float = 2.0
    num_threads: int = 2
    refine_edges: bool = True
    # WPILib's default is 300, which silently drops small/far tags once decimation
    # is on. 20 kept every tag a 1.0 decimation run found, at ~1/3 the CPU.
    min_cluster_pixels: int = 20


def make_detector(ds: DetectorSettings) -> AprilTagDetector:
    det = AprilTagDetector()
    det.addFamily(ds.family)
    cfg = det.getConfig()
    cfg.quadDecimate = ds.quad_decimate
    cfg.numThreads = ds.num_threads
    cfg.refineEdges = ds.refine_edges
    det.setConfig(cfg)
    q = det.getQuadThresholdParameters()
    q.minClusterPixels = ds.min_cluster_pixels
    det.setQuadThresholdParameters(q)
    return det


@dataclass
class SolverSettings:
    tag_size_m: float = 0.1651
    max_hamming: int = 0
    min_decision_margin: float = 25.0
    max_single_tag_ambiguity: float = 0.2
    max_tag_distance_m: float = 6.0
    field_border_margin_m: float = 0.5
    max_z_error_m: float = 0.75
    xy_coefficient: float = 0.01
    theta_coefficient: float = 0.03
    trust_single_tag_theta: bool = False


@dataclass
class PoseResult:
    robot_pose: Pose3d
    camera_pose: Pose3d
    tag_ids: list[int]
    avg_tag_distance: float
    ambiguity: float
    reprojection_error: float
    xy_std_dev: float
    theta_std_dev: float

    @property
    def tag_mask(self) -> int:
        mask = 0
        for tid in self.tag_ids:
            if 0 <= tid < 64:
                mask |= 1 << tid
        return mask


class PoseSolver:
    def __init__(self, layout: AprilTagFieldLayout, settings: SolverSettings):
        self.layout = layout
        self.s = settings
        self.last_reject: str | None = None  # why the last solve() returned None (for the stream overlay)
        self.field_length = layout.getFieldLength()
        self.field_width = layout.getFieldWidth()

        # Precompute field-frame corners for every tag, in the same order the
        # detector reports them (bottom-left, bottom-right, top-right, top-left
        # as seen by a viewer looking at the tag). Tag +X points out of the face.
        h = settings.tag_size_m / 2.0
        local = [(0.0, -h, -h), (0.0, h, -h), (0.0, h, h), (0.0, -h, h)]
        self.tag_corners: dict[int, np.ndarray] = {}
        self.tag_positions: dict[int, Translation3d] = {}
        for tag in layout.getTags():
            pts = []
            for x, y, z in local:
                p = tag.pose.transformBy(Transform3d(Translation3d(x, y, z), Rotation3d()))
                pts.append((p.X(), p.Y(), p.Z()))
            self.tag_corners[tag.ID] = np.array(pts, dtype=np.float64)
            self.tag_positions[tag.ID] = tag.pose.translation()

    def filter_detections(self, detections: list[AprilTagDetection]) -> list[AprilTagDetection]:
        out = []
        for d in detections:
            if d.getId() not in self.tag_corners:
                continue
            if d.getHamming() > self.s.max_hamming:
                continue
            if d.getDecisionMargin() < self.s.min_decision_margin:
                continue
            out.append(d)
        return out

    def solve(
        self,
        detections: list[AprilTagDetection],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        robot_to_camera: Transform3d,
        std_dev_factor: float = 1.0,
    ) -> PoseResult | None:
        self.last_reject = None
        dets = self.filter_detections(detections)
        if not dets:
            self.last_reject = "no usable tags (decision margin / hamming / unknown ID)" if detections else None
            return None

        ids = [d.getId() for d in dets]
        obj = np.concatenate([self.tag_corners[i] for i in ids])
        img = np.array(
            [[d.getCorner(k).x, d.getCorner(k).y] for d in dets for k in range(4)],
            dtype=np.float64,
        )

        ambiguity = 0.0
        if len(dets) == 1:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
            if n < 1:
                self.last_reject = "solvePnP failed"
                return None
            errs = np.asarray(errs).flatten()
            order = np.argsort(errs)
            best = int(order[0])
            if n > 1:
                alt = float(errs[order[1]])
                ambiguity = float(errs[best]) / alt if alt > 1e-9 else 1.0
                if ambiguity > self.s.max_single_tag_ambiguity:
                    self.last_reject = f"single-tag ambiguity {ambiguity:.2f} > {self.s.max_single_tag_ambiguity:.2f}"
                    return None
            rvec, tvec, reproj = rvecs[best], tvecs[best], float(errs[best])
        else:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj, img, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
            )
            if n < 1:
                self.last_reject = "solvePnP failed"
                return None
            rvec, tvec, reproj = rvecs[0], tvecs[0], float(np.asarray(errs).flatten()[0])

        # solvePnP gives field -> OpenCV-camera. Invert to get the camera pose in the field.
        R, _ = cv2.Rodrigues(rvec)
        position = (-R.T @ tvec).flatten()
        field_R_cam = R.T @ _CV_TO_WPI
        camera_pose = Pose3d(
            Translation3d(float(position[0]), float(position[1]), float(position[2])),
            Rotation3d(field_R_cam),
        )
        robot_pose = camera_pose.transformBy(robot_to_camera.inverse())

        cam_t = camera_pose.translation()
        avg_dist = sum(self.tag_positions[i].distance(cam_t) for i in ids) / len(ids)

        # Sanity filters: too far, off the field, or flying/underground.
        if avg_dist > self.s.max_tag_distance_m:
            self.last_reject = f"tags too far: {avg_dist:.1f} m > {self.s.max_tag_distance_m:.1f} m"
            return None
        m = self.s.field_border_margin_m
        if not (-m <= robot_pose.X() <= self.field_length + m and -m <= robot_pose.Y() <= self.field_width + m):
            self.last_reject = f"pose off field ({robot_pose.X():.1f}, {robot_pose.Y():.1f})"
            return None
        if abs(robot_pose.Z()) > self.s.max_z_error_m:
            self.last_reject = f"robot height {robot_pose.Z():.2f} m (check robot_to_camera?)"
            return None

        factor = (avg_dist**2) / (len(ids) ** 2) * std_dev_factor
        xy_std = self.s.xy_coefficient * factor
        if len(ids) == 1 and not self.s.trust_single_tag_theta:
            theta_std = math.inf
        else:
            theta_std = self.s.theta_coefficient * factor

        return PoseResult(
            robot_pose=robot_pose,
            camera_pose=camera_pose,
            tag_ids=ids,
            avg_tag_distance=avg_dist,
            ambiguity=ambiguity,
            reprojection_error=reproj,
            xy_std_dev=xy_std,
            theta_std_dev=theta_std,
        )
