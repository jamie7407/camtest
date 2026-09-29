"""Robot-side reader for the coprocessor's batched AprilTag vision pipeline.

Replaces polling one PhotonCamera per loop with a single NT4 struct-array read
(readQueue()) that returns every camera's new observations for that cycle in one
call, with no per-camera photonlibpy deserialization.

Drop-in for 7407-DriveCode-Rebuilt: copy this file and vision_types.py into
sensors/ (same API as the BatchedVisionSource already there). FieldOdometry
consumes poll() and owns the std devs.

    self.vision = BatchedVisionSource()
    for capture_time, obs in self.vision.poll():
        ...  # obs.robotPose, obs.stdDevFactor, obs.tagCount, obs.avgTagDistance, ...

Std devs: each observation carries stdDevFactor = avgTagDistance^2 / tagCount^2
times the camera's std_dev_factor (set on the coprocessor). Multiply by a tunable
coefficient on the robot, e.g. std_dev = coeff * obs.stdDevFactor.

BatchedVision below is a minimal all-in-one consumer (Preferences-tunable std devs)
for projects without their own odometry wrapper.
"""

from __future__ import annotations

import math

import ntcore
from wpilib import DriverStation, Preferences, Timer

try:  # inside a package (e.g. the robot's sensors/)
    from .vision_types import VisionObservation
except ImportError:  # run from this folder
    from vision_types import VisionObservation


class BatchedVisionSource:
    def __init__(
        self,
        topic: str = "/Vision/observations",
        inst: ntcore.NetworkTableInstance | None = None,
    ):
        inst = inst or ntcore.NetworkTableInstance.getDefault()
        opts = ntcore.PubSubOptions(sendAll=True, keepDuplicates=True, pollStorage=32)
        self._sub = inst.getStructArrayTopic(topic, VisionObservation).subscribe([], opts)
        base = topic.rsplit("/", 1)[0] or "/Vision"
        self._heartbeat = inst.getIntegerTopic(f"{base}/heartbeat").subscribe(0)
        self._last_hb = 0
        self._last_hb_time = -math.inf

    def is_connected(self, timeout_s: float = 1.5) -> bool:
        """True if the coprocessor has sent a heartbeat recently (it beats once a second)."""
        hb = self._heartbeat.get()
        if hb != self._last_hb:
            self._last_hb = hb
            self._last_hb_time = Timer.getFPGATimestamp()
        return Timer.getFPGATimestamp() - self._last_hb_time < timeout_s

    def poll(self) -> list[tuple[float, VisionObservation]]:
        """All new observations from every camera since the last poll, oldest first,
        as (capture_time, observation)."""
        entries = self._sub.readQueue()  # the single NT call per loop
        pending = []
        for e in entries:
            msg_time = e.time / 1e6  # NT4 already converted this to the robot's clock
            for obs in e.value:
                pending.append((msg_time - obs.ageSeconds, obs))
        pending.sort(key=lambda p: p[0])
        return pending


class BatchedVision:
    """Minimal consumer: feeds every observation into a WPILib pose estimator, with
    std devs from Preferences (tunable in Elastic under /Preferences/Vision/, and
    persisted on the roboRIO):

        Vision/XYStdDevCoefficient      xy std dev (m)      = coefficient * stdDevFactor
        Vision/ThetaStdDevCoefficient   theta std dev (rad) = coefficient * stdDevFactor
        Vision/TrustSingleTagTheta      false: single-tag heading ignored (gyro wins)
    """

    XY_KEY = "Vision/XYStdDevCoefficient"
    THETA_KEY = "Vision/ThetaStdDevCoefficient"
    TRUST_THETA_KEY = "Vision/TrustSingleTagTheta"

    def __init__(self, source: BatchedVisionSource | None = None, max_single_tag_jump_m: float | None = 1.0):
        """max_single_tag_jump_m: while enabled, reject single-tag observations more than
        this far from the current estimate (a cheap guard against a bad flip). None disables it."""
        self.source = source or BatchedVisionSource()
        self._max_jump = max_single_tag_jump_m
        # init only sets a value if the key doesn't exist yet, so tuned values survive deploys.
        Preferences.initDouble(self.XY_KEY, 0.01)
        Preferences.initDouble(self.THETA_KEY, 0.03)
        Preferences.initBoolean(self.TRUST_THETA_KEY, False)
        self.last_observations: list[tuple[float, VisionObservation, bool]] = []

    def is_connected(self) -> bool:
        return self.source.is_connected()

    def std_devs(self, obs: VisionObservation) -> tuple[float, float, float]:
        xy = Preferences.getDouble(self.XY_KEY, 0.01) * obs.stdDevFactor
        if obs.tagCount == 1 and not Preferences.getBoolean(self.TRUST_THETA_KEY, False):
            theta = math.inf  # one tag's heading is unreliable; let the gyro own heading
        else:
            theta = Preferences.getDouble(self.THETA_KEY, 0.03) * obs.stdDevFactor
        return xy, xy, theta

    def update(self, estimator) -> int:
        """Feed all new observations into the estimator. Returns the number accepted."""
        current = estimator.getEstimatedPosition()
        accepted = 0
        self.last_observations = []
        for capture_time, obs in self.source.poll():
            pose2d = obs.robotPose.toPose2d()
            ok = True
            if obs.tagCount == 1 and self._max_jump is not None and DriverStation.isEnabled():
                ok = pose2d.translation().distance(current.translation()) <= self._max_jump
            if ok:
                estimator.addVisionMeasurement(pose2d, capture_time, self.std_devs(obs))
                accepted += 1
            self.last_observations.append((capture_time, obs, ok))
        return accepted
