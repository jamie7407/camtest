"""RobotPy side: reads the coprocessor's batched observations and feeds the pose estimator.

One NT read per robot loop (readQueue) regardless of camera count, and struct
decoding instead of photonlibpy's full PhotonPipelineResult object graph.

Usage in your drivetrain subsystem:

    from batched_vision import BatchedVision
    self.vision = BatchedVision()

    def periodic(self):
        self.pose_estimator.update(gyro_rotation, module_positions)
        self.vision.update(self.pose_estimator)

Std devs: the coprocessor sends stdDevFactor (avgTagDistance^2 / tagCount^2, times its
per-camera factor). The trust coefficients are WPILib Preferences, so they're tunable
from Elastic (under /Preferences/Vision/...) and persist on the roboRIO across reboots:

    Vision/XYStdDevCoefficient      xy std dev (m)    = coefficient * stdDevFactor
    Vision/ThetaStdDevCoefficient   theta std dev (rad) = coefficient * stdDevFactor
    Vision/TrustSingleTagTheta      false: single-tag heading ignored (gyro wins)

Lower coefficient = trust vision more (the pose snaps to vision faster, and jitters
more). The defaults are 6328's starting point, not tuned for any particular robot.
"""

from __future__ import annotations

import math

import ntcore
from wpilib import DriverStation, Preferences, Timer

from vision_types import VisionObservation


class BatchedVision:
    XY_KEY = "Vision/XYStdDevCoefficient"
    THETA_KEY = "Vision/ThetaStdDevCoefficient"
    TRUST_THETA_KEY = "Vision/TrustSingleTagTheta"

    def std_devs(self, obs: VisionObservation) -> tuple[float, float, float]:
        """(x, y, theta) std devs for one observation, from the current Preferences."""
        xy = Preferences.getDouble(self.XY_KEY, 0.01) * obs.stdDevFactor
        if obs.tagCount == 1 and not Preferences.getBoolean(self.TRUST_THETA_KEY, False):
            theta = math.inf  # one tag's heading is unreliable; let the gyro own heading
        else:
            theta = Preferences.getDouble(self.THETA_KEY, 0.03) * obs.stdDevFactor
        return xy, xy, theta

    def __init__(
        self,
        topic: str = "/Vision/observations",
        max_single_tag_jump_m: float | None = 1.0,
        inst: ntcore.NetworkTableInstance | None = None,
    ):
        """
        max_single_tag_jump_m: reject single-tag observations that disagree with the
            current estimate by more than this (a cheap guard against a bad flip on
            one tag). Multi-tag observations are always accepted. Only applied while
            enabled, so single tags can still seed the pose while disabled on the field.
            None disables it.
        """
        inst = inst or ntcore.NetworkTableInstance.getDefault()
        opts = ntcore.PubSubOptions(sendAll=True, keepDuplicates=True, pollStorage=32)
        self._sub = inst.getStructArrayTopic(topic, VisionObservation).subscribe([], opts)
        base = topic.rsplit("/", 1)[0] or "/Vision"
        self._heartbeat = inst.getIntegerTopic(f"{base}/heartbeat").subscribe(0)
        self._last_hb = 0
        self._last_hb_time = -math.inf
        self._max_jump = max_single_tag_jump_m

        # Tunable from Elastic; init only sets a value if the key doesn't exist yet,
        # so values tuned on the robot survive code deploys.
        Preferences.initDouble(self.XY_KEY, 0.01)
        Preferences.initDouble(self.THETA_KEY, 0.03)
        Preferences.initBoolean(self.TRUST_THETA_KEY, False)

        # Most recent batch, handy for logging/dashboards: list of (capture_time, obs, accepted)
        self.last_observations: list[tuple[float, VisionObservation, bool]] = []

    def is_connected(self, timeout_s: float = 0.5) -> bool:
        """True if the coprocessor has sent a heartbeat recently."""
        hb = self._heartbeat.get()
        if hb != self._last_hb:
            self._last_hb = hb
            self._last_hb_time = Timer.getFPGATimestamp()
        return Timer.getFPGATimestamp() - self._last_hb_time < timeout_s + 1.0

    def update(self, estimator) -> int:
        """Feed all new observations into a WPILib pose estimator. Returns the number accepted."""
        entries = self._sub.readQueue()  # the single NT call per loop
        pending = []
        for e in entries:
            msg_time = e.time / 1e6  # NT4 already converted this to the robot's clock
            for obs in e.value:
                pending.append((msg_time - obs.ageSeconds, obs))
        pending.sort(key=lambda p: p[0])

        current = estimator.getEstimatedPosition()
        accepted = 0
        self.last_observations = []
        for capture_time, obs in pending:
            pose2d = obs.robotPose.toPose2d()
            ok = True
            if obs.tagCount == 1 and self._max_jump is not None and DriverStation.isEnabled():
                ok = pose2d.translation().distance(current.translation()) <= self._max_jump
            if ok:
                estimator.addVisionMeasurement(pose2d, capture_time, self.std_devs(obs))
                accepted += 1
            self.last_observations.append((capture_time, obs, ok))
        return accepted
