"""Drop-in FieldOdometry for 7407-DriveCode-Rebuilt, fed by the batched coprocessor
instead of PhotonVision.

Copy into sensors/ with batched_vision.py and vision_types.py. Construct it with a
BatchedVisionSource instead of the camera list:

    self.vision = BatchedVisionSource()
    self.field_odometry = FieldOdometry(self.drivetrain, self.vision, self.backup_gyro)

Std devs: xy = XY coefficient * stdDevFactor, where the coprocessor computes
stdDevFactor = avgTagDistance^2 / tagCount^2 * that camera's std_dev_factor.
The coefficients are WPILib Preferences: tune them in Elastic under
/Preferences/Vision/, and they persist on the roboRIO across reboots and deploys.
To trust one camera less than the other, raise its std_dev_factor on the
coprocessor's settings page instead of special-casing it here.
"""

import math

from phoenix6.hardware import Pigeon2
from wpilib import Preferences, SmartDashboard
from wpimath.geometry import Rotation2d

import robot_constants
from subsystems import CommandSwerveDrivetrain

from .batched_vision import BatchedVisionSource
from .vision_types import VisionObservation


class FieldOdometry:
    XY_COEFF_KEY = "Vision/XY Std Dev Coeff"
    THETA_COEFF_KEY = "Vision/Theta Std Dev Coeff"
    SINGLE_TAG_THETA_KEY = "Vision/Single Tag Theta Std Dev"

    # Starting points, roughly matching the old hand-tuned tiers (3 tags at 3 m -> 0.2 m,
    # 1 tag at 2 m -> 0.8 m). Tune on the field; Preferences keep tuned values.
    XY_COEFF_DEFAULT = 0.2
    THETA_COEFF_DEFAULT = 5.0  # 3 tags at 3 m -> 5 rad, like the old best case
    SINGLE_TAG_THETA_DEFAULT = 100.0  # effectively "gyro owns heading"

    def __init__(self, drivetrain: CommandSwerveDrivetrain, vision: BatchedVisionSource, backup_gyro: Pigeon2):
        self.drivetrain = drivetrain
        self.vision = vision
        self.backup_gyro = backup_gyro

        self.use_vision = True

        # init only writes a value if the key doesn't exist yet, so values tuned in
        # Elastic survive code deploys.
        Preferences.initDouble(self.XY_COEFF_KEY, self.XY_COEFF_DEFAULT)
        Preferences.initDouble(self.THETA_COEFF_KEY, self.THETA_COEFF_DEFAULT)
        Preferences.initDouble(self.SINGLE_TAG_THETA_KEY, self.SINGLE_TAG_THETA_DEFAULT)

    def enable(self):
        self.use_vision = True

    def disable(self):
        self.use_vision = False

    def std_devs(self, obs: VisionObservation) -> tuple[float, float, float]:
        xy = Preferences.getDouble(self.XY_COEFF_KEY, self.XY_COEFF_DEFAULT) * obs.stdDevFactor
        if obs.tagCount == 1:
            theta = Preferences.getDouble(self.SINGLE_TAG_THETA_KEY, self.SINGLE_TAG_THETA_DEFAULT)
        else:
            theta = Preferences.getDouble(self.THETA_COEFF_KEY, self.THETA_COEFF_DEFAULT) * obs.stdDevFactor
        return xy, xy, theta

    def add_vision_measure(self, capture_time: float, obs: VisionObservation):
        if obs.tagCount == 0:
            return

        if obs.tagCount == 1 and (
            obs.avgTagDistance > robot_constants.odometry_tag_distance
            or obs.ambiguity > getattr(robot_constants, "odometry_single_tag_ambiguity", 0.2)
        ):
            return

        self.drivetrain.add_vision_measurement(obs.robotPose.toPose2d(), capture_time, self.std_devs(obs))

    def update(self):
        SmartDashboard.putBoolean("Vision/Coprocessor Connected", self.vision.is_connected())

        # Always drain the queue, even while disabled, so re-enabling doesn't replay
        # a backlog of stale observations.
        observations = self.vision.poll()
        if not self.use_vision:
            return

        # Every camera's new observations since the last loop, oldest first.
        for capture_time, obs in observations:
            self.add_vision_measure(capture_time, obs)

        if self.drivetrain.gyro_broken:
            self.drivetrain.reset_rotation(Rotation2d(math.radians(self.backup_gyro.get_yaw().value)))
