# nml/processing/__init__.py
from nml.processing.activity import MotorTwitchHandler
from nml.processing.arm_incremental_kinematics import (
    ArmIncrementalKinematics,
    euler_xyz_rad_to_quaternion,
    wrist_position_decomposition,
)
from nml.processing.envelope import EnvelopeSmoother
from nml.processing.gyros import GyroProcessor
from nml.processing.hpf import HighPassFilter
from nml.processing.imu_ar import ImuArModel, ImuArFeatureExtractor
from nml.processing.mlp_window import MlpWindowExtractor
from nml.processing.orientation import OrientationEstimator
from nml.processing.spatial import SpatialReference, MontageMode
from nml.processing.zca import ZcaHandler, ZcaParams

# Defaults
DEFAULT_ZCA_PARAMS = ZcaParams(num_channels=8, buffer_duration_sec=5.0, tikhonov_epsilon=1.5, enable=True)