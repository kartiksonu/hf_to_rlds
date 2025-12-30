"""
Forward Kinematics converter.

Converts joint positions to end-effector poses and computes deltas.
Supports pluggable FK backends for different robots.
"""

import numpy as np
from typing import Tuple, Optional, Protocol, runtime_checkable
from pathlib import Path
from loguru import logger


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Uses ZYX convention (yaw-pitch-roll), which is common in robotics.

    Args:
        R: 3x3 rotation matrix

    Returns:
        np.ndarray: [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Abstract FK Interface
# =============================================================================

@runtime_checkable
class FKInterface(Protocol):
    """
    Protocol for Forward Kinematics backends.

    Any FK module must implement this interface to work with FKConverter.

    Example implementation:
        class MyRobotFK:
            def __init__(self, urdf_path: str):
                # Load your robot model
                pass

            def compute(self, joint_angles: np.ndarray) -> np.ndarray:
                # Return 4x4 transformation matrix
                return np.eye(4)
    """

    def compute(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            joint_angles: Joint angles (units depend on implementation,
                         typically degrees for SO101).

        Returns:
            4x4 homogeneous transformation matrix of end-effector pose.
        """
        ...


# =============================================================================
# SO101 FK Backend (Default)
# =============================================================================

def load_so101_fk(urdf_path: Optional[str] = None) -> FKInterface:
    """
    Load SO101 Forward Kinematics backend.

    Requires: pip install git+https://github.com/kartiksonu/so101_ik_fk.git

    Args:
        urdf_path: Path to SO101 URDF file. If None, uses package default.

    Returns:
        FK interface for SO101 robot.

    Raises:
        ImportError: If so101_ik_fk package not installed.
    """
    try:
        from so101_ik_fk.lib.so101_kinematics import SO101ForwardKinematics
    except ImportError as e:
        raise ImportError(
            "so101_ik_fk package not installed.\n"
            "Install with: pip install git+https://github.com/kartiksonu/so101_ik_fk.git\n"
            f"Original error: {e}"
        )

    # so101_ik_fk includes URDF and meshes internally
    # Only pass urdf_path if user explicitly provides one
    if urdf_path:
        return SO101ForwardKinematics(urdf_path)
    else:
        return SO101ForwardKinematics()  # Uses package's internal URDF


# =============================================================================
# Main FK Converter
# =============================================================================

class FKConverter:
    """
    Converts joint positions to EEF poses and computes action deltas.

    Supports pluggable FK backends for different robots.

    Usage with default SO101:
        converter = FKConverter()

    Usage with custom FK backend:
        my_fk = MyRobotFK("/path/to/urdf")
        converter = FKConverter(fk_backend=my_fk)

    Usage with custom FK class:
        converter = FKConverter(
            fk_class=MyRobotFK,
            urdf_path="/path/to/urdf"
        )
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        fk_backend: Optional[FKInterface] = None,
        fk_class: Optional[type] = None,
        num_arm_joints: int = 5,
        joint_units: str = "degrees",
    ):
        """
        Initialize the FK converter.

        Args:
            urdf_path: Path to URDF file (for default SO101 or custom fk_class).
            fk_backend: Pre-initialized FK backend instance.
            fk_class: Custom FK class to instantiate (must implement FKInterface).
            num_arm_joints: Number of arm joints (excluding gripper). Default: 5 for SO101.
            joint_units: Input joint units - "degrees" or "radians". Default: "degrees".

        Raises:
            ValueError: If both fk_backend and fk_class provided.
            FileNotFoundError: If URDF or mesh assets missing.
        """
        if fk_backend is not None and fk_class is not None:
            raise ValueError("Provide either fk_backend OR fk_class, not both.")

        self.num_arm_joints = num_arm_joints
        self.joint_units = joint_units

        if fk_backend is not None:
            # Use provided FK backend
            self.fk = fk_backend
            logger.info(f"Using provided FK backend: {type(fk_backend).__name__}")
        elif fk_class is not None:
            # Instantiate custom FK class
            self.fk = fk_class(urdf_path) if urdf_path else fk_class()
            logger.info(f"Using custom FK class: {fk_class.__name__}")
        else:
            # Default: SO101
            self.fk = load_so101_fk(urdf_path)
            logger.info("Using default SO101 FK backend")

    def joint_to_eef_pose(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert joint angles to end-effector pose.

        Args:
            joint_angles: Joint angles. Shape (num_arm_joints,) or larger
                         (extra values like gripper are ignored for FK).

        Returns:
            Tuple of (xyz, rpy):
                - xyz: Position [x, y, z] in meters
                - rpy: Orientation [roll, pitch, yaw] in radians
        """
        # Take only arm joints, convert to float64 for placo compatibility
        arm_joints = np.array(joint_angles[:self.num_arm_joints], dtype=np.float64)

        # Compute FK
        pose_matrix = self.fk.compute(arm_joints)

        # Extract position
        xyz = pose_matrix[:3, 3]

        # Extract orientation as euler angles
        rotation_matrix = pose_matrix[:3, :3]
        rpy = rotation_matrix_to_euler(rotation_matrix)

        return xyz, rpy

    def compute_episode_deltas(
        self,
        joint_sequence: np.ndarray,
        gripper_sequence: np.ndarray,
        binarize_gripper: bool = True,
        gripper_threshold: float = None
    ) -> np.ndarray:
        """
        Compute action deltas for an entire episode.

        Args:
            joint_sequence: Array of joint angles, shape (T, num_joints).
            gripper_sequence: Array of gripper values, shape (T,).
            binarize_gripper: Whether to binarize gripper values.
            gripper_threshold: Threshold for binarization. If None, uses midpoint.

        Returns:
            np.ndarray: Action deltas, shape (T, 7) where each row is
                       [dx, dy, dz, droll, dpitch, dyaw, gripper].
                       First row has zero deltas (no previous frame).
        """
        T = len(joint_sequence)
        actions = np.zeros((T, 7), dtype=np.float32)

        # Compute EEF poses for all timesteps
        poses = []
        for t in range(T):
            xyz, rpy = self.joint_to_eef_pose(joint_sequence[t])
            poses.append((xyz, rpy))

        # Process gripper
        if binarize_gripper:
            if gripper_threshold is None:
                gripper_threshold = (gripper_sequence.min() + gripper_sequence.max()) / 2
            gripper_binary = (gripper_sequence > gripper_threshold).astype(np.float32)
        else:
            g_min, g_max = gripper_sequence.min(), gripper_sequence.max()
            if g_max - g_min > 1e-6:
                gripper_binary = (gripper_sequence - g_min) / (g_max - g_min)
            else:
                gripper_binary = np.zeros_like(gripper_sequence)

        # First frame: zero deltas, current gripper
        actions[0, 6] = gripper_binary[0]

        # Compute deltas for remaining frames
        for t in range(1, T):
            prev_xyz, prev_rpy = poses[t - 1]
            curr_xyz, curr_rpy = poses[t]

            delta_xyz = curr_xyz - prev_xyz
            delta_rpy = np.array([
                wrap_angle(curr_rpy[0] - prev_rpy[0]),
                wrap_angle(curr_rpy[1] - prev_rpy[1]),
                wrap_angle(curr_rpy[2] - prev_rpy[2])
            ])

            actions[t, :3] = delta_xyz
            actions[t, 3:6] = delta_rpy
            actions[t, 6] = gripper_binary[t]

        return actions

    def compute_episode_states(
        self,
        joint_sequence: np.ndarray,
        gripper_sequence: np.ndarray,
        binarize_gripper: bool = True,
        gripper_threshold: float = None
    ) -> np.ndarray:
        """
        Compute absolute EEF states for an entire episode.

        Args:
            joint_sequence: Array of joint angles, shape (T, num_joints).
            gripper_sequence: Array of gripper values, shape (T,).
            binarize_gripper: Whether to binarize gripper values.
            gripper_threshold: Threshold for binarization.

        Returns:
            np.ndarray: States, shape (T, 7) where each row is
                       [x, y, z, roll, pitch, yaw, gripper].
        """
        T = len(joint_sequence)
        states = np.zeros((T, 7), dtype=np.float32)

        # Process gripper
        if binarize_gripper:
            if gripper_threshold is None:
                gripper_threshold = (gripper_sequence.min() + gripper_sequence.max()) / 2
            gripper_values = (gripper_sequence > gripper_threshold).astype(np.float32)
        else:
            g_min, g_max = gripper_sequence.min(), gripper_sequence.max()
            if g_max - g_min > 1e-6:
                gripper_values = (gripper_sequence - g_min) / (g_max - g_min)
            else:
                gripper_values = np.zeros_like(gripper_sequence)

        for t in range(T):
            xyz, rpy = self.joint_to_eef_pose(joint_sequence[t])
            states[t, :3] = xyz
            states[t, 3:6] = rpy
            states[t, 6] = gripper_values[t]

        return states
