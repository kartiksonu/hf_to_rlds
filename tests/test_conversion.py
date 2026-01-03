#!/usr/bin/env python3
"""
Unit tests for HF to RLDS conversion.

Run with:
    cd hf_to_rlds
    python -m pytest tests/test_conversion.py -v

Or standalone:
    python tests/test_conversion.py
"""

import sys
from pathlib import Path

# Add module root to path
MODULE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image
import io
from loguru import logger

from lib.config import ConversionConfig
from lib.converter import convert_lerobot_to_rlds
from lib.fk_converter import FKConverter, rotation_matrix_to_euler


# Test data paths
TEST_DATA_DIR = MODULE_ROOT / "test_data" / "lerobot"
OUTPUT_DIR = MODULE_ROOT / "test_data" / "rlds"


def euler_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to 3x3 rotation matrix.

    Uses ZYX convention (same as rotation_matrix_to_euler).
    """
    roll, pitch, yaw = rpy

    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # ZYX convention: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def test_conversion_single_episode():
    """Test converting a single episode with 1 camera."""
    output_path = OUTPUT_DIR / "test_single_episode.tfrecord"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        hf_repo_id="sapanostic/so101_offline_eval",
        output_path=str(output_path),
        local_data_dir=str(TEST_DATA_DIR),
        num_images=1,
        max_episodes=1,
        image_size=(224, 224),
    )

    num_converted = convert_lerobot_to_rlds(config, verbose=False)

    # Assertions
    assert num_converted == 1, f"Expected 1 episode, got {num_converted}"
    assert output_path.exists(), "Output file not created"
    assert output_path.stat().st_size > 0, "Output file is empty"

    # Verify TFRecord structure
    dataset = tf.data.TFRecordDataset(str(output_path))
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Check required keys exist
        assert 'episode_metadata/episode_id' in features
        assert 'steps/action' in features
        assert 'steps/observation/state' in features
        assert 'steps/language_instruction' in features
        assert 'steps/observation/image_0' in features

        # Check action shape (should be 7D per step)
        actions = np.array(features['steps/action'].float_list.value)
        num_steps = len(actions) // 7
        assert num_steps > 0, "No steps in episode"
        assert len(actions) % 7 == 0, "Actions not 7D"

        # Check images
        images = features['steps/observation/image_0'].bytes_list.value
        assert len(images) == num_steps, "Image count mismatch"

        # Verify first image is valid JPEG
        first_img = Image.open(io.BytesIO(images[0]))
        assert first_img.size == (224, 224), f"Wrong image size: {first_img.size}"

        break  # Only check first episode

    logger.success("test_conversion_single_episode passed")
    return True


def test_action_values():
    """Test that action values are reasonable (deltas should be small)."""
    output_path = OUTPUT_DIR / "test_action_values.tfrecord"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        hf_repo_id="sapanostic/so101_offline_eval",
        output_path=str(output_path),
        local_data_dir=str(TEST_DATA_DIR),
        num_images=1,
        max_episodes=1,
        image_size=(224, 224),
    )

    convert_lerobot_to_rlds(config, verbose=False)

    dataset = tf.data.TFRecordDataset(str(output_path))
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        actions = np.array(features['steps/action'].float_list.value)
        num_steps = len(actions) // 7
        actions = actions.reshape(num_steps, 7)

        # Position deltas should be small (< 0.1m typically)
        pos_deltas = actions[:, :3]
        assert np.all(np.abs(pos_deltas) < 0.5), "Position deltas too large"

        # Rotation deltas should be small (< 1 rad typically)
        rot_deltas = actions[:, 3:6]
        assert np.all(np.abs(rot_deltas) < np.pi), "Rotation deltas too large"

        # Gripper should be binary (0 or 1)
        gripper = actions[:, 6]
        assert np.all((gripper == 0) | (gripper == 1)), "Gripper not binary"

        break

    logger.success("test_action_values passed")
    return True


def test_state_values():
    """Test that state values are reasonable (absolute EEF pose)."""
    output_path = OUTPUT_DIR / "test_state_values.tfrecord"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        hf_repo_id="sapanostic/so101_offline_eval",
        output_path=str(output_path),
        local_data_dir=str(TEST_DATA_DIR),
        num_images=1,
        max_episodes=1,
        image_size=(224, 224),
    )

    convert_lerobot_to_rlds(config, verbose=False)

    dataset = tf.data.TFRecordDataset(str(output_path))
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        states = np.array(features['steps/observation/state'].float_list.value)
        num_steps = len(states) // 7
        states = states.reshape(num_steps, 7)

        # EEF position should be within robot workspace (roughly 0-0.5m from base)
        positions = states[:, :3]
        assert np.all(np.abs(positions) < 1.0), "EEF position outside workspace"

        # Orientation should be valid euler angles
        orientations = states[:, 3:6]
        assert np.all(np.abs(orientations) < 2 * np.pi), "Invalid euler angles"

        break

    logger.success("test_state_values passed")
    return True


def test_delta_roundtrip():
    """
    Test that delta computation is correct via roundtrip reconstruction.

    This test verifies the SO(3) delta computation by:
    1. Loading joint trajectory from test data
    2. Computing ground truth EEF poses
    3. Computing deltas using FKConverter
    4. Reconstructing trajectory by accumulating deltas in SE(3)
    5. Comparing reconstructed trajectory with ground truth

    If the delta computation is correct (proper SO(3) relative rotation),
    the reconstructed trajectory should match the original.
    """
    import pandas as pd

    # Load test parquet
    parquet_path = TEST_DATA_DIR / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.exists():
        logger.warning(f"Test data not found at {parquet_path}, skipping test")
        return True

    df = pd.read_parquet(parquet_path)

    # Get first episode
    first_episode = df['episode_index'].min()
    episode_data = df[df['episode_index'] == first_episode].sort_values('frame_index')

    T = len(episode_data)
    assert T > 10, f"Episode too short: {T} frames"

    # Extract joint sequences
    joint_sequence = np.array([row for row in episode_data['observation.state']])
    gripper_sequence = joint_sequence[:, 5]  # Gripper is index 5 for SO101

    # Initialize FK converter
    fk_converter = FKConverter(num_arm_joints=5)

    # Step 1: Compute ground truth EEF poses
    ground_truth_poses = []
    for t in range(T):
        xyz, R = fk_converter.joint_to_eef_pose_with_rotation(joint_sequence[t])
        ground_truth_poses.append((xyz.copy(), R.copy()))

    # Step 2: Compute deltas using the converter
    deltas = fk_converter.compute_episode_deltas(
        joint_sequence,
        gripper_sequence,
        binarize_gripper=True
    )

    # Step 3: Reconstruct trajectory by accumulating deltas in SE(3)
    reconstructed_poses = []

    # First pose is the starting point (no delta to apply)
    curr_xyz = ground_truth_poses[0][0].copy()
    curr_R = ground_truth_poses[0][1].copy()
    reconstructed_poses.append((curr_xyz.copy(), curr_R.copy()))

    for t in range(1, T):
        delta_xyz = deltas[t, :3]
        delta_rpy = deltas[t, 3:6]

        # Convert delta euler to rotation matrix
        R_delta = euler_to_rotation_matrix(delta_rpy)

        # Accumulate in SE(3):
        # - Position: add delta
        # - Rotation: R_new = R_delta @ R_prev (since delta = R_curr @ R_prev.T)
        curr_xyz = curr_xyz + delta_xyz
        curr_R = R_delta @ curr_R

        reconstructed_poses.append((curr_xyz.copy(), curr_R.copy()))

    # Step 4: Compare reconstructed with ground truth
    position_errors = []
    rotation_errors = []

    for t in range(T):
        gt_xyz, gt_R = ground_truth_poses[t]
        recon_xyz, recon_R = reconstructed_poses[t]

        # Position error (Euclidean distance)
        pos_error = np.linalg.norm(gt_xyz - recon_xyz)
        position_errors.append(pos_error)

        # Rotation error (Frobenius norm of difference, or angle)
        # R_error = R_gt @ R_recon.T should be identity if perfect
        R_error = gt_R @ recon_R.T
        # Angle from axis-angle: trace(R) = 1 + 2*cos(angle)
        trace = np.trace(R_error)
        # Clamp for numerical stability
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_error = np.abs(np.arccos(cos_angle))
        rotation_errors.append(angle_error)

    position_errors = np.array(position_errors)
    rotation_errors = np.array(rotation_errors)

    # Log statistics
    logger.info(f"  Trajectory length: {T} frames")
    logger.info(f"  Position error - max: {position_errors.max():.6f}m, mean: {position_errors.mean():.6f}m")
    logger.info(f"  Rotation error - max: {np.degrees(rotation_errors.max()):.4f}°, mean: {np.degrees(rotation_errors.mean()):.4f}°")

    # Assertions - errors should be near zero (floating point tolerance)
    # Position: should be < 1mm accumulated error
    assert position_errors.max() < 1e-3, f"Position reconstruction error too large: {position_errors.max():.6f}m"

    # Rotation: should be < 0.1 degree accumulated error
    assert np.degrees(rotation_errors.max()) < 0.1, f"Rotation reconstruction error too large: {np.degrees(rotation_errors.max()):.4f}°"

    logger.success("test_delta_roundtrip passed")
    return True


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("Running HF to RLDS Conversion Tests")
    logger.info("=" * 60)
    logger.info(f"Test data: {TEST_DATA_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    if not TEST_DATA_DIR.exists():
        logger.error(f"Test data not found at {TEST_DATA_DIR}")
        logger.error("Please ensure test_data/lerobot/ contains sample data.")
        sys.exit(1)

    tests = [
        test_conversion_single_episode,
        test_action_values,
        test_state_values,
        test_delta_roundtrip,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            logger.error(f"{test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"{test.__name__} ERROR: {e}")
            failed += 1

    logger.info("=" * 60)
    if failed == 0:
        logger.success(f"Results: {passed} passed, {failed} failed")
    else:
        logger.error(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
