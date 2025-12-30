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


# Test data paths
TEST_DATA_DIR = MODULE_ROOT / "test_data" / "lerobot"
OUTPUT_DIR = MODULE_ROOT / "test_data" / "rlds"


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
