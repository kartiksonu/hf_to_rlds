# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts LeRobot (HuggingFace) robotics datasets to RLDS (TFRecord) format for use with vision-language-action models (SpatialVLA, OpenVLA, RT-X, Octo, etc.).

Key transformations:
- **Actions**: Joint positions → End-effector pose deltas (7D: xyz + rpy + gripper)
- **Images**: MP4 videos → JPEG-encoded frames per timestep
- **Format**: Parquet + MP4 → TFRecord (one Example per episode)

## Commands

```bash
# Install with SO101 robot support
pip install -e ".[so101]"

# Run tests (requires test_data/lerobot/ sample data)
python -m pytest tests/test_conversion.py -v
# Or standalone:
python tests/test_conversion.py

# Download and convert in one command
python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval

# Convert locally downloaded dataset
python scripts/convert.py --repo_id user/dataset --local_dir ./data --output out.tfrecord

# Lint
ruff check .
black --check .
```

## Architecture

The conversion pipeline has four core components:

```
LeRobot Parquet → VideoLoader → FKConverter → RLDSWriter → TFRecord
                  (frames)      (FK + deltas)  (serialize)
```

### Core Library (`lib/`)

- **`config.py`**: `ConversionConfig` dataclass with all settings (repo ID, cameras, image size, FK options)
- **`converter.py`**: Main orchestration - loads HF dataset, coordinates other modules, writes output
- **`video_loader.py`**: Handles LeRobot's chunked video format where multiple episodes are concatenated into single MP4 files. Builds episode→file mapping from parquet, extracts frames with pyav (for AV1 codec support)
- **`fk_converter.py`**: Forward kinematics - converts joint angles to EEF poses using pluggable FK backends. Computes action deltas between consecutive frames. Default: SO101 via `so101_ik_fk` package
- **`tfrecord_writer.py`**: Serializes episodes to RLDS-compatible TFRecord format

### Scripts (`scripts/`)

- **`download_and_convert.py`**: One-command download + convert
- **`convert.py`**: Convert pre-downloaded datasets

## Key Technical Details

### Video Chunking
LeRobot concatenates episodes into chunked video files. Episode 15 is NOT in `file-015.mp4` — it's inside a chunk file at a specific frame offset. `VideoLoader._build_episode_map()` parses parquet files to determine `episode_index → (file_index, start_frame, num_frames)`.

### Pluggable FK Backends
The `FKConverter` supports three ways to provide forward kinematics:
1. Default SO101 (requires `pip install -e ".[so101]"`)
2. Custom FK class implementing `FKInterface` protocol (must have `compute(joint_angles) -> 4x4 matrix`)
3. Pre-initialized FK backend instance

### RLDS Output Format
Each TFRecord Example contains one complete episode:
- `steps/action`: Flattened `(T, 7)` EEF pose deltas
- `steps/observation/state`: Flattened `(T, 7)` absolute EEF poses
- `steps/observation/image_N`: List of JPEG bytes (up to 4 cameras)
- `steps/language_instruction`: Task instruction repeated per step
