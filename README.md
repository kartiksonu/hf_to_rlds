# HF to RLDS Converter

Converts LeRobot (HuggingFace) datasets to RLDS (TFRecord) format.

## Why This Exists

**RLDS** (Reinforcement Learning Datasets) is the standard format for robotics datasets in the Open X-Embodiment ecosystem. Most vision-language-action models (SpatialVLA, OpenVLA, RT-X, Octo, etc.) expect:
- End-effector (EEF) pose deltas as actions (7D: xyz + rpy + gripper)
- JPEG-encoded images per timestep
- One TFRecord example per episode

**LeRobot** (HuggingFace) stores data differently:
- Joint positions as actions (robot-specific, e.g., 6D for SO101)
- MP4 videos (not individual frames)
- Parquet files with one row per timestep

This module bridges that gap — making LeRobot data usable for RLDS-based training pipelines.

---

## Source Data Structure (LeRobot on HuggingFace)

```
hf_lerobot/
├── data/chunk-000/
│   ├── file-000.parquet    # Episodes 0, 1, 2 (612 rows)
│   ├── file-001.parquet    # Episodes 3, 4, 5, 6, 7 (1125 rows)
│   └── ...
├── videos/
│   ├── observation.images.front/chunk-000/
│   │   ├── file-000.mp4    # Episodes 0, 1, 2 concatenated
│   │   ├── file-001.mp4    # Episodes 3, 4, 5, 6, 7 concatenated
│   │   └── ...
│   ├── observation.images.top/...
│   └── observation.images.wrist/...
└── meta/
    ├── tasks.parquet       # task_index → instruction text
    └── info.json           # fps, total_episodes, feature shapes
```

**Key insight**: Videos contain MULTIPLE episodes concatenated. i.e Episode 15 is NOT in `file-015.mp4`.

---

## Conversion Pipeline (Detailed)

```
INPUT: LeRobot Parquet Row
┌────────────────────────────────────────────────────────────────┐
│ observation.state: [37.4,  7.1,     -5.0,  90.7,  0.2, 18.1]   │
│                     ↑       ↑         ↑     ↑      ↑    ↑      │
│                   shoulder shoulder elbow  wrist wrist gripper │
│                     pan    lift     flex   flex  roll  (deg)   │
│                                                                │
│ episode_index: 15    frame_index: 42    task_index: 0          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      STEP 1: VideoLoader                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Build episode map from parquet:                            │
│     episode_15 → (file_idx=3, start_frame=450, num_frames=225) │
│                                                                │
│  2. Open file-003.mp4, seek to frame 450                       │
│                                                                │
│  3. Decode 225 frames using pyav (AV1 codec)                   │
│                                                                │
│  4. Resize each frame: 640x480 → 224x224                       │
│                                                                │
│  5. Encode to JPEG bytes                                       │
│                                                                │
│  Output: List[bytes] - 225 JPEG images                         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      STEP 2: FKConverter                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Forward Kinematics: Joint angles → End-effector pose          │
│                                                                │
│  Uses: placo library + SO101 URDF file                         │
│        placo loads robot model, computes transform chain       │
│                                                                │
│  Input:  [37.4, 7.1, -5.0, 90.7, 0.2] degrees (5 arm joints)   │
│                       │                                        │
│                       ▼                                        │
│          ┌─────────────────────────┐                           │
│          │   URDF Kinematic Chain  │                           │
│          │   base → shoulder →     │                           │
│          │   elbow → wrist → EEF   │                           │
│          └─────────────────────────┘                           │
│                       │                                        │
│                       ▼                                        │
│  Output: 4x4 transform matrix → extract xyz position           │
│                                 extract rpy orientation        │
│                                                                │
│  For actions, compute DELTAS between consecutive frames:       │
│  action[t] = pose[t] - pose[t-1]                               │
│                                                                │
│  Gripper: Binarize continuous value (threshold at midpoint)    │
│           18.1° → 0 (closed) or 1 (open)                       │
│                                                                │
│  Output per frame:                                             │
│    state:  [x, y, z, roll, pitch, yaw, gripper]  (absolute)    │
│    action: [dx, dy, dz, dr, dp, dy, gripper]     (delta)       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    STEP 3: TFRecordWriter                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Serialize entire episode into single tf.train.Example:        │
│                                                                │
│  episode_metadata/                                             │
│    episode_id: "15"                                            │
│    has_image_0: 1                                              │
│    has_image_1: 1  (if num_images >= 2, set in config)         │
│    has_image_2: 1  (if num_images >= 3)                        │
│                                                                │
│  steps/observation/                                            │
│    image_0: [jpeg_bytes × 225]     # front camera              │
│    image_1: [jpeg_bytes × 225]     # top camera                │
│    image_2: [jpeg_bytes × 225]     # wrist camera              │
│    state: [x,y,z,r,p,y,g × 225]    # flattened floats          │
│                                                                │
│  steps/                                                        │
│    action: [dx,dy,dz,dr,dp,dy,g × 225]                         │
│    language_instruction: ["put potato in bowl" × 225]          │
│    is_first: [1, 0, 0, ..., 0]                                 │
│    is_last:  [0, 0, 0, ..., 1]                                 │
│    is_terminal: [0, 0, 0, ..., 1]                              │
│    reward: [0.0 × 225]                                         │
│    discount: [1.0 × 225]                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
OUTPUT: RLDS TFRecord file (one Example per episode)
```

---

## Module Structure

```
hf_to_rlds/
├── __init__.py            # Package exports
├── pyproject.toml         # pip install config
├── README.md
├── lib/                   # Core library
│   ├── __init__.py
│   ├── config.py          # ConversionConfig dataclass
│   ├── converter.py       # Main conversion logic
│   ├── fk_converter.py    # placo FK, delta computation
│   ├── video_loader.py    # Episode→file mapping, pyav decode
│   └── tfrecord_writer.py # tf.train.Example serialization
├── scripts/               # CLI tools
│   ├── __init__.py
│   └── convert.py         # CLI wrapper
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_conversion.py # Conversion tests
└── test_data/             # Sample data
    ├── lerobot/           # Input: 3 episodes, front camera
    └── rlds/              # Output: test results
```

---

## Quick Start

```bash
# Run tests with included test data
cd hf_to_rlds
python tests/test_conversion.py
```

This converts sample episodes and verifies the output structure, action values, and state values.

---

## Usage

```bash
# 1. Download your dataset first
python -c "
from huggingface_hub import snapshot_download
snapshot_download('your-username/your-dataset', repo_type='dataset', local_dir='./my_data')
"

# 2. Convert with 1 camera (default)
python scripts/convert.py \
  --repo_id your-username/your-dataset \
  --local_dir ./my_data \
  --output converted.tfrecord

# 3. Convert with all 3 cameras, limit to 10 episodes
python scripts/convert.py \
  --repo_id your-username/your-dataset \
  --local_dir ./my_data \
  --num_images 3 \
  --max_episodes 10 \
  --output my_data.tfrecord

# Full options
python scripts/convert.py --help
```

**Note**: Update `--repo_id` and `--local_dir` to match your dataset.

---

## Key Technical Details

### 1. Video Chunking Problem

LeRobot concatenates episodes into chunked video files for storage efficiency:
```
file-000.mp4: frames 0-611    → episodes 0, 1, 2
file-001.mp4: frames 0-1124   → episodes 3, 4, 5, 6, 7
file-003.mp4: frames 0-1124   → episodes 13, 14, 15, 16, 17
                                         ↑
                              episode 15 starts at frame 450
```

**Solution**: Parse parquet files to build `episode_index → (file_idx, start_frame, num_frames)` mapping before extraction.

### 2. AV1 Video Codec

LeRobot encodes videos in **AV1** — a modern, efficient codec that OpenCV cannot decode by default.

**What is AV1?** A royalty-free video codec developed by Alliance for Open Media. ~30% smaller files than H.264 but requires specific decoder support.

**Solution**: Use `imageio` with `pyav` backend, which links against ffmpeg and supports AV1.

```python
import imageio.v3 as iio
frames = iio.imread("video.mp4", plugin="pyav")  # Works with AV1
```

### 3. Joint-to-EEF Transform

Most Open X-Embodiment datasets (Bridge, RT-1, DROID, etc.) store actions as **end-effector pose deltas** — the change in position/orientation between consecutive frames. This is the de facto standard for training vision-language-action models.

However, LeRobot stores **joint positions** (the raw motor angles). To convert:

1. **Forward Kinematics (FK)**: Given joint angles + robot URDF, compute the EEF pose (4x4 transform matrix)
2. **Delta computation**: `action[t] = pose[t] - pose[t-1]`

**What is placo?** A C++ robotics library (with Python bindings) for kinematics. Given a URDF robot model and joint angles, it computes the 4x4 transformation matrix of any link in the kinematic chain.

```python
from so101_ik_fk.lib.so101_kinematics import SO101ForwardKinematics

fk = SO101ForwardKinematics("/path/to/robot.urdf")
pose_matrix = fk.compute([37.4, 7.1, -5.0, 90.7, 0.2])  # joint angles in degrees
# pose_matrix[:3, 3] = [x, y, z] position in meters
# pose_matrix[:3, :3] = rotation matrix → convert to roll, pitch, yaw
```

### 4. Gripper Binarization

LeRobot stores gripper as continuous angle (e.g., 0-50 degrees).
RLDS convention uses binary: 0 = closed, 1 = open.

**Solution**: Auto-compute threshold as midpoint of observed range, then binarize.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` | TFRecord read/write |
| `imageio[pyav]` | AV1 video decoding |
| `so101_ik_fk` | Forward kinematics for SO101 (optional, includes URDF + meshes) |
| `pandas` | Parquet file reading |
| `huggingface_hub` | Dataset downloading |
| `loguru` | Logging |

Install:
```bash
# Base installation
pip install -e .

# With SO101 robot support
pip install -e ".[so101]"
```

---

## Requirements

- **Local dataset**: Download via `huggingface_hub.snapshot_download()` first
- **SO101 FK**: Installed via `so101_ik_fk` package (includes URDF and meshes)

---

## Using with Different Robots

The converter supports **pluggable FK backends**.

### SO101 Robot (Default)

```bash
pip install -e ".[so101]"
```

Then just use the default config - no extra setup needed.

### Option 1: Custom URDF (if your FK module is similar to SO101)

```python
config = ConversionConfig(
    urdf_path="/path/to/your_robot.urdf",
    num_arm_joints=6,        # Adjust for your robot
    gripper_joint_index=6,   # Index of gripper in observation.state
)
```

### Option 2: Custom FK Class

Create a class that implements the `FKInterface` protocol:

```python
class MyRobotFK:
    def __init__(self, urdf_path: str):
        # Load your robot model
        self.model = load_model(urdf_path)

    def compute(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return 4x4 homogeneous transformation matrix."""
        return self.model.forward_kinematics(joint_angles)

# Use it:
config = ConversionConfig(
    fk_class=MyRobotFK,
    urdf_path="/path/to/your_robot.urdf",
    num_arm_joints=7,
)
```

### Option 3: Pre-initialized FK Backend

```python
from my_robot_lib import MyRobotKinematics

my_fk = MyRobotKinematics("/path/to/urdf")

config = ConversionConfig(
    fk_backend=my_fk,  # Pass the instance directly
    num_arm_joints=6,
)
```

### FK Interface Requirements

Your FK class must have a `compute()` method:

```python
def compute(self, joint_angles: np.ndarray) -> np.ndarray:
    """
    Args:
        joint_angles: Array of shape (num_arm_joints,)

    Returns:
        4x4 homogeneous transformation matrix (np.ndarray)
        [[R R R x]
         [R R R y]
         [R R R z]
         [0 0 0 1]]
    """
```
