"""
Configuration for LeRobot to RLDS conversion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path


@dataclass
class ConversionConfig:
    """Configuration for converting LeRobot dataset to RLDS format.

    Attributes:
        hf_repo_id: Hugging Face dataset repository ID.
        output_path: Path to output TFRecord file.
        local_data_dir: Path to locally downloaded HF dataset.
        num_images: Number of camera images to include (1-4).
        camera_order: Order of cameras to use from the LeRobot dataset.
        max_episodes: Maximum number of episodes to convert. None = all.
        image_size: Target image size (height, width).
        binarize_gripper: Whether to binarize gripper values.
        gripper_threshold: Threshold for binarizing gripper. None = auto.
        fps: Frames per second of the source dataset.
        urdf_path: Path to robot URDF for FK computation. None = default SO101.
        fk_backend: Pre-initialized FK backend instance (overrides urdf_path).
        fk_class: Custom FK class to use instead of SO101.
        num_arm_joints: Number of arm joints (excluding gripper). Default: 5.
        gripper_joint_index: Index of gripper in observation.state. Default: 5.
    """

    hf_repo_id: str = "sapanostic/so101_offline_eval"
    output_path: str = "converted_dataset.tfrecord"
    local_data_dir: Optional[str] = None

    # Image configuration
    num_images: int = 1
    camera_order: List[str] = field(default_factory=lambda: ["front", "top", "wrist"])
    image_size: tuple = (224, 224)

    # Episode limits
    max_episodes: Optional[int] = None

    # Gripper configuration
    binarize_gripper: bool = True
    gripper_threshold: Optional[float] = None

    # Source dataset info
    fps: int = 15

    # Robot/FK configuration
    urdf_path: Optional[str] = None
    fk_backend: Optional[Any] = None  # Pre-initialized FK instance
    fk_class: Optional[type] = None   # Custom FK class
    num_arm_joints: int = 5           # Joints for FK (excludes gripper)
    gripper_joint_index: int = 5      # Index in observation.state

    def __post_init__(self):
        if self.num_images < 1 or self.num_images > 4:
            raise ValueError(f"num_images must be 1-4, got {self.num_images}")

        self.output_path = Path(self.output_path)

        if self.fk_backend is not None and self.fk_class is not None:
            raise ValueError("Provide either fk_backend OR fk_class, not both.")
