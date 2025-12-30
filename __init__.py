"""
HF to RLDS Converter

Converts LeRobot (HuggingFace) datasets to RLDS (TFRecord) format.

Usage:
    from hf_to_rlds import ConversionConfig, convert_lerobot_to_rlds

    config = ConversionConfig(
        hf_repo_id="your-username/your-dataset",
        local_data_dir="./my_data",
        output_path="converted.tfrecord",
        num_images=1,
    )
    convert_lerobot_to_rlds(config)
"""

from .lib import (
    ConversionConfig,
    convert_lerobot_to_rlds,
    FKConverter,
    VideoLoader,
    RLDSWriter,
)

__version__ = "0.1.0"

__all__ = [
    "ConversionConfig",
    "convert_lerobot_to_rlds",
    "FKConverter",
    "VideoLoader",
    "RLDSWriter",
]
