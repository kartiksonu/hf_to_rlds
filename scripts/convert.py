#!/usr/bin/env python3
"""
CLI for LeRobot -> RLDS conversion.

Usage:
    python scripts/convert.py --help
    python scripts/convert.py --repo_id your/dataset --local_dir ./data --output out.tfrecord
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).parents[1]))

from lib.config import ConversionConfig
from lib.converter import convert_lerobot_to_rlds


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot (HuggingFace) dataset to RLDS (TFRecord) format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with 1 camera (front only)
  python scripts/convert.py --repo_id user/dataset --local_dir ./data --output out.tfrecord

  # Convert with all 3 cameras, limit to 5 episodes
  python scripts/convert.py --repo_id user/dataset --local_dir ./data --num_images 3 --max_episodes 5
        """
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Path to locally downloaded HF dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output TFRecord file path"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of camera images to include (1-4). Default: 1"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum episodes to convert. Default: all"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Image size (height width). Default: 224 224"
    )
    parser.add_argument(
        "--no_binarize_gripper",
        action="store_true",
        help="Do not binarize gripper values"
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help="Gripper binarization threshold. Default: auto"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default=None,
        help="Path to robot URDF file. Default: SO101"
    )

    args = parser.parse_args()

    config = ConversionConfig(
        hf_repo_id=args.repo_id,
        output_path=args.output,
        local_data_dir=args.local_dir,
        num_images=args.num_images,
        max_episodes=args.max_episodes,
        image_size=tuple(args.image_size),
        binarize_gripper=not args.no_binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        urdf_path=args.urdf_path,
    )

    num_converted = convert_lerobot_to_rlds(config)

    if num_converted == 0:
        print("ERROR: No episodes converted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
