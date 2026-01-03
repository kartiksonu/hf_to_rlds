#!/usr/bin/env python3
"""
Download and convert a HuggingFace LeRobot dataset to RLDS format.

Usage:
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --output_dir ./output

    # With specific number of cameras
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --num_images 3

    # Limit episodes (for testing)
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --max_episodes 5
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from huggingface_hub import snapshot_download
from lib.config import ConversionConfig
from lib.converter import convert_lerobot_to_rlds


def download_dataset(repo_id: str, local_dir: Path) -> Path:
    """
    Download dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "sapanostic/so101_offline_eval")
        local_dir: Local directory to download to

    Returns:
        Path to downloaded dataset
    """
    logger.info(f"Downloading dataset: {repo_id}")
    logger.info(f"Destination: {local_dir}")

    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    logger.success(f"Download complete: {dataset_path}")
    return Path(dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert HuggingFace LeRobot dataset to RLDS format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and convert full dataset
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval

    # Specify output directory
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --output_dir ./my_output

    # Use 3 cameras
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --num_images 3

    # Convert only first 5 episodes (for testing)
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --max_episodes 5

    # Skip download if already downloaded
    python scripts/download_and_convert.py --repo_id sapanostic/so101_offline_eval --skip_download --local_dir ./data/so101
        """
    )

    # Required
    parser.add_argument(
        "--repo_id",
        type=str,
        default="sapanostic/so101_offline_eval",
        help="HuggingFace dataset repository ID"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for RLDS TFRecord files"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (default: {repo_name}.tfrecord)"
    )

    # Download options
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory for downloaded dataset (default: ./data/{repo_name})"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download if dataset already exists locally"
    )

    # Conversion options
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of camera images to include (1-4)"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert (default: all)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Target image size (default: 224 224)"
    )
    parser.add_argument(
        "--camera_order",
        type=str,
        nargs="+",
        default=["front", "top", "wrist"],
        help="Camera order for image slots (default: front top wrist)"
    )

    args = parser.parse_args()

    # Determine paths
    repo_name = args.repo_id.split("/")[-1]

    if args.local_dir:
        local_dir = Path(args.local_dir)
    else:
        local_dir = Path("./data") / repo_name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        output_path = output_dir / f"{repo_name}.tfrecord"

    # Download dataset
    if args.skip_download and local_dir.exists():
        logger.info(f"Skipping download, using existing: {local_dir}")
        dataset_path = local_dir
    else:
        dataset_path = download_dataset(args.repo_id, local_dir)

    # Configure conversion
    config = ConversionConfig(
        hf_repo_id=args.repo_id,
        output_path=str(output_path),
        local_data_dir=str(dataset_path),
        num_images=args.num_images,
        camera_order=args.camera_order,
        image_size=tuple(args.image_size),
        max_episodes=args.max_episodes,
    )

    # Run conversion
    logger.info("=" * 60)
    logger.info("Starting conversion")
    logger.info("=" * 60)
    logger.info(f"  Source: {dataset_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Cameras: {args.num_images} ({args.camera_order[:args.num_images]})")
    logger.info(f"  Image size: {args.image_size}")
    if args.max_episodes:
        logger.info(f"  Max episodes: {args.max_episodes}")
    logger.info("=" * 60)

    convert_lerobot_to_rlds(config)

    logger.success("=" * 60)
    logger.success("Conversion complete!")
    logger.success(f"Output: {output_path}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
