"""
Core conversion logic: LeRobot (HuggingFace) -> RLDS (TFRecord).
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
from loguru import logger

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import ConversionConfig
from .fk_converter import FKConverter
from .video_loader import VideoLoader
from .tfrecord_writer import RLDSWriter


def load_task_mapping(hf_repo_id: str) -> Dict[int, str]:
    """
    Load task index to instruction text mapping.

    Args:
        hf_repo_id: Hugging Face dataset repository ID.

    Returns:
        Dictionary mapping task_index to instruction string.
    """
    try:
        tasks_path = hf_hub_download(
            hf_repo_id,
            "meta/tasks.parquet",
            repo_type="dataset"
        )
        tasks_df = pd.read_parquet(tasks_path)

        task_map = {}
        for task_name in tasks_df.index:
            task_idx = int(tasks_df.loc[task_name, "task_index"])
            task_map[task_idx] = str(task_name)

        return task_map
    except Exception as e:
        logger.warning(f"Could not load task mapping: {e}")
        return {}


def convert_lerobot_to_rlds(config: ConversionConfig, verbose: bool = True) -> int:
    """
    Convert a LeRobot dataset to RLDS format.

    Args:
        config: Conversion configuration.
        verbose: Enable logging output.

    Returns:
        Number of episodes converted.
    """
    if not verbose:
        logger.disable(__name__)

    logger.info("=" * 60)
    logger.info("LeRobot -> RLDS Conversion")
    logger.info("=" * 60)
    logger.info(f"Source: {config.hf_repo_id}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Num Images: {config.num_images}")
    logger.info(f"Image Size: {config.image_size}")

    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load HuggingFace dataset
    logger.info("[1/5] Loading HuggingFace dataset...")
    dataset = load_dataset(config.hf_repo_id, split="train")

    episode_indices = np.array(dataset["episode_index"])
    unique_episodes = np.unique(episode_indices)

    if config.max_episodes is not None:
        unique_episodes = unique_episodes[:config.max_episodes]

    logger.info(f"  Total frames: {len(dataset)}")
    logger.info(f"  Episodes to convert: {len(unique_episodes)}")

    # 2. Load task mapping
    logger.info("[2/5] Loading task mapping...")
    task_map = load_task_mapping(config.hf_repo_id)
    if task_map:
        logger.info(f"  Found {len(task_map)} tasks:")
        for idx, text in task_map.items():
            logger.info(f"    {idx}: {text}")
    else:
        logger.info("  No task mapping found, using default instruction.")

    # 3. Initialize FK converter
    logger.info("[3/5] Initializing FK converter...")
    fk_converter = FKConverter(
        urdf_path=config.urdf_path,
        fk_backend=config.fk_backend,
        fk_class=config.fk_class,
        num_arm_joints=config.num_arm_joints,
    )
    logger.info("  Forward Kinematics ready.")

    # 4. Initialize video loader
    logger.info("[4/5] Initializing video loader...")
    video_loader = VideoLoader(
        config.hf_repo_id,
        local_dir=config.local_data_dir,
        image_size=config.image_size,
    )
    logger.info(f"  Camera order: {config.camera_order[:config.num_images]}")
    if config.local_data_dir:
        logger.info(f"  Using local data: {config.local_data_dir}")

    # 5. Convert episodes
    logger.info("[5/5] Converting episodes...")

    # Compute gripper threshold if needed
    gripper_idx = config.gripper_joint_index
    gripper_threshold = config.gripper_threshold
    if config.binarize_gripper and gripper_threshold is None:
        all_gripper = []
        for idx in unique_episodes[:min(10, len(unique_episodes))]:
            mask = episode_indices == idx
            subset = dataset.select(np.where(mask)[0])
            for row in subset:
                all_gripper.append(row["observation.state"][gripper_idx])
        gripper_threshold = (min(all_gripper) + max(all_gripper)) / 2
        logger.info(f"  Auto-computed gripper threshold: {gripper_threshold:.2f}")

    with RLDSWriter(str(config.output_path), config.num_images) as writer:
        iterator = tqdm(unique_episodes, desc="Converting") if verbose else unique_episodes
        for ep_idx in iterator:
            mask = episode_indices == ep_idx
            frame_indices = np.where(mask)[0]
            subset = dataset.select(frame_indices)
            frame_order = np.argsort(subset["frame_index"])

            T = len(frame_order)
            joint_sequence = np.zeros((T, 6), dtype=np.float32)
            gripper_sequence = np.zeros(T, dtype=np.float32)
            task_idx = 0

            for i, order_idx in enumerate(frame_order):
                row = subset[int(order_idx)]
                joint_sequence[i] = row["observation.state"]
                gripper_sequence[i] = row["observation.state"][gripper_idx]
                task_idx = row["task_index"]

            instruction = task_map.get(task_idx, "perform the task")

            actions = fk_converter.compute_episode_deltas(
                joint_sequence,
                gripper_sequence,
                binarize_gripper=config.binarize_gripper,
                gripper_threshold=gripper_threshold,
            )

            states = fk_converter.compute_episode_states(
                joint_sequence,
                gripper_sequence,
                binarize_gripper=config.binarize_gripper,
                gripper_threshold=gripper_threshold,
            )

            images: Dict[int, List[bytes]] = {}
            for img_slot in range(config.num_images):
                if img_slot < len(config.camera_order):
                    camera = config.camera_order[img_slot]
                    images[img_slot] = video_loader.extract_frames(int(ep_idx), camera, T)
                else:
                    images[img_slot] = video_loader._generate_black_frames(T)

            writer.write_episode(
                episode_id=int(ep_idx),
                images=images,
                actions=actions,
                states=states,
                instruction=instruction,
                file_path=f"{config.hf_repo_id}/episode_{ep_idx}",
            )

    logger.success("=" * 60)
    logger.success("Conversion complete!")
    logger.success(f"  Episodes converted: {writer.get_episodes_written()}")
    logger.success(f"  Output file: {config.output_path}")
    logger.success(f"  File size: {config.output_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.success("=" * 60)

    if not verbose:
        logger.enable(__name__)

    return writer.get_episodes_written()
