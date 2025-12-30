"""
TFRecord writer for RLDS format.

Creates TFRecord files compatible with SpatialVLA's data loading pipeline.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Any


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values: List[bytes]) -> tf.train.Feature:
    """Returns a bytes_list from a list of bytes."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_list_feature(values: List[float]) -> tf.train.Feature:
    """Returns a float_list from a list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_list_feature(values: List[int]) -> tf.train.Feature:
    """Returns an int64_list from a list of ints."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class RLDSWriter:
    """
    Writes episodes to TFRecord files in RLDS format.
    
    RLDS Format (per episode/record):
        - episode_metadata/episode_id: bytes (episode identifier)
        - episode_metadata/file_path: bytes (source path, optional)
        - episode_metadata/has_image_X: int64 (flags for image availability)
        - steps/observation/image_X: bytes_list (JPEG-encoded images)
        - steps/observation/state: float_list (flattened states)
        - steps/action: float_list (flattened actions)
        - steps/language_instruction: bytes_list (repeated instruction)
        - steps/is_first: int64_list (first step flag)
        - steps/is_last: int64_list (last step flag)
        - steps/is_terminal: int64_list (terminal flag)
        - steps/reward: float_list (rewards)
        - steps/discount: float_list (discounts)
    """
    
    def __init__(self, output_path: str, num_images: int = 1):
        """
        Initialize the writer.
        
        Args:
            output_path: Path to output TFRecord file.
            num_images: Number of image slots (1-4).
        """
        self.output_path = str(output_path)
        self.num_images = num_images
        self.writer = None
        self.episodes_written = 0
    
    def __enter__(self):
        self.writer = tf.io.TFRecordWriter(self.output_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()
        return False
    
    def write_episode(
        self,
        episode_id: int,
        images: Dict[int, List[bytes]],  # {image_idx: [frame_bytes]}
        actions: np.ndarray,  # (T, 7)
        states: np.ndarray,  # (T, 7)
        instruction: str,
        file_path: str = "lerobot_converted",
    ):
        """
        Write a single episode to the TFRecord file.
        
        Args:
            episode_id: Unique episode identifier.
            images: Dictionary mapping image index (0-3) to list of JPEG bytes.
            actions: Action array of shape (T, 7).
            states: State array of shape (T, 7).
            instruction: Language instruction for this episode.
            file_path: Source file path (metadata).
        """
        T = len(actions)
        
        # Build feature dictionary
        feature = {}
        
        # Episode metadata
        feature["episode_metadata/episode_id"] = _bytes_feature(
            str(episode_id).encode("utf-8")
        )
        feature["episode_metadata/file_path"] = _bytes_feature(
            file_path.encode("utf-8")
        )
        
        # Image availability flags and data
        for img_idx in range(4):
            has_image = img_idx in images and len(images[img_idx]) > 0
            feature[f"episode_metadata/has_image_{img_idx}"] = _int64_list_feature(
                [1 if has_image else 0]
            )
            
            if has_image:
                feature[f"steps/observation/image_{img_idx}"] = _bytes_list_feature(
                    images[img_idx]
                )
            else:
                # Empty placeholder - black frames will be handled by data loader
                # But we need the key to exist for some loaders
                pass
        
        # Actions - flatten to 1D
        actions_flat = actions.flatten().astype(np.float32).tolist()
        feature["steps/action"] = _float_list_feature(actions_flat)
        
        # States - flatten to 1D
        states_flat = states.flatten().astype(np.float32).tolist()
        feature["steps/observation/state"] = _float_list_feature(states_flat)
        
        # Language instruction - repeated for each step
        instruction_bytes = instruction.encode("utf-8")
        feature["steps/language_instruction"] = _bytes_list_feature(
            [instruction_bytes] * T
        )
        
        # Control flow flags
        is_first = [1] + [0] * (T - 1)
        is_last = [0] * (T - 1) + [1]
        is_terminal = [0] * (T - 1) + [1]  # Assume all episodes end normally
        
        feature["steps/is_first"] = _int64_list_feature(is_first)
        feature["steps/is_last"] = _int64_list_feature(is_last)
        feature["steps/is_terminal"] = _int64_list_feature(is_terminal)
        
        # Rewards and discounts (placeholder values)
        feature["steps/reward"] = _float_list_feature([0.0] * T)
        feature["steps/discount"] = _float_list_feature([1.0] * T)
        
        # Create and write example
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.episodes_written += 1
    
    def get_episodes_written(self) -> int:
        """Return number of episodes written."""
        return self.episodes_written

