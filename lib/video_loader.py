"""
Video frame extraction from LeRobot datasets.

Loads MP4 video files and extracts frames as JPEG bytes.
Handles LeRobot's chunked video format where multiple episodes are concatenated.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from huggingface_hub import hf_hub_download
import io
from PIL import Image
from loguru import logger

try:
    import imageio.v3 as iio
    USE_IMAGEIO = True
except ImportError:
    import cv2
    USE_IMAGEIO = False


class VideoLoader:
    """
    Loads video frames from LeRobot dataset hosted on Hugging Face.

    LeRobot stores multiple episodes in each video file. This class handles
    the mapping from episode index to the correct video file and frame offset.
    """

    # Camera key mapping for LeRobot SO101 dataset
    CAMERA_KEYS = {
        "front": "observation.images.front",
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
        "ego": "observation.images.ego",
        "realsense": "observation.images.realsense",
    }

    def __init__(
        self,
        hf_repo_id: str,
        local_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize the video loader.

        Args:
            hf_repo_id: Hugging Face dataset repository ID.
            local_dir: Local directory containing downloaded dataset.
                       If provided, uses local files instead of downloading.
            image_size: Target image size (height, width) for resizing.
        """
        self.hf_repo_id = hf_repo_id
        self.local_dir = Path(local_dir) if local_dir else None
        self.image_size = image_size
        self._video_cache = {}
        self._episode_map = None  # Maps episode_index -> (file_index, start_frame, num_frames)

    def _build_episode_map(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Build mapping from episode index to (file_index, start_frame, num_frames).

        Returns:
            Dictionary mapping episode_index to (file_index, start_frame_in_file, num_frames).
        """
        if self._episode_map is not None:
            return self._episode_map

        self._episode_map = {}

        # Load episodes metadata
        if self.local_dir:
            meta_dir = self.local_dir / "meta" / "episodes" / "chunk-000"
        else:
            # Download meta files
            meta_dir = None

        # Read all episode parquet files to build the mapping
        file_idx = 0
        while True:
            try:
                if self.local_dir:
                    parquet_path = self.local_dir / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
                    if not parquet_path.exists():
                        break
                    df = pd.read_parquet(parquet_path)
                else:
                    parquet_path = hf_hub_download(
                        self.hf_repo_id,
                        f"data/chunk-000/file-{file_idx:03d}.parquet",
                        repo_type="dataset",
                    )
                    df = pd.read_parquet(parquet_path)

                # Group by episode to get frame counts
                for ep_idx in df['episode_index'].unique():
                    ep_data = df[df['episode_index'] == ep_idx]
                    # start_frame is the cumulative frame count within this file
                    # before this episode
                    start_frame = len(df[df['episode_index'] < ep_idx])
                    num_frames = len(ep_data)
                    self._episode_map[int(ep_idx)] = (file_idx, start_frame, num_frames)

                file_idx += 1
            except Exception as e:
                break

        return self._episode_map

    def _get_video_path(self, file_index: int, camera: str) -> Optional[str]:
        """
        Get path to video file.

        Args:
            file_index: Index of the video file (file-000, file-001, etc.).
            camera: Camera name ("front", "top", or "wrist").

        Returns:
            Local path to the video file.
        """
        camera_key = self.CAMERA_KEYS.get(camera, camera)

        if self.local_dir:
            video_path = self.local_dir / "videos" / camera_key / "chunk-000" / f"file-{file_index:03d}.mp4"
            if video_path.exists():
                return str(video_path)
            return None

        # Download from HuggingFace
        video_filename = f"videos/{camera_key}/chunk-000/file-{file_index:03d}.mp4"

        try:
            local_path = hf_hub_download(
                self.hf_repo_id,
                video_filename,
                repo_type="dataset",
            )
            return local_path
        except Exception as e:
            logger.warning(f"Could not get video file-{file_index:03d} for {camera}: {e}")
            return None

    def extract_frames(
        self,
        episode_index: int,
        camera: str,
        num_frames: Optional[int] = None,
    ) -> List[bytes]:
        """
        Extract frames for a specific episode from the concatenated video file.

        Args:
            episode_index: Episode index.
            camera: Camera name.
            num_frames: Expected number of frames. If provided, will pad/truncate.

        Returns:
            List of JPEG-encoded frame bytes.
        """
        # Build episode map if needed
        episode_map = self._build_episode_map()

        if episode_index not in episode_map:
            logger.warning(f"Episode {episode_index} not found in episode map")
            return self._generate_black_frames(num_frames or 1)

        file_index, start_frame, ep_num_frames = episode_map[episode_index]

        video_path = self._get_video_path(file_index, camera)

        if video_path is None:
            return self._generate_black_frames(num_frames or ep_num_frames)

        if USE_IMAGEIO:
            frames = self._extract_frames_imageio(video_path, start_frame, ep_num_frames)
        else:
            frames = self._extract_frames_opencv(video_path, start_frame, ep_num_frames)

        # Handle frame count mismatch
        if num_frames is not None:
            if len(frames) < num_frames:
                padding = self._generate_black_frames(num_frames - len(frames))
                frames.extend(padding)
            elif len(frames) > num_frames:
                frames = frames[:num_frames]

        return frames

    def _extract_frames_imageio(
        self, video_path: str, start_frame: int, num_frames: int
    ) -> List[bytes]:
        """Extract frames using imageio (better AV1 support)."""
        frames = []
        try:
            # Read all frames and slice - imageio handles AV1 via ffmpeg
            all_frames = iio.imread(video_path, plugin="pyav")
            end_frame = start_frame + num_frames

            for frame in all_frames[start_frame:end_frame]:
                # Resize
                img = Image.fromarray(frame)
                img_resized = img.resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.Resampling.LANCZOS
                )

                # Encode to JPEG
                jpeg_bytes = self._encode_jpeg(np.array(img_resized))
                frames.append(jpeg_bytes)
        except Exception as e:
            logger.warning(f"imageio failed to read video: {e}")
            return self._generate_black_frames(num_frames)

        return frames

    def _extract_frames_opencv(
        self, video_path: str, start_frame: int, num_frames: int
    ) -> List[bytes]:
        """Extract frames using OpenCV (fallback)."""
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb,
                    (self.image_size[1], self.image_size[0]),
                    interpolation=cv2.INTER_AREA
                )

                jpeg_bytes = self._encode_jpeg(frame_resized)
                frames.append(jpeg_bytes)
        finally:
            cap.release()

        return frames

    def _encode_jpeg(self, frame: np.ndarray, quality: int = 95) -> bytes:
        """
        Encode numpy array to JPEG bytes.

        Args:
            frame: RGB image as numpy array (H, W, 3).
            quality: JPEG quality (0-100).

        Returns:
            JPEG-encoded bytes.
        """
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    def _generate_black_frames(self, num_frames: int) -> List[bytes]:
        """
        Generate black (padding) frames.

        Args:
            num_frames: Number of frames to generate.

        Returns:
            List of JPEG-encoded black frames.
        """
        black_frame = np.zeros(
            (self.image_size[0], self.image_size[1], 3),
            dtype=np.uint8
        )
        black_jpeg = self._encode_jpeg(black_frame)
        return [black_jpeg] * num_frames

    def get_black_frame(self) -> bytes:
        """Get a single black (padding) frame."""
        return self._generate_black_frames(1)[0]
