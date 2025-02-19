"""
data.py

This module defines the functions to define the dataset and to do the necessary processing.

Authors: Sandro Mikautadze, Elio Samaha.
"""

import os
import json
import subprocess
from typing import Optional, Dict

def _load_json(json_path: str) -> Dict:
    """
    Loads the JSON annotation file.

    Args:
        json_path (str): Path to the JSON annotation file.

    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def _extract_frame(movie_id: str, shot_id: str, scale_label: str, split: str, video_root: str, output_dir: str) -> Optional[str]:
    """
    Extracts a single middle frame from a given video and saves it in the corresponding split directory.

    Args:
        movie_id (str): Unique identifier for a movie.
        shot_id (str): Unique identifier for a shot within a movie.
        scale_label (str): The shot scale label (e.g., 'CS', 'MS', 'FS', etc.).
        split (str): Dataset split ('train', 'val', or 'test').
        video_root (str): Path to the root directory containing video files.
        output_dir (str): Path to the root directory where extracted frames will be saved.

    Returns:
        str or None: Path to the saved frame image if successful, otherwise None.
    """
    video_path = os.path.join(video_root, movie_id, f"shot_{shot_id}.mp4")
    scale_dir = os.path.join(output_dir, split, scale_label)  # Save in the split folder
    output_path = os.path.join(scale_dir, f"{movie_id}_{shot_id}.jpg")

    os.makedirs(scale_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "select=eq(n\\,floor(n/2))",  # Extract middle frame
        "-vsync", "vfr",
        "-q:v", "10",  # Quality setting
        output_path,
        "-y"  # Overwrite existing file
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Frame saved to: {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        print(f"Error extracting frame from {video_path}")
        return None

def process_all_splits(video_root: str, json_path: str, output_dir: str) -> None:
    """
    Processes train, val, and test splits and extracts frames for all available videos.

    Args:
        video_root (str): Path to the root directory containing video files.
        json_path (str): Path to the JSON annotation file.
        output_dir (str): Path to the root directory where extracted frames will be saved.
    """
    data = _load_json(json_path)

    for split in ["train", "val", "test"]:
        if split not in data:
            print(f"Warning: No {split} split found in JSON")
            continue

        for movie_id, shots in data[split].items():
            for shot_id, metadata in shots.items():
                scale_label = metadata["scale"]["label"]
                _extract_frame(movie_id, shot_id, scale_label, split, video_root, output_dir)