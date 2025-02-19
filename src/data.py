"""
data.py

This module defines the functions to get the dataset and to do the necessary processing.

Authors: Sandro Mikautadze, Elio Samaha.
"""

import os
import json
import subprocess
from typing import Optional, Dict, Callable, Tuple
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

#######################
### DATA EXTRACTION ###
#######################

def _load_json(json_path: str) -> Dict:
    """
    Load the JSON annotation file.

    Args:
        json_path (str): Path to the JSON annotation file.

    Returns:
        Dict: Parsed JSON data as a dictionary.
    """
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading JSON file {json_path}: {e}")

def _extract_frame(
    movie_id: str, shot_id: str, scale_label: str, split: str, 
    video_root: str, image_quality: int, output_dir: str
) -> Optional[str]:
    """
    Extract a single middle frame from a given video and save it in the corresponding split directory.

    Args:
        movie_id (str): Unique identifier for a movie.
        shot_id (str): Unique identifier for a shot within a movie.
        scale_label (str): The shot scale label (e.g., 'CS', 'MS', 'FS', etc.).
        split (str): Dataset split ('train', 'val', or 'test').
        video_root (str): Path to the root directory containing video files.
        image_quality (int): Quality factor for the frame extraction. Minimum value is 2, maximum is 31; 
                             the smaller the value, the higher the quality.
        output_dir (str): Path to the root directory where extracted frames will be saved.

    Returns:
        Optional[str]: Path to the saved frame image if successful, otherwise None.
    """
    
    if not (2 <= image_quality <= 31):
        raise ValueError("image_quality must be between 2 (best) and 31 (worst).")

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
        "-q:v", f"{image_quality}",  # Quality setting
        output_path,
        "-y"  # Overwrite existing file
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frame from {video_path}: {e.stderr.decode()}")  # Proper error handling
        return None

def process_all_splits(video_root: str, json_path: str, image_quality: int, output_dir: str) -> None:
    """
    Process train, val, and test splits and extract frames for all available videos, saving them in a directory.

    Args:
        video_root (str): Path to the root directory containing video files.
        json_path (str): Path to the JSON annotation file.
        image_quality (int): Quality factor for the frame extraction. Minimum value is 2, maximum is 31; 
                             the smaller the value, the higher the quality.
        output_dir (str): Path to the root directory where extracted frames will be saved.

    Returns:
        None
    """
    data = _load_json(json_path)
    
    total_videos = sum(len(shots) for split in ["train", "val", "test"] if split in data for shots in data[split].values())

    with tqdm(total=total_videos, desc="Extracting Frames", unit="frame") as pbar:
        for split in ["train", "val", "test"]:
            if split not in data:
                print(f"Warning: No {split} split found in JSON")
                continue

            for movie_id, shots in data[split].items():
                for shot_id, metadata in shots.items():
                    scale_label = metadata["scale"]["label"]
                    _extract_frame(movie_id, shot_id, scale_label, split, video_root, image_quality, output_dir)
                    pbar.update(1)

######################
### DATASET OBJECT ###
######################

class LensTypeDataset(Dataset):
    """
    A PyTorch dataset class for movie shots and their corresponding shot lens type labels.
    Labes are
    - 0 for Extreme Close-Up Shot (ECS)
    - 1 for Close-Up Shot (CS)
    - 2 for Medium Shot (CS)
    - 3 for Full-Shot (FS)
    - 4 for Long-Shot (LS)
    
    It works for the dataset from "A Unified Framework for Shot Type Classification Based on Subject Centric Lens", ECCV 2020.
    """

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None):
        """
        Initialize the dataset by loading image paths and labels.

        Args:
            root_dir (str): Root directory of the dataset (e.g., "./data").
            split (str): One of "train", "val", or "test".
            transform (Optional[Callable]): A function/transform that takes in an image 
                                            and returns a transformed version.
        """
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.label_mapping = {"ECS": 0, "CS": 1, "MS": 2, "FS": 3, "LS": 4}

        for label in self.label_mapping.keys():
            label_dir = os.path.join(self.split_dir, label)
            if not os.path.isdir(label_dir):
                continue  # Skip if folder is missing
            
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.label_mapping[label])

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Retrieve an image and its corresponding label by index.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[Image.Image, int]: The image and its corresponding label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label