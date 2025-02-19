"""
data.py

This module defines the functions to get the dataset and to do the necessary processing.

Authors: Sandro Mikautadze, Elio Samaha.
"""

import os
import random
import json
import subprocess
from typing import Optional, Dict, Callable, Tuple, List
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
        
    Raises:
        RuntimeError: If the file is missing or not a valid JSON.
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


class AddGaussianNoise:
    """
    Applies Gaussian noise to an image with a given standard deviation.
    The noise is applied to a fixed proportion of images, ensuring an equal number of images for each noise level.

    Attributes:
        noise_stds (List[float]): List of standard deviations for noise levels.
        noisy_percentage (float): Proportion (0 to 1) of images that receive noise.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, noisy_percentage: float = 0.3, noise_stds: List[float] = [0.05, 0.1, 0.2, 0.4, 0.5], seed: int = 42):
        self.noisy_percentage = noisy_percentage  # Total fraction of noisy images
        self.noise_stds = noise_stds  # List of standard deviations
        self.seed = seed
        random.seed(seed)

    def assign_noise_levels(self, dataset_size: int) -> List[float]:
        """
        Assigns noise levels to a subset of images in the dataset to ensure equal distribution.

        Args:
            dataset_size (int): Total number of images in the dataset.

        Returns:
            List[float]: A list where each element corresponds to the noise level for an image.
                         If an image is clean, its noise level is set to 0.
        """
        num_noisy = int(self.noisy_percentage * dataset_size)  # Total noisy images
        num_per_level = num_noisy // len(self.noise_stds)  # Equal distribution per noise level
        remaining = num_noisy % len(self.noise_stds)  # Handle remainder

        # Assign equal noise levels
        noise_levels = []
        for std in self.noise_stds:
            noise_levels.extend([std] * num_per_level)

        # Distribute remaining images randomly
        noise_levels.extend(random.choices(self.noise_stds, k=remaining))

        # Pad with zeros for clean images
        noise_levels.extend([0] * (dataset_size - len(noise_levels)))

        # Shuffle to ensure randomness
        random.shuffle(noise_levels)

        return noise_levels
    
    def __call__(self, img: torch.Tensor, noise_std: float) -> torch.Tensor:
        """
        Apply noise to an image based on a given noise standard deviation.

        Args:
            img (torch.Tensor): Image tensor.
            noise_std (float): Standard deviation of the noise. If 0, no noise is added.

        Returns:
            torch.Tensor: Noisy or clean image.
        """
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        if noise_std > 0:
            noise = torch.randn_like(img) * noise_std
            img = torch.clamp(img + noise, 0, 1)

        return img

class LensTypeDataset(Dataset):
    """
    A PyTorch dataset class for movie shots and their corresponding shot lens type labels.
    Labels are:
    - 0 for Extreme Close-Up Shot (ECS)
    - 1 for Close-Up Shot (CS)
    - 2 for Medium Shot (MS)
    - 3 for Full-Shot (FS)
    - 4 for Long-Shot (LS)
    
    Ensures that a fixed proportion of images are noisy and distributes noise levels equally.

    Args:
        root_dir (str): Root dataset directory.
        split (str): Dataset split ('train', 'val', or 'test').
        transform (Optional[Callable]): Image transformations.
        noisy_percentage (float): Proportion of images with noise.
        noise_stds (List[float]): List of noise standard deviations.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None,
                 noisy_percentage: float = 0.3, noise_stds: List[float] = [0.05, 0.1, 0.2, 0.4, 0.5], seed: int = 42):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.add_noise = AddGaussianNoise(noisy_percentage, noise_stds, seed)
        self.image_paths = []
        self.labels = []

        self.label_mapping = {"ECS": 0, "CS": 1, "MS": 2, "FS": 3, "LS": 4}

        for label in self.label_mapping.keys():
            label_dir = os.path.join(self.split_dir, label)
            if not os.path.isdir(label_dir):
                continue  
            
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.label_mapping[label])

        # Assign fixed noise levels to dataset
        self.noise_levels = self.add_noise.assign_noise_levels(len(self.image_paths))

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve an image and its corresponding label by index.

        Args:
            idx (int): Index of the image.

        Returns:
            Tuple[torch.Tensor, int]: The image (clean or noisy) and its corresponding label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        noise_std = self.noise_levels[idx]  # Precomputed noise level

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        image = self.add_noise(image, noise_std)

        return image, label