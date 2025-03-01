"""
character_sheet.py

This module defines the function to generate a storyboard-style character sheet
based on a given character description using a diffusion model pipeline with ControlNet.
The generated images are saved in a structured folder hierarchy:
    stories/{normalized_story_name}/charactersheet/{normalized_character_name}/

Authors: Sandro Mikautadze, Elio Samaha.
"""

from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline

def generate_character_sheet(
    story_name: str,
    character_description: dict,
    pipe: StableDiffusionControlNetPipeline,
    device: torch.device = "cpu",
    guidance_scale: float = 8.5,
    num_inference_steps: int = 25,
    seed: int = 42,
    extra_negative_keywords: list = None
) -> list:
    """
    Generates a storyboard-style character sheet based on a character description.
    
    Generated images are saved in the directory structure:
        stories/{normalized_story_name}/charactersheet/{normalized_character_name}/

    Parameters:
        story_name (str): Name of the story (used for folder organization; will be normalized).
        character_description (dict): Dictionary containing character attributes. Must include:
            - "name" (str): Character's name.
            - "age" (str): Age descriptor (e.g., "young", "middle-aged", "elderly").
            - "gender" (str): Gender descriptor (must be "male" or "female").
            - "hair" (str): Hair description (e.g., "long curly red hair").
            - "clothing" (str): Clothing description.
            - "body_type" (str): Body type (e.g., "slim", "muscular", "fat").
            - "facial_hair" (str, required if male): Facial hair description for male characters.
            - "accessories" (str, optional): Accessories (e.g., "glasses", "hat", "scarf").
            - "ethnicity" (str, optional): Ethnicity (e.g., "Asian", "Black", "Caucasian").
        pipe (StableDiffusionControlNetPipeline): Diffusion model pipeline for image generation.
        device (torch.device): Torch device to be used (e.g., torch.device('cuda') or torch.device('cpu')).
        guidance_scale (float, optional): How closely the model follows the prompt (default: 8.5).
        num_inference_steps (int, optional): Number of inference steps for diffusion (default: 25).
        seed (int, optional): Random seed for reproducibility (default: 42).
        extra_negative_keywords (list, optional): Additional negative keywords to avoid.

    Returns:
        list: A list of file paths for the generated images.

    Raises:
        ValueError: If required keys are missing or invalid.
    """

    # Validate required keys
    required_keys = {"name", "age", "gender", "hair", "clothing", "body_type"}
    if not required_keys.issubset(character_description.keys()):
        raise ValueError(f"character_description must contain keys: {required_keys}")

    # Validate gender
    valid_genders = {"male", "female"}
    if character_description["gender"].lower() not in valid_genders:
        raise ValueError(f"Invalid gender '{character_description['gender']}'. Must be 'male' or 'female'.")

    # If the character is male, facial hair is required
    if character_description["gender"].lower() == "male" and "facial_hair" not in character_description:
        raise ValueError(f"Male characters must have a 'facial_hair' key in the description.")

    # Normalize story and character names
    normalized_story_name = story_name.replace(" ", "_").lower()
    normalized_character_name = character_description["name"].replace(" ", "_").lower()

    # Create save directory
    base_path = Path("stories") / normalized_story_name / "charactersheet"
    character_folder = base_path / normalized_character_name
    character_folder.mkdir(parents=True, exist_ok=True)

    # Build character features for prompts.
    character_features = (
        f"{character_description['body_type']} {character_description['age']} {character_description['gender']}, "
        f"{character_description['hair']}, {character_description['clothing']}"
    )

    # Add facial hair if male
    if character_description["gender"].lower() == "male":
        character_features += f", {character_description['facial_hair']}"

    # Add accessories if provided
    if "accessories" in character_description and character_description["accessories"]:
        character_features += f", wearing {character_description['accessories']}"

    # Add ethnicity if provided
    if "ethnicity" in character_description and character_description["ethnicity"]:
        character_features = f"{character_description['ethnicity']} {character_features}"

    # Fixed art style (hidden)
    art_style = "J.C. Leyendecker"

    # Define the pose image paths and shot descriptions
    pose_image_paths = [
        "./poses/front.png",     # Full-body, front view
        "./poses/side.png",      # Medium shot, side view
        "./poses/walking.png",   # Full-body, walking pose
        "./poses/greetings.png", # Medium shot, waving
        "./poses/head.png",      # Close-up, neutral
        "./poses/head2.png"      # Close-up, slight smile
    ]
    shot_types = [
        "full-body shot, front view",
        "medium shot, side view",
        "full-body shot, walking pose, back-side view",
        "medium shot, waving in greeting, three-quarters view",
        "close-up shot, neutral expression, face fully visible",
        "close-up shot, slight smile, looking away"
    ]

    # Create prompts dynamically
    prompts = [
        f"rough b&w pencil sketch of {character_features}, {shot}, simple sketch lines, "
        f"minimal shading, rough hatching, draft-style, white background, {art_style} style"
        for shot in shot_types
    ]

    # Define base negative prompt
    negative_prompt_base = (
        "photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, "
        "fine art, NSFW, shirtless, nudity"
    )
    negative_prompt = (
        negative_prompt_base + ", " + ", ".join(extra_negative_keywords)
        if extra_negative_keywords else negative_prompt_base
    )

    generated_images = []
    for i, (pose_path, prompt) in enumerate(zip(pose_image_paths, prompts)):
        try:
            pose_image = Image.open(pose_path).convert("RGB")
        except Exception as e:
            print(f"Error opening pose image {pose_path}: {e}")
            continue

        generator = torch.Generator(device).manual_seed(seed)

        try:
            generated_image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                image=pose_image,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                generator=generator
            ).images[0]
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            continue

        output_path = character_folder / f"{normalized_character_name}_pose_{i+1}.png"
        try:
            generated_image.save(output_path)
            generated_images.append(str(output_path))
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")

    # Final print when the sheet is done
    if generated_images:
        print(f"Character sheet generated for '{character_description['name']}'.")
    else:
        print(f"No images were generated for '{character_description['name']}' due to errors.")

    return generated_images