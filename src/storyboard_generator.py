"""
storyboard_generator.py

This module defines the StoryboardGenerator class for generating storyboard scenes.

Authors: Sandro Mikautadze, Elio Samaha.
"""



import os
import json
import logging
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
import torch
from PIL import Image
from together import Together
from src.prompt_scheme import SceneList
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from src.models import MultiPromptPipelineApproach1

# Set up module-level logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(stream_handler)


class StoryboardGenerator:
    ORIENTATIONS: List[str] = [
        "Front View", "Profile View", "Back View", "From Behind", "From Above",
        "From Below", "Three-Quarters View", "Long Shot", "Three-Quarters Rear View"
    ]

    CAMERA_SHOTS: List[str] = [
        "Aerial View", "Bird’s-Eye View", "Close-Up", "Cowboy Shot", "Dolly Zoom",
        "Dutch Angle", "Establishing Shot", "Extreme Close-Up", "Extreme Long Shot",
        "Full Shot", "Long Shot", "Medium Close-Up", "Medium Long Shot", "Medium Shot",
        "Over-the-Shoulder Shot", "Point-of-View Shot", "Two-Shot", "Fisheye Shot",
        "Worm's Eye", "Low-Angle Shot", "Macro Shot", "Tilt-Shift Shot", "Telephoto Shot"
    ]
    
    def __init__(
        self, 
        script: str, 
        characters: Dict[str, Dict[str, str]], 
        style: str = "storyboard", 
        prompt_weights: List[float] = [2, 1.0, 1.2, 1.5, 0.9],  # used only for 'prompt_weights' generation and 'modified-cfg'
        temperature: float = 0.7,
        device: str = "cpu", 
        seed: int = 42
    ) -> None:
        load_dotenv()
        self.together = Together()
        self.script: str = script
        self.characters: Dict[str, Dict[str, str]] = characters
        self.style: str = style
        self.prompt_weights: List[float] = prompt_weights
        self.temperature: float = temperature
        self.device: str = device
        self.seed: int = seed
        self.scenes: Any = None
        self.formatted_prompts: Any = None
        
        # valid_generation_types = {"unique", "prompt_weights", "modified-cfg"}
        self.current_generation_type: str = ""
        # if self.generation_type not in valid_generation_types:
            # raise ValueError(f"Invalid generation_type: {self.generation_type}. Must be one of {valid_generation_types}")
        
        self.pipe: Any = None
        
    def _setup_pipeline(self, generation_type: str) -> Any:
        if generation_type in {"unique", "prompt_weights"}:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            )
        elif generation_type == "modified-cfg":
            pipe = MultiPromptPipelineApproach1.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")
        
        pipe = pipe.to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        return pipe
    
    def _get_pipeline(self, generation_type: str) -> Any:
        # Reload only if the requested type is different from the current one.
        if self.current_generation_type != generation_type or self.pipe is None:
            self.pipe = self._setup_pipeline(generation_type)
            self.current_generation_type = generation_type
        return self.pipe


    def _build_character_description(self, char_info: Dict[str, str]) -> str:
        """
        Generates a textual description for a character given its attribute dictionary.
        """
        features = [
            char_info.get("ethnicity", ""),
            char_info.get("age", ""),
            char_info.get("gender", ""),
            char_info.get("hair", ""),
            char_info.get("facial_hair", ""),
            char_info.get("body_type", ""),
            f"wearing {char_info.get('clothing', '')}",
            f"with {char_info.get('accessories', '')}" if char_info.get("accessories") else ""
        ]
        return ", ".join(filter(None, features))
    
    def input_to_json(self) -> List[Dict[str, Any]]:
        """
        Converts the script and character descriptions into a JSON structure for storyboard generation.
        """
        character_descriptions = {
            name: self._build_character_description(desc) 
            for name, desc in self.characters.items()
        }
        script_section = f"Here is the film script: \n{self.script}"
        characters_section = f"The characters in the script have the following descriptions: \n{json.dumps(character_descriptions, indent=2)}"
        instructions = f"""
    ### Storyboard Generation Instructions
    1. **Number of Scenes**: Divide the entire script into a reasonable number of scenes (typically between 4 to 7 scenes), not too many or too few.
    2. **Single Distinct Moment**: Each scene captures a single moment.
    3. **Camera Angles & Orientation**: Choose from these shot types: {', '.join(self.CAMERA_SHOTS)}.  
    Choose from these orientations: {', '.join(self.ORIENTATIONS)}.
    4. **Location & Time**: Clearly derive environment from the script (e.g. INT DAY, DON'S OFFICE, etc.). Describe it in its details (size, lighting, mood, organization of the objects, etc.). Notice that if it's the same across the different scenes, it must be written in the same way
    5. **Characters**:
    - List only characters relevant to the single moment in each scene.
    - Each character must have the name and a short description (consistent from provided descriptions).
    6. Clearly describe the scene including actions, character positions (foreground, background, left, right), emotions, and expressions.
    7. **Scene Format**: Return JSON with a key 'scenes' as an array of structured objects:
    - "scene_number": integer
    - "shot_type": camera shot type (from provided list) 
    - "orientation": orientation (from provided list)
    - "characters": list of objects with:
            - "name": character's name, not as they appear on the script but as they were given to you in the description.
    - "environment": short description of the location
    - "description": short, vivid description focusing on actions, expressions, emotions of each single character. Also their relative position is clearly described. The description must be succint, without extra articles or words, it should be visual and useful for an image generation prompt. Ensure it makes sense with the shot type (e.g., if it's medium shot, don't say that the face is covering the full image, otherwise it should be a close up). Don't write the words they say, since they occupy tokens, unless it's a fundamental part of the script. Avoid useless adjectives or adverbs, be concise and clear.

    Follow the above instructions very carefully. Notice that the scenes have no knowledge of each other's contents. So in case something is necessary, describe it again. 
    """

        example_input = """
    ### Example
    Input: 
    - Script is 
    INT DAY: DON'S OFFICE (SUMMER 1945)

            DON CORLEONE
    ACT LIKE A MAN!  By Christ in
    Heaven, is it possible you turned
    out no better than a Hollywood
    finocchio.

    Both HAGEN and JOHNNY cannot refrain from laughing.  The DON
    smiles.  SONNY enters as noiselessly as possible, still
    adjusting his clothes.

            DON CORLEONE
    All right, Hollywood...Now tell me
    about this Hollywood Pezzonovanta
    who won't let you work.

            JOHNNY
    He owns the studio.  Just a month
    ago he bought the movie rights to
    this book, a best seller.  And the
    main character is a guy just like
    me.  I wouldn't even have to act,
    just be myself.

    The DON is silent, stern.

            DON CORLEONE
    You take care of your family?

            JOHNNY
    Sure.

    He glances at SONNY, who makes himself as inconspicuous as
    he can.

            DON CORLEONE
    You look terrible.  I want you to
    eat well, to rest.  And spend time
    with your family.  And then, at the
    end of the month, this big shot
    will give you the part you want.

            JOHNNY
    It's too late.  All the contracts
    have been signed, they're almost
    ready to shoot.

            DON CORLEONE
    I'll make him an offer he can't
    refuse.

    He takes JOHNNY to the door, pinching his cheek hard enough
    to hurt.

            DON CORLEONE
    Now go back to the party and leave
    it to me.

    He closes the door, smiling to himself.  Turns to HAGEN.

            DON CORLEONE
    When does my daughter leave with
    her bridegroom?

            HAGEN
    They'll cut the cake in a few
    minutes...leave right after that.
    Your new son-in-law, do we give him
    something important?

            DON CORLEONE
    No, give him a living.  But never
    let him know the family's business.
    What else, Tom?

            HAGEN
    I've called the hospital; they've
    notified Consigliere Genco's family
    to come and wait.  He won't last
    out the night.

    This saddens the DON.  He sighs.

            DON CORLEONE
    Genco will wait for me.  Santino,
    tell your brothers they will come
    with me to the hospital to see
    Genco.  Tell Fredo to drive the big
    car, and ask Johnny to come with us.

            SONNY
    And Michael?

            DON CORLEONE
    All my sons.
            (to HAGEN)
    Tom, I want you to go to California
    tonight.  Make the arrangements.
    But don't leave until I come back
    from the hospital and speak to you.
    Understood?

            HAGEN
    Understood.

    - Characters description from the dictionary gives
            - Don Vito Corleone: 'Italian-American, early 60s, male, slicked-back gray-black hair, stocky, slightly hunched posture, wearing dark three-piece suit, with gold ring on right hand, pocket watch'
            - Johnny Fontane: 'late 30s, male, short, slicked-back black hair, clean shaven, slim and fit, wearing dark, stylish suit with an open collar, with gold ring, cigarette'
            - Tom Hagen: 'German-Irish, early 40s, male, short, neatly combed brown hair, clean-shaven, medium build, upright posture, wearing gray suit, dark tie'
            - Sonny: 'Italian-American, early 30s, male, curly, dark brown hair, clean-shaven, athletic build, wearing formal suit, slightly disheveled'
    """

        example_output = """
    Example Output:
    {
    "scenes": [
    {
    "scene_number": 1,
    "shot_type": "Medium Shot",
    "orientation": "Front View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            },
            {
            "name": "Johnny Fontane"
            },
            {
            "name": "Tom Hagen"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Don Corleone stands imposingly behind desk, face stern with righteous anger, pointing finger at Johnny. Johnny appears embarrassed, head slightly bowed. Hagen stands to the right, barely containing laughter. Tension and amusement mix in intimate office atmosphere."
    },
    {
    "scene_number": 2,
    "shot_type": "Two-Shot",
    "orientation": "Profile View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            },
            {
            "name": "Johnny Fontane"
            },
            {
            "name": "Tom Hagen"
            },
            {
            "name": "Sonny"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Sonny quietly enters room from right, adjusting disheveled clothes. Don leans forward at desk, expression softening to business-like focus. Johnny stands center, straightening posture. Hagen observes from left corner. Atmosphere shifts from personal rebuke to business discussion."
    },
    {
    "scene_number": 3,
    "shot_type": "Close-Up",
    "orientation": "Front View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Don Corleone's face fills frame, stern and contemplative. Eyes narrowed, jaw set firmly. Saying 'I'll make him an offer he can't refuse' with quiet, confident menace. Power and authority emanate from his expression."
    },
    {
    "scene_number": 4,
    "shot_type": "Medium Close-Up",
    "orientation": "Three-Quarters View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            },
            {
            "name": "Johnny Fontane"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Don Corleone escorts Johnny to door, pinching his cheek firmly. Don's expression shows affection mixed with dominance. Johnny winces slightly at pain while showing relief and gratitude. Door frame visible on right edge of shot."
    },
    {
    "scene_number": 5,
    "shot_type": "Medium Shot",
    "orientation": "Front View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            },
            {
            "name": "Tom Hagen"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Don Corleone turns from closed door, small smile fading to serious business expression. Hagen stands attentively near desk, notepad ready. Don moves toward chair, shoulders slightly hunched, gold ring catching light as he gestures."
    },
    {
    "scene_number": 6,
    "shot_type": "Over-the-Shoulder Shot",
    "orientation": "Profile View",
    "characters": [
            {
            "name": "Don Vito Corleone"
            },
            {
            "name": "Tom Hagen"
            },
            {
            "name": "Sonny"
            }
    ],
    "environment": "Don's office, daytime, summer 1945. Elegant wood-paneled room with large desk, leather chairs, warm lighting filtering through venetian blinds.",
    "description": "Camera over Don's shoulder, facing Hagen and Sonny. Don's gray-black hair and dark suit visible in foreground. Hagen's face shows respectful attention. Sonny stands beside him, now composed. Don's voice carries weight as he issues final instructions about hospital visit."
    }
    ]
    }
    """

        user_content = f"{script_section}\n\n{characters_section}\n\n{instructions}\n{example_input}\n{example_output}"
        
        messages = [
            {"role": "system", "content": (
                "You are an AI specialized in creating structured storyboard scenes from a film script "
                "for image generation (e.g., stable diffusion). Each scene must capture a single distinct moment, "
                "should list relevant characters with consistent appearances, specify the environment, camera shot, "
                "and orientation, and provide direct clues for a diffusion model to generate images."
                )},
            {"role": "user", "content": user_content}
        ]
        
        response = self.together.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=10000,
            temperature=self.temperature,
            response_format={"type": "json_object", "schema": SceneList.model_json_schema()}
        )

        try:
            output_json = response.choices[0].message.content
            self.scenes = json.loads(output_json)["scenes"]
            return self.scenes
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Error parsing JSON output: %s", e)
            return []
        
    def scenes_to_formatted_prompts(self) -> List[Tuple[List[str], List[float]]]:
        """
        Converts a list of scenes into structured diffusion model prompts with weights.
        
        Returns:
            List[Tuple[List[str], List[float]]]: Each tuple contains subprompt texts and their corresponding weights.
        """
        if self.formatted_prompts is not None:
            return self.formatted_prompts
        logger.info("Generating formatted prompts...")
        weight_map = {
            "style": self.prompt_weights[0],
            "environment": self.prompt_weights[1],
            "shot": self.prompt_weights[2],
            "description": self.prompt_weights[3]
        }
        character_weight = self.prompt_weights[4]

        style_value = ("rough b&w pencil sketch, simple sketch lines, minimal shading, rough hatching, draft-style, "
                       "J.C. Leyendecker style") if self.style == "storyboard" else self.style

        formatted_results: List[Tuple[List[str], List[float]]] = []
        if self.scenes is None:
            self.input_to_json()

        for scene in self.scenes:
            subprompts: Dict[str, str] = {}
            for i, char in enumerate(scene["characters"]):
                char_name = char["name"]
                char_info = self.characters.get(char_name)
                if not char_info:
                    matching_keys = [key for key in self.characters if char_name in key]
                    if matching_keys:
                        char_info = self.characters.get(matching_keys[0],
                                                        {"age": "unknown", "gender": "unknown", "hair": "unknown",
                                                         "clothing": "unknown", "body_type": "unknown"})
                    else:
                        char_info = {"age": "unknown", "gender": "unknown", "hair": "unknown", "clothing": "unknown", "body_type": "unknown"}
                char_desc = self._build_character_description(char_info)
                subprompts[f"character{i+1}"] = f"{char_name}: {char_desc}"
            
            subprompts["style"] = style_value
            subprompts["environment"] = scene["environment"]
            subprompts["shot"] = f"{scene['shot_type']}, {scene['orientation']}"
            subprompts["description"] = scene["description"]

            subprompt_texts: List[str] = []
            subprompt_weights: List[float] = []
            for key, text in subprompts.items():
                subprompt_texts.append(text)
                if key.startswith("character"):
                    subprompt_weights.append(character_weight)
                else:
                    subprompt_weights.append(weight_map.get(key, 1.0))
            formatted_results.append((subprompt_texts, subprompt_weights))
        self.formatted_prompts = formatted_results
        logger.info("Formatted prompts generated successfully.")
        return formatted_results
    
    def _save_image(self, image: Image.Image, image_path: str) -> None:
        """
        Saves the generated image to the specified path.
        """
        image.save(image_path)
        logger.info("Image saved to %s", image_path)
    
    # unique prompt
    def build_unique_prompts(self) -> List[str]:
        """
        Builds unique prompt strings for each scene by concatenating the style, shot, and description.
        
        Returns:
            List[str]: List of unique prompt strings.
        """
        if self.formatted_prompts is None:
            self.scenes_to_formatted_prompts()
        style_override = "rough b&w simple pencil sketch, J.C. Leyendecker style," if self.style == "storyboard" else self.style
        unique_prompts: List[str] = []
        for subprompt_texts, _ in self.formatted_prompts:
            if len(subprompt_texts) < 2:
                logger.error("Insufficient subprompt texts to build unique prompt.")
                continue
            # Assumes the penultimate text is the shot prompt and the last is the description.
            shot_prompt = subprompt_texts[-2]
            description = subprompt_texts[-1]
            unique_prompt = f"{style_override} {shot_prompt}: {description}"
            unique_prompts.append(unique_prompt)
        return unique_prompts
        
    def generate_and_save_images_unique_prompts(
        self,
        save_dir: str, 
        generation_type: str = "unique", 
        negative_prompt: str = "low quality, photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, fine art, NSFW", 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5,
        )-> List[Image.Image]:
        """
        Generates images using unique prompts and saves them.
        
        Returns:
            List[Image.Image]: List of generated images.
        """
        pipe = self._get_pipeline(generation_type)  # generation_type should be "unique" here
        os.makedirs(save_dir, exist_ok=True)
        unique_prompts = self.build_unique_prompts()
        generated_images: List[Image.Image] = []
        for i, unique_prompt in enumerate(unique_prompts):
            with torch.no_grad():
                output = pipe(
                    prompt=unique_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            generated_image = output.images[0]
            generated_images.append(generated_image)
            image_path = os.path.join(save_dir, f"image_{i+1}.png")
            self._save_image(generated_image, image_path)
        return generated_images
    
    # prompt weights
    def weighted_sum_prompt_embeddings(
        self, 
        subprompt_texts: List[str], 
        subprompt_weights: List[float], 
        num_images_per_prompt: int = 1
    ) -> torch.Tensor:
        """
        Computes a weighted sum of text embeddings for a list of subprompts.
        
        Returns:
            torch.Tensor: Combined prompt embeddings.
        """
        encoded_prompts = []
        for text in subprompt_texts:
            text_inputs = self.pipe.tokenizer(
                text,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device) if "attention_mask" in text_inputs else None
            text_embeds = self.pipe.text_encoder(input_ids, attention_mask=attention_mask)[0]
            encoded_prompts.append(text_embeds)
        weighted_embedding = sum(weight * embeds for weight, embeds in zip(subprompt_weights, encoded_prompts))
        weight_total = sum(subprompt_weights)
        combined_embedding = weighted_embedding / weight_total
        batch_size, seq_len, embed_dim = combined_embedding.shape
        combined_embedding = combined_embedding.repeat(1, num_images_per_prompt, 1)
        combined_embedding = combined_embedding.view(batch_size * num_images_per_prompt, seq_len, embed_dim)
        return combined_embedding
    
    def generate_and_save_images_prompt_weights(
        self, 
        save_dir: str, 
        generation_type: str = "prompt_weights", 
        negative_prompt: str = "low quality, photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, fine art, NSFW", 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5,
    ) -> List[Image.Image]:
        """
        Generates images using weighted prompt embeddings and saves them.
        """
        pipe = self._get_pipeline(generation_type)  # generation_type should be "prompt_weights" here
        os.makedirs(save_dir, exist_ok=True)
        generated_images: List[Image.Image] = []
        formatted_prompts = self.scenes_to_formatted_prompts()
        for i, (subprompt_texts, subprompt_weights) in enumerate(formatted_prompts):
            combined_embeddings = self.weighted_sum_prompt_embeddings(subprompt_texts, subprompt_weights)
            with torch.no_grad():
                output = pipe(
                    prompt_embeds=combined_embeddings,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            generated_image = output.images[0]
            generated_images.append(generated_image)
            image_path = os.path.join(save_dir, f"image_{i+1}.png")
            self._save_image(generated_image, image_path)
        return generated_images
    
    # modified-cfg
    def encode_subprompt(self, text: str) -> torch.Tensor:
        """
        Tokenizes and encodes a single subprompt into an embedding.
        """
        text_inputs = self.pipe.tokenizer(
            text,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeds = self.pipe.text_encoder(
            text_inputs.input_ids.to(self.device),
            attention_mask=text_inputs.attention_mask.to(self.device)
        )[0]
        return text_embeds
        
    def generate_and_save_images_multi_prompt(
        self, 
        save_dir: str, 
        generation_type: str = "modified-cfg", 
        negative_prompt: str = "low quality, photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, fine art, NSFW", 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5,
    ) -> List[Image.Image]:
        """
        Generates images using the Multi-Prompt pipeline and saves them.
        """
        pipe = self._get_pipeline(generation_type)  # generation_type should be "modified-cfg" here
        os.makedirs(save_dir, exist_ok=True)
        generated_images: List[Image.Image] = []
        uncond_embeds = self.encode_subprompt(negative_prompt)
        formatted_prompts = self.scenes_to_formatted_prompts()
        for i, (subprompt_texts, subprompt_weights) in enumerate(formatted_prompts):
            subprompt_embeds = [self.encode_subprompt(sp) for sp in subprompt_texts]
            logger.info("Generating image for scene %d...", i+1)
            with torch.no_grad():
                output = pipe(
                    subprompt_embeds=subprompt_embeds,
                    subprompt_weights=subprompt_weights,
                    uncond_embeds=uncond_embeds,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            generated_image = output.images[0]
            generated_images.append(generated_image)
            image_path = os.path.join(save_dir, f"image_{i+1}.png")
            self._save_image(generated_image, image_path)
        return generated_images
    
    def generate_and_save_images(
        self, 
        save_dir: str, 
        generation_type: str, 
        negative_prompt: str = "low quality, photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, fine art, NSFW", 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5,
    ) -> List[Image.Image]:
        """
        Unified method that generates and saves images based on the provided generation type.
        """
        if generation_type == "unique":
            return self.generate_and_save_images_unique_prompts(save_dir, generation_type, negative_prompt, num_inference_steps, guidance_scale)
        elif generation_type == "prompt_weights":
            return self.generate_and_save_images_prompt_weights(save_dir, generation_type, negative_prompt, num_inference_steps, guidance_scale)
        elif generation_type == "modified-cfg":
            return self.generate_and_save_images_multi_prompt(save_dir, generation_type, negative_prompt, num_inference_steps, guidance_scale)
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")
        
    def generate_and_save_prompts_txt(self, save_dir: str, generation_type: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "prompts.txt")
        lines = []
        if generation_type == "unique":
            unique_prompts = self.build_unique_prompts()
            for i, prompt in enumerate(unique_prompts):
                lines.append(f"Scene {i+1} Unique Prompt:\n{prompt}\n")
        else:
            # For prompt_weights and modified-cfg, save subprompts breakdown.
            if self.formatted_prompts is None:
                self.scenes_to_formatted_prompts()
            for i, (subprompt_texts, subprompt_weights) in enumerate(self.formatted_prompts):
                lines.append(f"Scene {i+1} Subprompts:")
                for j, text in enumerate(subprompt_texts):
                    weight = subprompt_weights[j]
                    lines.append(f"  Subprompt {j+1} (weight {weight}): {text}")
                lines.append("")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Prompts saved to %s", file_path)
        
    def generate_and_save(
        self, 
        save_dir: str, 
        generation_type: str, 
        negative_prompt: str = "low quality, photorealistic, 3d render, overly detailed, digital art, painting, vibrant colors, fine art, NSFW", 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5,
    ) -> None:
        """
        Unified method that generates and saves images and prompt text.
        """
        self.generate_and_save_images(save_dir, generation_type, negative_prompt, num_inference_steps, guidance_scale)
        self.generate_and_save_prompts_txt(save_dir, generation_type)
        logger.info("Storyboard generation completed successfully.")
        
    def __repr__(self) -> str:
        return (
            f"StoryboardGenerator(script={self.script[:50]}..., "
            f"characters={list(self.characters.keys())}, "
            f"style={self.style}, "
            f"prompt_weights={self.prompt_weights}, "
            f"temperature={self.temperature}, "
            f"device={self.device}, "
            f"seed={self.seed})"
        )