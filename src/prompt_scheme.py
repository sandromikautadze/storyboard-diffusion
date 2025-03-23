"""
prompt_scheme.py

This module defines the Pydantic models for the prompt scheme used to get the JSON from the LLM.

Authors: Sandro Mikautadze, Elio Samaha.
"""


from pydantic import BaseModel
from typing import List

class Character(BaseModel):
    name: str
    
class Scene(BaseModel):
    scene_number: int
    shot_type: str
    orientation: str
    characters: List[Character]
    environment: str
    description: str
    
class SceneList(BaseModel):
    scenes: List[Scene]
    