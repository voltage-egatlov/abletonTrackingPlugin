"""YAML configuration loader and Pydantic models."""
from __future__ import annotations
import yaml
from pydantic import BaseModel
from cv_midi.mapper import MappingConfig


class MidiConfig(BaseModel):
    port_name: str = "CV-MIDI"
    rate_limit: int = 60


class CameraConfig(BaseModel):
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


class DetectorConfig(BaseModel):
    motion: bool = True
    pose: bool = False
    hands: bool = False
    color: bool = False


class AppConfig(BaseModel):
    midi: MidiConfig = MidiConfig()
    camera: CameraConfig = CameraConfig()
    detectors: DetectorConfig = DetectorConfig()
    mappings: dict[str, MappingConfig] = {}


def load_config(path: str) -> AppConfig:
    """Load and validate config from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
