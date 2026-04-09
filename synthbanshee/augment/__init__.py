"""Audio augmentation pipeline: room simulation, device profiles, noise mixing, preprocessing."""

from synthbanshee.augment.device_profiles import DeviceProfiler
from synthbanshee.augment.noise_mixer import NoiseMixer
from synthbanshee.augment.preprocessing import PreprocessingResult, preprocess, validate_audio
from synthbanshee.augment.room_sim import RoomSimulator
from synthbanshee.augment.types import AugmentationResult, AugmentedEvent

__all__ = [
    "AugmentationResult",
    "AugmentedEvent",
    "DeviceProfiler",
    "NoiseMixer",
    "PreprocessingResult",
    "RoomSimulator",
    "preprocess",
    "validate_audio",
]
