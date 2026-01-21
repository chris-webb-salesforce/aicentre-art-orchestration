"""
Configuration loader and typed config classes for the Robot Portrait System.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class MyCobotConfig:
    port: str = "/dev/ttyAMA0"
    baud_rate: int = 115200
    home_angles: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    tracking_angles: List[float] = field(default_factory=lambda: [0, 20, -30, 0, 30, 0])
    speed: int = 30
    pan_sensitivity: float = 0.05
    tilt_sensitivity: float = 0.05


@dataclass
class DexArmConfig:
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    feedrate: int = 2000
    travel_feedrate: int = 3000


@dataclass
class DrawingConfig:
    x_min: float = -60.0
    x_max: float = 60.0
    y_min: float = 240.0
    y_max: float = 360.0
    size_mm: float = 80.0
    z_up: float = 10.0
    z_down: float = 0.0
    safe_position: dict = field(default_factory=lambda: {"x": 0.0, "y": 300.0, "z": 30.0})


@dataclass
class LogoConfig:
    enabled: bool = False
    path: str = "config/logo.png"
    x_min: float = -30.0
    x_max: float = 30.0
    y_min: float = 345.0
    y_max: float = 385.0


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    capture_width: int = 1280
    capture_height: int = 720


@dataclass
class FaceTrackingConfig:
    center_threshold: float = 0.1
    stable_duration: float = 1.5
    max_tracking_time: float = 30.0
    min_face_size: float = 0.1


@dataclass
class OpenAIConfig:
    model: str = "gpt-image-1"
    prompt: str = "Transform this portrait into a minimalist single continuous line drawing."
    size: str = "1024x1024"
    max_retries: int = 3
    retry_delay: float = 5.0


@dataclass
class ContourConfig:
    canny_low: int = 50
    canny_high: int = 150
    min_area: int = 100
    simplify_epsilon: float = 2.0


@dataclass
class PathOptimizationConfig:
    enabled: bool = True
    min_travel_distance: float = 5.0


@dataclass
class PracticeStrokesConfig:
    enabled: bool = True
    height_offset: float = 20.0
    radius: float = 15.0
    speed: int = 1500


@dataclass
class CuriousTiltsConfig:
    enabled: bool = True
    angle_range: float = 5.0
    interval_min: float = 3.0
    interval_max: float = 6.0


@dataclass
class PersonalityConfig:
    practice_strokes: PracticeStrokesConfig = field(default_factory=PracticeStrokesConfig)
    curious_tilts: CuriousTiltsConfig = field(default_factory=CuriousTiltsConfig)


@dataclass
class SystemConfig:
    output_dir: str = "output"
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration container."""
    mycobot: MyCobotConfig = field(default_factory=MyCobotConfig)
    dexarm: DexArmConfig = field(default_factory=DexArmConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    logo: LogoConfig = field(default_factory=LogoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_tracking: FaceTrackingConfig = field(default_factory=FaceTrackingConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    path_optimization: PathOptimizationConfig = field(default_factory=PathOptimizationConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def _dict_to_dataclass(data: dict, cls):
    """Convert a dictionary to a dataclass, handling nested structures."""
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Check if field type is a dataclass
            if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, looks for config/settings.yaml
                    relative to the project root.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        # Find config relative to this file's location
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        config_path = os.path.join(project_root, "config", "settings.yaml")

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return Config()

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if data is None:
        return Config()

    # Build config object from YAML data
    config = Config(
        mycobot=_dict_to_dataclass(data.get('mycobot'), MyCobotConfig),
        dexarm=_dict_to_dataclass(data.get('dexarm'), DexArmConfig),
        drawing=_dict_to_dataclass(data.get('drawing'), DrawingConfig),
        logo=_dict_to_dataclass(data.get('logo'), LogoConfig),
        camera=_dict_to_dataclass(data.get('camera'), CameraConfig),
        face_tracking=_dict_to_dataclass(data.get('face_tracking'), FaceTrackingConfig),
        openai=_dict_to_dataclass(data.get('openai'), OpenAIConfig),
        contour=_dict_to_dataclass(data.get('contour_extraction'), ContourConfig),
        path_optimization=_dict_to_dataclass(data.get('path_optimization'), PathOptimizationConfig),
        personality=_dict_to_dataclass(data.get('personality'), PersonalityConfig),
        system=_dict_to_dataclass(data.get('system'), SystemConfig),
    )

    return config


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration values.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Validate drawing bounds
    x_range = config.drawing.x_max - config.drawing.x_min
    y_range = config.drawing.y_max - config.drawing.y_min

    if x_range <= 0:
        errors.append(f"Invalid X range: {config.drawing.x_min} to {config.drawing.x_max}")
    if y_range <= 0:
        errors.append(f"Invalid Y range: {config.drawing.y_min} to {config.drawing.y_max}")

    # Validate Z heights
    if config.drawing.z_down >= config.drawing.z_up:
        errors.append(f"z_down ({config.drawing.z_down}) must be less than z_up ({config.drawing.z_up})")

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable not set")

    # Validate camera index
    if config.camera.index < 0:
        errors.append(f"Invalid camera index: {config.camera.index}")

    # Validate face tracking thresholds
    if not 0 < config.face_tracking.center_threshold < 1:
        errors.append(f"center_threshold must be between 0 and 1: {config.face_tracking.center_threshold}")

    return errors


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Loaded configuration:")
    print(f"  MyCobot port: {config.mycobot.port}")
    print(f"  DexArm port: {config.dexarm.port}")
    print(f"  Drawing bounds: X({config.drawing.x_min}, {config.drawing.x_max}), Y({config.drawing.y_min}, {config.drawing.y_max})")
    print(f"  Z heights: up={config.drawing.z_up}, down={config.drawing.z_down}")

    errors = validate_config(config)
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration valid!")
