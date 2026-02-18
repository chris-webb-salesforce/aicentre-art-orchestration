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
    acceleration: int = 200           # Drawing acceleration (mm/s²)
    travel_acceleration: int = 400    # Travel acceleration (mm/s²)
    jerk: float = 5.0                 # Jerk limit (mm/s) - lower = smoother


@dataclass
class DrawingConfig:
    x_min: float = -60.0
    x_max: float = 60.0
    y_min: float = 240.0
    y_max: float = 360.0
    size_mm: float = 80.0
    z_up: float = 10.0
    z_down: float = 0.0
    flip_x: bool = True   # Mirror horizontally (fixes camera mirror effect)
    flip_y: bool = False  # Mirror vertically
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
class StyleContourConfig:
    """Contour extraction overrides for a style. All fields are optional."""
    method: str = None
    canny_low: int = None
    canny_high: int = None
    min_area: int = None
    simplify_epsilon: float = None
    blur_kernel: int = None
    min_contour_points: int = None
    thickness_threshold: int = None
    density_threshold: float = None
    skeleton_simplify: float = None
    thinning_threshold: int = None
    thinning_cleanup: bool = None
    thinning_cleanup_kernel: int = None
    min_straightness: float = None
    min_length: float = None
    merge_distance: float = None
    merge_enabled: bool = None
    region_aware: bool = None
    detail_simplify_epsilon: float = None
    detail_min_length: float = None
    detail_min_area: int = None
    detail_region_padding: int = None


@dataclass
class StyleConfig:
    """Configuration for an art style."""
    name: str = ""
    prompt: str = ""
    contour: StyleContourConfig = None  # Optional contour overrides for this style


@dataclass
class OpenAIConfig:
    model: str = "gpt-image-1"
    prompt: str = "Transform this portrait into a minimalist single continuous line drawing."
    size: str = "1024x1024"
    max_retries: int = 3
    retry_delay: float = 5.0
    default_style: str = "minimal"
    reference_image: str = None  # Path to style reference image
    styles: dict = field(default_factory=dict)  # Dict of style_name -> StyleConfig


@dataclass
class ContourConfig:
    method: str = "adaptive"          # "canny", "skeleton", "adaptive", "hybrid", or "thinning"
    canny_low: int = 30
    canny_high: int = 100
    min_area: int = 50
    simplify_epsilon: float = 0.8
    blur_kernel: int = 3              # Gaussian blur kernel size (3 or 5)
    min_contour_points: int = 5       # Minimum points to keep a contour
    thickness_threshold: int = 3      # For adaptive: lines thicker than this use skeleton
    density_threshold: float = 0.3    # For adaptive: density above this triggers skeleton
    skeleton_simplify: float = 1.0    # Simplification for skeleton contours
    # Thinning parameters (Zhang-Suen)
    thinning_threshold: int = 127     # Binary threshold for thinning (0-255)
    thinning_cleanup: bool = True     # Post-thinning morphological cleanup
    thinning_cleanup_kernel: int = 2  # Kernel size for cleanup
    # Noise filtering
    min_straightness: float = 0.15    # Min bbox_diagonal / path_length (filters squiggly noise)
    # Speed optimizations
    min_length: float = 10.0          # Minimum contour length in pixels
    merge_distance: float = 5.0       # Merge contours with endpoints within this distance
    merge_enabled: bool = True        # Enable contour merging to reduce pen lifts
    # Region-aware processing (preserves detail in facial features)
    region_aware: bool = False        # Enable face/eye detection for detail preservation
    detail_simplify_epsilon: float = 0.3   # Lower epsilon for detail regions
    detail_min_length: float = 3.0    # Keep smaller contours in detail regions
    detail_min_area: int = 10         # Keep smaller areas in detail regions
    detail_region_padding: int = 20   # Pixels to expand around detected features


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
    openai_config = _dict_to_dataclass(data.get('openai'), OpenAIConfig)

    # Convert styles dict entries to StyleConfig objects
    openai_data = data.get('openai', {})
    if 'styles' in openai_data and isinstance(openai_data['styles'], dict):
        styles = {}
        for style_name, style_data in openai_data['styles'].items():
            if isinstance(style_data, dict):
                # Parse contour overrides if present
                contour_config = None
                if 'contour' in style_data and isinstance(style_data['contour'], dict):
                    contour_data = style_data['contour']
                    contour_config = StyleContourConfig(
                        method=contour_data.get('method'),
                        canny_low=contour_data.get('canny_low'),
                        canny_high=contour_data.get('canny_high'),
                        min_area=contour_data.get('min_area'),
                        simplify_epsilon=contour_data.get('simplify_epsilon'),
                        blur_kernel=contour_data.get('blur_kernel'),
                        min_contour_points=contour_data.get('min_contour_points'),
                        thickness_threshold=contour_data.get('thickness_threshold'),
                        density_threshold=contour_data.get('density_threshold'),
                        skeleton_simplify=contour_data.get('skeleton_simplify'),
                        thinning_threshold=contour_data.get('thinning_threshold'),
                        thinning_cleanup=contour_data.get('thinning_cleanup'),
                        thinning_cleanup_kernel=contour_data.get('thinning_cleanup_kernel'),
                        min_straightness=contour_data.get('min_straightness'),
                        min_length=contour_data.get('min_length'),
                        merge_distance=contour_data.get('merge_distance'),
                        merge_enabled=contour_data.get('merge_enabled'),
                        region_aware=contour_data.get('region_aware'),
                        detail_simplify_epsilon=contour_data.get('detail_simplify_epsilon'),
                        detail_min_length=contour_data.get('detail_min_length'),
                        detail_min_area=contour_data.get('detail_min_area'),
                        detail_region_padding=contour_data.get('detail_region_padding'),
                    )

                styles[style_name] = StyleConfig(
                    name=style_data.get('name', style_name),
                    prompt=style_data.get('prompt', ''),
                    contour=contour_config
                )
        openai_config.styles = styles

    config = Config(
        mycobot=_dict_to_dataclass(data.get('mycobot'), MyCobotConfig),
        dexarm=_dict_to_dataclass(data.get('dexarm'), DexArmConfig),
        drawing=_dict_to_dataclass(data.get('drawing'), DrawingConfig),
        logo=_dict_to_dataclass(data.get('logo'), LogoConfig),
        camera=_dict_to_dataclass(data.get('camera'), CameraConfig),
        face_tracking=_dict_to_dataclass(data.get('face_tracking'), FaceTrackingConfig),
        openai=openai_config,
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

    # Test styles loading
    print(f"\n  OpenAI default style: {config.openai.default_style}")
    print(f"  Loaded {len(config.openai.styles)} art styles:")
    for name, style in config.openai.styles.items():
        prompt_preview = style.prompt[:50].replace('\n', ' ') + "..." if len(style.prompt) > 50 else style.prompt
        print(f"    - {name}: {style.name} -> {prompt_preview}")

    errors = validate_config(config)
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration valid!")
