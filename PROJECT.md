# Robot Portrait Drawing System

An artistic robot installation that creates single-line portrait drawings of people using two coordinated robot arms.

## Overview

This system uses two robot arms working together:

- **MyCobot320 Pi** (The "Cameraman/Processor"): Tracks faces, captures photos, and watches the drawing process with curious movements
- **Rotrics DexArm** (The "Artist"): Draws the portrait as a single continuous line

### The Experience

1. A visitor steps in front of the camera and presses the spacebar
2. The MyCobot arm actively tracks their face, adjusting its position to center them in frame
3. While this happens, the DexArm performs gentle practice strokes in the air (warming up!)
4. Once the face is centered and stable, a photo is captured
5. The photo is sent to OpenAI to generate a minimalist single-line artistic rendition
6. The line art is processed to extract drawable paths
7. The DexArm draws the portrait while the MyCobot watches with curious head tilts
8. The finished portrait is complete!

## System Requirements

### Hardware

- **MyCobot320 Pi** by Elephant Robotics
  - Raspberry Pi 4 (built into the base)
  - USB webcam connected to the Pi
  - Serial connection: `/dev/ttyAMA0`

- **Rotrics DexArm**
  - Connected via USB-to-Serial adapter
  - Pen/drawing module attached
  - Serial connection: Configurable (e.g., `/dev/ttyUSB0`)

### Software

- Python 3.8+
- OpenCV
- OpenAI API key (for `gpt-image-1` model)
- pymycobot library
- pyserial

## Installation

### 1. Clone the repository

```bash
cd /home/pi  # Or your preferred directory
git clone <repository-url> aicentre-art-orchestration
cd aicentre-art-orchestration
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Add this to your `.bashrc` or `.profile` for persistence:

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

### 5. Configure serial ports

Check your connected devices:

```bash
ls /dev/tty*
```

The MyCobot320 Pi typically uses `/dev/ttyAMA0` (built-in UART).
The DexArm typically appears as `/dev/ttyUSB0` or `/dev/ttyACM0`.

Update `config/settings.yaml` with your ports:

```yaml
mycobot:
  port: "/dev/ttyAMA0"

dexarm:
  port: "/dev/ttyUSB0"  # Update this!
```

### 6. Set serial port permissions

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyAMA0
sudo chmod 666 /dev/ttyUSB0
```

Log out and back in for group changes to take effect.

## Configuration

All configuration is in `config/settings.yaml`. Key settings:

### Drawing Bounds

```yaml
drawing:
  x_min: -40.0      # Left edge (mm)
  x_max: 40.0       # Right edge (mm)
  y_min: 260.0      # Near edge (mm)
  y_max: 340.0      # Far edge (mm)
  size_mm: 80       # Target drawing size
  z_up: 10.0        # Pen up height
  z_down: 0.0       # Pen down height - ADJUST THIS FOR YOUR SETUP
```

**Important**: You'll need to calibrate `z_down` for your paper/pen setup. Start with a higher value and decrease until the pen touches paper correctly.

### Face Tracking

```yaml
face_tracking:
  center_threshold: 0.1    # How centered (0=perfect, 0.1=10% margin)
  stable_duration: 1.5     # Seconds to hold still before capture
  max_tracking_time: 30.0  # Timeout in seconds
```

### Personality Animations

```yaml
personality:
  practice_strokes:
    enabled: true
    height_offset: 20.0  # mm above paper for air practice
    radius: 15.0         # mm radius of practice circles
  curious_tilts:
    enabled: true
    angle_range: 5.0     # degrees of curious movement
    interval_min: 3.0    # seconds between tilts
    interval_max: 6.0
```

## Usage

### Running the System

```bash
# From the project directory
source venv/bin/activate
python -m src.main
```

### Command Line Options

```bash
# Test mode (no OpenAI API calls)
python -m src.main --mock

# Disable personality animations
python -m src.main --no-personality

# Use a different config file
python -m src.main --config /path/to/custom/settings.yaml
```

### Controls

- **SPACEBAR**: Start portrait capture
- **q**: Quit the system

### Output

Each portrait session creates a timestamped folder in `output/`:

```
output/
└── 20240115_143022/
    ├── portrait.jpg      # Original captured photo
    ├── lineart.png       # AI-generated line art
    └── drawing.gcode     # Generated drawing commands
```

## Testing Components

Use the test script to verify each component works before running the full system:

```bash
python scripts/test_components.py <component> [options]
```

### Available Tests

| Component | Description | Command |
|-----------|-------------|---------|
| `camera` | Test camera connection, live preview, capture | `python scripts/test_components.py camera` |
| `face` | Test face detection with live overlay | `python scripts/test_components.py face` |
| `mycobot` | Test MyCobot connection and movements | `python scripts/test_components.py mycobot` |
| `dexarm` | Test DexArm connection and pen movements | `python scripts/test_components.py dexarm` |
| `openai` | Test OpenAI API line art generation | `python scripts/test_components.py openai` |
| `contours` | Test contour extraction from image | `python scripts/test_components.py contours` |
| `gcode` | Test GCode generation from contours | `python scripts/test_components.py gcode` |
| `personality` | Test animation system (mock hardware) | `python scripts/test_components.py personality` |
| `pipeline` | Run full pipeline | `python scripts/test_components.py pipeline` |

### Test Options

```bash
# Override serial port
python scripts/test_components.py dexarm --port /dev/ttyUSB1

# Use a specific image for testing
python scripts/test_components.py contours --image my_lineart.png
python scripts/test_components.py openai --image my_photo.jpg

# Test OpenAI without API calls (uses edge detection)
python scripts/test_components.py openai --mock

# Test DexArm with actual drawing (pen touches paper!)
python scripts/test_components.py dexarm --draw-test

# Run full pipeline in mock mode
python scripts/test_components.py pipeline --mock
```

### Recommended Test Order

1. **Camera** - Verify the camera works
2. **Face** - Verify face detection
3. **MyCobot** - Test arm movements (ensure clear space!)
4. **DexArm** - Test arm movements (no drawing)
5. **DexArm with --draw-test** - Verify pen calibration
6. **OpenAI --mock** - Test image processing
7. **OpenAI** - Test real API (requires key)
8. **Pipeline --mock** - Full test without API
9. **Pipeline** - Full production test

## Architecture

```
src/
├── main.py                 # Main orchestrator
├── config.py               # Configuration management
├── personality.py          # Animation controllers
├── hardware/
│   ├── camera_controller.py    # USB camera interface
│   ├── mycobot_controller.py   # MyCobot320 control
│   └── dexarm_controller.py    # DexArm control
├── vision/
│   ├── face_tracker.py         # Face detection & tracking
│   └── contour_extractor.py    # Line art → paths
├── ai/
│   └── openai_client.py        # OpenAI API integration
└── planning/
    └── gcode_generator.py      # Paths → GCode
```

### Data Flow

```
┌─────────────────┐
│ User presses    │
│ SPACEBAR        │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Face Tracking   │◄────│ DexArm Practice │
│ (MyCobot moves) │     │ Strokes (air)   │
└────────┬────────┘     └─────────────────┘
         ▼
┌─────────────────┐
│ Capture Photo   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ OpenAI API      │
│ Generate Line   │
│ Art (PNG)       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Extract         │
│ Contours        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Generate        │
│ GCode           │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐
│ DexArm Draws    │◄────│ MyCobot Curious │
│ Portrait        │     │ Tilts (watching)│
└────────┬────────┘     └─────────────────┘
         ▼
┌─────────────────┐
│ Complete!       │
└─────────────────┘
```

## Calibration Guide

### 1. Paper Position

Place paper on a flat surface within the DexArm's reach. The default drawing area is centered at (0, 300) with an 80x80mm target area.

### 2. Pen Height Calibration

1. Run the DexArm test script:
   ```bash
   python -c "
   from src.hardware.dexarm_controller import DexArmController
   arm = DexArmController(port='/dev/ttyUSB0')
   arm.initialize()
   arm.move_to(0, 300, 0)  # Move to center, z=0
   "
   ```

2. Check if pen touches paper. If not:
   - If pen is too high, decrease `z_down` in config (e.g., -2.0)
   - If pen is too low, increase `z_down` (e.g., 2.0)

3. Test drawing a square:
   ```bash
   python -c "
   from src.hardware.dexarm_controller import DexArmController
   arm = DexArmController()
   arm.initialize()
   test_gcode = [
       'G0 Z10', 'G0 X-30 Y270', 'G1 Z0',
       'G1 X30 Y270', 'G1 X30 Y330', 'G1 X-30 Y330', 'G1 X-30 Y270',
       'G0 Z10'
   ]
   arm.stream_gcode(test_gcode)
   arm.release()
   "
   ```

### 3. Camera Calibration

Test face detection:
```bash
python src/vision/face_tracker.py
```

A window will show the camera feed with face detection overlay. Press 'q' to quit.

### 4. MyCobot Calibration

Test joint positions:
```bash
python src/hardware/mycobot_controller.py
```

This will test home position, tracking position, and curious tilts.

## Troubleshooting

### Camera not found

```
Error: Failed to open camera at index 0
```

**Solutions:**
- Check camera is connected: `ls /dev/video*`
- Try different index in config: `camera: index: 1`
- Check permissions: `sudo chmod 666 /dev/video0`

### Serial port access denied

```
Error: Permission denied: '/dev/ttyUSB0'
```

**Solutions:**
- Add user to dialout group: `sudo usermod -a -G dialout $USER`
- Log out and back in
- Or temporarily: `sudo chmod 666 /dev/ttyUSB0`

### OpenAI API errors

```
Error: OPENAI_API_KEY environment variable not set
```

**Solutions:**
- Set the API key: `export OPENAI_API_KEY="sk-..."`
- Use mock mode for testing: `python -m src.main --mock`

### DexArm not responding

**Solutions:**
- Check the correct port: `ls /dev/tty*`
- Ensure DexArm is powered on
- Check baud rate (should be 115200)
- Try resetting the DexArm

### MyCobot not moving

**Solutions:**
- Check `/dev/ttyAMA0` exists
- Ensure the arm is powered
- Wait 2 seconds after initialization
- Check for physical obstructions

### Drawing is offset or wrong size

**Solutions:**
- Recalibrate paper position
- Check `drawing` bounds in config match physical setup
- Ensure paper is flat and secure

## API Reference

### Main Classes

#### `PortraitSystem`
Main orchestrator class.

```python
from src.main import PortraitSystem
from src.config import load_config

config = load_config()
system = PortraitSystem(config, use_mock_ai=False)
system.initialize()
system.wait_for_trigger()
system.run_portrait_pipeline()
system.shutdown()
```

#### `DexArmController`
Controls the drawing arm.

```python
from src.hardware.dexarm_controller import DexArmController

arm = DexArmController(port="/dev/ttyUSB0", z_up=10, z_down=0)
arm.initialize()
arm.pen_up()
arm.move_to(0, 300)
arm.pen_down()
arm.move_to(10, 310, drawing=True)
arm.stream_gcode(["G0 X0 Y300", "G1 Z0", "G1 X10 Y310"])
arm.release()
```

#### `MyCobotController`
Controls the camera arm.

```python
from src.hardware.mycobot_controller import MyCobotController

robot = MyCobotController(port="/dev/ttyAMA0")
robot.initialize()
robot.go_home()
robot.go_to_tracking_position()
robot.adjust_for_face(pan_offset=0.1, tilt_offset=-0.05)
robot.perform_curious_tilt(angle_range=5.0)
robot.release()
```

#### `GCodeGenerator`
Converts contours to drawing commands.

```python
from src.planning.gcode_generator import GCodeGenerator, DrawingBounds
from src.vision.contour_extractor import ContourExtractor

bounds = DrawingBounds(x_min=-40, x_max=40, y_min=260, y_max=340)
generator = GCodeGenerator(bounds)

extractor = ContourExtractor()
contours = extractor.extract_from_file("lineart.png")
image_bounds = extractor.get_bounds(contours)
contours = extractor.optimize_order(contours)

gcode = generator.generate(contours, image_bounds)
generator.save_to_file(gcode, "output.gcode")
```

## License

[Your license here]

## Credits

- Built with [MyCobot320](https://www.elephantrobotics.com/) by Elephant Robotics
- Drawing with [Rotrics DexArm](https://www.rotrics.com/)
- AI art generation by [OpenAI](https://openai.com/)
