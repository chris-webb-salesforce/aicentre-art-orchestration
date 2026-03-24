"""
Remote client that connects to the Heroku relay server.

Runs on the local machine alongside the robot hardware. Connects outbound
to the Heroku server via WebSocket, receives photo submissions, and triggers
the portrait drawing pipeline.

Each arm runs as a separate process with its own --arm-id.

Usage:
    python -m src.web.remote_client --server https://your-app.herokuapp.com --arm-id arm-1

    # With mock AI (no OpenAI calls):
    python -m src.web.remote_client --server https://your-app.herokuapp.com --arm-id arm-1 --mock
"""

import os
import sys
import base64
import logging
import argparse
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import socketio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.config import load_config, validate_config
from src.main import PortraitSystem
from src.planning.gcode_generator import DrawingBounds

logger = logging.getLogger(__name__)


class RemoteClient:
    """
    Connects to the Heroku relay server and processes incoming photo jobs.
    """

    def __init__(self, server_url: str, portrait_system: PortraitSystem,
                 arm_id: str, label: str = '', token: str = '',
                 config_path: str = None):
        self.server_url = server_url.rstrip('/')
        self.system = portrait_system
        self.arm_id = arm_id
        self.label = label or arm_id
        self.token = token
        self.config_path = config_path
        self._is_busy = False
        self._is_testing = False

        self.sio = socketio.Client(
            reconnection=True,
            reconnection_delay=2,
            reconnection_delay_max=30,
            logger=False,
        )

        self._register_handlers()

    def _register_handlers(self):
        @self.sio.on('connect', namespace='/robot')
        def on_connect():
            logger.info(f"Connected to relay server as '{self.arm_id}'")
            self.sio.emit('ready', {
                'arm_id': self.arm_id,
                'label': self.label,
            }, namespace='/robot')
            # Report config on connect so operator panel has it immediately
            self._handle_get_config()

        @self.sio.on('disconnect', namespace='/robot')
        def on_disconnect():
            logger.warning("Disconnected from relay server")

        @self.sio.on('new_job', namespace='/robot')
        def on_new_job(data):
            self._handle_job(data)

        @self.sio.on('get_config', namespace='/robot')
        def on_get_config(data=None):
            self._handle_get_config()

        @self.sio.on('config_update', namespace='/robot')
        def on_config_update(data):
            self._handle_config_update(data)

        @self.sio.on('config_save', namespace='/robot')
        def on_config_save(data=None):
            self._handle_config_save()

        @self.sio.on('test_pen_start', namespace='/robot')
        def on_test_pen_start(data=None):
            try:
                logger.info(f"[{self.arm_id}] Received test_pen_start")
                self._handle_test_pen_start()
            except Exception as e:
                logger.error(f"[{self.arm_id}] test_pen_start error: {e}", exc_info=True)

        @self.sio.on('test_pen_move', namespace='/robot')
        def on_test_pen_move(data):
            try:
                logger.info(f"[{self.arm_id}] Received test_pen_move: {data}")
                self._handle_test_pen_move(data)
            except Exception as e:
                logger.error(f"[{self.arm_id}] test_pen_move error: {e}", exc_info=True)

        @self.sio.on('test_pen_stop', namespace='/robot')
        def on_test_pen_stop(data=None):
            try:
                logger.info(f"[{self.arm_id}] Received test_pen_stop")
                self._handle_test_pen_stop()
            except Exception as e:
                logger.error(f"[{self.arm_id}] test_pen_stop error: {e}", exc_info=True)

    def connect(self):
        """Connect to the Heroku relay server."""
        params = {}
        if self.token:
            params['token'] = self.token
        params['arm_id'] = self.arm_id

        url = f"{self.server_url}?{urlencode(params)}"

        logger.info(f"Connecting to {self.server_url} as arm '{self.arm_id}'...")
        self.sio.connect(url, namespaces=['/robot'],
                         transports=['websocket', 'polling'])

    def wait(self):
        """Block until disconnected."""
        self.sio.wait()

    def _send_status(self, status: str, message: str, percent: int = 0):
        """Send a status update back to the relay server."""
        try:
            self.sio.emit('status_update', {
                'status': status,
                'message': message,
                'percent': percent,
            }, namespace='/robot')
        except Exception as e:
            logger.warning(f"Failed to send status: {e}")

    def _handle_get_config(self):
        """Report current config back to server."""
        config = self.system.dexarm.get_config()
        # Add drawing bounds from the gcode generator
        if self.system.gcode_generator:
            b = self.system.gcode_generator.bounds
            config['drawing'].update({
                'x_min': b.x_min, 'x_max': b.x_max,
                'y_min': b.y_min, 'y_max': b.y_max,
            })
        self.sio.emit('arm_config', {
            'arm_id': self.arm_id,
            'config': config,
        }, namespace='/robot')
        logger.info(f"[{self.arm_id}] Reported config")

    def _handle_config_update(self, data):
        """Apply config changes at runtime."""
        config = data.get('config', {})
        drawing = config.get('drawing', {})
        dexarm = config.get('dexarm', {})

        # Update DexArmController
        self.system.dexarm.update_config(
            z_up=drawing.get('z_up'),
            z_down=drawing.get('z_down'),
            feedrate=dexarm.get('feedrate'),
            travel_feedrate=dexarm.get('travel_feedrate'),
            acceleration=dexarm.get('acceleration'),
            travel_acceleration=dexarm.get('travel_acceleration'),
            jerk=dexarm.get('jerk'),
        )

        # Update GCodeGenerator bounds if drawing config changed
        if drawing and self.system.gcode_generator:
            b = self.system.gcode_generator.bounds
            self.system.gcode_generator.bounds = DrawingBounds(
                x_min=drawing.get('x_min', b.x_min),
                x_max=drawing.get('x_max', b.x_max),
                y_min=drawing.get('y_min', b.y_min),
                y_max=drawing.get('y_max', b.y_max),
                z_up=drawing.get('z_up', b.z_up),
                z_down=drawing.get('z_down', b.z_down),
                feedrate=dexarm.get('feedrate', b.feedrate),
                travel_feedrate=dexarm.get('travel_feedrate', b.travel_feedrate),
                flip_x=drawing.get('flip_x', b.flip_x),
                flip_y=drawing.get('flip_y', b.flip_y),
            )

        logger.info(f"[{self.arm_id}] Config updated: {config}")
        # Report back the new config
        self._handle_get_config()

    def _handle_config_save(self):
        """Persist current config to settings.yaml."""
        import yaml

        success = False
        try:
            config_path = self.config_path
            if not config_path:
                config_path = str(Path(__file__).parent.parent.parent / 'config' / 'settings.yaml')

            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            # Find and update this arm's config
            arms_list = data.get('arms', [])
            found = False
            for arm_data in arms_list:
                if arm_data.get('id') == self.arm_id:
                    found = True
                    # Update drawing values
                    if 'drawing' not in arm_data:
                        arm_data['drawing'] = {}
                    arm_data['drawing']['z_up'] = self.system.dexarm.z_up
                    arm_data['drawing']['z_down'] = self.system.dexarm.z_down
                    # Update dexarm values
                    if 'dexarm' not in arm_data:
                        arm_data['dexarm'] = {}
                    arm_data['dexarm']['feedrate'] = self.system.dexarm.feedrate
                    arm_data['dexarm']['travel_feedrate'] = self.system.dexarm.travel_feedrate
                    arm_data['dexarm']['acceleration'] = self.system.dexarm.acceleration
                    arm_data['dexarm']['travel_acceleration'] = self.system.dexarm.travel_acceleration
                    arm_data['dexarm']['jerk'] = self.system.dexarm.jerk
                    break

            if not found:
                logger.error(f"[{self.arm_id}] Arm ID not found in YAML arms list — config NOT saved")
            else:
                with open(config_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                success = True
                logger.info(f"[{self.arm_id}] Config saved to {config_path}")

        except Exception as e:
            logger.error(f"[{self.arm_id}] Failed to save config: {e}")

        self.sio.emit('config_saved', {
            'arm_id': self.arm_id,
            'success': success,
        }, namespace='/robot')

    def _handle_test_pen_start(self):
        """Enter pen test mode — move to center of drawing bounds."""
        if self._is_busy:
            return
        self._is_testing = True
        dexarm = self.system.dexarm
        logger.info(f"[{self.arm_id}] DexArm available: {dexarm.is_available()}, arm: {dexarm.arm}, initialized: {dexarm._is_initialized}")
        if self.system.gcode_generator:
            b = self.system.gcode_generator.bounds
            cx = (b.x_min + b.x_max) / 2
            cy = (b.y_min + b.y_max) / 2
        else:
            cx, cy = 0.0, 300.0
        result = dexarm.move_to(cx, cy, dexarm.z_up)
        logger.info(f"[{self.arm_id}] Pen test move_to({cx}, {cy}, {dexarm.z_up}) returned: {result}")
        self.sio.emit('test_pen_status', {
            'arm_id': self.arm_id,
            'z': dexarm.z_up,
            'mode': 'up',
        }, namespace='/robot')
        logger.info(f"[{self.arm_id}] Pen test started at ({cx}, {cy})")

    def _handle_test_pen_move(self, data):
        """Move pen to specific Z during test."""
        if not self._is_testing:
            logger.warning(f"[{self.arm_id}] test_pen_move ignored — not in testing mode")
            return
        z = data.get('z')
        if z is None:
            return
        z = float(z)
        result = self.system.dexarm.move_to_z(z)
        logger.info(f"[{self.arm_id}] move_to_z({z}) returned: {result}")
        mode = 'down' if z <= self.system.dexarm.z_down else 'up'
        self.sio.emit('test_pen_status', {
            'arm_id': self.arm_id,
            'z': z,
            'mode': mode,
        }, namespace='/robot')

    def _handle_test_pen_stop(self):
        """Exit pen test mode — return to safe position, then ack."""
        self._is_testing = False
        self.system.dexarm.pen_up()
        self.system.dexarm.go_to_safe_position(
            self.system.config.drawing.safe_position.get('x', 0),
            self.system.config.drawing.safe_position.get('y', 300),
            self.system.config.drawing.safe_position.get('z', 30),
        )
        # Ack to server — arm is now safe for dispatch
        self.sio.emit('test_pen_stopped', {}, namespace='/robot')
        logger.info(f"[{self.arm_id}] Pen test stopped")

    def _handle_job(self, data):
        """Process an incoming photo job."""
        if self._is_busy or self._is_testing:
            self.sio.emit('job_error', {
                'message': 'Robot is busy' + (' (testing)' if self._is_testing else '')
            }, namespace='/robot')
            return

        photo_b64 = data.get('photo')
        style = data.get('style', 'minimal')
        job_id = data.get('job_id', 'unknown')

        if not photo_b64:
            self.sio.emit('job_error', {
                'message': 'No photo data received'
            }, namespace='/robot')
            return

        self._is_busy = True

        try:
            # Decode the base64 photo
            self._send_status('processing', 'Decoding photo...', 5)

            # Strip data URL prefix if present
            if ',' in photo_b64:
                photo_b64 = photo_b64.split(',', 1)[1]

            photo_bytes = base64.b64decode(photo_b64)
            nparr = np.frombuffer(photo_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                self.sio.emit('job_error', {
                    'message': 'Failed to decode photo'
                }, namespace='/robot')
                return

            logger.info(f"[{self.arm_id}] Received job {job_id} ({image.shape}), style: {style}")

            # Run the portrait pipeline
            def status_callback(status, message, percent=0):
                self._send_status(status, message, percent)

            success = self.system.run_pipeline_from_image(
                image, style, status_callback
            )

            if success:
                self.sio.emit('job_complete', {}, namespace='/robot')
                logger.info(f"[{self.arm_id}] Drawing complete for job {job_id}")
            else:
                self.sio.emit('job_error', {
                    'message': 'Drawing pipeline failed'
                }, namespace='/robot')

        except Exception as e:
            logger.error(f"[{self.arm_id}] Job processing error: {e}")
            import traceback
            traceback.print_exc()
            self.sio.emit('job_error', {
                'message': f'Error: {str(e)}'
            }, namespace='/robot')

        finally:
            self._is_busy = False


def main():
    parser = argparse.ArgumentParser(description="Remote client for portrait relay server")
    parser.add_argument('--server', type=str, required=True,
                        help='Heroku relay server URL (e.g. https://your-app.herokuapp.com)')
    parser.add_argument('--arm-id', type=str, required=True,
                        help='Unique arm identifier (must match an entry in config arms list)')
    parser.add_argument('--token', type=str, default=os.environ.get('ROBOT_TOKEN', ''),
                        help='Authentication token for robot connection')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock AI client (no OpenAI API calls)')
    parser.add_argument('--no-personality', action='store_true',
                        help='Disable personality animations')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load and validate config
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        # Allow missing API key if using mock
        non_key_errors = [e for e in errors if 'OPENAI_API_KEY' not in e]
        if non_key_errors or (not args.mock and errors):
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            if not args.mock:
                sys.exit(1)

    # Resolve per-arm config
    arm_config = next((a for a in config.arms if a.id == args.arm_id), None)
    label = args.arm_id
    if arm_config:
        config.dexarm = arm_config.dexarm
        config.drawing = arm_config.drawing
        label = arm_config.label
        logger.info(f"Using arm config '{args.arm_id}': port={arm_config.dexarm.port}")
    else:
        logger.warning(f"No arm config found for '{args.arm_id}', using top-level defaults")

    # Create and initialize the portrait system (remote mode: DexArm only)
    system = PortraitSystem(
        config,
        use_mock_ai=args.mock,
        enable_personality=False,
        remote_mode=True
    )

    logger.info(f"Initializing portrait system for arm '{args.arm_id}'...")
    if not system.initialize():
        logger.error("Failed to initialize portrait system")
        sys.exit(1)

    # Create and connect the remote client
    client = RemoteClient(
        args.server, system,
        arm_id=args.arm_id,
        label=label,
        token=args.token,
        config_path=args.config,
    )

    try:
        client.connect()
        logger.info(f"Remote client '{args.arm_id}' running. Waiting for photo submissions...")
        client.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        system.shutdown()


if __name__ == '__main__':
    main()
