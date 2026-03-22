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

logger = logging.getLogger(__name__)


class RemoteClient:
    """
    Connects to the Heroku relay server and processes incoming photo jobs.
    """

    def __init__(self, server_url: str, portrait_system: PortraitSystem,
                 arm_id: str, label: str = '', token: str = ''):
        self.server_url = server_url.rstrip('/')
        self.system = portrait_system
        self.arm_id = arm_id
        self.label = label or arm_id
        self.token = token
        self._is_busy = False

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

        @self.sio.on('disconnect', namespace='/robot')
        def on_disconnect():
            logger.warning("Disconnected from relay server")

        @self.sio.on('new_job', namespace='/robot')
        def on_new_job(data):
            self._handle_job(data)

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

    def _handle_job(self, data):
        """Process an incoming photo job."""
        if self._is_busy:
            self.sio.emit('job_error', {
                'message': 'Robot is busy with another drawing'
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
