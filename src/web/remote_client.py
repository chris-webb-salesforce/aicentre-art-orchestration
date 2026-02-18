"""
Remote client that connects to the Heroku relay server.

Runs on the local machine alongside the robot hardware. Connects outbound
to the Heroku server via WebSocket, receives photo submissions, and triggers
the portrait drawing pipeline.

Usage:
    python -m src.web.remote_client --server https://your-app.herokuapp.com

    # With mock AI (no OpenAI calls):
    python -m src.web.remote_client --server https://your-app.herokuapp.com --mock
"""

import os
import sys
import base64
import logging
import argparse
import time
from pathlib import Path

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
from src.personality import IdleDance

logger = logging.getLogger(__name__)


class RemoteClient:
    """
    Connects to the Heroku relay server and processes incoming photo jobs.
    """

    def __init__(self, server_url: str, portrait_system: PortraitSystem, token: str = ''):
        self.server_url = server_url.rstrip('/')
        self.system = portrait_system
        self.token = token
        self._is_busy = False

        # Idle dance animation (runs while waiting for jobs)
        self.idle_dance = IdleDance(portrait_system.dexarm, speed=10000)

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
            logger.info("Connected to relay server")
            self.sio.emit('ready', {}, namespace='/robot')
            self.idle_dance.start()

        @self.sio.on('disconnect', namespace='/robot')
        def on_disconnect():
            logger.warning("Disconnected from relay server")
            self.idle_dance.stop()

        @self.sio.on('new_job', namespace='/robot')
        def on_new_job(data):
            self._handle_job(data)

    def connect(self):
        """Connect to the Heroku relay server."""
        url = self.server_url
        if self.token:
            url = f"{url}?token={self.token}"

        logger.info(f"Connecting to {self.server_url} ...")
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

        if not photo_b64:
            self.sio.emit('job_error', {
                'message': 'No photo data received'
            }, namespace='/robot')
            return

        self._is_busy = True
        self.idle_dance.stop()

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

            logger.info(f"Received photo ({image.shape}), style: {style}")

            # Run the portrait pipeline
            def status_callback(status, message, percent=0):
                self._send_status(status, message, percent)

            success = self.system.run_pipeline_from_image(
                image, style, status_callback
            )

            if success:
                self.sio.emit('job_complete', {}, namespace='/robot')
                logger.info("Drawing complete")
            else:
                self.sio.emit('job_error', {
                    'message': 'Drawing pipeline failed'
                }, namespace='/robot')

        except Exception as e:
            logger.error(f"Job processing error: {e}")
            import traceback
            traceback.print_exc()
            self.sio.emit('job_error', {
                'message': f'Error: {str(e)}'
            }, namespace='/robot')

        finally:
            self._is_busy = False
            self.idle_dance.start()


def main():
    parser = argparse.ArgumentParser(description="Remote client for portrait relay server")
    parser.add_argument('--server', type=str, required=True,
                        help='Heroku relay server URL (e.g. https://your-app.herokuapp.com)')
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

    # Create and initialize the portrait system (remote mode: DexArm only)
    system = PortraitSystem(
        config,
        use_mock_ai=args.mock,
        enable_personality=False,
        remote_mode=True
    )

    logger.info("Initializing portrait system...")
    if not system.initialize():
        logger.error("Failed to initialize portrait system")
        sys.exit(1)

    # Create and connect the remote client
    client = RemoteClient(args.server, system, token=args.token)

    try:
        client.connect()
        logger.info("Remote client running. Waiting for photo submissions...")
        client.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        system.shutdown()


if __name__ == '__main__':
    main()
