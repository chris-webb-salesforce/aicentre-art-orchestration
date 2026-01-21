"""
Web visualization server for real-time drawing preview.

Serves a web page that shows:
- Live drawing progress
- Path being traced
- Progress bar and stats

Can run standalone or be started from the drawing script.
"""

import os
import threading
import webbrowser
import time
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# Flask app setup
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'portrait-drawing-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
_server_thread = None
_server_running = False


class DrawingVisualizer:
    """
    Interface for sending drawing events to the web visualization.

    Usage:
        viz = DrawingVisualizer()
        viz.start_server(open_browser=True)

        # When drawing starts
        viz.drawing_started(style_name, gcode_lines, contours, bounds, photo_path, lineart_path)

        # During drawing (call from progress callback)
        viz.update_progress(current_line, total_lines, x, y, pen_down)

        # When complete
        viz.drawing_complete()
    """

    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self._drawing_data = {}

    def start_server(self, open_browser=True):
        """Start the web server in a background thread."""
        global _server_thread, _server_running

        if _server_running:
            logger.info("Server already running")
            if open_browser:
                webbrowser.open(f'http://{self.host}:{self.port}')
            return

        def run_server():
            global _server_running
            _server_running = True
            socketio.run(app, host=self.host, port=self.port,
                        debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

        _server_thread = threading.Thread(target=run_server, daemon=True)
        _server_thread.start()

        # Wait for server to start
        time.sleep(1)

        logger.info(f"Visualization server started at http://{self.host}:{self.port}")

        if open_browser:
            webbrowser.open(f'http://{self.host}:{self.port}')

    def drawing_started(self, style_name, gcode_lines, contours_data, bounds,
                       photo_path=None, lineart_path=None):
        """
        Notify that a new drawing has started.

        Args:
            style_name: Name of the art style
            gcode_lines: List of GCode commands
            contours_data: List of contour dicts with 'points' key
            bounds: Dict with x_min, x_max, y_min, y_max
            photo_path: Path to original photo (optional)
            lineart_path: Path to line art image (optional)
        """
        # Convert contours to serializable format
        paths = []
        for contour in contours_data:
            if hasattr(contour, 'points'):
                # Contour object
                points = [(float(p[0]), float(p[1])) for p in contour.points]
            elif isinstance(contour, dict) and 'points' in contour:
                points = [(float(p[0]), float(p[1])) for p in contour['points']]
            else:
                continue
            paths.append(points)

        self._drawing_data = {
            'style': style_name,
            'total_lines': len(gcode_lines),
            'paths': paths,
            'bounds': bounds,
            'photo_path': photo_path,
            'lineart_path': lineart_path
        }

        socketio.emit('drawing_started', self._drawing_data)
        logger.info(f"Drawing started: {style_name} with {len(paths)} paths")

    def update_progress(self, current_line, total_lines, x, y, pen_down=True):
        """Send progress update to web clients."""
        data = {
            'current_line': current_line,
            'total_lines': total_lines,
            'progress': current_line / total_lines if total_lines > 0 else 0,
            'x': float(x),
            'y': float(y),
            'pen_down': pen_down
        }
        socketio.emit('progress', data)

    def drawing_complete(self):
        """Notify that drawing is complete."""
        socketio.emit('drawing_complete', {})
        logger.info("Drawing complete")

    def send_status(self, message):
        """Send a status message to display."""
        socketio.emit('status', {'message': message})


# Global visualizer instance
visualizer = DrawingVisualizer()


# Flask routes
@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('visualization.html')


@app.route('/status')
def status():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'drawing': bool(visualizer._drawing_data)})


# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle new client connection."""
    logger.info("Client connected")
    # Send current state if drawing in progress
    if visualizer._drawing_data:
        emit('drawing_started', visualizer._drawing_data)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


# Standalone server
def run_standalone(host='127.0.0.1', port=5000, open_browser=True):
    """Run the server standalone (blocking)."""
    if open_browser:
        # Open browser after short delay
        def open_delayed():
            time.sleep(1)
            webbrowser.open(f'http://{host}:{port}')
        threading.Thread(target=open_delayed, daemon=True).start()

    print(f"\n{'='*50}")
    print("VISUALIZATION SERVER")
    print(f"{'='*50}")
    print(f"Open http://{host}:{port} in your browser")
    print("Waiting for drawing to start...")
    print(f"{'='*50}\n")

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_standalone(open_browser=True)
