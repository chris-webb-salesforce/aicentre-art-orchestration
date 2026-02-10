"""
Heroku-hosted relay server for iPad photo capture.

Serves the iPad capture interface and relays photos to the local machine
running the robot portrait system via WebSocket.

Usage:
    # Local development
    python -m src.web.capture_app

    # Heroku (via Procfile)
    gunicorn --worker-class eventlet -w 1 src.web.capture_app:app
"""

import os
import logging
import time
import threading
from datetime import timedelta
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# Flask app setup
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent' if os.environ.get('DYNO') else 'threading',
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB for photo uploads
    ping_timeout=60,
    ping_interval=25
)

CAPTURE_PASSWORD = os.environ.get('CAPTURE_PASSWORD', 'portrait')

# Art styles (matches settings.yaml)
ART_STYLES = {
    'simple': {'name': 'Simple Line Art', 'description': 'Clean, flowing cartoon line drawing'},
    'minimal': {'name': 'Minimalist', 'description': 'Single continuous line, abstract but recognizable'},
    'vangogh': {'name': 'Van Gogh', 'description': 'Expressive swirling line work'},
    'ghibli': {'name': 'Studio Ghibli', 'description': 'Anime-style with gentle lines'},
    'picasso': {'name': 'Picasso Cubist', 'description': 'Geometric, angular interpretation'},
    'sketch': {'name': 'Quick Sketch', 'description': 'Loose, gestural 30-second sketch'},
    'caricature': {'name': 'Caricature', 'description': 'Exaggerated features, playful'},
    'geometric': {'name': 'Geometric', 'description': 'Straight lines, low-poly angular'},
    'contour': {'name': 'Blind Contour', 'description': 'Wandering continuous line, imperfect'},
}


class JobManager:
    """Manages the current drawing job state."""

    IDLE = 'idle'
    QUEUED = 'queued'
    PROCESSING = 'processing'
    DRAWING = 'drawing'
    COMPLETE = 'complete'
    ERROR = 'error'

    def __init__(self):
        self.status = self.IDLE
        self.message = ''
        self.percent = 0
        self.robot_sid = None  # SocketIO session ID of connected robot
        self._lock = threading.Lock()

    @property
    def robot_connected(self):
        return self.robot_sid is not None

    @property
    def is_busy(self):
        return self.status in (self.QUEUED, self.PROCESSING, self.DRAWING)

    def get_state(self):
        return {
            'status': self.status,
            'message': self.message,
            'percent': self.percent,
            'robot_connected': self.robot_connected,
        }

    def submit(self):
        with self._lock:
            if self.is_busy:
                return False
            self.status = self.QUEUED
            self.message = 'Photo sent to robot...'
            self.percent = 0
            return True

    def update(self, status, message='', percent=0):
        with self._lock:
            self.status = status
            self.message = message
            self.percent = percent

    def reset(self):
        with self._lock:
            self.status = self.IDLE
            self.message = ''
            self.percent = 0


job = JobManager()


# --- Authentication ---

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# --- Routes ---

@app.route('/')
def index():
    if session.get('authenticated'):
        return redirect(url_for('capture'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == CAPTURE_PASSWORD:
            session['authenticated'] = True
            session.permanent = True
            return redirect(url_for('capture'))
        error = 'Invalid password'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/capture')
@require_auth
def capture():
    return render_template('capture.html')


@app.route('/api/status')
def api_status():
    return jsonify(job.get_state())


# --- SocketIO: iPad clients (/capture namespace) ---

@socketio.on('connect', namespace='/capture')
def capture_connect():
    logger.info("iPad client connected")
    emit('state', job.get_state())


@socketio.on('disconnect', namespace='/capture')
def capture_disconnect():
    logger.info("iPad client disconnected")


@socketio.on('submit_photo', namespace='/capture')
def handle_submit_photo(data):
    """Handle photo submission from iPad."""
    photo = data.get('photo')
    style = data.get('style', 'minimal')

    if not photo:
        emit('error', {'message': 'No photo data received'})
        return

    if style not in ART_STYLES:
        style = 'minimal'

    if not job.robot_connected:
        emit('error', {'message': 'Robot is not connected. Please wait.'})
        return

    if not job.submit():
        emit('error', {'message': 'Robot is busy drawing. Please wait.'})
        return

    # Relay photo to robot
    socketio.emit('new_job', {
        'photo': photo,
        'style': style,
    }, namespace='/robot', to=job.robot_sid)

    # Notify all iPad clients
    socketio.emit('status', {
        'status': 'queued',
        'message': 'Photo sent to robot...',
        'percent': 0,
    }, namespace='/capture')

    logger.info(f"Photo submitted with style '{style}', relayed to robot")


# --- SocketIO: Robot connection (/robot namespace) ---

@socketio.on('connect', namespace='/robot')
def robot_connect():
    token = request.args.get('token', '')
    expected_token = os.environ.get('ROBOT_TOKEN', '')
    if expected_token and token != expected_token:
        logger.warning("Robot connection rejected: invalid token")
        return False

    job.robot_sid = request.sid
    logger.info(f"Robot connected (sid: {request.sid})")

    # Notify iPad clients
    socketio.emit('state', job.get_state(), namespace='/capture')


@socketio.on('disconnect', namespace='/robot')
def robot_disconnect():
    if request.sid == job.robot_sid:
        job.robot_sid = None
        job.reset()
        logger.info("Robot disconnected")
        socketio.emit('state', job.get_state(), namespace='/capture')


@socketio.on('status_update', namespace='/robot')
def robot_status_update(data):
    """Robot sends progress updates."""
    status = data.get('status', 'processing')
    message = data.get('message', '')
    percent = data.get('percent', 0)

    job.update(status, message, percent)

    # Broadcast to all iPad clients
    socketio.emit('status', {
        'status': status,
        'message': message,
        'percent': percent,
    }, namespace='/capture')


@socketio.on('job_complete', namespace='/robot')
def robot_job_complete(data=None):
    """Robot signals drawing is complete."""
    job.update(JobManager.COMPLETE, 'Portrait complete!', 100)

    socketio.emit('status', {
        'status': 'complete',
        'message': 'Portrait complete!',
        'percent': 100,
    }, namespace='/capture')

    logger.info("Drawing complete")

    # Reset after a delay (only if no new job has started)
    def reset_later():
        time.sleep(10)
        with job._lock:
            if not job.is_busy:
                job.status = JobManager.IDLE
                job.message = ''
                job.percent = 0
        socketio.emit('state', job.get_state(), namespace='/capture')

    threading.Thread(target=reset_later, daemon=True).start()


@socketio.on('job_error', namespace='/robot')
def robot_job_error(data):
    """Robot signals an error."""
    message = data.get('message', 'An error occurred')
    job.update(JobManager.ERROR, message)

    socketio.emit('status', {
        'status': 'error',
        'message': message,
        'percent': 0,
    }, namespace='/capture')

    logger.error(f"Robot error: {message}")

    # Reset after a delay (only if no new job has started)
    def reset_later():
        time.sleep(5)
        with job._lock:
            if not job.is_busy:
                job.status = JobManager.IDLE
                job.message = ''
                job.percent = 0
        socketio.emit('state', job.get_state(), namespace='/capture')

    threading.Thread(target=reset_later, daemon=True).start()


# --- Main ---

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', 5001))
    print(f"\n{'='*50}")
    print("PORTRAIT CAPTURE SERVER")
    print(f"{'='*50}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Password: {CAPTURE_PASSWORD}")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
