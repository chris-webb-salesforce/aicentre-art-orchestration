"""
Heroku-hosted relay server for iPad photo capture.

Serves the iPad capture interface and relays photos to the local machine
running the robot portrait system via WebSocket. Supports multiple robot
arms with a FIFO job queue and auto-dispatch.

Usage:
    # Local development
    python -m src.web.capture_app

    # Heroku (via Procfile)
    gunicorn --worker-class eventlet -w 1 src.web.capture_app:app
"""

import os
import logging
import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field
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
    async_mode='eventlet' if os.environ.get('DYNO') else 'threading',
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


# --- Multi-arm state management ---

@dataclass
class ArmSlot:
    arm_id: str
    sid: str
    label: str = ''
    status: str = 'idle'  # idle | processing | drawing | complete | error


@dataclass
class QueuedJob:
    job_id: str
    photo: str
    style: str
    requester_sid: str
    enqueued_at: float = field(default_factory=time.time)


class ArmPool:
    """Manages multiple robot arms and a FIFO job queue."""

    def __init__(self):
        self.arms: dict = {}                # arm_id -> ArmSlot
        self.queue: deque = deque()         # pending QueuedJobs
        self.active_jobs: dict = {}         # arm_id -> QueuedJob
        self._lock = threading.Lock()

    @property
    def any_connected(self):
        return len(self.arms) > 0

    def register(self, arm_id: str, sid: str, label: str = ''):
        with self._lock:
            existing = self.arms.get(arm_id)
            if existing and existing.status not in ('idle', 'complete', 'error'):
                # Arm reconnected mid-job — mark the old job as lost
                self.active_jobs.pop(arm_id, None)
            self.arms[arm_id] = ArmSlot(arm_id=arm_id, sid=sid, label=label or arm_id)

    def unregister(self, sid: str):
        """Remove arm by SID. Returns (arm_id, lost_job) tuple."""
        with self._lock:
            for arm_id, slot in list(self.arms.items()):
                if slot.sid == sid:
                    del self.arms[arm_id]
                    lost_job = self.active_jobs.pop(arm_id, None)
                    return arm_id, lost_job
        return None, None

    def _get_idle_arm(self):
        """Return first idle arm. Must be called with _lock held."""
        for slot in self.arms.values():
            if slot.status == 'idle':
                return slot
        return None

    def enqueue(self, photo: str, style: str, requester_sid: str):
        job = QueuedJob(
            job_id=uuid.uuid4().hex[:8],
            photo=photo,
            style=style,
            requester_sid=requester_sid,
        )
        with self._lock:
            self.queue.append(job)
        return job

    def queue_position(self, job_id: str):
        """1-based position in queue. 0 if not found (already dispatched)."""
        with self._lock:
            for i, j in enumerate(self.queue):
                if j.job_id == job_id:
                    return i + 1
        return 0

    def try_dispatch(self):
        """Pop next job and assign to an idle arm. Returns (slot, job) or None."""
        with self._lock:
            if not self.queue:
                return None
            slot = self._get_idle_arm()
            if slot is None:
                return None
            job = self.queue.popleft()
            slot.status = 'processing'
            self.active_jobs[slot.arm_id] = job
            return slot, job

    def get_active_job(self, arm_id: str):
        """Get the active job for an arm, if any."""
        with self._lock:
            return self.active_jobs.get(arm_id)

    def update_arm_status(self, arm_id: str, status: str):
        with self._lock:
            if arm_id in self.arms:
                self.arms[arm_id].status = status

    def mark_arm_complete(self, arm_id: str):
        """Mark arm as complete (waiting for paper confirmation)."""
        with self._lock:
            if arm_id in self.arms:
                self.arms[arm_id].status = 'complete'
            self.active_jobs.pop(arm_id, None)

    def confirm_arm_ready(self, arm_id: str):
        """Operator confirms paper loaded — arm becomes idle."""
        with self._lock:
            if arm_id in self.arms and self.arms[arm_id].status == 'complete':
                self.arms[arm_id].status = 'idle'
                return True
        return False

    def mark_arm_error(self, arm_id: str):
        with self._lock:
            if arm_id in self.arms:
                self.arms[arm_id].status = 'error'
            self.active_jobs.pop(arm_id, None)

    def reset_arm_error(self, arm_id: str):
        """Reset an errored arm back to idle."""
        with self._lock:
            if arm_id in self.arms and self.arms[arm_id].status == 'error':
                self.arms[arm_id].status = 'idle'
                return True
        return False

    def arm_from_sid(self, sid: str):
        with self._lock:
            for slot in self.arms.values():
                if slot.sid == sid:
                    return slot
        return None

    def get_overview(self):
        with self._lock:
            return {
                'arms': [
                    {'arm_id': s.arm_id, 'label': s.label, 'status': s.status}
                    for s in self.arms.values()
                ],
                'queue_depth': len(self.queue),
                'any_connected': self.any_connected,
            }


pool = ArmPool()


def _try_dispatch():
    """Attempt to dispatch next queued job to an idle arm."""
    result = pool.try_dispatch()
    if result is None:
        return
    slot, job = result
    socketio.emit('new_job', {
        'photo': job.photo,
        'style': job.style,
        'job_id': job.job_id,
    }, namespace='/robot', to=slot.sid)
    socketio.emit('overview', pool.get_overview(), namespace='/capture')
    logger.info(f"Dispatched job {job.job_id} to arm '{slot.arm_id}' ({slot.label})")


def _broadcast_overview():
    socketio.emit('overview', pool.get_overview(), namespace='/capture')


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
    light_mode = os.environ.get('LIGHT_MODE', 'false').lower() == 'true'
    return render_template('capture.html', light_mode=light_mode)


@app.route('/api/status')
def api_status():
    return jsonify(pool.get_overview())


# --- SocketIO: iPad clients (/capture namespace) ---

@socketio.on('connect', namespace='/capture')
def capture_connect():
    logger.info("iPad client connected")
    emit('overview', pool.get_overview())


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

    if not pool.any_connected:
        emit('error', {'message': 'No robots connected. Please wait.'})
        return

    job = pool.enqueue(photo, style, requester_sid=request.sid)

    # Tell submitter their queue position before dispatch may dequeue it
    pos = pool.queue_position(job.job_id)
    emit('queued', {'job_id': job.job_id, 'position': pos})

    # Try immediate dispatch
    _try_dispatch()
    _broadcast_overview()

    logger.info(f"Photo submitted with style '{style}', job {job.job_id} (queue pos: {pos})")


@socketio.on('confirm_ready', namespace='/capture')
def handle_confirm_ready(data):
    """Operator confirms paper is loaded on an arm."""
    arm_id = data.get('arm_id')
    if not arm_id:
        return

    if pool.confirm_arm_ready(arm_id):
        logger.info(f"Arm '{arm_id}' confirmed ready by operator")
        _broadcast_overview()
        _try_dispatch()
    elif pool.reset_arm_error(arm_id):
        logger.info(f"Arm '{arm_id}' error reset by operator")
        _broadcast_overview()
        _try_dispatch()


# --- SocketIO: Robot connection (/robot namespace) ---

@socketio.on('connect', namespace='/robot')
def robot_connect():
    token = request.args.get('token', '')
    expected_token = os.environ.get('ROBOT_TOKEN', '')
    if expected_token and token != expected_token:
        logger.warning("Robot connection rejected: invalid token")
        return False
    logger.info(f"Robot socket connected (sid: {request.sid})")


@socketio.on('ready', namespace='/robot')
def robot_ready(data):
    """Robot identifies itself after connecting."""
    arm_id = data.get('arm_id', request.sid[:8])
    label = data.get('label', arm_id)
    pool.register(arm_id, request.sid, label)
    logger.info(f"Arm '{arm_id}' ({label}) ready")
    _broadcast_overview()
    _try_dispatch()


@socketio.on('disconnect', namespace='/robot')
def robot_disconnect():
    arm_id, lost_job = pool.unregister(request.sid)
    if arm_id:
        logger.info(f"Arm '{arm_id}' disconnected")
        if lost_job:
            # Notify the submitter that their job was lost
            socketio.emit('status', {
                'arm_id': arm_id,
                'job_id': lost_job.job_id,
                'status': 'error',
                'message': f'{arm_id} disconnected during drawing',
                'percent': 0,
            }, namespace='/capture', to=lost_job.requester_sid)
            logger.warning(f"Job {lost_job.job_id} lost due to arm '{arm_id}' disconnect")
        _broadcast_overview()


@socketio.on('status_update', namespace='/robot')
def robot_status_update(data):
    """Robot sends progress updates."""
    slot = pool.arm_from_sid(request.sid)
    if not slot:
        return

    status = data.get('status', 'processing')
    message = data.get('message', '')
    percent = data.get('percent', 0)

    pool.update_arm_status(slot.arm_id, status)

    active_job = pool.get_active_job(slot.arm_id)
    job_id = active_job.job_id if active_job else None

    # Send to the specific submitter if known, otherwise broadcast
    payload = {
        'arm_id': slot.arm_id,
        'label': slot.label,
        'job_id': job_id,
        'status': status,
        'message': message,
        'percent': percent,
    }
    if active_job:
        socketio.emit('status', payload, namespace='/capture', to=active_job.requester_sid)
    else:
        socketio.emit('status', payload, namespace='/capture')


@socketio.on('job_complete', namespace='/robot')
def robot_job_complete(data=None):
    """Robot signals drawing is complete."""
    slot = pool.arm_from_sid(request.sid)
    if not slot:
        return

    # Get the job before marking complete (which removes it from active_jobs)
    active_job = pool.get_active_job(slot.arm_id)
    job_id = active_job.job_id if active_job else None
    requester_sid = active_job.requester_sid if active_job else None

    # Mark as complete (waiting for paper confirmation)
    pool.mark_arm_complete(slot.arm_id)

    payload = {
        'arm_id': slot.arm_id,
        'label': slot.label,
        'job_id': job_id,
        'status': 'complete',
        'message': 'Portrait complete!',
        'percent': 100,
    }
    if requester_sid:
        socketio.emit('status', payload, namespace='/capture', to=requester_sid)
    else:
        socketio.emit('status', payload, namespace='/capture')

    _broadcast_overview()
    logger.info(f"Arm '{slot.arm_id}' completed drawing — waiting for paper confirmation")


@socketio.on('job_error', namespace='/robot')
def robot_job_error(data):
    """Robot signals an error."""
    slot = pool.arm_from_sid(request.sid)
    if not slot:
        return

    message = data.get('message', 'An error occurred')
    active_job = pool.get_active_job(slot.arm_id)
    job_id = active_job.job_id if active_job else None
    requester_sid = active_job.requester_sid if active_job else None

    pool.mark_arm_error(slot.arm_id)

    payload = {
        'arm_id': slot.arm_id,
        'label': slot.label,
        'job_id': job_id,
        'status': 'error',
        'message': message,
        'percent': 0,
    }
    if requester_sid:
        socketio.emit('status', payload, namespace='/capture', to=requester_sid)
    else:
        socketio.emit('status', payload, namespace='/capture')

    _broadcast_overview()
    logger.error(f"Arm '{slot.arm_id}' error: {message}")


# --- Main ---

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', 5001))
    print(f"\n{'='*50}")
    print("PORTRAIT CAPTURE SERVER (Multi-Arm)")
    print(f"{'='*50}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Password: {CAPTURE_PASSWORD}")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
