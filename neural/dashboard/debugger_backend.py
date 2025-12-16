"""
Backend support for the integrated debugger.
Handles debug commands, breakpoint management, and execution control.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set


logger = logging.getLogger(__name__)

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SocketIO = None
    SOCKETIO_AVAILABLE = False


class DebuggerBackend:
    """Backend for integrated debugger functionality."""

    def __init__(self, socketio: Optional[SocketIO] = None):
        self.socketio = socketio
        self.is_running = False
        self.is_paused = False
        self.current_layer = None
        self.current_step = 'idle'
        self.breakpoints: Set[int] = set()
        self.execution_callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_pause': [],
            'on_stop': [],
            'on_step': [],
            'on_breakpoint': [],
        }
        self.trace_buffer: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}
        self.execution_progress = {
            'forwardPass': 0,
            'backwardPass': 0,
            'currentPhase': 'idle',
        }
        self._lock = threading.Lock()

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a debug event."""
        if event in self.execution_callbacks:
            self.execution_callbacks[event].append(callback)

    def emit_state_change(self):
        """Emit current debugger state to connected clients."""
        if self.socketio and SOCKETIO_AVAILABLE:
            with self._lock:
                self.socketio.emit('state_change', {
                    'type': 'state_change',
                    'isRunning': self.is_running,
                    'isPaused': self.is_paused,
                    'currentLayer': self.current_layer,
                    'currentStep': self.current_step,
                })

    def emit_trace_update(self):
        """Emit trace data update to connected clients."""
        if self.socketio and SOCKETIO_AVAILABLE:
            with self._lock:
                self.socketio.emit('trace_update', {
                    'type': 'trace_update',
                    'trace': self.trace_buffer,
                })

    def emit_variable_update(self):
        """Emit variable data update to connected clients."""
        if self.socketio and SOCKETIO_AVAILABLE:
            with self._lock:
                self.socketio.emit('variable_update', {
                    'type': 'variable_update',
                    'variables': self.variables,
                })

    def emit_progress_update(self):
        """Emit execution progress update to connected clients."""
        if self.socketio and SOCKETIO_AVAILABLE:
            with self._lock:
                self.socketio.emit('execution_progress', {
                    'type': 'execution_progress',
                    'progress': self.execution_progress,
                })

    def handle_start(self):
        """Handle start debugging command."""
        with self._lock:
            self.is_running = True
            self.is_paused = False
            self.current_step = 'running'
            self.execution_progress['currentPhase'] = 'forward'

        for callback in self.execution_callbacks['on_start']:
            callback()

        self.emit_state_change()
        self.emit_progress_update()

    def handle_pause(self):
        """Handle pause debugging command."""
        with self._lock:
            self.is_paused = True
            self.current_step = 'paused'

        for callback in self.execution_callbacks['on_pause']:
            callback()

        self.emit_state_change()

    def handle_stop(self):
        """Handle stop debugging command."""
        with self._lock:
            self.is_running = False
            self.is_paused = False
            self.current_layer = None
            self.current_step = 'idle'
            self.trace_buffer = []
            self.variables = {}
            self.execution_progress = {
                'forwardPass': 0,
                'backwardPass': 0,
                'currentPhase': 'idle',
            }

        for callback in self.execution_callbacks['on_stop']:
            callback()

        self.emit_state_change()
        self.emit_trace_update()
        self.emit_variable_update()
        self.emit_progress_update()

    def handle_step(self):
        """Handle step command."""
        with self._lock:
            self.current_step = 'stepping'

        for callback in self.execution_callbacks['on_step']:
            callback()

        self.emit_state_change()

    def handle_continue(self):
        """Handle continue command."""
        with self._lock:
            self.is_paused = False
            self.current_step = 'running'

        self.emit_state_change()

    def handle_add_breakpoint(self, line_number: int):
        """Add a breakpoint at the specified line number."""
        with self._lock:
            self.breakpoints.add(line_number)

    def handle_remove_breakpoint(self, line_number: int):
        """Remove a breakpoint at the specified line number."""
        with self._lock:
            if line_number in self.breakpoints:
                self.breakpoints.remove(line_number)

    def handle_clear_all_breakpoints(self):
        """Clear all breakpoints."""
        with self._lock:
            self.breakpoints.clear()

    def check_breakpoint(self, line_number: int) -> bool:
        """Check if a breakpoint is set at the given line number."""
        with self._lock:
            return line_number in self.breakpoints

    def update_trace_data(self, trace_entry: Dict[str, Any]):
        """Update trace data with a new entry."""
        with self._lock:
            self.trace_buffer.append(trace_entry)

        self.emit_trace_update()

    def update_variables(self, variables: Dict[str, Any]):
        """Update variable data."""
        with self._lock:
            self.variables.update(variables)

        self.emit_variable_update()

    def update_execution_progress(
        self,
        forward_pass: Optional[float] = None,
        backward_pass: Optional[float] = None,
        phase: Optional[str] = None,
    ):
        """Update execution progress."""
        with self._lock:
            if forward_pass is not None:
                self.execution_progress['forwardPass'] = forward_pass
            if backward_pass is not None:
                self.execution_progress['backwardPass'] = backward_pass
            if phase is not None:
                self.execution_progress['currentPhase'] = phase

        self.emit_progress_update()

    def set_current_layer(self, layer: Any):
        """Set the current layer being executed."""
        with self._lock:
            self.current_layer = layer

        self.emit_state_change()

    def wait_if_paused(self):
        """Wait if execution is paused (for step debugging)."""
        while True:
            with self._lock:
                if not self.is_paused:
                    break
            time.sleep(0.1)


def setup_debugger_routes(app, socketio: Optional[SocketIO], debugger: DebuggerBackend):
    """Set up debugger routes and WebSocket handlers."""
    if not socketio or not SOCKETIO_AVAILABLE:
        return

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info('Debugger client connected')
        debugger.emit_state_change()

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info('Debugger client disconnected')

    @socketio.on('message')
    def handle_message(data):
        """Handle incoming WebSocket messages."""
        try:
            if isinstance(data, str):
                data = json.loads(data)

            command = data.get('command')

            if command == 'start':
                debugger.handle_start()
            elif command == 'pause':
                debugger.handle_pause()
            elif command == 'stop':
                debugger.handle_stop()
            elif command == 'step':
                debugger.handle_step()
            elif command == 'step_over':
                debugger.handle_step()
            elif command == 'step_into':
                debugger.handle_step()
            elif command == 'step_out':
                debugger.handle_step()
            elif command == 'continue':
                debugger.handle_continue()
            elif command == 'add_breakpoint':
                line_number = data.get('lineNumber')
                if line_number is not None:
                    debugger.handle_add_breakpoint(line_number)
            elif command == 'remove_breakpoint':
                line_number = data.get('lineNumber')
                if line_number is not None:
                    debugger.handle_remove_breakpoint(line_number)
            elif command == 'clear_all_breakpoints':
                debugger.handle_clear_all_breakpoints()

        except Exception as e:
            logger.error(f'Error handling debugger command: {e}', exc_info=True)

    @app.route('/api/debugger/status')
    def get_debugger_status():
        """Get current debugger status."""
        return {
            'isRunning': debugger.is_running,
            'isPaused': debugger.is_paused,
            'currentLayer': debugger.current_layer,
            'currentStep': debugger.current_step,
            'breakpoints': list(debugger.breakpoints),
        }

    @app.route('/api/debugger/trace')
    def get_trace_data():
        """Get current trace data."""
        return {'trace': debugger.trace_buffer}

    @app.route('/api/debugger/variables')
    def get_variables():
        """Get current variable data."""
        return {'variables': debugger.variables}

    @app.route('/api/debugger/progress')
    def get_progress():
        """Get current execution progress."""
        return {'progress': debugger.execution_progress}


def create_debugger_backend(socketio: Optional[SocketIO] = None) -> DebuggerBackend:
    """Create and return a new debugger backend instance."""
    return DebuggerBackend(socketio)
