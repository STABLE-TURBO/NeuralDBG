"""Terminal handler for WebSocket-based shell integration."""

import asyncio
import logging
import os
import platform
import shlex
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TerminalSession:
    """Manages a single terminal session with shell process."""

    def __init__(self, session_id: str, shell: str = "bash"):
        self.session_id = session_id
        self.shell = shell
        self.process: Optional[subprocess.Popen] = None
        self.command_history: List[str] = []
        self.current_dir = os.getcwd()
        self.env = os.environ.copy()
        
    def start(self):
        """Start the shell process."""
        try:
            system = platform.system()
            
            if self.shell == "powershell":
                if system == "Windows":
                    cmd = ["powershell", "-NoLogo", "-NoExit"]
                else:
                    cmd = ["pwsh", "-NoLogo", "-NoExit"]
            elif self.shell == "cmd":
                cmd = ["cmd", "/K"]
            elif self.shell == "zsh":
                cmd = ["zsh", "-i"]
            elif self.shell == "sh":
                cmd = ["sh", "-i"]
            else:
                cmd = ["bash", "-i"]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.current_dir,
                env=self.env,
                bufsize=0,
                universal_newlines=False,
            )
            
            logger.info(f"Terminal session {self.session_id} started with shell: {self.shell}")
            return True
        except Exception as e:
            logger.error(f"Failed to start terminal session: {e}")
            return False

    async def execute_command(self, command: str) -> str:
        """Execute a command in the shell."""
        if not self.process or self.process.poll() is not None:
            return "Error: Shell process not running\r\n"
        
        try:
            self.command_history.append(command)
            
            if command.strip().startswith("cd "):
                return await self._handle_cd(command)
            
            cmd_bytes = (command + "\n").encode('utf-8')
            self.process.stdin.write(cmd_bytes)
            self.process.stdin.flush()
            
            await asyncio.sleep(0.1)
            
            output = self._read_output()
            return output
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return f"Error: {str(e)}\r\n"

    async def _handle_cd(self, command: str) -> str:
        """Handle directory change commands."""
        try:
            parts = shlex.split(command)
            if len(parts) > 1:
                new_dir = parts[1]
                new_path = Path(new_dir)
                
                if not new_path.is_absolute():
                    new_path = Path(self.current_dir) / new_path
                
                new_path = new_path.resolve()
                
                if new_path.exists() and new_path.is_dir():
                    self.current_dir = str(new_path)
                    os.chdir(self.current_dir)
                    
                    if self.process:
                        cd_cmd = f"cd {shlex.quote(self.current_dir)}\n"
                        self.process.stdin.write(cd_cmd.encode('utf-8'))
                        self.process.stdin.flush()
                    
                    return f"{self.current_dir}\r\n"
                else:
                    return f"cd: no such file or directory: {new_dir}\r\n"
            else:
                home = Path.home()
                self.current_dir = str(home)
                os.chdir(self.current_dir)
                
                if self.process:
                    cd_cmd = f"cd {shlex.quote(self.current_dir)}\n"
                    self.process.stdin.write(cd_cmd.encode('utf-8'))
                    self.process.stdin.flush()
                
                return f"{self.current_dir}\r\n"
        except Exception as e:
            return f"cd: {str(e)}\r\n"

    def _read_output(self, timeout: float = 0.5) -> str:
        """Read available output from the shell process."""
        if not self.process or not self.process.stdout:
            return ""
        
        try:
            output_bytes = b""
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                if self.process.stdout.readable():
                    chunk = self.process.stdout.read(4096)
                    if chunk:
                        output_bytes += chunk
                    else:
                        break
                else:
                    break
            
            return output_bytes.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error reading output: {e}")
            return ""

    def get_autocomplete_suggestions(self, partial_command: str) -> List[str]:
        """Get autocomplete suggestions for partial command."""
        suggestions = []
        
        try:
            if not partial_command:
                return []
            
            parts = partial_command.split()
            if len(parts) == 0:
                return []
            
            if len(parts) == 1:
                suggestions = self._get_command_suggestions(parts[0])
            else:
                suggestions = self._get_path_suggestions(parts[-1])
            
        except Exception as e:
            logger.error(f"Autocomplete error: {e}")
        
        return suggestions[:10]

    def _get_command_suggestions(self, prefix: str) -> List[str]:
        """Get command name suggestions."""
        neural_commands = [
            'neural',
            'neural compile',
            'neural run',
            'neural visualize',
            'neural debug',
            'neural hpo',
            'neural automl',
        ]
        
        common_commands = [
            'ls', 'cd', 'pwd', 'cat', 'grep', 'find', 'echo',
            'python', 'pip', 'git', 'npm', 'node',
        ]
        
        all_commands = neural_commands + common_commands
        return [cmd for cmd in all_commands if cmd.startswith(prefix)]

    def _get_path_suggestions(self, prefix: str) -> List[str]:
        """Get file/directory path suggestions."""
        try:
            if prefix.startswith('~/'):
                base_dir = Path.home()
                relative_path = prefix[2:]
            elif prefix.startswith('/'):
                base_dir = Path('/')
                relative_path = prefix[1:]
            else:
                base_dir = Path(self.current_dir)
                relative_path = prefix
            
            if '/' in relative_path:
                dir_part = relative_path.rsplit('/', 1)[0]
                file_part = relative_path.rsplit('/', 1)[1]
                search_dir = base_dir / dir_part
            else:
                search_dir = base_dir
                file_part = relative_path
            
            if not search_dir.exists():
                return []
            
            suggestions = []
            for item in search_dir.iterdir():
                if item.name.startswith(file_part):
                    if item.is_dir():
                        suggestions.append(item.name + '/')
                    else:
                        suggestions.append(item.name)
            
            return sorted(suggestions)[:10]
        except Exception as e:
            logger.error(f"Path suggestion error: {e}")
            return []

    def change_shell(self, new_shell: str) -> bool:
        """Change the shell type."""
        try:
            self.stop()
            self.shell = new_shell
            return self.start()
        except Exception as e:
            logger.error(f"Failed to change shell: {e}")
            return False

    def stop(self):
        """Stop the shell process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping terminal session: {e}")
                try:
                    self.process.kill()
                except:
                    pass
            finally:
                self.process = None


class TerminalManager:
    """Manages multiple terminal sessions."""

    def __init__(self):
        self.sessions: Dict[str, TerminalSession] = {}

    def create_session(self, session_id: str, shell: str = "bash") -> TerminalSession:
        """Create a new terminal session."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        session = TerminalSession(session_id, shell)
        if session.start():
            self.sessions[session_id] = session
            return session
        else:
            raise RuntimeError(f"Failed to start terminal session {session_id}")

    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """Get an existing terminal session."""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        """Remove and cleanup a terminal session."""
        session = self.sessions.pop(session_id, None)
        if session:
            session.stop()

    def cleanup_all(self):
        """Cleanup all terminal sessions."""
        for session in self.sessions.values():
            session.stop()
        self.sessions.clear()
