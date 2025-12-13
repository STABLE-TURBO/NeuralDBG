"""Tests for terminal handler."""

import asyncio
import pytest
from terminal_handler import TerminalSession, TerminalManager


class TestTerminalSession:
    """Test cases for TerminalSession class."""

    def test_session_creation(self):
        """Test creating a terminal session."""
        session = TerminalSession("test-session-1", shell="bash")
        assert session.session_id == "test-session-1"
        assert session.shell == "bash"
        assert session.process is None
        assert len(session.command_history) == 0

    def test_session_start(self):
        """Test starting a terminal session."""
        session = TerminalSession("test-session-2", shell="bash")
        result = session.start()
        
        assert result is True
        assert session.process is not None
        assert session.process.poll() is None
        
        session.stop()

    @pytest.mark.asyncio
    async def test_execute_command(self):
        """Test executing a command in terminal session."""
        session = TerminalSession("test-session-3", shell="bash")
        session.start()
        
        output = await session.execute_command("echo 'Hello World'")
        
        assert output is not None
        assert isinstance(output, str)
        assert len(session.command_history) == 1
        assert session.command_history[0] == "echo 'Hello World'"
        
        session.stop()

    @pytest.mark.asyncio
    async def test_cd_command(self):
        """Test directory change command."""
        session = TerminalSession("test-session-4", shell="bash")
        session.start()
        
        import os
        original_dir = os.getcwd()
        
        output = await session.execute_command("cd /tmp")
        
        assert "/tmp" in output or "tmp" in output
        
        os.chdir(original_dir)
        session.stop()

    def test_autocomplete_commands(self):
        """Test command autocomplete suggestions."""
        session = TerminalSession("test-session-5", shell="bash")
        
        suggestions = session.get_autocomplete_suggestions("neur")
        
        assert isinstance(suggestions, list)
        assert any("neural" in s for s in suggestions)

    def test_autocomplete_paths(self):
        """Test path autocomplete suggestions."""
        session = TerminalSession("test-session-6", shell="bash")
        
        suggestions = session.get_autocomplete_suggestions("./")
        
        assert isinstance(suggestions, list)

    def test_change_shell(self):
        """Test changing shell type."""
        session = TerminalSession("test-session-7", shell="bash")
        session.start()
        
        result = session.change_shell("sh")
        
        assert result is True
        assert session.shell == "sh"
        
        session.stop()

    def test_stop_session(self):
        """Test stopping a terminal session."""
        session = TerminalSession("test-session-8", shell="bash")
        session.start()
        
        assert session.process is not None
        
        session.stop()
        
        if session.process:
            assert session.process.poll() is not None


class TestTerminalManager:
    """Test cases for TerminalManager class."""

    def test_manager_creation(self):
        """Test creating a terminal manager."""
        manager = TerminalManager()
        assert len(manager.sessions) == 0

    def test_create_session(self):
        """Test creating a session through manager."""
        manager = TerminalManager()
        
        session = manager.create_session("manager-test-1", shell="bash")
        
        assert session is not None
        assert session.session_id == "manager-test-1"
        assert "manager-test-1" in manager.sessions
        
        manager.cleanup_all()

    def test_get_session(self):
        """Test retrieving an existing session."""
        manager = TerminalManager()
        
        created_session = manager.create_session("manager-test-2", shell="bash")
        retrieved_session = manager.get_session("manager-test-2")
        
        assert retrieved_session is created_session
        
        manager.cleanup_all()

    def test_get_nonexistent_session(self):
        """Test retrieving a session that doesn't exist."""
        manager = TerminalManager()
        
        session = manager.get_session("nonexistent")
        
        assert session is None

    def test_remove_session(self):
        """Test removing a session."""
        manager = TerminalManager()
        
        manager.create_session("manager-test-3", shell="bash")
        assert "manager-test-3" in manager.sessions
        
        manager.remove_session("manager-test-3")
        assert "manager-test-3" not in manager.sessions

    def test_cleanup_all(self):
        """Test cleaning up all sessions."""
        manager = TerminalManager()
        
        manager.create_session("manager-test-4", shell="bash")
        manager.create_session("manager-test-5", shell="bash")
        
        assert len(manager.sessions) == 2
        
        manager.cleanup_all()
        
        assert len(manager.sessions) == 0


class TestShellCommands:
    """Test cases for shell-specific commands."""

    @pytest.mark.asyncio
    async def test_bash_echo(self):
        """Test echo command in bash."""
        session = TerminalSession("bash-test-1", shell="bash")
        session.start()
        
        output = await session.execute_command("echo 'test'")
        
        assert "test" in output.lower() or output != ""
        
        session.stop()

    @pytest.mark.asyncio
    async def test_pwd_command(self):
        """Test pwd command."""
        session = TerminalSession("bash-test-2", shell="bash")
        session.start()
        
        output = await session.execute_command("pwd")
        
        assert output is not None
        assert len(output) > 0
        
        session.stop()


class TestNeuralCLIAutocomplete:
    """Test cases for Neural CLI autocomplete."""

    def test_neural_command_suggestions(self):
        """Test autocomplete for neural commands."""
        session = TerminalSession("neural-test-1", shell="bash")
        
        suggestions = session.get_autocomplete_suggestions("neural ")
        
        assert len(suggestions) > 0
        assert any("compile" in s for s in suggestions)
        assert any("run" in s for s in suggestions)

    def test_neural_compile_suggestion(self):
        """Test autocomplete for neural compile."""
        session = TerminalSession("neural-test-2", shell="bash")
        
        suggestions = session.get_autocomplete_suggestions("neural comp")
        
        assert any("compile" in s for s in suggestions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
