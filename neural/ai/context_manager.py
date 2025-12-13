"""
Context Manager for AI Assistant

Maintains conversation history, session state, and provides context retention
across interactions for more coherent AI assistance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path


class ConversationMessage:
    """Represents a single message in conversation."""
    
    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize conversation message.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (timestamp, intent, etc.)
        """
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMessage:
        """Create from dictionary."""
        msg = cls(data['role'], data['content'], data.get('metadata'))
        msg.timestamp = datetime.fromisoformat(data['timestamp'])
        return msg


class SessionContext:
    """
    Maintains context for a single conversation session.
    """
    
    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Initialize session context.
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or self._generate_session_id()
        self.messages: List[ConversationMessage] = []
        self.model_state: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.context_variables: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add message to conversation history."""
        message = ConversationMessage(role, content, metadata)
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def get_recent_messages(self, n: int = 10) -> List[ConversationMessage]:
        """Get n most recent messages."""
        return self.messages[-n:]
    
    def update_model_state(self, updates: Dict[str, Any]) -> None:
        """Update current model state."""
        self.model_state.update(updates)
        self.last_activity = datetime.now()
    
    def update_context(self, key: str, value: Any) -> None:
        """Update context variable."""
        self.context_variables[key] = value
        self.last_activity = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context variable."""
        return self.context_variables.get(key, default)
    
    def clear_context(self) -> None:
        """Clear context variables."""
        self.context_variables.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'model_state': self.model_state,
            'user_preferences': self.user_preferences,
            'context_variables': self.context_variables,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionContext:
        """Create from dictionary."""
        session = cls(data['session_id'])
        session.messages = [
            ConversationMessage.from_dict(msg) for msg in data['messages']
        ]
        session.model_state = data.get('model_state', {})
        session.user_preferences = data.get('user_preferences', {})
        session.context_variables = data.get('context_variables', {})
        session.created_at = datetime.fromisoformat(data['created_at'])
        session.last_activity = datetime.fromisoformat(data['last_activity'])
        return session


class ContextManager:
    """
    Manages conversation context and history across sessions.
    """
    
    def __init__(self, persistence_dir: Optional[str] = None) -> None:
        """
        Initialize context manager.
        
        Args:
            persistence_dir: Directory to persist sessions (optional)
        """
        self.current_session: Optional[SessionContext] = None
        self.sessions: Dict[str, SessionContext] = {}
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None
        
        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, session_id: Optional[str] = None) -> SessionContext:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional session ID (generates one if not provided)
            
        Returns:
            New session context
        """
        session = SessionContext(session_id)
        self.current_session = session
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session by ID."""
        # Try memory first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try loading from disk
        if self.persistence_dir:
            return self._load_session(session_id)
        
        return None
    
    def save_session(self, session_id: Optional[str] = None) -> bool:
        """
        Save session to disk.
        
        Args:
            session_id: Session to save (uses current if not specified)
            
        Returns:
            True if saved successfully
        """
        if not self.persistence_dir:
            return False
        
        session = self.current_session if session_id is None else self.sessions.get(session_id)
        if not session:
            return False
        
        try:
            filepath = self.persistence_dir / f"{session.session_id}.json"
            with open(filepath, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception:
            return False
    
    def _load_session(self, session_id: str) -> Optional[SessionContext]:
        """Load session from disk."""
        if not self.persistence_dir:
            return None
        
        try:
            filepath = self.persistence_dir / f"{session_id}.json"
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            session = SessionContext.from_dict(data)
            self.sessions[session_id] = session
            return session
        except Exception:
            return None
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        session_ids = set(self.sessions.keys())
        
        # Add persisted sessions
        if self.persistence_dir and self.persistence_dir.exists():
            for filepath in self.persistence_dir.glob("*.json"):
                session_id = filepath.stem
                session_ids.add(session_id)
        
        return sorted(session_ids)
    
    def resume_session(self, session_id: str) -> Optional[SessionContext]:
        """
        Resume a previous session.
        
        Args:
            session_id: ID of session to resume
            
        Returns:
            Resumed session or None if not found
        """
        session = self.get_session(session_id)
        if session:
            self.current_session = session
        return session
    
    def get_conversation_summary(
        self,
        session: Optional[SessionContext] = None
    ) -> str:
        """
        Generate summary of conversation.
        
        Args:
            session: Session to summarize (uses current if not specified)
            
        Returns:
            Conversation summary
        """
        session = session or self.current_session
        if not session or not session.messages:
            return "No conversation history."
        
        summary = f"Session: {session.session_id}\n"
        summary += f"Started: {session.created_at.strftime('%Y-%m-%d %H:%M')}\n"
        summary += f"Messages: {len(session.messages)}\n\n"
        
        # Summarize key topics
        topics = self._extract_topics(session)
        if topics:
            summary += "Topics discussed:\n"
            for topic in topics:
                summary += f"- {topic}\n"
        
        return summary
    
    def _extract_topics(self, session: SessionContext) -> List[str]:
        """Extract discussed topics from messages."""
        topics = []
        
        # Simple keyword-based topic extraction
        keywords = {
            'overfitting': 'Overfitting and regularization',
            'learning rate': 'Learning rate tuning',
            'architecture': 'Model architecture design',
            'transfer learning': 'Transfer learning',
            'augmentation': 'Data augmentation',
            'debugging': 'Debugging and troubleshooting'
        }
        
        for message in session.messages:
            content_lower = message.content.lower()
            for keyword, topic in keywords.items():
                if keyword in content_lower and topic not in topics:
                    topics.append(topic)
        
        return topics
    
    def build_context_for_llm(
        self,
        current_query: str,
        include_history: bool = True,
        max_messages: int = 5
    ) -> Dict[str, Any]:
        """
        Build context dictionary for LLM prompt.
        
        Args:
            current_query: Current user query
            include_history: Whether to include conversation history
            max_messages: Maximum number of historical messages to include
            
        Returns:
            Context dictionary for prompt building
        """
        context: Dict[str, Any] = {
            'current_query': current_query,
            'session_id': self.current_session.session_id if self.current_session else None
        }
        
        if self.current_session:
            # Add model state if available
            if self.current_session.model_state:
                context['model_state'] = self.current_session.model_state
            
            # Add conversation history if requested
            if include_history:
                recent_messages = self.current_session.get_recent_messages(max_messages)
                context['conversation_history'] = [
                    {'role': msg.role, 'content': msg.content}
                    for msg in recent_messages[:-1]  # Exclude current query
                ]
            
            # Add relevant context variables
            context['context_variables'] = self.current_session.context_variables
        
        return context
    
    def clear_session(self) -> None:
        """Clear current session."""
        if self.current_session:
            self.current_session = None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted successfully
        """
        # Remove from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Remove from disk
        if self.persistence_dir:
            try:
                filepath = self.persistence_dir / f"{session_id}.json"
                if filepath.exists():
                    filepath.unlink()
            except Exception:
                pass
        
        return True
    
    def get_context_summary(self) -> str:
        """Get summary of current context."""
        if not self.current_session:
            return "No active session."
        
        summary = f"Session: {self.current_session.session_id}\n"
        summary += f"Messages: {len(self.current_session.messages)}\n"
        
        if self.current_session.model_state:
            summary += "\nCurrent Model:\n"
            model_name = self.current_session.model_state.get('name', 'Unknown')
            num_layers = len(self.current_session.model_state.get('layers', []))
            summary += f"- Name: {model_name}\n"
            summary += f"- Layers: {num_layers}\n"
        
        if self.current_session.context_variables:
            summary += "\nContext Variables:\n"
            for key, value in self.current_session.context_variables.items():
                summary += f"- {key}: {value}\n"
        
        return summary
