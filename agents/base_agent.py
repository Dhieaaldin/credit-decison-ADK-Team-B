"""
Base Agent Class for Multi-Agent Infrastructure

All agents inherit from this base class to ensure consistent interface
and enable agent communication patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response structure from agent processing."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Each agent:
    - Has a unique name and role
    - Processes messages and returns responses
    - Can communicate with other agents
    - Maintains its own state
    """
    
    def __init__(self, agent_name: str, agent_role: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Unique name for this agent instance
            agent_role: Role/type of this agent
        """
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.state: Dict[str, Any] = {}
        self.message_history: list[AgentMessage] = []
        
    @abstractmethod
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process a message and return a response.
        
        Args:
            message: Input message to process
            
        Returns:
            AgentResponse with processing results
        """
        pass
    
    def send_message(
        self,
        receiver: str,
        message_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Create and return a message to send to another agent.
        
        Args:
            receiver: Name of receiving agent
            message_type: Type of message
            payload: Message data
            metadata: Optional metadata
            
        Returns:
            AgentMessage ready to send
        """
        return AgentMessage(
            sender=self.agent_name,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            metadata=metadata or {}
        )
    
    def log_message(self, message: AgentMessage):
        """Log a message for audit purposes."""
        self.message_history.append(message)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return self.state.copy()
    
    def update_state(self, updates: Dict[str, Any]):
        """Update agent state."""
        self.state.update(updates)
