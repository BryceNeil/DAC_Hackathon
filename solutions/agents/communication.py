"""
Agent Communication Protocol
============================

This module defines the communication protocol for inter-agent messaging
in the ASU Tapeout LangGraph system.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import defaultdict


class Priority(Enum):
    """Priority levels for agent messages"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    receiver: str
    task: str
    context: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default="")
    
    def __post_init__(self):
        """Generate unique message ID if not provided"""
        if not self.message_id:
            self.message_id = f"{self.sender}_{self.receiver}_{self.timestamp.timestamp()}"


class AgentCoordinator:
    """Coordinates communication between agents in the LangGraph system"""
    
    def __init__(self):
        """Initialize the coordinator"""
        self.message_queue = asyncio.Queue()
        self.pending_tasks = defaultdict(list)
        self.completed_tasks = {}
        self.agent_status = {}
        
    async def route_task(self, message: AgentMessage) -> Dict[str, Any]:
        """Route tasks to appropriate agents
        
        Args:
            message: AgentMessage containing task details
            
        Returns:
            Task routing information
        """
        # Add to message queue based on priority
        await self.message_queue.put((message.priority.value, message))
        
        # Track pending task
        self.pending_tasks[message.receiver].append(message.message_id)
        
        # Update agent status
        self.agent_status[message.receiver] = "busy"
        
        return {
            "routed": True,
            "message_id": message.message_id,
            "queue_size": self.message_queue.qsize()
        }
    
    async def collect_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Aggregate results from multiple agents
        
        Args:
            task_id: ID of the task to collect results for
            
        Returns:
            Aggregated results or None if not complete
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check if task is still pending
        for agent, tasks in self.pending_tasks.items():
            if task_id in tasks:
                return None  # Task still pending
        
        return None
    
    def get_agent_status(self, agent_name: str) -> str:
        """Get current status of an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent status (idle/busy/error)
        """
        return self.agent_status.get(agent_name, "idle")
    
    def mark_task_complete(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as complete with results
        
        Args:
            task_id: ID of the completed task
            result: Task execution results
        """
        self.completed_tasks[task_id] = result
        
        # Remove from pending tasks
        for agent, tasks in self.pending_tasks.items():
            if task_id in tasks:
                tasks.remove(task_id)
                if not tasks:  # No more tasks for this agent
                    self.agent_status[agent] = "idle"
                break
    
    async def broadcast_message(self, sender: str, task: str, context: Dict[str, Any], 
                               agents: List[str], priority: Priority = Priority.NORMAL):
        """Broadcast a message to multiple agents
        
        Args:
            sender: Sending agent name
            task: Task description
            context: Task context
            agents: List of receiving agents
            priority: Message priority
        """
        tasks = []
        for receiver in agents:
            message = AgentMessage(
                sender=sender,
                receiver=receiver,
                task=task,
                context=context,
                priority=priority
            )
            tasks.append(self.route_task(message))
        
        await asyncio.gather(*tasks)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the message queue
        
        Returns:
            Queue statistics
        """
        return {
            "queue_size": self.message_queue.qsize(),
            "pending_tasks": {agent: len(tasks) for agent, tasks in self.pending_tasks.items()},
            "completed_tasks": len(self.completed_tasks),
            "agent_status": self.agent_status.copy()
        } 