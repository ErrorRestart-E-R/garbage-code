"""
ConversationHistory: Conversation history management

Manages up to N conversations in JSON format.
This history is used for LLM Judge and response generation.
"""

import datetime
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple


@dataclass
class ConversationEntry:
    """Conversation entry"""
    speaker: str
    text: str
    timestamp: str
    raw_timestamp: float = 0.0
    is_ai: bool = False


class ConversationHistory:
    """
    Conversation history management class
    
    - Maintains up to max_size conversations
    - Can be passed to LLM in JSON format
    - Tracks participants
    """
    
    def __init__(self, max_size: int = 20, ai_name: str = "LLM"):
        """
        Args:
            max_size: Maximum number of conversations to keep
            ai_name: AI name (used to distinguish AI utterances)
        """
        self.history: deque = deque(maxlen=max_size)
        self.ai_name = ai_name
        self.participants: Set[str] = set()
    
    def add(self, speaker: str, text: str) -> ConversationEntry:
        """
        Add conversation entry
        
        Args:
            speaker: Speaker name
            text: Utterance content
            
        Returns:
            Added ConversationEntry
        """
        entry = ConversationEntry(
            speaker=speaker,
            text=text,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S"),
            raw_timestamp=time.time(),
            is_ai=(speaker == self.ai_name)
        )
        self.history.append(entry)
        
        # Track participants (excluding AI)
        if speaker != self.ai_name:
            self.participants.add(speaker)
        
        return entry
    
    def add_ai_response(self, text: str) -> ConversationEntry:
        """
        Add AI response (convenience method)
        
        Args:
            text: AI response content
        """
        return self.add(self.ai_name, text)
    
    def get_history_and_current(self) -> Tuple[str, Optional[str], Optional[str], float]:
        """
        Get conversation history (excluding last) and current message separately.
        This prevents LLM from confusing past messages with the current one.
        
        Returns:
            (history_text, current_speaker, current_message, current_timestamp)
            - history_text: Past conversations in simple text format
            - current_speaker: Last speaker name (or None if empty)
            - current_message: Last message text (or None if empty)
            - current_timestamp: Raw timestamp of the current message
        """
        if not self.history:
            return "(no previous conversation)", None, None, 0.0
        
        # Separate history (all but last) from current (last)
        history_list = list(self.history)
        
        if len(history_list) == 1:
            # Only one message - no history, just current
            current = history_list[0]
            return "(no previous conversation)", current.speaker, current.text, current.raw_timestamp
        
        # History = all except last (simple text format)
        lines = []
        for entry in history_list[:-1]:
            lines.append(f"[{entry.timestamp}] {entry.speaker}: {entry.text}")
        
        # Current = last entry
        current = history_list[-1]
        
        return (
            "\n".join(lines),
            current.speaker,
            current.text,
            current.raw_timestamp
        )
    
    def to_json(self) -> str:
        """
        Convert conversation history to JSON string
        
        Returns:
            Conversation history in JSON format
        """
        entries = []
        for entry in self.history:
            entries.append({
                "speaker": entry.speaker,
                "text": entry.text,
                "time": entry.timestamp
            })
        return json.dumps(entries, ensure_ascii=False, indent=2)
    
    def to_list(self) -> List[dict]:
        """
        Return conversation history as list
        
        Returns:
            Conversation history list
        """
        return [
            {
                "speaker": entry.speaker,
                "text": entry.text,
                "time": entry.timestamp
            }
            for entry in self.history
        ]
    
    def get_last_entry(self) -> Optional[ConversationEntry]:
        """Return last conversation entry"""
        return self.history[-1] if self.history else None
    
    def get_last_speaker(self) -> Optional[str]:
        """Return last speaker"""
        last = self.get_last_entry()
        return last.speaker if last else None
    
    def get_last_ai_spoke(self) -> bool:
        """Check if last speaker was AI"""
        last = self.get_last_entry()
        return last.is_ai if last else False
    
    def get_participant_count(self) -> int:
        """Return participant count (excluding AI)"""
        return len(self.participants)
    
    def add_participant(self, name: str):
        """Add participant"""
        if name != self.ai_name:
            self.participants.add(name)
    
    def remove_participant(self, name: str):
        """Remove participant"""
        self.participants.discard(name)
    
    def clear_participants(self):
        """Clear participant list"""
        self.participants.clear()
    
    def get_context_summary(self) -> str:
        """
        Get context summary for LLM
        
        Returns:
            Context summary string
        """
        participant_list = ", ".join(self.participants) if self.participants else "None"
        count = self.get_participant_count()
        
        summary = f"Current participants: {participant_list} ({count} people)\n"
        summary += f"Conversation history ({len(self.history)} entries):\n"
        summary += self.to_json()
        
        return summary
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
    
    def __len__(self) -> int:
        return len(self.history)
    
    def __bool__(self) -> bool:
        return len(self.history) > 0
