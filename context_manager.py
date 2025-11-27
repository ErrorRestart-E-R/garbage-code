from collections import deque
import datetime
from enum import Enum
from typing import Optional, Set, List, Dict
from dataclasses import dataclass, field


class AddressType(Enum):
    """Speech target types"""
    BROADCAST = "broadcast"        # Everyone ("다들", "여러분")
    DIRECT = "direct"              # Specific person ("@철수", "철수야")
    AI_DIRECT = "ai_direct"        # Direct AI call ("LLM아", "@LLM")
    CONTINUATION = "continuation"  # Continuing previous conversation
    SELF_TALK = "self_talk"        # Self-talk/monologue
    UNKNOWN = "unknown"            # Cannot classify


class ThreadState(Enum):
    """Conversation thread states"""
    IDLE = "idle"                      # Idle state
    QUESTION_PENDING = "question_pending"  # Waiting for answer after question
    DISCUSSION = "discussion"          # Discussion in progress
    AI_RESPONDED = "ai_responded"      # AI just responded


@dataclass
class Message:
    """Message data class"""
    user: str
    text: str
    timestamp: datetime.datetime
    address_type: AddressType = AddressType.UNKNOWN
    targets: Set[str] = field(default_factory=set)
    is_question: bool = False
    responded_by: Set[str] = field(default_factory=set)


@dataclass
class ConversationThread:
    """Conversation thread (conversation flow for specific topic/question)"""
    initiator: str
    started_at: datetime.datetime
    state: ThreadState = ThreadState.IDLE
    expected_responders: Set[str] = field(default_factory=set)
    actual_responders: Set[str] = field(default_factory=set)
    topic_message: Optional[Message] = None
    last_activity: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def is_expired(self, timeout_seconds: float = 30.0) -> bool:
        """Check if thread has timed out"""
        elapsed = (datetime.datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds


class ConversationManager:
    def __init__(self, max_history: int = 20, ai_name: str = "LLM"):
        self.participants: Set[str] = set()
        self.history: deque = deque(maxlen=max_history)
        self.last_speaker: Optional[str] = None
        self.last_interaction_time: datetime.datetime = datetime.datetime.now()
        self.ai_name: str = ai_name
        
        # Thread management
        self.active_threads: List[ConversationThread] = []
        self.current_thread: Optional[ConversationThread] = None
        
        # Conversation pattern statistics
        self.speaker_stats: Dict[str, Dict] = {}  # Per-speaker stats
        self.pair_stats: Dict[str, int] = {}      # Speaker pair conversation frequency

    def add_participant(self, user_name: str):
        self.participants.add(user_name)
        if user_name not in self.speaker_stats:
            self.speaker_stats[user_name] = {
                "message_count": 0,
                "questions_asked": 0,
                "ai_interactions": 0,
                "last_spoke": None
            }

    def remove_participant(self, user_name: str):
        if user_name in self.participants:
            self.participants.remove(user_name)

    def add_message(self, user_name: str, text: str, 
                    address_type: AddressType = AddressType.UNKNOWN,
                    targets: Optional[Set[str]] = None,
                    is_question: bool = False) -> Message:
        """Add message and update statistics"""
        timestamp = datetime.datetime.now()
        
        msg = Message(
            user=user_name,
            text=text,
            timestamp=timestamp,
            address_type=address_type,
            targets=targets or set(),
            is_question=is_question
        )
        
        self.history.append(msg)
        
        # Update speaker statistics
        if user_name in self.speaker_stats:
            self.speaker_stats[user_name]["message_count"] += 1
            self.speaker_stats[user_name]["last_spoke"] = timestamp
            if is_question:
                self.speaker_stats[user_name]["questions_asked"] += 1
            if self.ai_name in (targets or set()) or address_type == AddressType.AI_DIRECT:
                self.speaker_stats[user_name]["ai_interactions"] += 1
        
        # Update speaker pair statistics
        if self.last_speaker and self.last_speaker != user_name:
            pair_key = tuple(sorted([self.last_speaker, user_name]))
            self.pair_stats[str(pair_key)] = self.pair_stats.get(str(pair_key), 0) + 1
        
        self.last_speaker = user_name
        self.last_interaction_time = timestamp
        
        return msg

    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get last N messages"""
        return list(self.history)[-count:]

    def get_last_message(self) -> Optional[Message]:
        """Get last message"""
        return self.history[-1] if self.history else None

    def get_messages_by_user(self, user_name: str, count: int = 3) -> List[Message]:
        """Get recent messages by specific user"""
        user_msgs = [m for m in self.history if m.user == user_name]
        return user_msgs[-count:]

    def get_active_participants_count(self) -> int:
        return len(self.participants)

    def get_recent_active_speakers(self, seconds: int = 30) -> Set[str]:
        """Get participants who spoke in last N seconds"""
        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
        return {m.user for m in self.history if m.timestamp > cutoff}

    def analyze_situation(self) -> str:
        count = self.get_active_participants_count()
        if count <= 1:
            return "1:1"
        elif count <= 4:
            return "Small Group"
        else:
            return "Large Crowd"

    def get_conversation_flow(self) -> str:
        """Analyze recent conversation flow"""
        if len(self.history) < 2:
            return "minimal"
        
        recent = list(self.history)[-5:]
        speakers = [m.user for m in recent]
        unique_speakers = len(set(speakers))
        
        if unique_speakers == 1:
            return "monologue"
        elif unique_speakers == 2:
            return "dialogue"
        else:
            return "group_discussion"

    def is_ai_recently_responded(self, seconds: int = 5) -> bool:
        """Check if AI responded recently"""
        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=seconds)
        for msg in reversed(list(self.history)):
            if msg.timestamp < cutoff:
                break
            if msg.user == self.ai_name:
                return True
        return False

    def get_unanswered_questions(self) -> List[Message]:
        """Get list of unanswered questions"""
        return [m for m in self.history if m.is_question and not m.responded_by]

    def mark_question_answered(self, question_msg: Message, responder: str):
        """Record answer to question"""
        question_msg.responded_by.add(responder)

    def get_system_context(self) -> str:
        """Generate system context for LLM"""
        situation = self.analyze_situation()
        participant_count = self.get_active_participants_count()
        participants_list = ", ".join(self.participants) if self.participants else "No one"
        flow = self.get_conversation_flow()
        
        context = f"Current Situation: {situation} conversation.\n"
        context += f"Total Participants: {participant_count} people in voice channel.\n"
        context += f"Participants: {participants_list}.\n"
        context += f"Conversation Flow: {flow}.\n"
        
        # Current thread state
        if self.current_thread and not self.current_thread.is_expired():
            context += f"\nActive Thread: Started by {self.current_thread.initiator}, "
            context += f"State: {self.current_thread.state.value}\n"
            if self.current_thread.expected_responders:
                context += f"Expected responders: {', '.join(self.current_thread.expected_responders)}\n"
        
        if self.history:
            context += "\nRecent Conversation History:\n"
            for msg in list(self.history)[-10:]:
                time_str = msg.timestamp.strftime("%H:%M:%S")
                addr_info = ""
                if msg.address_type != AddressType.UNKNOWN:
                    addr_info = f" [{msg.address_type.value}]"
                    if msg.targets:
                        addr_info += f" -> {', '.join(msg.targets)}"
                context += f"[{time_str}] {msg.user}{addr_info}: {msg.text}\n"
            
            last_msg = self.history[-1]
            time_diff = (datetime.datetime.now() - last_msg.timestamp).total_seconds()
            if time_diff > 60:
                context += "\nIt has been a while since the last message.\n"
        
        return context

    # Thread management methods
    def start_thread(self, initiator: str, message: Message, 
                     expected_responders: Optional[Set[str]] = None) -> ConversationThread:
        """Start new conversation thread"""
        thread = ConversationThread(
            initiator=initiator,
            started_at=datetime.datetime.now(),
            state=ThreadState.QUESTION_PENDING if message.is_question else ThreadState.DISCUSSION,
            expected_responders=expected_responders or set(),
            topic_message=message
        )
        self.active_threads.append(thread)
        self.current_thread = thread
        return thread

    def update_thread(self, responder: str):
        """Add responder to thread"""
        if self.current_thread:
            self.current_thread.actual_responders.add(responder)
            self.current_thread.last_activity = datetime.datetime.now()
            
            # If all expected responders have responded, close thread
            if (self.current_thread.expected_responders and 
                self.current_thread.expected_responders <= self.current_thread.actual_responders):
                self.current_thread.state = ThreadState.IDLE

    def cleanup_expired_threads(self, timeout_seconds: float = 30.0):
        """Clean up expired threads"""
        self.active_threads = [t for t in self.active_threads if not t.is_expired(timeout_seconds)]
        if self.current_thread and self.current_thread.is_expired(timeout_seconds):
            self.current_thread = None
