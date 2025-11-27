"""
TurnManager: Turn and timing management

Determines when AI should join the conversation and how long to wait.
"""

import datetime
import asyncio
from typing import Optional, Set, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from context_manager import ConversationManager, AddressType, ThreadState, Message


class ResponseDecision(Enum):
    """Response decision types"""
    RESPOND_NOW = "respond_now"           # Respond immediately
    WAIT_FOR_OTHERS = "wait_for_others"   # Wait for others then respond
    OBSERVE_ONLY = "observe_only"         # Observe only (no response)
    DEFER = "defer"                       # Defer judgment


@dataclass
class TurnDecision:
    """Turn decision result"""
    decision: ResponseDecision
    wait_seconds: float = 0.0
    priority: float = 0.0  # 0.0 ~ 1.0
    reason: str = ""
    can_interrupt: bool = False  # Can interrupt during wait
    is_broadcast_response: bool = False  # Response to broadcast question


@dataclass
class PendingResponse:
    """Pending response"""
    message: Message
    decision: TurnDecision
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    cancelled: bool = False
    others_responded: int = 0  # Number of others who responded
    original_wait_seconds: float = 0.0  # Original wait time


class TurnManager:
    """Turn manager"""
    
    def __init__(self, 
                 ai_name: str = "LLM",
                 base_wait_time: float = 2.0,
                 max_wait_time: float = 8.0,
                 min_wait_time: float = 0.5,
                 broadcast_extra_wait: float = 1.5):
        """
        Args:
            ai_name: AI name
            base_wait_time: Base wait time (seconds)
            max_wait_time: Maximum wait time (seconds)
            min_wait_time: Minimum wait time (seconds)
            broadcast_extra_wait: Extra wait time after others respond to broadcast
        """
        self.ai_name = ai_name
        self.base_wait_time = base_wait_time
        self.max_wait_time = max_wait_time
        self.min_wait_time = min_wait_time
        self.broadcast_extra_wait = broadcast_extra_wait
        
        # Pending response
        self.pending_response: Optional[PendingResponse] = None
        
        # Last AI response time
        self.last_ai_response_time: Optional[datetime.datetime] = None
        
        # Consecutive response counter (to avoid talking too much)
        self.consecutive_responses = 0
        self.max_consecutive = 3
    
    def calculate_wait_time(self, 
                            priority: float,
                            participant_count: int,
                            is_question: bool,
                            address_type: AddressType) -> float:
        """
        Calculate wait time
        
        Higher priority and fewer participants = shorter wait time
        """
        # Base wait time
        wait = self.base_wait_time
        
        # Adjust by priority (higher = faster)
        wait *= (1.0 - priority * 0.7)
        
        # Adjust by participant count (more = longer wait)
        if participant_count > 2:
            wait *= 1.0 + (participant_count - 2) * 0.3
        
        # AI direct call = minimum wait
        if address_type == AddressType.AI_DIRECT:
            wait = self.min_wait_time
        
        # Broadcast question = wait longer
        elif address_type == AddressType.BROADCAST and is_question:
            wait *= 1.5
        
        # Consecutive responses = wait longer
        if self.consecutive_responses > 0:
            wait *= 1.0 + self.consecutive_responses * 0.5
        
        # Clamp to range
        return max(self.min_wait_time, min(self.max_wait_time, wait))
    
    def decide_turn(self,
                    message: Message,
                    ai_included: bool,
                    priority: float,
                    conversation_manager: ConversationManager) -> TurnDecision:
        """
        Decide turn
        
        Args:
            message: Analyzed message
            ai_included: Whether AI is included in targets
            priority: AI response priority (0.0 ~ 1.0)
            conversation_manager: Conversation manager
        """
        # If AI is not the target, observe only
        if not ai_included:
            return TurnDecision(
                decision=ResponseDecision.OBSERVE_ONLY,
                reason="AI is not the target of this message"
            )
        
        # If self-talk, observe only
        if message.address_type == AddressType.SELF_TALK:
            return TurnDecision(
                decision=ResponseDecision.OBSERVE_ONLY,
                reason="Self-talk detected"
            )
        
        participant_count = conversation_manager.get_active_participants_count()
        
        # AI direct call = respond immediately
        if message.address_type == AddressType.AI_DIRECT:
            return TurnDecision(
                decision=ResponseDecision.RESPOND_NOW,
                wait_seconds=self.min_wait_time,
                priority=1.0,
                reason="Direct call to AI"
            )
        
        # UNKNOWN = might be calling someone else, observe only
        # (could be a name not in participant list)
        if message.address_type == AddressType.UNKNOWN:
            return TurnDecision(
                decision=ResponseDecision.OBSERVE_ONLY,
                reason="Target unclear - might be calling someone else"
            )
        
        # 1:1 conversation = respond immediately (only when AI is clearly the target)
        if participant_count <= 1:
            return TurnDecision(
                decision=ResponseDecision.RESPOND_NOW,
                wait_seconds=self.min_wait_time,
                priority=0.9,
                reason="One-on-one conversation"
            )
        
        # 2 people = quick response (small group)
        if participant_count == 2:
            return TurnDecision(
                decision=ResponseDecision.WAIT_FOR_OTHERS,
                wait_seconds=self.min_wait_time * 2,  # About 1 second
                priority=0.7,
                reason="Small group (2 people), quick response"
            )
        
        # Check consecutive response limit
        if self.consecutive_responses >= self.max_consecutive:
            return TurnDecision(
                decision=ResponseDecision.DEFER,
                wait_seconds=self.max_wait_time,
                priority=0.1,
                reason="Too many consecutive responses, backing off"
            )
        
        # Calculate wait time
        wait_time = self.calculate_wait_time(
            priority=priority,
            participant_count=participant_count,
            is_question=message.is_question,
            address_type=message.address_type
        )
        
        # Broadcast message (question)
        if message.address_type == AddressType.BROADCAST:
            return TurnDecision(
                decision=ResponseDecision.WAIT_FOR_OTHERS,
                wait_seconds=wait_time,
                priority=priority,
                reason="Broadcast message, waiting for others first then respond",
                can_interrupt=False,
                is_broadcast_response=True  # Mark as broadcast question
            )
        
        # Continuation conversation where AI is target
        if message.address_type == AddressType.CONTINUATION:
            return TurnDecision(
                decision=ResponseDecision.RESPOND_NOW,
                wait_seconds=self.min_wait_time,
                priority=0.8,
                reason="Continuing conversation with AI"
            )
        
        # Default: wait briefly then respond
        return TurnDecision(
            decision=ResponseDecision.WAIT_FOR_OTHERS,
            wait_seconds=wait_time,
            priority=priority,
            reason="Default: wait briefly before responding",
            can_interrupt=True
        )
    
    def should_cancel_pending(self, new_message: Message) -> bool:
        """
        Determine if pending response should be cancelled due to new message
        
        For BROADCAST questions, don't cancel even if others respond,
        maintain AI's turn.
        """
        if not self.pending_response:
            return False
        
        pending = self.pending_response
        
        # If new message is AI direct call, cancel previous and respond to new
        if new_message.address_type == AddressType.AI_DIRECT:
            return True
        
        # If pending message's speaker speaks again, cancel (topic change)
        if new_message.user == pending.message.user:
            return True
        
        # If waiting for broadcast question
        if pending.decision.is_broadcast_response:
            # Don't cancel even if others respond!
            # Just increment counter and extend wait time
            if new_message.user != self.ai_name:
                pending.others_responded += 1
                # Extra wait for each person who responds (being polite)
                return False  # Don't cancel
        
        # General case: cancel if someone else responded first
        if (pending.decision.decision == ResponseDecision.WAIT_FOR_OTHERS and
            not pending.decision.is_broadcast_response and
            new_message.user != self.ai_name and
            pending.message.is_question):
            return True
        
        return False
    
    def extend_wait_for_broadcast(self) -> float:
        """
        Return extra wait time when others respond to broadcast question
        """
        if not self.pending_response:
            return 0.0
        
        if self.pending_response.decision.is_broadcast_response:
            # Extra wait for each person who responds
            return self.broadcast_extra_wait
        
        return 0.0
    
    def set_pending(self, message: Message, decision: TurnDecision):
        """Set pending response"""
        self.pending_response = PendingResponse(
            message=message,
            decision=decision,
            original_wait_seconds=decision.wait_seconds
        )
    
    def cancel_pending(self):
        """Cancel pending response"""
        if self.pending_response:
            self.pending_response.cancelled = True
            self.pending_response = None
    
    def clear_pending(self):
        """Clear pending after wait complete"""
        self.pending_response = None
    
    def record_ai_response(self):
        """Record AI response"""
        self.last_ai_response_time = datetime.datetime.now()
        self.consecutive_responses += 1
    
    def record_human_response(self):
        """Record human response (reset consecutive counter)"""
        self.consecutive_responses = 0
    
    async def wait_for_turn(self, 
                            decision: TurnDecision,
                            check_cancelled: Callable[[], bool],
                            get_others_responded: Optional[Callable[[], int]] = None) -> bool:
        """
        Wait until it's AI's turn
        
        Args:
            decision: Turn decision
            check_cancelled: Callback to check if cancelled
            get_others_responded: Callback to check others' response count (for broadcast)
            
        Returns:
            True: Should respond, False: Cancelled
        """
        if decision.decision == ResponseDecision.RESPOND_NOW:
            await asyncio.sleep(decision.wait_seconds)
            return not check_cancelled()
        
        if decision.decision == ResponseDecision.OBSERVE_ONLY:
            return False
        
        if decision.decision == ResponseDecision.DEFER:
            return False
        
        # WAIT_FOR_OTHERS: Check cancellation at intervals while waiting
        elapsed = 0.0
        check_interval = 0.2  # Check every 200ms
        target_wait = decision.wait_seconds
        last_others_count = 0
        
        while elapsed < target_wait:
            await asyncio.sleep(check_interval)
            elapsed += check_interval
            
            if check_cancelled():
                return False
            
            # For broadcast questions, extend wait if others respond
            if decision.is_broadcast_response and get_others_responded:
                current_others = get_others_responded()
                if current_others > last_others_count:
                    # New response = extra wait
                    additional_wait = (current_others - last_others_count) * self.broadcast_extra_wait
                    target_wait = min(target_wait + additional_wait, self.max_wait_time)
                    last_others_count = current_others
        
        return not check_cancelled()
    
    def get_silence_duration(self, conversation_manager: ConversationManager) -> float:
        """Get silence duration since last message (seconds)"""
        last_msg = conversation_manager.get_last_message()
        if not last_msg:
            return float('inf')
        return (datetime.datetime.now() - last_msg.timestamp).total_seconds()
    
    def should_break_silence(self, 
                             conversation_manager: ConversationManager,
                             silence_threshold: float = 30.0) -> bool:
        """
        Determine if AI should break long silence
        """
        silence = self.get_silence_duration(conversation_manager)
        
        if silence < silence_threshold:
            return False
        
        # If participants exist and silence is long, start conversation
        if conversation_manager.get_active_participants_count() > 0:
            return True
        
        return False
    
    def get_pending_others_responded(self) -> int:
        """Get count of others who responded to pending message"""
        if self.pending_response:
            return self.pending_response.others_responded
        return 0
