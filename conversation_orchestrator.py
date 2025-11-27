"""
ConversationOrchestrator: Integrated conversation algorithm orchestrator

Integrates AddressDetector, TurnManager, and ConversationManager
to manage natural conversation flow in multi-user environments.
Uses score-based address detection for more accurate AI response decisions.
"""

import asyncio
import datetime
from typing import Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass
from enum import Enum

from context_manager import ConversationManager, AddressType, Message, ThreadState
from address_detector import AddressDetector, AddressAnalysis
from turn_manager import TurnManager, TurnDecision, ResponseDecision
from logger import setup_logger


class OrchestratorAction(Enum):
    """Orchestrator actions"""
    RESPOND = "respond"           # Execute response
    WAIT = "wait"                 # Wait (may respond later)
    SKIP = "skip"                 # Skip this message
    CANCEL_PENDING = "cancel"     # Cancel pending response


@dataclass
class OrchestratorResult:
    """Orchestrator result"""
    action: OrchestratorAction
    message: Optional[Message] = None
    wait_seconds: float = 0.0
    reason: str = ""
    address_analysis: Optional[AddressAnalysis] = None
    turn_decision: Optional[TurnDecision] = None
    score_explanation: str = ""  # Score breakdown explanation


class ConversationOrchestrator:
    """Conversation orchestrator with score-based address detection"""
    
    def __init__(self, 
                 ai_name: str = "LLM",
                 logger_file: Optional[str] = None,
                 log_level: int = 20):
        """
        Args:
            ai_name: AI name
            logger_file: Log file path
            log_level: Log level
        """
        self.ai_name = ai_name
        self.logger = setup_logger(__name__, logger_file, log_level)
        
        # Core components
        self.conversation_manager = ConversationManager(ai_name=ai_name)
        self.address_detector = AddressDetector(ai_name=ai_name)
        self.turn_manager = TurnManager(ai_name=ai_name)
        
        # State
        self._pending_task: Optional[asyncio.Task] = None
        self._cancelled = False
        
        # Callbacks
        self._on_respond: Optional[Callable[[Message, str], Awaitable[None]]] = None
        
        # Settings
        self.enable_proactive_chat = True  # Initiate conversation during silence
        self.silence_threshold = 60.0      # Silence threshold (seconds)
        self.enable_score_logging = True   # Log score breakdown
    
    def set_respond_callback(self, callback: Callable[[Message, str], Awaitable[None]]):
        """Set response callback"""
        self._on_respond = callback
    
    def add_participant(self, user_name: str):
        """Add participant"""
        self.conversation_manager.add_participant(user_name)
        self.address_detector.update_participants(self.conversation_manager.participants)
        self.logger.debug(f"Participant added: {user_name}")
    
    def remove_participant(self, user_name: str):
        """Remove participant"""
        self.conversation_manager.remove_participant(user_name)
        self.address_detector.update_participants(self.conversation_manager.participants)
        self.logger.debug(f"Participant removed: {user_name}")
    
    def _cancel_pending(self):
        """Cancel pending response"""
        self._cancelled = True
        self.turn_manager.cancel_pending()
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None
    
    def _is_cancelled(self) -> bool:
        """Check if cancelled"""
        return self._cancelled
    
    def _get_others_responded(self) -> int:
        """Get others' response count (for broadcast questions)"""
        return self.turn_manager.get_pending_others_responded()
    
    async def process_message(self, user_name: str, text: str) -> OrchestratorResult:
        """
        Process new message using score-based address detection.
        
        Full pipeline:
        1. Analyze speech target with scoring (AddressDetector)
        2. Record message (ConversationManager)
        3. Decide turn based on score (TurnManager)
        4. Wait or respond immediately
        
        Args:
            user_name: Speaker name
            text: Speech content
            
        Returns:
            OrchestratorResult: Processing result with score explanation
        """
        self.logger.debug(f"Processing message from {user_name}: {text[:50]}...")
        
        # Get participant count for scoring
        participant_count = self.conversation_manager.get_active_participants_count()
        
        # 1. Analyze speech target with scoring
        recent_messages = self.conversation_manager.get_recent_messages(5)
        address_analysis = self.address_detector.analyze(
            text=text,
            speaker=user_name,
            recent_messages=recent_messages,
            participant_count=participant_count
        )
        
        # Get score explanation
        score_explanation = ""
        if address_analysis.score_breakdown:
            score_explanation = self.address_detector.get_score_explanation(
                address_analysis.score_breakdown
            )
        
        # Log score breakdown
        if self.enable_score_logging:
            self.logger.info(
                f"[SCORE] {user_name}: \"{text[:30]}...\" → {score_explanation}"
            )
        
        self.logger.debug(
            f"Address analysis: type={address_analysis.address_type.value}, "
            f"score={address_analysis.confidence:.2f}, "
            f"targets={address_analysis.targets}"
        )
        
        # 2. Record message
        message = self.conversation_manager.add_message(
            user_name=user_name,
            text=text,
            address_type=address_analysis.address_type,
            targets=address_analysis.targets,
            is_question=address_analysis.is_question
        )
        
        # 3. Check pending response handling
        should_cancel = self.turn_manager.should_cancel_pending(message)
        
        if should_cancel:
            self._cancel_pending()
            self.logger.debug("Cancelled pending response due to new message")
        elif self.turn_manager.pending_response:
            if self.turn_manager.pending_response.decision.is_broadcast_response:
                others = self.turn_manager.pending_response.others_responded
                self.logger.debug(
                    f"Broadcast question: {others} others have responded, "
                    f"AI still waiting for turn"
                )
        
        # 4. Check if AI should respond based on score
        ai_included, priority = self.address_detector.should_ai_be_included(address_analysis)
        
        # Log decision
        threshold = self.address_detector.RESPONSE_THRESHOLD
        self.logger.debug(
            f"AI inclusion: {ai_included}, priority={priority:.2f}, "
            f"threshold={threshold}"
        )
        
        # If already waiting for broadcast question, ignore new analysis
        if (self.turn_manager.pending_response and 
            self.turn_manager.pending_response.decision.is_broadcast_response and
            not should_cancel):
            self.logger.debug("Maintaining existing broadcast wait, new message recorded only")
            return OrchestratorResult(
                action=OrchestratorAction.SKIP,
                message=message,
                reason="Already waiting for broadcast response turn",
                address_analysis=address_analysis,
                score_explanation=score_explanation
            )
        
        if not ai_included:
            self.turn_manager.record_human_response()
            return OrchestratorResult(
                action=OrchestratorAction.SKIP,
                message=message,
                reason=f"Score below threshold: {address_analysis.reason}",
                address_analysis=address_analysis,
                score_explanation=score_explanation
            )
        
        # 5. Decide turn
        turn_decision = self.turn_manager.decide_turn(
            message=message,
            ai_included=ai_included,
            priority=priority,
            conversation_manager=self.conversation_manager
        )
        
        self.logger.debug(
            f"Turn decision: {turn_decision.decision.value}, "
            f"wait={turn_decision.wait_seconds:.1f}s, "
            f"is_broadcast={turn_decision.is_broadcast_response}, "
            f"reason={turn_decision.reason}"
        )
        
        # 6. Action based on decision
        if turn_decision.decision == ResponseDecision.RESPOND_NOW:
            return OrchestratorResult(
                action=OrchestratorAction.RESPOND,
                message=message,
                wait_seconds=turn_decision.wait_seconds,
                reason=turn_decision.reason,
                address_analysis=address_analysis,
                turn_decision=turn_decision,
                score_explanation=score_explanation
            )
        
        elif turn_decision.decision == ResponseDecision.WAIT_FOR_OTHERS:
            self._cancelled = False
            self.turn_manager.set_pending(message, turn_decision)
            
            return OrchestratorResult(
                action=OrchestratorAction.WAIT,
                message=message,
                wait_seconds=turn_decision.wait_seconds,
                reason=turn_decision.reason,
                address_analysis=address_analysis,
                turn_decision=turn_decision,
                score_explanation=score_explanation
            )
        
        elif turn_decision.decision == ResponseDecision.OBSERVE_ONLY:
            self.turn_manager.record_human_response()
            return OrchestratorResult(
                action=OrchestratorAction.SKIP,
                message=message,
                reason=turn_decision.reason,
                address_analysis=address_analysis,
                turn_decision=turn_decision,
                score_explanation=score_explanation
            )
        
        else:  # DEFER
            return OrchestratorResult(
                action=OrchestratorAction.SKIP,
                message=message,
                reason=turn_decision.reason,
                address_analysis=address_analysis,
                turn_decision=turn_decision,
                score_explanation=score_explanation
            )
    
    async def wait_and_respond(self, result: OrchestratorResult) -> bool:
        """
        Wait then execute response
        
        Args:
            result: process_message result
            
        Returns:
            True: Responded, False: Cancelled
        """
        if result.action == OrchestratorAction.RESPOND:
            await asyncio.sleep(result.wait_seconds)
            if not self._is_cancelled():
                self.turn_manager.record_ai_response()
                self.turn_manager.clear_pending()
                return True
            return False
        
        elif result.action == OrchestratorAction.WAIT:
            should_respond = await self.turn_manager.wait_for_turn(
                decision=result.turn_decision,
                check_cancelled=self._is_cancelled,
                get_others_responded=self._get_others_responded
            )
            
            if should_respond:
                self.turn_manager.record_ai_response()
                self.turn_manager.clear_pending()
                return True
            
            self.logger.debug("Response cancelled during wait")
            return False
        
        return False
    
    def record_ai_response(self, response_text: str):
        """Record AI response"""
        self.conversation_manager.add_message(
            user_name=self.ai_name,
            text=response_text,
            address_type=AddressType.DIRECT,
            targets=set()
        )
    
    def get_system_context(self) -> str:
        """Get system context for LLM"""
        return self.conversation_manager.get_system_context()
    
    def get_conversation_summary(self) -> dict:
        """Get conversation state summary"""
        pending_info = None
        if self.turn_manager.pending_response:
            pending_info = {
                "is_broadcast": self.turn_manager.pending_response.decision.is_broadcast_response,
                "others_responded": self.turn_manager.pending_response.others_responded
            }
        
        return {
            "participants": list(self.conversation_manager.participants),
            "participant_count": self.conversation_manager.get_active_participants_count(),
            "situation": self.conversation_manager.analyze_situation(),
            "flow": self.conversation_manager.get_conversation_flow(),
            "message_count": len(self.conversation_manager.history),
            "has_pending": self.turn_manager.pending_response is not None,
            "pending_info": pending_info,
            "consecutive_responses": self.turn_manager.consecutive_responses,
            "response_threshold": self.address_detector.RESPONSE_THRESHOLD,
            "high_priority_threshold": self.address_detector.HIGH_PRIORITY_THRESHOLD
        }
    
    async def check_silence_break(self) -> Optional[str]:
        """
        Detect silence and suggest conversation start
        
        Returns:
            Conversation start suggestion message or None
        """
        if not self.enable_proactive_chat:
            return None
        
        if self.turn_manager.should_break_silence(
            self.conversation_manager, 
            self.silence_threshold
        ):
            participants = self.conversation_manager.participants
            if participants:
                return f"Silence has lasted over {self.silence_threshold} seconds. Can initiate conversation."
        
        return None
    
    def cleanup_expired_threads(self):
        """Clean up expired threads"""
        self.conversation_manager.cleanup_expired_threads()
    
    def set_response_threshold(self, threshold: float):
        """
        Adjust response threshold dynamically.
        
        Lower threshold = AI responds more often
        Higher threshold = AI responds less often
        
        Args:
            threshold: New threshold (0.0 ~ 1.0)
        """
        self.address_detector.RESPONSE_THRESHOLD = max(0.0, min(1.0, threshold))
        self.logger.info(f"Response threshold set to {threshold:.2f}")


# Factory function for convenient use
def create_orchestrator(ai_name: str = "LLM", 
                        logger_file: Optional[str] = None,
                        log_level: int = 20) -> ConversationOrchestrator:
    """Create orchestrator factory"""
    return ConversationOrchestrator(
        ai_name=ai_name,
        logger_file=logger_file,
        log_level=log_level
    )
