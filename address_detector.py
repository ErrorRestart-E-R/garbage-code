"""
AddressDetector: Rule-based speech target classifier

Determines speech targets using regex, keywords, and conversation pattern analysis
without LLM calls.
"""

import re
from typing import Set, Tuple, Optional, List
from dataclasses import dataclass
from context_manager import AddressType, Message


@dataclass
class AddressAnalysis:
    """Speech target analysis result"""
    address_type: AddressType
    targets: Set[str]
    confidence: float  # 0.0 ~ 1.0
    is_question: bool
    reason: str


class AddressDetector:
    """Rule-based speech target detector"""
    
    # Broadcast keywords (Korean/English)
    BROADCAST_KEYWORDS = [
        "다들", "여러분", "모두", "전부", "다같이", "우리", "얘들아", "애들아",
        "everyone", "everybody", "all", "guys", "folks", "y'all"
    ]
    
    # Question patterns
    QUESTION_PATTERNS = [
        r"[?？]$",                          # Ends with question mark
        r"^(누가|뭐|뭘|어디|언제|왜|어떻게|몇|얼마)",  # Starts with interrogative
        r"(할래|할까|하자|갈래|갈까|먹을래|볼래|해볼래)\s*[?？]?$",  # Suggestion/question ending
        r"(인가요|인가|일까|일까요|인지|는지|나요|니|냐)\s*[?？]?$",  # Interrogative ending
        r"(맞아|맞지|그래|그렇지|아니야|아닌가)\s*[?？]?$",  # Confirmation question
        r"^(혹시|그래서|근데|그런데)",       # Question starters
    ]
    
    # Self-talk/monologue patterns
    SELF_TALK_PATTERNS = [
        r"^(아|어|음|흠|에|오|와|헐|ㅋ|ㅎ|ㄷ|ㅇ)+$",  # Interjections only
        r"^\.{2,}$",                        # Dots only
        r"^(뭐지|뭐야|뭐냐|왜지|왜이래|어라|이상하네|신기하네)",  # Self-talk expressions
    ]
    
    # Short responses (conversation break signals)
    SHORT_RESPONSES = [
        "ok", "okay", "yes", "no", "yeah", "yep", "nope",
        "응", "어", "아", "네", "예", "아니", "그래", "알겠어"
    ]
    
    # AI call patterns (AI name added dynamically)
    AI_CALL_PATTERNS = [
        r"@{ai_name}",
        r"{ai_name}[야]",
        r"{ai_name}[님]",
        r"^{ai_name}[,\s]",
        r"[,\s]{ai_name}$",
    ]
    
    # Generic name call pattern (for detecting non-participant names)
    # Calls ending with "~아", "~야", "~님", "~씨" etc.
    GENERIC_NAME_CALL_PATTERN = r"^([가-힣a-zA-Z]+)[아야님씨]\s"
    
    def __init__(self, ai_name: str = "LLM", participants: Optional[Set[str]] = None):
        self.ai_name = ai_name
        self.participants = participants or set()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.question_regex = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self.self_talk_regex = [re.compile(p, re.IGNORECASE) for p in self.SELF_TALK_PATTERNS]
        
        # Compile AI call patterns
        ai_patterns = [p.format(ai_name=re.escape(self.ai_name)) for p in self.AI_CALL_PATTERNS]
        self.ai_call_regex = [re.compile(p, re.IGNORECASE) for p in ai_patterns]
        
        # Compile generic name call pattern
        self.generic_name_call_regex = re.compile(self.GENERIC_NAME_CALL_PATTERN)
    
    def update_participants(self, participants: Set[str]):
        """Update participant list"""
        self.participants = participants
    
    def update_ai_name(self, ai_name: str):
        """Update AI name and recompile patterns"""
        self.ai_name = ai_name
        self._compile_patterns()
    
    def _is_question(self, text: str) -> bool:
        """Determine if text is a question"""
        return any(regex.search(text) for regex in self.question_regex)
    
    def _is_self_talk(self, text: str) -> bool:
        """Determine if text is self-talk"""
        # Too short text
        if len(text.strip()) <= 3:
            return True
        # Self-talk pattern matching
        return any(regex.match(text.strip()) for regex in self.self_talk_regex)
    
    def _is_short_response(self, text: str) -> bool:
        """Determine if text is a short response"""
        normalized = text.strip().lower()
        return normalized in [s.lower() for s in self.SHORT_RESPONSES] or len(normalized) <= 2
    
    def _is_ai_direct_call(self, text: str) -> bool:
        """Determine if text is a direct AI call"""
        return any(regex.search(text) for regex in self.ai_call_regex)
    
    def _is_broadcast(self, text: str) -> bool:
        """Determine if text is a broadcast message"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.BROADCAST_KEYWORDS)
    
    def _extract_mentioned_users(self, text: str) -> Set[str]:
        """Extract mentioned users from text"""
        mentioned = set()
        
        for participant in self.participants:
            if participant == self.ai_name:
                continue
            
            # Check @mention
            if f"@{participant}" in text:
                mentioned.add(participant)
                continue
            
            # Check name + honorific (e.g., 철수야, 철수님, 철수씨)
            name_patterns = [
                rf"\b{re.escape(participant)}[아야님씨]?\b",
                rf"^{re.escape(participant)}[,\s]",
                rf"[,\s]{re.escape(participant)}$",
            ]
            for pattern in name_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    mentioned.add(participant)
                    break
        
        return mentioned
    
    def _detect_unknown_name_call(self, text: str) -> Optional[str]:
        """
        Detect name calls not in participant list
        
        For cases like "수민아 내일 뭐해?" where "수민" is not in participants,
        detect that someone is being called.
        
        Returns:
            Detected name or None
        """
        # Pattern starting with "~아", "~야"
        match = self.generic_name_call_regex.match(text)
        if match:
            name = match.group(1)
            # If not AI name and not in participants, it's an unknown person call
            if name.lower() != self.ai_name.lower() and name not in self.participants:
                return name
        
        return None
    
    def _analyze_continuation(self, text: str, recent_messages: List[Message], 
                              speaker: str) -> Tuple[bool, Set[str]]:
        """Analyze if text is continuing a previous conversation"""
        if not recent_messages:
            return False, set()
        
        last_msg = recent_messages[-1]
        
        # If previous message was addressed to current speaker, it's a continuation
        if speaker in last_msg.targets:
            return True, {last_msg.user}
        
        # If previous message was a question not yet answered
        if last_msg.is_question and not last_msg.responded_by:
            # If someone other than the questioner speaks, treat as answer
            if speaker != last_msg.user:
                return True, {last_msg.user}
        
        # If last 2 messages are between same two people, it's a continuation
        if len(recent_messages) >= 2:
            recent_speakers = {recent_messages[-1].user, recent_messages[-2].user}
            if speaker in recent_speakers and len(recent_speakers) == 2:
                other = (recent_speakers - {speaker}).pop()
                return True, {other}
        
        return False, set()
    
    def analyze(self, text: str, speaker: str, 
                recent_messages: Optional[List[Message]] = None) -> AddressAnalysis:
        """
        Main speech target analysis function
        
        Priority:
        1. AI direct call (@LLM, LLM아)
        2. Specific user mention (@철수, 철수야)
        3. Broadcast keywords (다들, 여러분)
        4. Previous conversation continuation
        5. Self-talk/monologue
        6. Default: UNKNOWN
        """
        text = text.strip()
        recent = recent_messages or []
        
        # 1. Check self-talk/short response first
        if self._is_self_talk(text) or self._is_short_response(text):
            # But if it's a response to previous question, treat as continuation
            is_cont, targets = self._analyze_continuation(text, recent, speaker)
            if is_cont and targets:
                return AddressAnalysis(
                    address_type=AddressType.CONTINUATION,
                    targets=targets,
                    confidence=0.6,
                    is_question=False,
                    reason="Short response to previous message"
                )
            return AddressAnalysis(
                address_type=AddressType.SELF_TALK,
                targets=set(),
                confidence=0.8,
                is_question=False,
                reason="Self-talk or minimal response"
            )
        
        is_question = self._is_question(text)
        
        # 2. Check AI direct call
        if self._is_ai_direct_call(text):
            return AddressAnalysis(
                address_type=AddressType.AI_DIRECT,
                targets={self.ai_name},
                confidence=0.95,
                is_question=is_question,
                reason=f"Direct call to {self.ai_name}"
            )
        
        # 3. Check specific user mention (in participant list)
        mentioned = self._extract_mentioned_users(text)
        if mentioned:
            return AddressAnalysis(
                address_type=AddressType.DIRECT,
                targets=mentioned,
                confidence=0.9,
                is_question=is_question,
                reason=f"Direct mention of {', '.join(mentioned)}"
            )
        
        # 3.5. Detect name call not in participant list
        # For cases like "수민아 내일 뭐해?" where caller is not a participant
        unknown_name = self._detect_unknown_name_call(text)
        if unknown_name:
            return AddressAnalysis(
                address_type=AddressType.DIRECT,
                targets={unknown_name},  # Unknown person
                confidence=0.85,
                is_question=is_question,
                reason=f"Call to '{unknown_name}' (not in participant list)"
            )
        
        # 4. Check broadcast keywords
        if self._is_broadcast(text):
            # If broadcast, include AI and all participants
            all_targets = self.participants.copy()
            return AddressAnalysis(
                address_type=AddressType.BROADCAST,
                targets=all_targets,
                confidence=0.85,
                is_question=is_question,
                reason="Broadcast keyword detected"
            )
        
        # 5. Check conversation continuation
        is_cont, cont_targets = self._analyze_continuation(text, recent, speaker)
        if is_cont:
            return AddressAnalysis(
                address_type=AddressType.CONTINUATION,
                targets=cont_targets,
                confidence=0.7,
                is_question=is_question,
                reason="Continuing previous conversation"
            )
        
        # 6. If question without specific target, treat as broadcast
        if is_question:
            return AddressAnalysis(
                address_type=AddressType.BROADCAST,
                targets=self.participants.copy(),
                confidence=0.5,
                is_question=True,
                reason="Question without specific target"
            )
        
        # 7. Default
        return AddressAnalysis(
            address_type=AddressType.UNKNOWN,
            targets=set(),
            confidence=0.3,
            is_question=is_question,
            reason="Could not determine target"
        )
    
    def should_ai_be_included(self, analysis: AddressAnalysis) -> Tuple[bool, float]:
        """
        Determine if AI should be included in speech targets
        
        Returns:
            (should_include, priority_score)
            priority_score: 0.0 ~ 1.0, higher means AI should respond first
        """
        # AI direct call
        if analysis.address_type == AddressType.AI_DIRECT:
            return True, 1.0
        
        # Broadcast (AI included)
        if analysis.address_type == AddressType.BROADCAST:
            if self.ai_name in analysis.targets:
                # Lower priority for broadcast (let others answer first)
                return True, 0.3
        
        # Specific user target (AI excluded)
        if analysis.address_type == AddressType.DIRECT:
            if self.ai_name in analysis.targets:
                return True, 0.9
            return False, 0.0
        
        # Continuation conversation
        if analysis.address_type == AddressType.CONTINUATION:
            if self.ai_name in analysis.targets:
                return True, 0.8
            return False, 0.0
        
        # Self-talk
        if analysis.address_type == AddressType.SELF_TALK:
            return False, 0.0
        
        # Unknown - include with low priority
        return True, 0.2
