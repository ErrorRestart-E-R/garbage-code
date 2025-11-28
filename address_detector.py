"""
AddressDetector: Score-based speech target classifier

Determines speech targets using a scoring system that combines multiple factors
(regex, keywords, conversation patterns) to calculate AI relevance probability.
"""

import re
from typing import Set, Tuple, Optional, List
from dataclasses import dataclass, field
from context_manager import AddressType, Message


@dataclass
class ScoreBreakdown:
    """Detailed score breakdown for debugging/transparency"""
    ai_direct_call: float = 0.0
    request_pattern: float = 0.0
    question_pattern: float = 0.0
    one_on_one: float = 0.0
    continuation_with_ai: float = 0.0
    broadcast_keyword: float = 0.0
    other_user_mention: float = 0.0
    self_talk_penalty: float = 0.0
    short_response_penalty: float = 0.0
    unknown_name_call: float = 0.0
    
    def total(self) -> float:
        """Calculate total score"""
        return (
            self.ai_direct_call +
            self.request_pattern +
            self.question_pattern +
            self.one_on_one +
            self.continuation_with_ai +
            self.broadcast_keyword +
            self.other_user_mention +  # Usually negative
            self.self_talk_penalty +   # Usually negative
            self.short_response_penalty +  # Usually negative
            self.unknown_name_call     # Usually negative
        )
    
    def clamped_total(self) -> float:
        """Get total clamped to 0.0 ~ 1.0"""
        return max(0.0, min(1.0, self.total()))


@dataclass
class AddressAnalysis:
    """Speech target analysis result"""
    address_type: AddressType
    targets: Set[str]
    confidence: float  # 0.0 ~ 1.0 (now represents AI relevance score)
    is_question: bool
    reason: str
    score_breakdown: Optional[ScoreBreakdown] = None  # Detailed scores


class AddressDetector:
    """Score-based speech target detector"""
    
    # === Score Weights (Tunable) ===
    SCORE_AI_DIRECT_CALL = 0.5       # "@LLM", "LLM아"
    SCORE_REQUEST_PATTERN = 0.4     # "~해줘", "~알려줘" 
    SCORE_QUESTION = 0.1            # Question mark, interrogatives
    SCORE_ONE_ON_ONE = 0.3          # Only 1 participant (1:1 with AI)
    SCORE_SMALL_GROUP = 0.1         # 2-3 participants
    SCORE_CONTINUATION_AI = 0.25    # AI just spoke, continuing conversation
    SCORE_BROADCAST = 0.15          # "다들", "여러분"
    
    PENALTY_OTHER_USER_MENTION = -0.5   # Mentioned another user by name
    PENALTY_SELF_TALK = -0.4            # Self-talk patterns
    PENALTY_SHORT_RESPONSE = -0.2       # Very short responses like "응", "어"
    PENALTY_UNKNOWN_NAME = -0.5         # Called someone not in participants
    
    # Response threshold
    RESPONSE_THRESHOLD = 0.3        # Respond if score >= this (lowered for better responsiveness)
    HIGH_PRIORITY_THRESHOLD = 0.7   # Respond immediately if score >= this
    
    # === Pattern Definitions ===
    
    # Broadcast keywords (Korean/English)
    BROADCAST_KEYWORDS = [
        "다들", "여러분", "모두", "전부", "다같이", "우리", "얘들아", "애들아",
        "everyone", "everybody", "all", "guys", "folks", "y'all"
    ]
    
    # Request/Command patterns (Korean)
    REQUEST_PATTERNS = [
        r"(해줘|해주세요|해줄래|해줄수|해줄 수)",
        r"(알려줘|알려주세요|알려줄래)",
        r"(말해줘|말해주세요|말해봐)",
        r"(찾아줘|찾아주세요|찾아봐)",
        r"(검색해줘|검색해주세요|검색해봐)",
        r"(설명해줘|설명해주세요|설명해봐)",
        r"(추천해줘|추천해주세요|추천해봐)",
        r"(보여줘|보여주세요|보여줄래)",
        r"(읽어줘|읽어주세요|읽어봐)",
        r"(번역해줘|번역해주세요|번역해봐)",
        r"(가르쳐줘|가르쳐주세요|가르쳐봐)",
        r"(도와줘|도와주세요|도움)",
    ]
    
    # Question patterns
    QUESTION_PATTERNS = [
        r"[?？]$",                          # Ends with question mark
        r"^(누가|뭐|뭘|어디|언제|왜|어떻게|몇|얼마)",  # Starts with interrogative
        r"(할래|할까|하자|갈래|갈까|먹을래|볼래|해볼래)\s*[?？]?$",
        r"(인가요|인가|일까|일까요|인지|는지|나요|니|냐)\s*[?？]?$",
        r"(맞아|맞지|그래|그렇지|아니야|아닌가)\s*[?？]?$",
        r"^(혹시|그래서|근데|그런데)",
    ]
    
    # Self-talk/monologue patterns
    SELF_TALK_PATTERNS = [
        r"^(아|어|음|흠|에|오|와|헐|ㅋ|ㅎ|ㄷ|ㅇ)+$",
        r"^\.{2,}$",
        r"^(뭐지|뭐야|뭐냐|왜지|왜이래|어라|이상하네|신기하네)",
    ]
    
    # Short responses
    SHORT_RESPONSES = [
        "ok", "okay", "yes", "no", "yeah", "yep", "nope",
        "응", "어", "아", "네", "예", "아니", "그래", "알겠어", "ㅇㅇ", "ㄴㄴ"
    ]
    
    # AI call patterns (AI name added dynamically)
    AI_CALL_PATTERNS = [
        r"@{ai_name}",
        r"{ai_name}[야아]",
        r"{ai_name}[님]",
        r"^{ai_name}[,\s]",
        r"[,\s]{ai_name}$",
        r"^{ai_name}$",
    ]
    
    # Generic name call pattern
    GENERIC_NAME_CALL_PATTERN = r"^([가-힣a-zA-Z]+)[아야님씨]\s"
    
    def __init__(self, ai_name: str = "LLM", participants: Optional[Set[str]] = None):
        self.ai_name = ai_name
        self.participants = participants or set()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.question_regex = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self.self_talk_regex = [re.compile(p, re.IGNORECASE) for p in self.SELF_TALK_PATTERNS]
        self.request_regex = [re.compile(p, re.IGNORECASE) for p in self.REQUEST_PATTERNS]
        
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
    
    # === Pattern Detection Methods ===
    
    def _is_question(self, text: str) -> bool:
        """Determine if text is a question"""
        return any(regex.search(text) for regex in self.question_regex)
    
    def _is_request(self, text: str) -> bool:
        """Determine if text is a request/command"""
        return any(regex.search(text) for regex in self.request_regex)
    
    def _is_self_talk(self, text: str) -> bool:
        """Determine if text is self-talk"""
        if len(text.strip()) <= 2:
            return True
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
            
            if f"@{participant}" in text:
                mentioned.add(participant)
                continue
            
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
        """Detect name calls not in participant list"""
        match = self.generic_name_call_regex.match(text)
        if match:
            name = match.group(1)
            if name.lower() != self.ai_name.lower() and name not in self.participants:
                return name
        return None
    
    def _check_continuation_with_ai(self, recent_messages: List[Message], speaker: str) -> bool:
        """Check if this is a continuation of conversation with AI"""
        if not recent_messages:
            return False
        
        last_msg = recent_messages[-1]
        
        # AI just spoke to this user
        if last_msg.user == self.ai_name:
            return True
        
        # User was talking to AI in recent messages
        for msg in reversed(recent_messages[-3:]):
            if msg.user == speaker and self.ai_name in msg.targets:
                return True
            if msg.user == self.ai_name and speaker in msg.targets:
                return True
        
        return False
    
    # === Main Scoring Method ===
    
    def calculate_ai_relevance_score(self, text: str, speaker: str,
                                      recent_messages: Optional[List[Message]] = None,
                                      participant_count: int = 0) -> ScoreBreakdown:
        """
        Calculate AI relevance score using multiple factors.
        
        Returns a ScoreBreakdown with individual component scores.
        """
        text = text.strip()
        recent = recent_messages or []
        breakdown = ScoreBreakdown()
        
        # 1. AI Direct Call (+0.5)
        if self._is_ai_direct_call(text):
            breakdown.ai_direct_call = self.SCORE_AI_DIRECT_CALL
        
        # 2. Request/Command Pattern (+0.25)
        if self._is_request(text):
            breakdown.request_pattern = self.SCORE_REQUEST_PATTERN
        
        # 3. Question Pattern (+0.1)
        if self._is_question(text):
            breakdown.question_pattern = self.SCORE_QUESTION
        
        # 4. Participant Count Factor
        if participant_count <= 1:
            # 1:1 conversation with AI
            breakdown.one_on_one = self.SCORE_ONE_ON_ONE
        elif participant_count <= 3:
            # Small group
            breakdown.one_on_one = self.SCORE_SMALL_GROUP
        
        # 5. Continuation with AI (+0.25)
        if self._check_continuation_with_ai(recent, speaker):
            breakdown.continuation_with_ai = self.SCORE_CONTINUATION_AI
        
        # 6. Broadcast Keywords (+0.15)
        if self._is_broadcast(text):
            breakdown.broadcast_keyword = self.SCORE_BROADCAST
        
        # 7. Other User Mention (-0.5)
        mentioned_users = self._extract_mentioned_users(text)
        if mentioned_users:
            breakdown.other_user_mention = self.PENALTY_OTHER_USER_MENTION
        
        # 8. Self-talk Penalty (-0.4)
        if self._is_self_talk(text):
            breakdown.self_talk_penalty = self.PENALTY_SELF_TALK
        
        # 9. Short Response Penalty (-0.2)
        # But not if continuing conversation with AI
        if self._is_short_response(text) and breakdown.continuation_with_ai == 0:
            breakdown.short_response_penalty = self.PENALTY_SHORT_RESPONSE
        
        # 10. Unknown Name Call (-0.5)
        unknown_name = self._detect_unknown_name_call(text)
        if unknown_name:
            breakdown.unknown_name_call = self.PENALTY_UNKNOWN_NAME
        
        return breakdown
    
    def analyze(self, text: str, speaker: str,
                recent_messages: Optional[List[Message]] = None,
                participant_count: int = 0) -> AddressAnalysis:
        """
        Main speech target analysis function using scoring system.
        
        Calculates AI relevance score and determines address type based on
        the score and detected patterns.
        """
        text = text.strip()
        recent = recent_messages or []
        
        # Calculate score
        breakdown = self.calculate_ai_relevance_score(
            text, speaker, recent, participant_count
        )
        score = breakdown.clamped_total()
        is_question = self._is_question(text)
        
        # Determine AddressType based on patterns and score
        mentioned_users = self._extract_mentioned_users(text)
        unknown_name = self._detect_unknown_name_call(text)
        
        # Priority 1: AI Direct Call
        if self._is_ai_direct_call(text):
            return AddressAnalysis(
                address_type=AddressType.AI_DIRECT,
                targets={self.ai_name},
                confidence=score,
                is_question=is_question,
                reason=f"Direct call to {self.ai_name} (score: {score:.2f})",
                score_breakdown=breakdown
            )
        
        # Priority 2: Other user mentioned
        if mentioned_users:
            return AddressAnalysis(
                address_type=AddressType.DIRECT,
                targets=mentioned_users,
                confidence=score,
                is_question=is_question,
                reason=f"Direct mention of {', '.join(mentioned_users)} (score: {score:.2f})",
                score_breakdown=breakdown
            )
        
        # Priority 3: Unknown name called
        if unknown_name:
            return AddressAnalysis(
                address_type=AddressType.DIRECT,
                targets={unknown_name},
                confidence=score,
                is_question=is_question,
                reason=f"Call to '{unknown_name}' not in participants (score: {score:.2f})",
                score_breakdown=breakdown
            )
        
        # Priority 4: Self-talk
        if self._is_self_talk(text) and score < self.RESPONSE_THRESHOLD:
            return AddressAnalysis(
                address_type=AddressType.SELF_TALK,
                targets=set(),
                confidence=score,
                is_question=False,
                reason=f"Self-talk detected (score: {score:.2f})",
                score_breakdown=breakdown
            )
        
        # Priority 5: Broadcast
        if self._is_broadcast(text):
            all_targets = self.participants.copy()
            return AddressAnalysis(
                address_type=AddressType.BROADCAST,
                targets=all_targets,
                confidence=score,
                is_question=is_question,
                reason=f"Broadcast keyword detected (score: {score:.2f})",
                score_breakdown=breakdown
            )
        
        # Priority 6: High score = likely talking to AI
        if score >= self.RESPONSE_THRESHOLD:
            # Determine if it's continuation or general
            if breakdown.continuation_with_ai > 0:
                return AddressAnalysis(
                    address_type=AddressType.CONTINUATION,
                    targets={self.ai_name},
                    confidence=score,
                    is_question=is_question,
                    reason=f"Continuing conversation with AI (score: {score:.2f})",
                    score_breakdown=breakdown
                )
            
            # Request/question without specific target = likely to AI
            if breakdown.request_pattern > 0 or is_question:
                return AddressAnalysis(
                    address_type=AddressType.BROADCAST,
                    targets=self.participants.copy() | {self.ai_name},
                    confidence=score,
                    is_question=is_question,
                    reason=f"Request/question with high AI relevance (score: {score:.2f})",
                    score_breakdown=breakdown
                )
        
        # Priority 7: Low score = unknown/not for AI
        return AddressAnalysis(
            address_type=AddressType.UNKNOWN,
            targets=set(),
            confidence=score,
            is_question=is_question,
            reason=f"Low AI relevance (score: {score:.2f})",
            score_breakdown=breakdown
        )
    
    def should_ai_be_included(self, analysis: AddressAnalysis) -> Tuple[bool, float]:
        """
        Determine if AI should respond based on score.
        
        Returns:
            (should_include, priority_score)
        """
        score = analysis.confidence
        
        # AI direct call = always respond with highest priority
        if analysis.address_type == AddressType.AI_DIRECT:
            return True, 1.0
        
        # Self-talk = never respond
        if analysis.address_type == AddressType.SELF_TALK:
            return False, 0.0
        
        # Other user mentioned (not AI) = don't respond
        if analysis.address_type == AddressType.DIRECT and self.ai_name not in analysis.targets:
            return False, 0.0
        
        # Score-based decision
        if score >= self.HIGH_PRIORITY_THRESHOLD:
            # High score = respond with high priority
            return True, min(0.9, score)
        
        elif score >= self.RESPONSE_THRESHOLD:
            # Medium score = respond but with lower priority
            return True, score * 0.8
        
        else:
            # Low score = don't respond
            return False, 0.0
    
    def get_score_explanation(self, breakdown: ScoreBreakdown) -> str:
        """Get human-readable explanation of score breakdown"""
        parts = []
        
        if breakdown.ai_direct_call > 0:
            parts.append(f"AI호출: +{breakdown.ai_direct_call:.2f}")
        if breakdown.request_pattern > 0:
            parts.append(f"요청패턴: +{breakdown.request_pattern:.2f}")
        if breakdown.question_pattern > 0:
            parts.append(f"질문: +{breakdown.question_pattern:.2f}")
        if breakdown.one_on_one > 0:
            parts.append(f"소규모대화: +{breakdown.one_on_one:.2f}")
        if breakdown.continuation_with_ai > 0:
            parts.append(f"AI대화이어가기: +{breakdown.continuation_with_ai:.2f}")
        if breakdown.broadcast_keyword > 0:
            parts.append(f"전체대상: +{breakdown.broadcast_keyword:.2f}")
        if breakdown.other_user_mention < 0:
            parts.append(f"다른사람멘션: {breakdown.other_user_mention:.2f}")
        if breakdown.self_talk_penalty < 0:
            parts.append(f"혼잣말: {breakdown.self_talk_penalty:.2f}")
        if breakdown.short_response_penalty < 0:
            parts.append(f"짧은응답: {breakdown.short_response_penalty:.2f}")
        if breakdown.unknown_name_call < 0:
            parts.append(f"알수없는이름: {breakdown.unknown_name_call:.2f}")
        
        total = breakdown.clamped_total()
        return f"[{' | '.join(parts)}] = {total:.2f}"
