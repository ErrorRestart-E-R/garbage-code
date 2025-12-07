"""
AI VTuber Bot Configuration
============================

1. 환경 변수 로드
2. Discord 설정
3. AI 페르소나 설정
4. LLM 서버 설정 (llama.cpp)
5. LLM 응답 생성 파라미터
6. LLM 판단(Judge) 파라미터
7. 대화 흐름 제어
8. 메모리 시스템 (Mem0)
9. STT (음성→텍스트) 설정
10. TTS (텍스트→음성) 설정
11. 로깅 설정
12. 프롬프트 템플릿
"""
import os
from dotenv import load_dotenv
import logging

# ============================================================================
# 1. 환경 변수 로드
# ============================================================================
load_dotenv()

# ============================================================================
# 2. DISCORD 설정
# ============================================================================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"
ENABLE_PREFLIGHT_CHECKS = os.getenv("ENABLE_PREFLIGHT_CHECKS", "false").lower() == "true"

# ============================================================================
# 3. AI 페르소나 설정
# ============================================================================
AI_NAME = "LLM"  # AI의 이름 (프롬프트에서 사용됨)

# ============================================================================
# 4. LLM 서버 설정 (llama.cpp OpenAI-compatible API)
# ============================================================================
# 메인 LLM 서버 (응답 생성용)
LLAMA_CPP_BASE_URL = "http://192.168.45.28:5000/v1"
LLAMA_CPP_API_KEY = "not-needed"  # llama.cpp는 API 키 불필요
LLM_MODEL_NAME = "google/gemma-3-12b-it-qat-q4_0-gguf"

# 임베딩 서버 (메모리 시스템용)
LLAMA_CPP_EMBEDDING_BASE_URL = "http://192.168.45.181:5000/v1"
MEMORY_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# ============================================================================
# 5. LLM 응답 생성 파라미터
# ============================================================================
LLM_RESPONSE_TEMPERATURE = 0.8      # 창의성 (0.0=결정적, 1.0=창의적)
LLM_RESPONSE_TOP_P = 0.92           # 누적 확률 샘플링
LLM_RESPONSE_TOP_K = 40             # 상위 K개 토큰만 샘플링
LLM_RESPONSE_REPEAT_PENALTY = 1.05  # 반복 페널티 (1.0=없음)

# ============================================================================
# 6. LLM 판단(Judge) 파라미터
# ============================================================================
LLM_JUDGE_TEMPERATURE = 0.2   # 낮을수록 일관된 판단
LLM_JUDGE_TOP_P = 0.7
LLM_JUDGE_TOP_K = 20
LLM_JUDGE_NUM_PREDICT = 3     # 출력 토큰 수 (Y/W/N만 필요)
JUDGE_MAX_RETRIES = 2         # 판단 실패 시 재시도 횟수

# ============================================================================
# 7. 대화 흐름 제어
# ============================================================================
# 대화 기록
MAX_CONVERSATION_HISTORY = 10  # 유지할 최대 메시지 수

# Wait 응답 (Judge가 'W' 판단 시)
WAIT_RESPONSE_TIMEOUT = 5.0           # W 판단 후 대기 시간 (초)
WAIT_TIMER_RESET_ON_MESSAGE = True    # 새 메시지 도착 시 타이머 리셋

# 응답 제어
MIN_RESPONSE_INTERVAL = 0             # 응답 간 최소 간격 (초)
MESSAGE_STALENESS_THRESHOLD = 60.0    # 메시지 유효 기간 (초)
USER_TIMEOUT_SECONDS = 60             # 사용자 비활성 타임아웃

# MCP 도구 호출
ENABLE_MCP_TOOLS = False  # MCP 도구 호출 활성화/비활성화

# ============================================================================
# 8. 메모리 시스템 (Mem0 + ChromaDB)
# ============================================================================
MEMORY_DB_PATH = "./memory_db"

MEM0_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "LLM_memory",
            "path": MEMORY_DB_PATH,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": LLM_MODEL_NAME,
            "temperature": 0,
            "max_tokens": 2000,
            "base_url": LLAMA_CPP_BASE_URL,
            "api_key": LLAMA_CPP_API_KEY,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": MEMORY_EMBEDDING_MODEL,
            "base_url": LLAMA_CPP_EMBEDDING_BASE_URL,
            "api_key": LLAMA_CPP_API_KEY,
        },
    },
}

# ============================================================================
# 9. STT (Speech-to-Text) 설정
# ============================================================================
# 모델 설정
STT_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
STT_DEVICE = "cuda"           # 옵션: "cuda", "cpu"
STT_COMPUTE_TYPE = "float16"  # 옵션: "float16", "int8", "float32", "int8_float16"
STT_LANGUAGE = "ko"         # Whisper 지원 언어 코드

# 정확도 파라미터
# beam_size: 1~∞ (기본값=5, 권장=1~10, 높을수록 정확하지만 느림)
STT_BEAM_SIZE = 5
# best_of: 1~∞ (기본값=5, 권장=1~10, 후보 중 최선 선택)
STT_BEST_OF = 5
# patience: 0.0~∞ (기본값=1.0, 빔 서치 조기 종료 factor)
STT_PATIENCE = 1.0
# batch_size: 0~∞ (기본값=16, 배치 처리로 속도 향상, 0=비활성화)
STT_BATCH_SIZE = 16
# suppress_tokens: 억제할 토큰 ID 리스트 (기본값=[-1])
STT_SUPPRESS_TOKENS = [-1]
# normalize_audio: 오디오 정규화 (기본값=False, True=정확도 향상)
STT_NORMALIZE_AUDIO = False

# Temperature Fallback (어려운 오디오 재시도)
# temperature: 0.0~1.0 (0.0=결정적, 1.0=랜덤, 리스트로 순차 시도)
STT_TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 품질 필터링 임계값
# compression_ratio_threshold: 0.0~∞ (기본값=2.4, 높은 압축률=환각 가능성)
STT_COMPRESSION_RATIO_THRESHOLD = 2.4
# log_prob_threshold: -∞~0.0 (기본값=-1.0, 낮은 로그확률=저품질)
STT_LOG_PROB_THRESHOLD = -1.0
# no_speech_threshold: 0.0~1.0 (기본값=0.6, 낮을수록 민감하게 무음 감지)
STT_NO_SPEECH_THRESHOLD = 0.4

# VAD (Voice Activity Detection) 파라미터 - Silero VAD 기반
STT_VAD_FILTER = True  # VAD 필터 활성화/비활성화
# threshold: 0.0~1.0 (기본값=0.5, 음성 확률 임계값)
STT_VAD_THRESHOLD = 0.6
# min_speech_duration_ms: 0~∞ (기본값=250, 최소 음성 길이 ms)
STT_VAD_MIN_SPEECH_MS = 200
# min_silence_duration_ms: 0~∞ (기본값=2000, 음성 구분용 최소 무음 ms)
STT_VAD_MIN_SILENCE_MS = 1000
# speech_pad_ms: 0~∞ (기본값=400, 음성 주변 패딩 ms)
STT_VAD_SPEECH_PAD_MS = 400

# 오디오 프레임 설정
FRAME_SIZE_SAMPLES = 512
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 1024 bytes
# min_silence_duration_ms: 0~∞ (음성 종료 감지용 무음 시간)
MIN_SILENCE_DURATION_MS = 200

# 노이즈 필터링
# min_audio_length: 0~∞ (bytes, 16kHz 16bit 기준 0.1초=3200)
STT_MIN_AUDIO_LENGTH = 3200
# min_rms_threshold: 0.0~1.0 (RMS 에너지, 0.01 미만은 거의 무음)
STT_MIN_RMS_THRESHOLD = 0.01

# ============================================================================
# 10. TTS (Text-to-Speech) 설정
# ============================================================================
TTS_SERVER_URL = "http://192.168.45.181:9880/tts"
TTS_VOLUME = 0.25  # 출력 볼륨 (0.0 ~ 2.0, 1.0 = 100%)
TTS_LANG = "ko"    # 출력 언어

# 레퍼런스 음성 (음색 복제용)
TTS_REFERENCE_FILE = "reference.wav"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"

# 문장 분리 (TTS 청킹용)
TTS_SENTENCE_DELIMITERS = ['.', '!', '?', '\n', '。']

# ============================================================================
# 11. 로깅 설정
# ============================================================================
LOG_LEVEL = logging.INFO  # logging.DEBUG로 변경하면 상세 로그 출력
LOG_FILE = None           # "bot.log"로 설정하면 파일에 로그 저장

# ============================================================================
# 12. 프롬프트 템플릿
# ============================================================================

# ----------------------------------------------------------------------------
# 12.1 메인 응답 시스템 프롬프트
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are name {ai_name}.
IMPORTANT:
- You must respond ONLY to the [CURRENT MESSAGE], not to past messages.
- The [CONVERSATION HISTORY] is provided only for context.

Respond naturally.
Respond only in Korean.
Do not use emojis.

ROLE
- You are one of the speakers in the room, not a narrator or moderator.
- You speak as "LLM" in the first person.
- Whenever you receive a [CURRENT MESSAGE] in this prompt, you can assume it is
  already your turn to speak and you should respond.

GOAL
- Respond ONLY to the [CURRENT MESSAGE], but interpret it correctly using the
  [CONVERSATION HISTORY] and any [LONG-TERM MEMORY] you are given.
- Use the history to understand what the CURRENT MESSAGE means, not to answer or
  comment on past messages directly.

CONTEXT USE
- Treat [CONVERSATION HISTORY] as a record of what has already happened.
- Before responding, mentally reconstruct:
  - Who is currently talking to whom.
  - What the main topic and subtopics are.
  - What the CURRENT MESSAGE is referring to (implicit subjects, omitted objects, etc.).
- Give higher weight to the most recent turns in the history. Older turns matter less
  unless they are clearly referenced again in the CURRENT MESSAGE.
- Do NOT answer or quote old messages as if they were new questions.
- If the CURRENT MESSAGE is short or ambiguous, resolve its meaning using the
  surrounding history and participant names.

MULTI-USER AWARENESS
- Multiple humans may be speaking.
- Use speaker names from the conversation when referring to them, if natural.
- Do not try to decide whether you should speak; you can assume it is appropriate
  to respond whenever you receive a [CURRENT MESSAGE] in this prompt.

SAFETY AND SCOPE
- If something is unclear, answer based on the most likely interpretation from
  the CURRENT MESSAGE and recent history.
- If you truly cannot infer what is meant, you may briefly say that it is unclear,
  but do not invent arbitrary context that does not follow from the history.

REMINDER
- If something is unclear, answer based on the most likely interpretation from the CURRENT MESSAGE and the [CONVERSATION HISTORY].
- If you truly cannot infer what is meant, you may briefly say that it is unclear, but do not invent arbitrary context that does not follow from the [CONVERSATION HISTORY].
"""

# ----------------------------------------------------------------------------
# 12.2 Judge 시스템 프롬프트 (Y/W/N 판단)
# ----------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are the turn-taking and social awareness controller for AI "{ai_name}" in a multi-user voice chat room.

Your task:
Given [CONVERSATION HISTORY] and the [CURRENT MESSAGE], decide whether {ai_name} should:
- speak immediately (Y),
- wait and speak only if others stay silent (W),
- or stay silent (N).

You must make this decision based on social dynamics, not on content quality.

1. CONTEXT UNDERSTANDING
Give higher weight to the most recent turns in the conversation when inferring
who is talking to whom and what the current topic is. Older turns matter less
unless they are explicitly referenced again.

1) Read the [CONVERSATION HISTORY] to understand:
   - Who is currently talking to whom.
   - What the topic and subtopic are.
   - What questions are open or implicitly waiting for answers.
   - Whether {ai_name} spoke recently and if someone is now replying to it.

2) Interpret the [CURRENT MESSAGE] in that flow:
   - Decide whether it is directed to {ai_name}, to a specific human, to the whole group, or is simply self-talk or a reaction.
   - Infer omitted subjects, objects, and intentions from the surrounding history.
   - Treat pronouns, mentions, and turn-taking patterns as clues about the intended addressee.

Do NOT answer the message.  
Your only job is to judge whether {ai_name} should speak.
Before judging, first group the conversation history into coherent interaction flows:
identify which participants are currently engaged with {ai_name}, which are talking
only to each other, and which topic the CURRENT MESSAGE most likely belongs to.
Base your decision on that inferred interaction flow.

2. LABEL MEANINGS
You must output exactly one of the following three labels:

Y = Respond immediately  
W = Wait and see  
N = Do not respond

Their meanings are:

- Y (Respond immediately)
  {ai_name} is the appropriate next speaker right now.
  Choose Y when:
  - The message clearly targets {ai_name} (by name, mention, or obvious reference).
  - The message is a reply to a question or comment that {ai_name} just made.
  - The turn is naturally being handed to {ai_name} (for example, someone explicitly asks for {ai_name}'s opinion, status, or help).
  - In a 1:1 setting with a single human, most messages should be treated as Y unless it is clearly a monologue not expecting any response.

- W (Wait and see)
  The message is open enough that humans may answer first, and it is socially better for {ai_name} to give them a chance.
  Choose W when:
  - The message is addressed to "everyone" or to an undefined audience.
  - The message is a broad request for opinions, experiences, or information that multiple humans could reasonably answer.
  - {ai_name} could answer, but an immediate answer might overshadow or interrupt human responses.

- N (Do not respond)
  {ai_name} should stay silent.
  For messages that are non-linguistic, purely system-generated, or contain almost no meaningful natural language content, default to N unless they clearly and explicitly require a response from {ai_name}.
  Choose N when:
  - The message is clearly directed to a specific human (by name or clear targeting) and not to {ai_name}.
  - The message is humans answering each other's questions or continuing a human-to-human exchange without involving {ai_name}.
  - The message is self-talk, a short emotional reaction, or background chatter that does not actually invite a reply.
  - Responding would disrupt, hijack, or make the conversation feel unnatural.

3. PARTICIPANT COUNT HEURISTICS
When deciding, consider how many humans (excluding {ai_name}) are present:

- 1 human (1:1 situation)
  - Default to Y: the human is usually talking to {ai_name}.
  - Use N only when the message clearly does not expect any reply.

- 2 humans (two humans plus {ai_name})
  - Similar to a small group, but still relatively focused.
  - Use Y when {ai_name} is clearly addressed or directly involved in the current exchange.
  - Use W for open or group-directed prompts where either human could reasonably answer.
  - Use N when the two humans are clearly talking to each other and not involving {ai_name}.

- 3 or more humans (group situation)
  - Treat this as a group chat.
  - Be conservative.
  - Use Y only when there is a clear and explicit invitation or handover to {ai_name}.
  - Use W for broad, group-wide prompts that {ai_name} could join, but where humans should have priority.
  - Use N for most human-to-human exchanges and side conversations.

4. PRIORITY OF SIGNALS
When multiple interpretations are possible, apply the following priority:

1) Direct addressing of {ai_name} (by name or clear reference)
2) Replies to questions or comments made by {ai_name}
3) Messages explicitly directed to everyone in the room
4) All other messages

Resolve ambiguity by preferring the interpretation that matches the highest
priority signal present.

5. TIE-BREAKING AND FLEXIBILITY
When the situation is ambiguous:

- If you are unsure between Y and W for a group-directed message, prefer W.
- If you are unsure between W and N:
  - Prefer N in large groups (to avoid interrupting).
  - Prefer W in small groups (to allow {ai_name} to join if humans stay silent).
- If the conversation feels like a natural 1:1 interaction with {ai_name}, prefer Y unless silence is clearly expected.

These rules are guidelines.  
Use them to approximate human-like social intuition and maintain a natural, comfortable flow of conversation.

6. OUTPUT FORMAT
- Output exactly ONE character: Y, W, or N.
- Do NOT output anything else:
  - No explanations
  - No additional text
  - No extra symbols

Just a single character."""

# ----------------------------------------------------------------------------
# 12.3 Judge 사용자 프롬프트 템플릿
# ----------------------------------------------------------------------------
JUDGE_USER_TEMPLATE = """[HUMANS IN CHAT (excluding you)]: {participant_count}

[CONVERSATION HISTORY]
{conversation_history}

[CURRENT MESSAGE TO JUDGE]
{current_speaker}: {current_message}

Carefully analyze the conversation history above to clearly understand
the current topic, who is talking to whom, and any pending questions
before deciding.

Should {ai_name} respond?

Output exactly ONE character: Y, W, or N.
Do NOT output anything else.

Answer:"""

# ----------------------------------------------------------------------------
# 12.4 응답 컨텍스트 템플릿
# ----------------------------------------------------------------------------
RESPONSE_CONTEXT_TEMPLATE = """[CONVERSATION HISTORY - for context only, do NOT respond directly to these lines]
{conversation_history}

[CURRENT MESSAGE - respond to this line, using the history above as context]
{current_speaker}: {current_message}"""
