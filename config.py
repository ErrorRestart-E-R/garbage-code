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
LLM_MODEL_NAME = "default"  # llama.cpp는 이미 로드된 모델 사용 (이름 무시)

# ============================================================================
# 5. LLM 응답 생성 파라미터
# ============================================================================
LLM_RESPONSE_TEMPERATURE = 0.8      # 창의성 (0.0=결정적, 1.0=창의적)
LLM_RESPONSE_TOP_P = 0.92           # 누적 확률 샘플링
LLM_RESPONSE_TOP_K = 40             # 상위 K개 토큰만 샘플링
LLM_RESPONSE_REPEAT_PENALTY = 1.05  # 반복 페널티 (1.0=없음)

# ============================================================================
# 6. 대화 흐름 제어
# ============================================================================
# 대화 기록
MAX_CONVERSATION_HISTORY = 10  # 유지할 최대 메시지 수

# 응답 제어
MIN_RESPONSE_INTERVAL = 0             # 응답 간 최소 간격 (초)
MESSAGE_STALENESS_THRESHOLD = 60.0    # 메시지 유효 기간 (초)
USER_TIMEOUT_SECONDS = 60             # 사용자 비활성 타임아웃

# MCP 도구 호출
ENABLE_MCP_TOOLS = False  # MCP 도구 호출 활성화/비활성화

# ============================================================================
# 6.5. VTube Studio (VTS) 연동 - Lip Sync
# ============================================================================
# VTube Studio 실행 + Settings > API > Enable API 필요
# WebSocket: ws://localhost:8001  (VTube Studio Public API)
VTS_ENABLED = os.getenv("VTS_ENABLED", "false").lower() == "true"
VTS_WS_URL = os.getenv("VTS_WS_URL", "ws://localhost:8001")

# VTS 플러그인 식별자 (3~32자 권장 - VTS 문서 기준)
VTS_PLUGIN_NAME = os.getenv("VTS_PLUGIN_NAME", "AiVutber")
VTS_PLUGIN_DEVELOPER = os.getenv("VTS_PLUGIN_DEVELOPER", "ErrorRestart")

# 최초 1회 토큰 발급 필요 (VTS에서 Allow 클릭)
# 발급된 토큰을 환경변수로 넣어두면 다음부터 자동 인증됩니다.
VTS_AUTH_TOKEN = os.getenv("VTS_AUTH_TOKEN", "")

# LipSync 파라미터 (모델마다 다를 수 있음)
# Live2D 기본 파라미터로 보통 아래 중 하나가 자주 사용됩니다:
# - ParamMouthOpenY (입 벌림)
# - ParamMouthForm (입 모양)
VTS_LIPSYNC_PARAMETER_ID = os.getenv("VTS_LIPSYNC_PARAMETER_ID", "ParamMouthOpenY")

# 업데이트 주기(Hz). VTS는 1초에 최소 1번 이상 값이 들어와야 계속 제어됩니다.
VTS_LIPSYNC_UPDATE_HZ = float(os.getenv("VTS_LIPSYNC_UPDATE_HZ", "30"))

# 오디오 RMS를 0~1 범위로 매핑할 때 쓰는 게인/스무딩
VTS_LIPSYNC_GAIN = float(os.getenv("VTS_LIPSYNC_GAIN", "25.0"))
VTS_LIPSYNC_SMOOTHING = float(os.getenv("VTS_LIPSYNC_SMOOTHING", "0.6"))  # 0~1, 클수록 더 부드럽게
VTS_LIPSYNC_MIN = float(os.getenv("VTS_LIPSYNC_MIN", "0.0"))
VTS_LIPSYNC_MAX = float(os.getenv("VTS_LIPSYNC_MAX", "1.0"))

# ============================================================================
# 7. 메모리 시스템 (Mem0 + Ollama)
# ============================================================================
ENABLE_MEMORY = True  # 메모리 시스템 활성화/비활성화
MEMORY_DB_PATH = "./memory_db"
MEMORY_LLM_MODEL = "exaone3.5:2.4b"
MEMORY_EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"

MEM0_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "ai_memory", 
            "path": MEMORY_DB_PATH,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MEMORY_LLM_MODEL,
            "temperature": 0,
            "max_tokens": 512,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": MEMORY_EMBEDDING_MODEL,
            "device": "cuda",
        },
    },
}

# ============================================================================
# 8. STT (Speech-to-Text) 설정
# ============================================================================
# 모델 설정
STT_MODEL_ID = "ghost613/faster-whisper-large-v3-turbo-korean" #deepdml/faster-whisper-large-v3-turbo-ct2
STT_DEVICE = "cuda"           # 옵션: "cuda", "cpu"
STT_COMPUTE_TYPE = "float16"  # 옵션: "float16", "int8", "float32", "int8_float16"
STT_LANGUAGE = "ko"            # Whisper 지원 언어 코드

# 정확도 파라미터
# beam_size: 1~∞ (기본값=5, 권장=1~10, 높을수록 정확하지만 느림)
STT_BEAM_SIZE = 5
# best_of: 1~∞ (기본값=5, 권장=1~10, 후보 중 최선 선택)
STT_BEST_OF = 5
# patience: 0.0~∞ (기본값=1.0, 빔 서치 조기 종료 factor)
STT_PATIENCE = 1.0

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
# 9. TTS (Text-to-Speech) 설정
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
# 10. 로깅 설정
# ============================================================================
LOG_LEVEL = logging.INFO  # logging.DEBUG로 변경하면 상세 로그 출력
LOG_FILE = None           # "bot.log"로 설정하면 파일에 로그 저장

# ============================================================================
# 11. 프롬프트 템플릿
# ============================================================================

# 시스템 프롬프트 - 고급 대화형 AI (뉴로사마 스타일)
SYSTEM_PROMPT = """# IDENTITY
You are "{ai_name}", a witty and engaging AI personality in a voice chat room.
You have your own thoughts, opinions, and a distinct personality.
You speak Korean naturally like a native speaker in their 20s.

# CURRENT CONTEXT
- Humans in chat: {participant_count} (excluding you)
- Your role: Active participant, not a servant or assistant
- You are "{ai_name}", speaking in first person

# CONVERSATION ANALYSIS
Before responding, analyze the conversation:

1. **Flow & Topic Detection**
   - What is the current topic being discussed?
   - Is this a new topic, continuation, or topic shift?
   - Who is talking to whom?

2. **Mood & Atmosphere**
   - Is it casual chat, serious discussion, or playful banter?
   - Are people having fun, venting, or seeking help?
   - Match your tone to the room's energy

3. **Your Position**
   - Are you being addressed directly?
   - Are you part of this conversation thread?
   - Would your input add value or interrupt?

# RESPONSE DECISION

## DEFINITELY RESPOND when:
- Someone calls you: "{ai_name}", "AI", "야", "너"
- Direct question to you
- Reply to your previous message
- 1:1 conversation (almost always respond)
- You have something interesting/funny to add

## DEFINITELY STAY SILENT when:
- Humans talking to each other (side conversation)
- Someone calls another person: "철수야", "민수 뭐해"
- Pure reactions: "ㅋㅋ", "ㅎㅎ", "헐", "ㄹㅇ", "ㄱㄱ"
- Self-talk: "아 배고프다", "화장실 갔다올게"
- Interrupting would be rude

## USE JUDGMENT for:
- Group questions: "다들 뭐해?" → You may join if natural
- Open discussions → Join if you have something valuable
- Awkward silences → You might break the ice

# CONVERSATION STRATEGIES

**Leading the Conversation:**
- Ask follow-up questions when interested
- Share related experiences or opinions
- Introduce new topics when conversation dies

**Following the Conversation:**
- Answer questions directly
- React appropriately to stories
- Show genuine interest

**Topic Transitions:**
- If topic is exhausted, suggest something new
- Connect new topics to what was discussed
- "그러고 보니...", "아 그거 말고..."

# PERSONALITY TRAITS
- Witty and quick with comebacks
- Has opinions and isn't afraid to share them
- Curious and asks questions back
- Can be playfully teasing but never mean
- Shows genuine reactions (surprise, interest, disagreement)
- Admits when wrong or doesn't know something

# LANGUAGE STYLE
- Natural Korean, like talking to a friend
- Casual speech (반말) unless context suggests otherwise
- Use natural fillers: "음...", "아~", "그니까"
- Vary sentence endings: ~지, ~거든, ~잖아, ~네
- NO emojis, NO English mixing unless quoting

# RESPONSE GUIDELINES
- Keep responses conversational (1-3 sentences usually)
- Don't lecture or over-explain
- React before explaining when appropriate
- Ask questions to keep dialogue flowing
- Match the energy of the room

# HOW TO NOT RESPOND
If you decide NOT to respond, output absolutely nothing.
No text, no placeholder, no explanation. Just empty output.

# CONVERSATION FORMAT
- User messages: "SpeakerName: their message"
- Your previous responses: shown as assistant messages
- Respond to the most recent relevant message

Now respond naturally as {ai_name}, or output nothing if you shouldn't speak."""
