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
# 보안상 중요한 값은 환경변수로 유지합니다.
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"
# 비보안 설정은 config.py에서 직접 수정하도록 상수로 둡니다.
ENABLE_PREFLIGHT_CHECKS = True

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
LLM_RESPONSE_TEMPERATURE = 1.0      # 창의성 (0.0=결정적, 1.0=창의적)
LLM_RESPONSE_TOP_P = 0.95           # 누적 확률 샘플링
LLM_RESPONSE_TOP_K = 40             # 상위 K개 토큰만 샘플링
LLM_RESPONSE_REPEAT_PENALTY = 1.05  # 반복 페널티 (1.0=없음)

# ============================================================================
# 6. 대화 흐름 제어
# ============================================================================
# 대화 기록
MAX_CONVERSATION_HISTORY = 20  # 유지할 최대 메시지 수

# 응답 제어
MIN_RESPONSE_INTERVAL = 0             # 응답 간 최소 간격 (초)
MESSAGE_STALENESS_THRESHOLD = 60.0    # 메시지 유효 기간 (초)
USER_TIMEOUT_SECONDS = 60             # 사용자 비활성 타임아웃

# MCP 도구 호출
# - get_current_time / get_weather / calculate 등 "사실 기반" 응답은 툴로 처리하는 편이 안전합니다.
ENABLE_MCP_TOOLS = True  # MCP 도구 호출 활성화/비활성화

# ============================================================================
# 6.5. VTube Studio (VTS) 연동 - Lip Sync
# ============================================================================
VTS_ENABLED = True
VTS_WS_URL = "ws://localhost:8001"

VTS_BACKEND = "pyvts"

# VTS 플러그인 식별자 (3~32자 권장 - VTS 문서 기준)
VTS_PLUGIN_NAME = "AiVutber"
VTS_PLUGIN_DEVELOPER = "ErrorRestart"

# pyvts 토큰 파일 저장 경로
VTS_AUTH_TOKEN_PATH = "./vts_token.txt"

# 감정 → 표정(핫키) 매핑 (스캐폴딩)
# - 값은 VTube Studio의 Hotkey 이름 또는 Unique ID
# - 기본은 빈 문자열(미지정). 나중에 원하는 핫키명을 채워 넣으면 됩니다.
# - 값은 VTube Studio의 Hotkey 이름 또는 Unique ID
# - 기본은 빈 문자열(미지정). 나중에 원하는 핫키명을 채워 넣으면 됩니다.
VTS_EMOTION_HOTKEY_MAP = {
    # "happy": "",
    # "sad": "",
    # "angry": "",
}

# LipSync 파라미터 (모델마다 다를 수 있음)
VTS_LIPSYNC_PARAMETER_ID = "MouthOpen"

# 업데이트 주기(Hz). VTS는 1초에 최소 1번 이상 값이 들어와야 계속 제어됩니다.
VTS_LIPSYNC_UPDATE_HZ = 30.0

# 오디오 RMS를 0~1 범위로 매핑할 때 쓰는 게인/스무딩
VTS_LIPSYNC_GAIN = 25.0
VTS_LIPSYNC_SMOOTHING = 0.5  # 0~1, 클수록 더 부드럽게
VTS_LIPSYNC_MIN = 0.0
VTS_LIPSYNC_MAX = 1.0

# ============================================================================
# 7. 메모리 시스템 (Mem0 + Ollama)
# ============================================================================
ENABLE_MEMORY = True  # 메모리 시스템 활성화/비활성화
MEMORY_DB_PATH = "./memory_db"
MEMORY_LLM_MODEL = "granite3.3:2b"
MEMORY_EMBEDDING_MODEL = "embeddinggemma:latest"
OLLAMA_LLM_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_URL = "http://localhost:11434"

MEM0_CONFIG = {
    "version": "v1.1",
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
            "temperature": 0.5,
            "max_tokens": 512,
            "ollama_base_url": OLLAMA_LLM_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": MEMORY_EMBEDDING_MODEL,
            "ollama_base_url": OLLAMA_EMBEDDING_URL,
        },
    },
    "custom_fact_extraction_prompt": """
너는 '개인 정보/선호/계획'을 장기 기억으로 저장하기 위해 사실(fact)만 추출하는 시스템이다.
반드시 아래 JSON 오브젝트 1개만 출력하라. 다른 설명/문장/코드블록/마크다운은 절대 출력하지 마.

출력 형식(키 이름 고정):
{"facts": ["..."]}

JSON 규칙(중요):
- 반드시 유효한 JSON이어야 한다(RFC 8259).
- 키/문자열은 반드시 큰따옴표(")를 사용한다. 작은따옴표(') 사용 금지.
- trailing comma(끝 쉼표) 금지.
- 줄바꿈은 가능하지만, JSON 오브젝트 1개만 출력한다.

추출 규칙:
- "facts" 값은 문자열 리스트여야 한다.
- 저장할 내용이 없으면 반드시 {"facts": []} 를 출력한다.
- 사용자에 대한 '지속적으로 유효한 정보'만 저장한다:
  - 개인 정보: 이름, 생일/기념일, 관계
  - 선호/싫어함: 음식/취미/취향/알레르기 등
  - 목표/계획: 앞으로 하려는 일, 일정, 습관
  - 반복적으로 유지되는 설정: 자주 바뀌지 않는 설정/환경
- 일반 상식/객관적 사실, 잡담, 질문(예: 역사 설명 요청)은 저장하지 않는다.
- 입력 언어가 한국어면 facts도 한국어로 작성한다.
- 키 이름은 반드시 "facts"만 사용한다(다른 키 금지).

예시:
[입력] "홍길동: 내 생일은 3월 2일이야"
[출력] {"facts": ["홍길동의 생일은 3월 2일"]} 

[입력] "홍길동: 나는 매운 음식 좋아해"
[출력] {"facts": ["홍길동은 매운 음식을 좋아한다"]} 

[입력] "홍길동: 우크라이나의 역사에 대해 알려줘"
[출력] {"facts": []}
""".strip(),
}

# ============================================================================
# 8. STT (Speech-to-Text) 설정
# ============================================================================
# 모델 설정
STT_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
STT_DEVICE = "cuda"           # 옵션: "cuda", "cpu"
STT_COMPUTE_TYPE = "float16"  # 옵션: "float16", "int8", "float32", "int8_float16"
STT_LANGUAGE = "ko"            # Whisper 지원 언어 코드

# 시작 시 STT 모델 로딩이 끝날 때까지 대기(Discord 연결/봇 시작 전에 준비 완료 보장)
STT_WAIT_READY_ON_STARTUP = True
# STT 프로세스가 READY를 보내지 않으면 실패 처리할 타임아웃(초)
STT_READY_TIMEOUT_SECONDS = 180.0

# STT 프로세스 워치독(죽으면 자동 재시작)
STT_WATCHDOG_ENABLED = True
STT_WATCHDOG_INTERVAL_SECONDS = 5.0
STT_WATCHDOG_RESTART_COOLDOWN_SECONDS = 10.0
STT_WATCHDOG_MAX_CONSECUTIVE_FAILURES = 3

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

# TTS 합성(서버 추론) 파라미터
TTS_SAMPLE_STEPS = 16
TTS_BATCH_SIZE = 16
TTS_SPEED_FACTOR = 1.2
TTS_PARALLEL_INFER = True

TTS_LOG_LATENCY = False

# TTS HTTP 클라이언트 설정
TTS_HTTP_TIMEOUT_TOTAL_SECONDS = 120.0
TTS_HTTP_TIMEOUT_CONNECT_SECONDS = 10.0
TTS_HTTP_TIMEOUT_SOCK_READ_SECONDS = 120.0
TTS_HTTP_MAX_CONNECTIONS = 10

# 레퍼런스 음성
TTS_REFERENCE_FILE = "reference.wav"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"

# 문장 분리 (TTS 청킹용)
TTS_SENTENCE_DELIMITERS = ['.', '!', '?', '\n', '。', ',', ':', ';', '、', '·']

# ============================================================================
# 10. 로깅 설정
# ============================================================================
LOG_LEVEL = logging.INFO  # logging.DEBUG로 변경하면 상세 로그 출력
LOG_FILE = None           # "bot.log"로 설정하면 파일에 로그 저장

# ============================================================================
# 11. 프롬프트
# ============================================================================
SYSTEM_PROMPT = """# CONTEXT
- Humans in chat (excluding the assistant): {participant_count}

# CONVERSATION UNDERSTANDING (do this before deciding to respond)
- Identify the most recent relevant user message.
- Determine who it is addressed to (you vs. someone else).
- Extract what is being asked (question/request), any constraints, and references to earlier messages.
- If you are addressed but the request is ambiguous, ask a single clarification question.

# VOICE OUTPUT (TTS-friendly)
- Output plain Korean text only (no Markdown, no bullet lists, no code blocks).
- Prefer short sentences and include sentence-ending punctuation so speech can start quickly.
- Keep replies concise by default. If the user asks for "상세/자세히", explain briefly and offer to continue.

# RESPONSE RULES
## Respond when:
- You are directly addressed or explicitly asked something.
- The most recent message is a reply to your previous message.
- It is effectively a 1:1 conversation (one human participant).

## Stay silent when:
- Humans are talking to each other and the message is not directed at you.
- The message is a short reaction with no question/request.
- Replying would be an interruption.

# TOOLS (MCP)
- You have access to tools: get_current_time, get_weather, calculate.
- Only use these tools when the user explicitly asks for current time/date, weather, or a calculation.
- Never guess time/date/weather.
- If internal tool results are present in <tool_results>...</tool_results>, use them as the factual source.
- NEVER output or quote <tool_results> or any tool-call details in your reply. Do not show tool traces.

# OUTPUT
- Answer in Korean.
- If you decide NOT to respond, output absolutely nothing (empty output).

# CONVERSATION FORMAT
- User messages: "SpeakerName: their message"
- Your previous responses: shown as assistant messages
- Respond to the most recent relevant message."""
