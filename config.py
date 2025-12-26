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
# 5.5. KTANE 모드 전용 응답 파라미터 (결정성/정확도 우선)
# ============================================================================
# KTANE(폭탄 해체)에서는 규칙 적용이 핵심이라, 일반 잡담보다 temperature를 낮추는 편이 안전합니다.
# llama.cpp 서버/모델에 따라 값은 튜닝 포인트입니다.
KTANE_LLM_TEMPERATURE = 0.2
KTANE_LLM_TOP_P = 0.9
KTANE_LLM_TOP_K = 40
KTANE_LLM_REPEAT_PENALTY = 1.05

# KTANE 모드에서 컨텍스트가 커져 llama.cpp가 거절할 때를 대비한 히스토리 트림 기준(메시지 개수)
KTANE_MAX_HISTORY_MESSAGES = 4

# ============================================================================
# 6. 대화 흐름 제어
# ============================================================================
# 대화 기록
MAX_CONVERSATION_HISTORY = 20  # 유지할 최대 메시지 수

# 응답 제어
MIN_RESPONSE_INTERVAL = 0             # 응답 간 최소 간격 (초)
MESSAGE_STALENESS_THRESHOLD = 60.0    # 메시지 유효 기간 (초)
USER_TIMEOUT_SECONDS = 60             # 사용자 비활성 타임아웃

# 바지인(barge-in) / 턴 합치기(voice turn consolidation)
# - LLM 응답 생성/재생 중 사용자 발화가 들어오면, 재생 전 단계(PREPLAY)에서는 현재 응답을 취소하고
#   최근 사용자 발화를 합쳐 다시 응답을 생성합니다.
# - 이미 음성 재생이 시작된 경우(PLAYING)에는 끊지 않고, 재생이 끝난 뒤 한 번에 처리합니다.
BARGE_IN_ENABLED = True
# 재생 전 바지인 시, 추가 발화를 모아 하나의 턴으로 확정하기 위한 디바운스(ms)
BARGE_IN_MERGE_WINDOW_MS = 500
# 너무 짧은 발화(잡음/추임새)로 바지인이 과도하게 트리거되는 것을 방지
BARGE_IN_MIN_CHARS = 3
# 선택: 무시할 짧은 발화 패턴(비워두면 미사용). 예: r\"^(어|음|잠깐)$\"
BARGE_IN_IGNORE_REGEX = ""

# MCP 도구 호출
# - get_current_time / get_weather / calculate 등 "사실 기반" 응답은 툴로 처리하는 편이 안전합니다.
ENABLE_MCP_TOOLS = True  # MCP 도구 호출 활성화/비활성화

# ============================================================================
# 6.2. GAME HUB (HTTP) - 게임 모드 플러그인 서비스
# ============================================================================
# 메인 코어는 GameHub에 HTTP로 연결해 게임별 프롬프트 패치(system_addendum/context_blocks)를 받습니다.
# GameHub는 별도 프로세스로 실행됩니다.
GAME_HUB_ENABLED = True
GAME_HUB_BASE_URL = "http://127.0.0.1:8765"
GAME_HUB_HTTP_TIMEOUT_TOTAL_SECONDS = 2.5
GAME_HUB_HTTP_TIMEOUT_CONNECT_SECONDS = 0.6
# GameHub에서 가져온 컨텍스트 블록을 시스템 프롬프트에 주입할 때의 최대 문자 수
GAME_HUB_RAG_MAX_CONTEXT_CHARS = 6000

# llama.cpp가 컨텍스트 초과를 반환할 때, 자동 재시도 시 남길 대화 히스토리 메시지 수
LLM_MAX_HISTORY_MESSAGES_ON_OVERFLOW = 10

# ============================================================================
# 6.25. GAME MODE: Keep Talking and Nobody Explodes (KTANE) - Manual RAG
# ============================================================================
# - 평소에는 일반 잡담 모드로 동작합니다.
# - KTANE_GAME_MODE_ENABLED=True 일 때만, 로컬에 저장된 "텍스트로 변환된 매뉴얼"을
#   RAG(검색) 컨텍스트로 LLM에 주입하여 해체 방법을 안내합니다.
#
# IMPORTANT:
# - 유저는 말로 상태를 설명하고, LLM은 "로컬 매뉴얼 텍스트"에서 근거를 찾아 안내합니다.
KTANE_GAME_MODE_ENABLED = True

# 매뉴얼 텍스트 파일 경로(여러 개 가능). PDF를 직접 읽지 않고, 사용자가 텍스트로 변환한 파일을 사용합니다.
# 예: ["./ktane_manual.txt"] 또는 ["./manuals/ktane_1.txt", "./manuals/ktane_2.txt"]
KTANE_MANUAL_TEXT_PATHS = ["./ktane_manual.txt"]

# RAG 검색 파라미터
KTANE_RAG_TOP_K = 4  # 검색으로 가져올 청크 수
KTANE_RAG_MAX_CONTEXT_CHARS = 6000  # 시스템 프롬프트에 주입할 최대 문자 수(과도한 토큰 방지)

# 임베딩 설정 (KTANE 매뉴얼 검색용)
# - provider="auto": KTANE_EMBEDDING_MODEL 값 형태로 자동 선택
# - provider="ollama": OLLAMA_EMBEDDING_URL(기존 Mem0 embedder와 동일)로 임베딩 요청
# - provider="sentence_transformers": 로컬 SentenceTransformers로 임베딩(최초 실행 시 모델 다운로드 필요)
KTANE_EMBEDDING_PROVIDER = "auto"  # "auto" | "ollama" | "sentence_transformers"
KTANE_EMBEDDING_MODEL = "embeddinggemma:latest"

# 게임 모드에서는 잡담/개인정보 메모리 시스템에 게임 내용이 섞이지 않도록 기본적으로 저장을 끕니다.
# (필요하면 True로 바꾸세요.)
KTANE_MEMORY_SAVE_ENABLED = True

# KTANE 모드에서 장기기억( Mem0 ) 컨텍스트를 시스템 프롬프트에 주입할지 여부
# - 기본 False 권장: 과거 오답/잡담이 규칙 적용을 오염시키는 것을 방지
KTANE_INJECT_LONG_TERM_MEMORY = False

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
OLLAMA_EMBEDDING_URL = "http://192.168.45.28:11434"

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
            # Fact extraction should be deterministic to avoid duplicated / hallucinated facts.
            "temperature": 0.0,
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
- facts 내에서 같은 내용은 절대 중복하지 않는다(중복 fact는 1개만).
- 질문/요청/명령문(예: "?","알려줘","기억해")은 저장하지 않는다: {"facts": []}
- 사용자에 대한 '지속적으로 유효한 정보'만 저장한다:
  - 개인 정보: 이름, 생일/기념일, 관계
  - 선호/싫어함: 음식/취미/취향/알레르기 등
  - 목표/계획: 앞으로 하려는 일, 일정, 습관
  - 반복적으로 유지되는 설정: 자주 바뀌지 않는 설정/환경
- 일반 상식/객관적 사실, 잡담, 질문(예: 역사 설명 요청)은 저장하지 않는다.
- 입력 문장에서 중요한 정보 만 요약해서 짦게 저장한다.
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
STT_TEMPERATURE = 0.0

# 품질 필터링 임계값
# compression_ratio_threshold: 0.0~∞ (기본값=2.4, 높은 압축률=환각 가능성)
STT_COMPRESSION_RATIO_THRESHOLD = 2.4
# log_prob_threshold: -∞~0.0 (기본값=-1.0, 낮은 로그확률=저품질)
STT_LOG_PROB_THRESHOLD = -1.0
# no_speech_threshold: 0.0~1.0 (기본값=0.6, 낮을수록 민감하게 무음 감지)
STT_NO_SPEECH_THRESHOLD = 0.2

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
# min_audio_seconds: 전사로 넘길 최소 발화 길이(초). 너무 짧은 잡음/클릭/호흡을 STT에 넘기지 않기 위함.
# 16kHz 16bit mono 기준: bytes ~= seconds * 16000 * 2
STT_MIN_AUDIO_SECONDS = 0.2
# min_rms_threshold: 0.0~1.0 (RMS 에너지, 너무 작은 소리/잡음은 무시)
# Discord 환경에서 0.01은 너무 민감할 수 있어 기본값을 약간 올립니다.
STT_MIN_RMS_THRESHOLD = 0.06

# STT 후처리(환각/잡음 억제) 파라미터
# - segment.no_speech_prob 가 이 값보다 크면(=무음 가능성 큼) 결과를 폐기
STT_POST_FILTER_NO_SPEECH_MARGIN = 0.15
# - segment.avg_logprob 평균이 너무 낮으면(=저품질/환각 가능성) 결과를 폐기 (faster-whisper가 제공할 때만 적용)
STT_POST_FILTER_MIN_AVG_LOGPROB = -0.85

# 이전 세그먼트 텍스트를 컨텍스트로 사용할지 여부.
# 잡음/짧은 소리에서 "감사합니다" 같은 이전 문장 끌려오는 현상을 줄이려면 False가 유리합니다.
STT_CONDITION_ON_PREVIOUS_TEXT = False

# ============================================================================
# 9. TTS (Text-to-Speech) 설정
# ============================================================================
TTS_SERVER_URL = "http://192.168.45.181:9880/tts"
TTS_VOLUME = 0.25  # 출력 볼륨 
TTS_LANG = "ko"    # 출력 언어

# TTS 합성(서버 추론) 파라미터
TTS_SAMPLE_STEPS = 4
TTS_BATCH_SIZE = 8
TTS_SPEED_FACTOR = 1.2
TTS_PARALLEL_INFER = False

TTS_LOG_LATENCY = False

# TTS HTTP 클라이언트 설정
TTS_HTTP_TIMEOUT_TOTAL_SECONDS = 120.0
TTS_HTTP_TIMEOUT_CONNECT_SECONDS = 10.0
TTS_HTTP_TIMEOUT_SOCK_READ_SECONDS = 120.0
TTS_HTTP_MAX_CONNECTIONS = 10

# 레퍼런스 음성
TTS_REFERENCE_FILE = "reference.wav"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ"
TTS_REFERENCE_PROMPT_LANG = "ja"

# 문장 분리 (TTS 청킹용)
TTS_SENTENCE_DELIMITERS = ['.', '!', '?', '\n', '。', ',', ':', ';', '、', '·']

# ============================================================================
# 10. 로깅 설정
# ============================================================================
LOG_LEVEL = logging.INFO  # logging.DEBUG로 변경하면 상세 로그 출력
LOG_FILE = None           # "bot.log"로 설정하면 파일에 로그 저장

# 파이프라인 레이턴시 측정 로그 (STT / mem 검색 / LLM / TTS 첫 오디오 / mem0 저장)
# - True: 각 턴 종료 시 한 줄 요약 로그 출력 + mem0 저장 시 별도 한 줄 로그 출력
# - False: 레이턴시 요약 로그 비활성화
PIPELINE_METRICS_ENABLED = True

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
