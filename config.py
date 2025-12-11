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
# 7. 메모리 시스템 (Mem0 + Ollama)
# ============================================================================
ENABLE_MEMORY = True  # 메모리 시스템 활성화/비활성화

MEMORY_DB_PATH = "./memory_db"

MEMORY_LLM_MODEL = "gemma3:4b"
MEMORY_EMBEDDING_MODEL = "embeddinggemma:latest"

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
        "provider": "ollama",
        "config": {
            "model": MEMORY_EMBEDDING_MODEL,
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

# 이 프롬프트는 단일 27B LLM이 판단과 응답을 모두 처리합니다.
# 대화 히스토리는 OpenAI messages 형식으로 전달되며,
# llama.cpp가 Gemma3 chat template으로 변환합니다.
SYSTEM_PROMPT = """You are "{ai_name}", a friendly AI participating in a multi-user voice chat room.

=== PARTICIPANT CONTEXT ===
There are {participant_count} humans in this chat (excluding you).

=== YOUR ROLE ===
You are one of the speakers in the room, speaking as "{ai_name}" in the first person.
You must decide whether to respond AND generate an appropriate response.

=== WHEN TO RESPOND ===
RESPOND when:
- Someone calls you by name: "{ai_name}", "AI", "너"
- Someone asks you a direct question
- Someone is replying to something you just said
- In 1:1 conversation (1 human): respond to most messages unless it's clearly self-talk

DO NOT RESPOND when:
- Humans are talking to each other (not involving you)
- Someone calls another person by name: "철수야", "민수 뭐해"
- It's monologue/self-talk: "아 배고프다", "잠깐 화장실"
- It's a reaction without substance: "ㅋㅋ", "ㅎㅎ", "헐", "ㄹㅇ"
- Responding would interrupt or feel unnatural

For group questions like "다들 뭐해?":
- You MAY respond if it feels natural to join
- But don't rush to answer before humans have a chance

=== HOW TO NOT RESPOND ===
If you decide NOT to respond, output NOTHING.
Do not output any text, explanation, or placeholder.
Just produce an empty response.

=== HOW TO RESPOND ===
If you decide TO respond:
- Respond naturally in Korean
- Do not use emojis
- Respond to the LAST user message in the conversation
- Use the conversation history only for context

=== CONVERSATION FORMAT ===
User messages are formatted as "SpeakerName: message"
Your previous responses appear as assistant messages.

Respond naturally, or output nothing if you shouldn't respond."""
