import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Discord Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"

# STT Configuration
STT_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"
STT_LANGUAGE = "ko"
STT_BEAM_SIZE = 1
STT_SILENCE_TIMEOUT = 0.5 # Seconds to wait after silence before processing

# VAD Configuration
VAD_REPO_OR_DIR = 'snakers4/silero-vad'
VAD_MODEL = 'silero_vad'
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 32 # Silero requires 32ms (512 samples)
FRAME_SIZE_SAMPLES = 512
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2 # 1024 bytes
RING_BUFFER_SIZE = 10 # ~320ms context

# Cleanup Configuration
USER_TIMEOUT_SECONDS = 60

# LLM Configuration
# Chat LLM (The main persona)
CHAT_API_BASE_URL = "http://localhost:1234/v1"
CHAT_API_KEY = "lm-studio"
CHAT_MODEL_NAME = "local-model" # Replace with specific model name if needed

# Judge LLM (The decision maker)
JUDGE_API_BASE_URL = "http://localhost:1234/v1"
JUDGE_API_KEY = "lm-studio"
JUDGE_MODEL_NAME = "local-model" # Can be a smaller/faster model

TTS_SERVER_URL = "http://192.168.45.49:9880/tts"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"
TTS_REFERENCE_FILE = "reference.wav"
TTS_LANG = "ko"
TTS_EARLY_CHUNKS = 1
TTS_EARLY_MIN_WORD_COUNT = 4
TTS_MIN_WORD_COUNT = 15

# Memory Configuration
MEMORY_DB_PATH = "./memory_db"
MEMORY_COLLECTION_NAME = "neuro_memory"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# LLM Timeouts
JUDGE_TIMEOUT = 5.0
IMPORTANCE_TIMEOUT = 10.0

# Context Configuration
MAX_HISTORY_MESSAGES = 10