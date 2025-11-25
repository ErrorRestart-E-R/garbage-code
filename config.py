import os
from dotenv import load_dotenv
import logging

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
# Ollama SDK Configuration
OLLAMA_HOST = "http://localhost:11434"

# LLM Model Configuration
# Specify the model to use with Ollama
LLM_MODEL_NAME = "gemma3:27b" 

# Logging Configuration
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for detailed logs
LOG_FILE = None  # Set to "bot.log" to enable file logging

# TTS Configuration
TTS_SERVER_URL = "http://192.168.45.49:9880/tts"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"
TTS_REFERENCE_FILE = "reference.wav"
TTS_LANG = "ko"