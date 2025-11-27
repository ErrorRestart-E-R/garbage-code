import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Discord Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"
# Optional startup diagnostics
ENABLE_PREFLIGHT_CHECKS = os.getenv("ENABLE_PREFLIGHT_CHECKS", "false").lower() == "true"

# STT Configuration
STT_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"
STT_LANGUAGE = "ko"
STT_BEAM_SIZE = 1

FRAME_SIZE_SAMPLES = 512
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2 # 1024 bytes
MIN_SILENCE_DURATION_MS = 500 # Wait 0.5s silence before cutting off

# Cleanup Configuration
USER_TIMEOUT_SECONDS = 60

# LLM Configuration
# Ollama SDK Configuration
OLLAMA_HOST = "http://192.168.45.28:11434"

# LLM Model Configuration
# Specify the model to use with Ollama
LLM_MODEL_NAME = "gemma3:4b" 

# Logging Configuration
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for detailed logs
LOG_FILE = None  # Set to "bot.log" to enable file logging

# TTS Configuration
TTS_SERVER_URL = "http://192.168.45.49:9880/tts"
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"
TTS_REFERENCE_FILE = "reference.wav"
TTS_LANG = "ko"

# AI Identity
AI_NAME = "LLM"

# System Prompts
SYSTEM_PROMPT = """
You are "LLM", an AI Assistant. 

Do not try to include motion or any other non-textual content.
Do not try to include emojis.
Do not try to include trailing questions if not necessary.
Please respond in Korean only.
"""

JUDGE_SYSTEM_PROMPT = """
You are a conversation analyzer for "LLM", an AI Assistant.
Your job is to decide if LLM should join the conversation.
Respond with ONLY 'Y' (Yes) or 'N' (No).

LLM is curious, friendly, and likes to chat.
She SHOULD respond if:
- The user is talking to her (obviously).
- The topic is interesting, funny, or something she can comment on.
- The user is expressing an opinion or asking a general question.
- She wants to join the banter.

She should NOT respond ONLY if:
- The input is just noise or very short (e.g. "ok", "hmm").
- The users are having a strictly private or technical conversation that doesn't concern her.

BE MORE PROACTIVE. If in doubt, say 'Y'.
"""

IMPORTANCE_SYSTEM_PROMPT = """
You are a memory assistant. Your job is to decide if a user's message contains important information worth saving to long-term memory.
Important information includes:
- Personal details (name, age, location, job).
- Preferences (likes, dislikes, hobbies, favorites).
- Specific facts about the user's life or history.
- Important context for future conversations.

Respond with ONLY 'Y' (Yes) or 'N' (No).
"""