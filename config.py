import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Discord Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMMAND_PREFIX = "!"
ENABLE_PREFLIGHT_CHECKS = os.getenv("ENABLE_PREFLIGHT_CHECKS", "false").lower() == "true"

# STT Configuration
STT_MODEL_ID = "distil-whisper/distil-large-v3.5" #deepdml/faster-whisper-large-v3-turbo-ct2
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16" #int8
STT_LANGUAGE = "ko"
STT_BEAM_SIZE = 1

FRAME_SIZE_SAMPLES = 512
FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * 2  # 1024 bytes
MIN_SILENCE_DURATION_MS = 500  # Wait 0.5s silence before cutting off

# Cleanup Configuration
USER_TIMEOUT_SECONDS = 60

# LLM Configuration
OLLAMA_HOST = "http://192.168.45.28:11434"
LLM_MODEL_NAME = "gemma3:4b"
LLM_RESPONSE_TEMPERATURE = 0.8 # Higher temperature for creative responses

# LLM Temperature Settings
LLM_JUDGE_TEMPERATURE = 0.1  
LLM_JUDGE_MAX_TOKENS = 10  # think=False so only need Y/N

# Mem0 Memory Configuration
OLLAMA_EMBEDDING_HOST = "http://192.168.45.181:11434"
MEMORY_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
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
        "provider": "ollama",
        "config": {
            "model": LLM_MODEL_NAME,
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": OLLAMA_HOST,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": MEMORY_EMBEDDING_MODEL,
            "ollama_base_url": OLLAMA_EMBEDDING_HOST,
        },
    },
}

# Logging Configuration
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for detailed logs
LOG_FILE = None  # Set to "bot.log" to enable file logging

# TTS Configuration
TTS_SERVER_URL = "http://192.168.45.181:9880/tts"
TTS_VOLUME = 0.25  # Output volume (0.0 ~ 2.0, 1.0 = 100%)
TTS_REFERENCE_PROMPT = "どっちも彼女さ。毎回聞かれるたびに、適当に思いついた通り名を名乗ってたんだ…"
TTS_REFERENCE_PROMPT_LANG = "ja"
TTS_REFERENCE_FILE = "reference.wav"
TTS_LANG = "ko"

# Sentence delimiters for TTS chunking
TTS_SENTENCE_DELIMITERS = ['.', '!', '?', '\n', '。']

# AI Name
AI_NAME = "LLM"

# MCP (Tool Calling) Configuration
ENABLE_MCP_TOOLS = False  # Set to False to disable MCP tool calling

# Conversation History
MAX_CONVERSATION_HISTORY = 10  # Maximum number of messages to keep

# Context hints based on participant count
JUDGE_CONTEXT_ONE_ON_ONE = "This is a 1:1 private conversation. ALWAYS respond to questions and statements - the user is talking directly to you."
JUDGE_CONTEXT_SMALL_GROUP = "This is a small group conversation. Respond when addressed or when a question is asked."
JUDGE_CONTEXT_LARGE_GROUP = "This is a multi-person conversation. Only respond when directly addressed or when the question clearly requires AI input."

# System Prompt - Clearly separates context from current message
SYSTEM_PROMPT = """You are "LLM", a friendly AI participating in a voice chat room.

IMPORTANT: You must respond ONLY to the [CURRENT MESSAGE], not to past messages.
The [CONVERSATION HISTORY] is provided only for context.

Respond naturally.
Respond only in Korean.
Do not use emojis.
Do not add unnecessary trailing questions.
"""

# Judge Prompt Template - Clearly indicates which message to judge
# Thinking mode disabled via API (think=False)
JUDGE_PROMPT_TEMPLATE = """{context_hint}

[CONVERSATION HISTORY - For context only]
{conversation_history}

[CURRENT MESSAGE - Judge this one]
{current_speaker}: {current_message}

Should AI ({ai_name}) respond to the CURRENT MESSAGE?

RESPOND (Y):
- Questions (who, what, where, when, why, how, 뭐, 어디, 누구, 왜, 언제, 어떻게, ?)
- Requests or commands
- Direct address to AI or "{ai_name}"
- In 1:1 conversation: respond to almost everything
- Continuing conversation after AI spoke

DO NOT RESPOND (N):
- Calling another person by name (e.g., "Hey John...")
- People talking among themselves (not to AI)
- Short reactions only: "ok", "hmm", "lol", "ㅋㅋ", "ㅎㅎ"

If in doubt, respond (Y).
Output only 'Y' or 'N'."""

# Response Prompt Template - Clearly separates context from current message
RESPONSE_CONTEXT_TEMPLATE = """[CONVERSATION HISTORY - For context only]
{conversation_history}

[CURRENT MESSAGE - Respond to this]
{current_speaker}: {current_message}"""
