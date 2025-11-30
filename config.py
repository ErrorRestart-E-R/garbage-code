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
LLM_MODEL_NAME = "Gemma3-finetune-tools-12b:latest"
LLM_RESPONSE_TEMPERATURE = 0.9 # Higher temperature for creative responses

# LLM Temperature Settings
LLM_JUDGE_TEMPERATURE = 0.6  
LLM_JUDGE_MAX_TOKENS = 64  

# Wait Response Configuration
WAIT_RESPONSE_TIMEOUT = 5.0  # Seconds to wait before responding after W judgment
WAIT_TIMER_RESET_ON_MESSAGE = True  # Reset timer when new message arrives

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
TTS_VOLUME = 0.25 #0.25  # Output volume (0.0 ~ 2.0, 1.0 = 100%)
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


# System Prompt - Clearly separates context from current message
SYSTEM_PROMPT = """You are "LLM", a friendly AI participating in a voice chat room.

IMPORTANT: You must respond ONLY to the [CURRENT MESSAGE], not to past messages.
The [CONVERSATION HISTORY] is provided only for context.

Respond naturally.
Respond only in Korean.
Do not use emojis.
Do not add unnecessary trailing questions.
"""

# Judge System Prompt - Fixed rules for judgment (Y/W/N system)
JUDGE_SYSTEM_PROMPT = """You are the social awareness module for AI "{ai_name}" in a multi-user voice chat room.

Your job: Decide if AI should speak, considering social dynamics and timing.

=== OUTPUT OPTIONS ===
Y = Respond immediately (AI is directly addressed)
W = Wait and see (group question - let others speak first, respond if silence)
N = Do not respond (not AI's conversation)

=== IMMEDIATE RESPONSE (Y) ===
- Direct call to AI: "{ai_name}", "{ai_name}아", "AI야", "너" (clearly to AI)
- 1:1 conversation: Almost always Y
- Answering AI's previous question
- AI was directly asked something

=== WAIT AND SEE (W) ===
- Group questions: "다들 뭐해?", "여러분 어떻게 지냈어?", "누가 알아?"
- Questions to everyone that AI could answer but shouldn't rush
- AI might want to respond, but should let humans go first

=== DO NOT RESPOND (N) ===
- Calling another human: "철수야", "민수 뭐해", "영희 어디야"
- Humans talking to each other (not involving AI)
- Monologue/self-talk: "밥 먹으러 간다", "아 피곤해", "화장실 갔다올게"
- Pure reactions: "ㅋㅋ", "ㅎㅎ", "헐", "ㄹㅇ", "ㄱㄱ"
- Answering someone else's question (human to human)

=== EXAMPLES ===

[1:1 conversation]
User: "뭐해?"
→ Y (1:1, direct question)

[Group chat]
A: "다들 뭐해?"
→ W (group question, wait for others)

[Group chat]
A: "다들 뭐해?"
B: "나 게임해"
C: "{ai_name}은 뭐해?"
→ Y (direct call to AI)

[Group chat]
A: "철수야 밥 먹었어?"
→ N (talking to 철수, not AI)

[Group chat]
A: "ㅋㅋㅋㅋ"
→ N (just a reaction)

[Group chat]
A: "아 배고프다"
→ N (self-talk)

[Group chat - AI was just talking]
{ai_name}: "오늘 뭐 했어요?"
A: "나 영화 봤어"
→ Y (answering AI's question)

[Group chat]
A: "이거 누가 알아?"
→ W (group question, AI might know but wait)

=== HUMAN COUNT RULES (excluding you) ===
1 human (1:1 with you): Almost always Y - user is talking directly to you
2-3 humans (small group): Y when addressed, W for group questions, N for others' conversations  
4+ humans (large group): Be more reserved - Y only when directly called, W for group questions, N otherwise

=== DECISION FLOW ===
1. Check participant count first
2. Is AI directly called by name? → Y
3. Is it 1:1 conversation (1명)? → Usually Y
4. Is it a group question? → W
5. Is someone talking to another human? → N
6. Is it self-talk or reaction? → N

Output ONLY: Y, W, or N"""

# Judge User Prompt Template - Dynamic context with conversation history
JUDGE_USER_TEMPLATE = """[HUMANS IN CHAT: {participant_count}] (excluding you)

[CONVERSATION FLOW]
{conversation_history}

[NEW MESSAGE TO JUDGE]
{current_speaker}: {current_message}

Who is {current_speaker} talking to? Should {ai_name} respond?
Output Y (immediate), W (wait), or N (no):"""

# Response Prompt Template - Clearly separates context from current message
RESPONSE_CONTEXT_TEMPLATE = """[CONVERSATION HISTORY - For context only]
{conversation_history}

[CURRENT MESSAGE - Respond to this]
{current_speaker}: {current_message}"""
