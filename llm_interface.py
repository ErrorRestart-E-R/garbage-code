import ollama
import config
import asyncio
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# System Prompt
SYSTEM_PROMPT = """
You are "Neuro", an AI VTuber. 

Do not try to include motion or any other non-textual content.
Do not try to include emojis.
Do not try to include trailing questions if not necessary.
Please respond in Korean only.
"""

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = """
You are a conversation analyzer for "Neuro", an AI VTuber.
Your job is to decide if Neuro should join the conversation.
Respond with ONLY 'Y' (Yes) or 'N' (No).

Neuro is curious, friendly, and likes to chat.
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

# Importance Judge Prompt
IMPORTANCE_SYSTEM_PROMPT = """
You are a memory assistant. Your job is to decide if a user's message contains important information worth saving to long-term memory.
Important information includes:
- Personal details (name, age, location, job).
- Preferences (likes, dislikes, hobbies, favorites).
- Specific facts about the user's life or history.
- Important context for future conversations.

Respond with ONLY 'Y' (Yes) or 'N' (No).
"""

async def should_respond(user_input, system_context):
    """
    Decides whether to respond to the user input using the Judge LLM.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT + "\n" + system_context},
            {'role': 'user', 'content': user_input}
        ]
        
        response = await client.chat(model=config.LLM_MODEL_NAME, messages=messages, options={"temperature": 0.1})
        
        # Handle both object and dict response types
        if hasattr(response, 'message'):
            response_text = response.message.content.strip().upper()
        else:
            response_text = response['message']['content'].strip().upper()
            
        return "Y" in response_text
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return False

async def is_important(user_input):
    """
    Decides if the input is worth saving to memory.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': IMPORTANCE_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_input}
        ]
        
        response = await client.chat(model=config.LLM_MODEL_NAME, messages=messages, options={"temperature": 0.1})
        
        # Handle both object and dict response types
        if hasattr(response, 'message'):
            response_text = response.message.content.strip().upper()
        else:
            response_text = response['message']['content'].strip().upper()
            
        return "Y" in response_text
    except Exception as e:
        logger.debug(f"Importance judge error: {e}")
        return False


async def get_neuro_response_stream(user_input_json, system_context):
    """
    Sends the user input to the LLM and yields the response chunks asynchronously.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT + "\n" + system_context},
            {'role': 'user', 'content': user_input_json}
        ]
        
        async for part in await client.chat(model=config.LLM_MODEL_NAME, messages=messages, stream=True, options={"temperature": 0.7}):
            # Handle both object and dict response types
            if hasattr(part, 'message'):
                content = part.message.content
            else:
                content = part.get('message', {}).get('content')
                
            if content:
                yield content
                    
    except Exception as e:
        logger.error(f"LLM stream error: {e}")
        yield None
