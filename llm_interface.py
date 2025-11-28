import ollama
import config
import re
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)



async def should_respond(user_input, system_context):
    """
    Decides whether to respond to the user input using the Judge LLM.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': config.JUDGE_SYSTEM_PROMPT + "\n" + system_context},
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

PERSONAL_KEYWORDS = [
    "이름", "나이", "사는", "사는곳", "직업", "회사", "학교", "취미", "좋아", "싫어",
    "favorite", "job", "work", "hobby", "age", "live", "location", "address"
]

def _looks_like_personal_fact(text: str) -> bool:
    lowered = text.lower()
    if any(keyword in lowered for keyword in PERSONAL_KEYWORDS):
        return True
    if re.search(r"\d{2}\s*살", text):
        return True
    if re.search(r"\d{4}", text):
        return True
    return False


async def is_important(user_input):
    """
    Lightweight heuristic check to decide if the input is worth saving.
    Avoids an additional LLM call to keep latency low.
    """
    if not user_input:
        return False
    
    text = user_input.strip()
    if len(text) < 10:
        return False
    
    word_count = len(text.split())
    if word_count >= 6 and _looks_like_personal_fact(text):
        return True
    
    return _looks_like_personal_fact(text)

async def get_llm_response_stream(user_input_json, system_context):
    """
    Sends the user input to the LLM and yields the response chunks asynchronously.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': config.SYSTEM_PROMPT + "\n" + system_context},
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
