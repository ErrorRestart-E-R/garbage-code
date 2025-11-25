import ollama
import config
import asyncio
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

async def is_important(user_input):
    """
    Decides if the input is worth saving to memory.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        messages = [
            {'role': 'system', 'content': config.IMPORTANCE_SYSTEM_PROMPT},
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
