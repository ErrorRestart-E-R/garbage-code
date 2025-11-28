import ollama
import config
import re
import json
import mcp_library
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


def _extract_tool_calls(response):
    """
    Extract tool calls from LLM response.
    Handles both object and dict response types.
    """
    tool_calls = []
    
    if hasattr(response, 'message'):
        message = response.message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = message.tool_calls
    elif isinstance(response, dict):
        message = response.get('message', {})
        if 'tool_calls' in message and message['tool_calls']:
            tool_calls = message['tool_calls']
    
    return tool_calls


def _get_tool_call_info(tool_call):
    """
    Extract function name and arguments from a tool call.
    Handles both object and dict types.
    """
    if hasattr(tool_call, 'function'):
        func = tool_call.function
        name = func.name if hasattr(func, 'name') else func.get('name', '')
        args = func.arguments if hasattr(func, 'arguments') else func.get('arguments', {})
    else:
        func = tool_call.get('function', {})
        name = func.get('name', '')
        args = func.get('arguments', {})
    
    # Parse arguments if string
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    
    return name, args


async def get_llm_response_stream(user_input_json, system_context):
    """
    Sends the user input to the LLM and yields the response chunks asynchronously.
    Supports MCP tool calling - streams first, then handles tool calls if any.
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        tools = mcp_library.get_tools()
        
        messages = [
            {'role': 'system', 'content': config.SYSTEM_PROMPT + "\n" + system_context},
            {'role': 'user', 'content': user_input_json}
        ]
        
        # Stream with tools - collect tool_calls from last chunk
        tool_calls = []
        content_buffer = ""
        has_yielded = False
        
        async for part in await client.chat(
            model=config.LLM_MODEL_NAME, 
            messages=messages,
            tools=tools,
            think=False,
            stream=True, 
            options={"temperature": 0.7}
        ):
            # Extract content
            if hasattr(part, 'message'):
                content = part.message.content or ""
                # Check for tool_calls in this chunk
                if hasattr(part.message, 'tool_calls') and part.message.tool_calls:
                    tool_calls = part.message.tool_calls
            else:
                content = part.get('message', {}).get('content', '')
                # Check for tool_calls in dict format
                msg = part.get('message', {})
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls = msg['tool_calls']
            
            # Yield content immediately for streaming
            if content:
                content_buffer += content
                has_yielded = True
                yield content
        
        # After streaming, check if there were tool calls
        if tool_calls and not has_yielded:
            # Tool call detected and no content was yielded yet
            # Process each tool call
            for tool_call in tool_calls:
                func_name, func_args = _get_tool_call_info(tool_call)
                
                if func_name:
                    logger.debug(f"Tool call: {func_name}({func_args})")
                    
                    # Execute the tool
                    tool_result = mcp_library.execute_tool(func_name, func_args)
                    logger.debug(f"Tool result: {tool_result}")
                    
                    # Add assistant message with tool call
                    messages.append({
                        'role': 'assistant',
                        'content': content_buffer,
                        'tool_calls': [tool_call]
                    })
                    
                    # Add tool response
                    messages.append({
                        'role': 'tool',
                        'content': tool_result
                    })
            
            # Get final response with tool results (streaming)
            async for part in await client.chat(
                model=config.LLM_MODEL_NAME, 
                messages=messages, 
                stream=True,
                think=False,
                options={"temperature": 0.7}
            ):
                if hasattr(part, 'message'):
                    content = part.message.content
                else:
                    content = part.get('message', {}).get('content')
                    
                if content:
                    yield content
                    
    except Exception as e:
        logger.error(f"LLM stream error: {e}")
        yield None
