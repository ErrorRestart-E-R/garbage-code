"""
LLM Interface: LLM-based conversation judgment and response generation

Handles both Judge and Response generation with a single LLM.
- Judge: Decides whether to join the conversation (Y/N)
- Response: Generates response (streaming)

IMPORTANT: Separates conversation history from current message
to prevent LLM from responding to past messages.
"""

import ollama
import config
import json
import mcp_library
from typing import Tuple, AsyncGenerator
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


async def judge_conversation(conversation_history: str,
                              current_speaker: str,
                              current_message: str,
                              participant_count: int = 1) -> Tuple[str, str]:
    """
    Analyzes conversation to determine if AI should respond to the CURRENT message.
    
    Args:
        conversation_history: Past conversation history in JSON format (excludes current)
        current_speaker: Current speaker name
        current_message: Current message to judge
        participant_count: Number of participants
        
    Returns:
        (decision, reason)
        - decision: "Y" (respond immediately), "W" (wait and see), "N" (don't respond)
        - reason: Judgment reason (for debugging)
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        
        # Build system prompt (fixed rules)
        system_content = config.JUDGE_SYSTEM_PROMPT.format(ai_name=config.AI_NAME)
        
        # Build user prompt (dynamic context with history)
        user_content = config.JUDGE_USER_TEMPLATE.format(
            participant_count=participant_count,
            conversation_history=conversation_history,
            current_speaker=current_speaker,
            current_message=current_message,
            ai_name=config.AI_NAME
        )
        
        logger.debug(f"Judge context: participants={participant_count}")

        # Retry loop for robustness
        for attempt in range(config.JUDGE_MAX_RETRIES):
            try:
                response = await client.chat(
                    model=config.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    think=False,  
                    options={
                        "temperature": config.LLM_JUDGE_TEMPERATURE,
                        "num_predict": config.LLM_JUDGE_MAX_TOKENS
                    }
                )
                
                # Parse response
                if hasattr(response, 'message'):
                    result = response.message.content.strip()
                else:
                    result = response['message']['content'].strip()
                    
                clean_result = result.strip().upper()
                
                # Determine decision: Y, W, or N
                if "Y" in clean_result:
                    decision = "Y"
                    reason = "Judge: respond immediately"
                    logger.debug(f"Judge raw: '{result}' -> decision: {decision}")
                    return decision, reason
                elif "W" in clean_result:
                    decision = "W"
                    reason = "Judge: wait and see"
                    logger.debug(f"Judge raw: '{result}' -> decision: {decision}")
                    return decision, reason
                elif "N" in clean_result:
                    decision = "N"
                    reason = "Judge: do not respond"
                    logger.debug(f"Judge raw: '{result}' -> decision: {decision}")
                    return decision, reason
                else:
                    logger.warning(f"Judge invalid response (attempt {attempt+1}): {result}")
                    # Retry if invalid
            
            except Exception as e:
                logger.warning(f"Judge attempt {attempt+1} failed: {e}")
        
        # Fallback after all retries
        logger.error("Judge failed all retries, defaulting to N")
        return "N", "Judge failed all retries"
        
    except Exception as e:
        logger.error(f"Judge critical error: {e}")
        # Default to not responding on error
        return "N", f"Judge critical error: {e}"


async def get_response_stream(user_name: str,
                               user_text: str,
                               conversation_history: str,
                               memory_context: str = "") -> AsyncGenerator[str, None]:
    """
    Generates LLM response via streaming.
    Clearly separates past conversation from current message.
    
    Args:
        user_name: Current speaker name
        user_text: Current message to respond to
        conversation_history: Past conversation history JSON (excludes current)
        memory_context: Long-term memory context (optional)
        
    Yields:
        Response text chunks
    """
    try:
        client = ollama.AsyncClient(host=config.OLLAMA_HOST)
        tools = mcp_library.get_tools() if config.ENABLE_MCP_TOOLS else None
        
        # Build system prompt (fixed personality/rules only)
        system_content = config.SYSTEM_PROMPT
        
        # Add long-term memory if available (static context)
        if memory_context:
            system_content += f"\n\n[LONG-TERM MEMORY]\n{memory_context}"
        
        # Build user content with dynamic context (history + current message)
        context_content = config.RESPONSE_CONTEXT_TEMPLATE.format(
            conversation_history=conversation_history,
            current_speaker=user_name,
            current_message=user_text
        )
        
        # User message contains history + current message + instruction
        user_content = (
            f"{context_content}\n\n"
            "Use the CONVERSATION HISTORY only to interpret the CURRENT MESSAGE, "
            "and then respond ONLY to the CURRENT MESSAGE in Korean."
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Streaming request
        chat_kwargs = {
            "model": config.LLM_MODEL_NAME,
            "messages": messages,
            "stream": True,
            "think": False, 
            "options": {"temperature": config.LLM_RESPONSE_TEMPERATURE}
        }
        if tools:
            chat_kwargs["tools"] = tools
        
        tool_calls = []
        content_buffer = ""
        has_yielded = False
        
        async for part in await client.chat(**chat_kwargs):
            # Extract content
            if hasattr(part, 'message'):
                content = part.message.content or ""
                # Check for tool calls
                if hasattr(part.message, 'tool_calls') and part.message.tool_calls:
                    tool_calls = part.message.tool_calls
            else:
                content = part.get('message', {}).get('content', '')
                if 'tool_calls' in part.get('message', {}):
                    tool_calls = part['message']['tool_calls']
            
            if content:
                content_buffer += content
                has_yielded = True
                yield content
        
        # Process tool calls (only if no content was yielded)
        if tool_calls and not has_yielded:
            for tool_call in tool_calls:
                func_name, func_args = _get_tool_call_info(tool_call)
                
                if func_name:
                    logger.debug(f"Tool call: {func_name}({func_args})")
                    tool_result = mcp_library.execute_tool(func_name, func_args)
                    logger.debug(f"Tool result: {tool_result}")
                    
                    messages.append({
                        "role": "assistant",
                        "content": content_buffer,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
            
            # Generate final response with tool results
            async for part in await client.chat(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                stream=True,
                think=False,
                options={"temperature": config.LLM_RESPONSE_TEMPERATURE}
            ):
                if hasattr(part, 'message'):
                    content = part.message.content
                else:
                    content = part.get('message', {}).get('content')
                
                if content:
                    yield content
                    
    except Exception as e:
        logger.error(f"Response stream error: {e}")
        yield None


def _get_tool_call_info(tool_call) -> Tuple[str, dict]:
    """
    Extract function name and arguments from tool call
    """
    if hasattr(tool_call, 'function'):
        func = tool_call.function
        name = func.name if hasattr(func, 'name') else func.get('name', '')
        args = func.arguments if hasattr(func, 'arguments') else func.get('arguments', {})
    else:
        func = tool_call.get('function', {})
        name = func.get('name', '')
        args = func.get('arguments', {})
    
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    
    return name, args
