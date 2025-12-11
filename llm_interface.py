"""
LLM Interface: LLM-based conversation judgment and response generation

Handles both Judge and Response generation with a single LLM.
- Judge: Decides whether to join the conversation (Y/N)
- Response: Generates response (streaming)

Using llama.cpp server with OpenAI-compatible API.

IMPORTANT: Separates conversation history from current message
to prevent LLM from responding to past messages.
"""

from openai import AsyncOpenAI
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
        client = AsyncOpenAI(
            base_url=config.LLAMA_CPP_BASE_URL,
            api_key=config.LLAMA_CPP_API_KEY
        )
        
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
                response = await client.chat.completions.create(
                    model=config.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=config.LLM_JUDGE_TEMPERATURE,
                    top_p=config.LLM_JUDGE_TOP_P,
                    max_tokens=config.LLM_JUDGE_NUM_PREDICT,
                    extra_body={
                        "top_k": config.LLM_JUDGE_TOP_K,
                    }
                )
                
                
                # Parse response (OpenAI format)
                result = response.choices[0].message.content.strip()
                    
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
        client = AsyncOpenAI(
            base_url=config.LLAMA_CPP_BASE_URL,
            api_key=config.LLAMA_CPP_API_KEY
        )
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
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Streaming request (OpenAI format with llama.cpp extra_body)
        chat_kwargs = {
            "model": config.LLM_MODEL_NAME,
            "messages": messages,
            "stream": True,
            "temperature": config.LLM_RESPONSE_TEMPERATURE,
            "top_p": config.LLM_RESPONSE_TOP_P,
            "extra_body": {
                "top_k": config.LLM_RESPONSE_TOP_K,
                "repeat_penalty": config.LLM_RESPONSE_REPEAT_PENALTY,
            }
        }
        if tools:
            chat_kwargs["tools"] = tools
        
        tool_calls_accumulated = []
        content_buffer = ""
        has_yielded = False
        
        stream = await client.chat.completions.create(**chat_kwargs)
        async for chunk in stream:
            # Extract content from streaming chunk
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                content = delta.content or ""
                
                # Check for tool calls (accumulated across chunks)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        # Accumulate tool call information
                        if tc.index >= len(tool_calls_accumulated):
                            tool_calls_accumulated.append({
                                "id": tc.id or "",
                                "function": {"name": "", "arguments": ""}
                            })
                        if tc.function:
                            if tc.function.name:
                                tool_calls_accumulated[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_accumulated[tc.index]["function"]["arguments"] += tc.function.arguments
                
                if content:
                    content_buffer += content
                    has_yielded = True
                    yield content
        
        # Process tool calls (only if no content was yielded)
        if tool_calls_accumulated and not has_yielded:
            for tool_call in tool_calls_accumulated:
                func_name = tool_call["function"]["name"]
                func_args_str = tool_call["function"]["arguments"]
                
                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    func_args = {}
                
                if func_name:
                    logger.debug(f"Tool call: {func_name}({func_args})")
                    tool_result = mcp_library.execute_tool(func_name, func_args)
                    logger.debug(f"Tool result: {tool_result}")
                    
                    messages.append({
                        "role": "assistant",
                        "content": content_buffer,
                        "tool_calls": [{
                            "id": tool_call["id"],
                            "type": "function",
                            "function": tool_call["function"]
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
            
            # Generate final response with tool results
            stream = await client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                stream=True,
                temperature=config.LLM_RESPONSE_TEMPERATURE,
                top_p=config.LLM_RESPONSE_TOP_P,
                extra_body={
                    "top_k": config.LLM_RESPONSE_TOP_K,
                    "repeat_penalty": config.LLM_RESPONSE_REPEAT_PENALTY,
                }
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
                    
    except Exception as e:
        logger.error(f"Response stream error: {e}")
        yield None

