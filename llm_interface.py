"""
LLM Interface: Single 27B LLM handles both judgment and response generation

The LLM decides whether to respond based on conversation context.
If it shouldn't respond, it outputs empty or minimal response.

Using llama.cpp server with OpenAI-compatible API.
Conversation history is passed in OpenAI messages format,
llama.cpp converts it to the appropriate chat template (e.g., Gemma3).
"""

from openai import AsyncOpenAI
import config
import json
import mcp_library
from typing import Tuple, AsyncGenerator, List, Dict, Optional
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


async def get_response_stream(
    messages: List[Dict[str, str]],
    participant_count: int = 1,
    memory_context: str = ""
) -> AsyncGenerator[Optional[str], None]:
    """
    Single LLM handles both judgment and response generation.
    
    The LLM receives conversation history in OpenAI messages format.
    llama.cpp converts it to chat template (e.g., Gemma3's <start_of_turn>).
    
    If the LLM decides not to respond, it outputs empty response.
    
    Args:
        messages: Conversation history in OpenAI messages format
                 [{"role": "user"/"assistant", "content": "..."}]
        participant_count: Number of human participants (for context)
        memory_context: Long-term memory context (optional)
        
    Yields:
        Response text chunks (empty if LLM decides not to respond)
    """
    try:
        # OpenAI Python SDK 권장: 단일 클라이언트로 요청 (base_url로 OpenAI-compatible 서버 사용)
        client = AsyncOpenAI(base_url=config.LLAMA_CPP_BASE_URL, api_key=config.LLAMA_CPP_API_KEY)
        tools = mcp_library.get_tools() if config.ENABLE_MCP_TOOLS else None
        
        # Build system prompt with participant count context
        system_content = config.SYSTEM_PROMPT.format(
            ai_name=config.AI_NAME,
            participant_count=participant_count
        )
        
        # Add long-term memory if available
        if memory_context:
            system_content += f"\n\n[LONG-TERM MEMORY]\n{memory_context}"
        
        # Build final messages list: system + conversation history
        final_messages = [{"role": "system", "content": system_content}] + messages
        
        logger.debug(f"LLM request: {len(messages)} messages, {participant_count} participants")
        
        # Streaming request (OpenAI format with llama.cpp extra_body)
        chat_kwargs = {
            "model": config.LLM_MODEL_NAME,
            "messages": final_messages,
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
        
        tool_calls_accumulated: list[dict] = []
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
                    
        if tool_calls_accumulated:
            tool_lines: list[str] = []
            for tool_call in tool_calls_accumulated:
                func_name = tool_call["function"]["name"]
                func_args_str = tool_call["function"]["arguments"]

                try:
                    func_args = json.loads(func_args_str) if func_args_str else {}
                except json.JSONDecodeError:
                    func_args = {}

                if not func_name:
                    continue

                logger.debug(f"Tool call: {func_name}({func_args})")
                tool_result = mcp_library.execute_tool(func_name, func_args)
                logger.debug(f"Tool result: {tool_result}")

                # Keep this compact — it gets injected into system prompt.
                try:
                    args_repr = json.dumps(func_args, ensure_ascii=False)
                except Exception:
                    args_repr = str(func_args)
                tool_lines.append(f"- {func_name}({args_repr}): {tool_result}")

            # Inject tool results into the system message (single system message only)
            system_with_tools = system_content
            if tool_lines:
                system_with_tools += "\n\n[TOOL RESULTS]\n" + "\n".join(tool_lines)

            # Re-run generation WITHOUT adding tool-role messages to keep strict alternation
            messages_with_tools = [{"role": "system", "content": system_with_tools}] + messages

            stream = await client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages_with_tools,
                stream=True,
                temperature=config.LLM_RESPONSE_TEMPERATURE,
                top_p=config.LLM_RESPONSE_TOP_P,
                extra_body={
                    "top_k": config.LLM_RESPONSE_TOP_K,
                    "repeat_penalty": config.LLM_RESPONSE_REPEAT_PENALTY,
                },
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    has_yielded = True
                    yield delta.content
                    
    except Exception as e:
        logger.error(f"Response stream error: {e}")
        yield None

