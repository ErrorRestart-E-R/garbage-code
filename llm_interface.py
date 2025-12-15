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
import re
from typing import Tuple, AsyncGenerator, List, Dict, Optional
from logger import setup_logger

# Setup logger
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


def _normalize_messages_for_llama(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    llama.cpp (some chat templates, e.g. Gemma3) requires strict alternation:
    user / assistant / user / assistant ...

    We defensively normalize:
    - keep only roles: user, assistant
    - drop empty content
    - merge consecutive same-role messages (join with newline)
    - ensure the first message is user (drop leading assistant blocks if any)
    """
    normalized: List[Dict[str, str]] = []

    for msg in messages or []:
        role = (msg.get("role") or "").strip()
        if role not in ("user", "assistant"):
            continue

        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n" + content
        else:
            normalized.append({"role": role, "content": content})

    # Ensure conversation starts with a user message
    while normalized and normalized[0]["role"] != "user":
        normalized.pop(0)

    return normalized


def _extract_last_user_text(messages: List[Dict[str, str]]) -> str:
    """
    Extract the most recent user utterance text (without 'SpeakerName: ' prefix).

    messages: OpenAI messages format list (system excluded here).
    """
    for msg in reversed(messages or []):
        if (msg.get("role") or "") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            return ""

        # Consecutive user messages can be merged with newlines; use the last line as "latest utterance"
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else content

        # Strip "SpeakerName: " prefix if present
        if ": " in last_line:
            _, last_line = last_line.split(": ", 1)
        return last_line.strip()

    return ""


def _should_enable_llm_tools(messages: List[Dict[str, str]]) -> bool:
    """
    Safety gate: only enable LLM tool-calling when the user explicitly asks for
    time/date, weather, or a calculation.

    This prevents accidental tool calls (and leaking tool traces into TTS).
    """
    if not getattr(config, "ENABLE_MCP_TOOLS", False):
        return False

    text = _extract_last_user_text(messages)
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)

    # Weather: require explicit '날씨'
    if "날씨" in text:
        return True

    # Time/date keywords
    if any(
        k in compact
        for k in (
            "현재시간",
            "지금시간",
            "현재시각",
            "지금시각",
            "지금몇시",
            "몇시야",
            "몇시냐",
            "시간알려줘",
            "오늘날짜",
            "오늘며칠",
            "오늘몇일",
            "날짜알려줘",
        )
    ):
        return True

    # Basic math expression
    if re.fullmatch(r"[0-9+\-*/(). ]+", text) and any(op in text for op in ("+", "*", "/")):
        return True

    return False


async def get_response_stream(
    messages: List[Dict[str, str]],
    participant_count: int = 1,
    memory_context: str = "",
    ktane_mode: bool = False,
    ktane_context: str = "",
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
        tools = mcp_library.get_tools() if _should_enable_llm_tools(messages) else None
        
        # Build system prompt with participant count context
        system_content = config.SYSTEM_PROMPT.format(
            ai_name=config.AI_NAME,
            participant_count=participant_count
        )
        
        # Add long-term memory if available
        if memory_context:
            system_content += f"\n\n[LONG-TERM MEMORY]\n{memory_context}"

        # KTANE game mode: rely on injected manual context, do not guess.
        if ktane_mode:
            system_content += (
                "\n\n[KTANE GAME MODE]\n"
                "- You are assisting with the game 'KEEP TALKING and NOBODY EXPLODES'.\n"
                "- The user describes what they see on the bomb in Korean.\n"
                "- You MUST rely only on the provided [KTANE MANUAL CONTEXT] text for defusal rules.\n"
                "- If the context is missing or insufficient, ask 1-2 short clarifying questions.\n"
                "- Never guess rules from general knowledge when KTANE mode is on.\n"
            )
            if ktane_context and ktane_context.strip():
                system_content += f"\n\n[KTANE MANUAL CONTEXT]\n{ktane_context.strip()}"
        
        normalized_messages = _normalize_messages_for_llama(messages)

        # Build final messages list: system + conversation history
        final_messages = [{"role": "system", "content": system_content}] + normalized_messages
        
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
                # Internal-only: tool results must NOT be repeated in assistant output
                system_with_tools += "\n\n<tool_results>\n" + "\n".join(tool_lines) + "\n</tool_results>"

            # Re-run generation WITHOUT adding tool-role messages to keep strict alternation
            messages_with_tools = [{"role": "system", "content": system_with_tools}] + normalized_messages

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

