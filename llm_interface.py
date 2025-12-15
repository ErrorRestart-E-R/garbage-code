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


def _is_exceed_context_size_error(exc: Exception) -> bool:
    """
    Detect llama.cpp/OpenAI-compatible context overflow errors.
    """
    s = str(exc) or ""
    return ("exceed_context_size_error" in s) or ("exceeds the available context size" in s)


def _trim_messages_keep_last(
    msgs: List[Dict[str, str]],
    max_messages: int,
) -> List[Dict[str, str]]:
    if not msgs:
        return []
    if max_messages <= 0 or len(msgs) <= max_messages:
        trimmed = list(msgs)
    else:
        trimmed = list(msgs[-max_messages:])

    # Ensure starts with user
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


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
        system_base = config.SYSTEM_PROMPT.format(
            ai_name=config.AI_NAME,
            participant_count=participant_count
        )

        def _build_system_content(mem_ctx: str, kt_mode: bool, kt_ctx: str) -> str:
            sc = system_base

            # Add long-term memory if available
            if mem_ctx:
                sc += f"\n\n[LONG-TERM MEMORY]\n{mem_ctx}"

            # KTANE game mode: rely on injected manual context, do not guess.
            if kt_mode:
                sc += (
                    "\n\n[KTANE GAME MODE]\n"
                    "- You are assisting with the game 'KEEP TALKING and NOBODY EXPLODES'.\n"
                    "- The user describes what they see on the bomb in Korean.\n"
                    "- You MUST rely only on the provided [KTANE MANUAL CONTEXT] text for defusal rules.\n"
                    "- If the context is missing or insufficient, ask 1-2 short clarifying questions.\n"
                    "- Never guess rules from general knowledge when KTANE mode is on.\n"
                    "- In KTANE mode, ALWAYS respond to the latest user message. Do not stay silent.\n"
                    "- In KTANE mode, NEVER output an empty response.\n"
                    "- Users may describe things inaccurately/colloquially. Map their description to the closest term/alias found in the manual context.\n"
                    "- If multiple candidates fit, ask a single short disambiguation question.\n"
                    "\n"
                    "[KTANE OUTPUT RULES]\n"
                    "- Do NOT explain how you found the rule.\n"
                    "- Do NOT quote or paraphrase the manual/context.\n"
                    "- Do NOT mention '매뉴얼', '문서', 'RAG', '컨텍스트' in your reply.\n"
                    "- Output only the final actionable instruction(s) the defuser must do now.\n"
                    "- For Keypads/키패드: if you don't have all 4 symbols with positions (좌상/우상/좌하/우하), ask for them.\n"
                    "- For Keypads/키패드 final answer: output press order as positions only (e.g., '좌상 -> 우하 -> ...').\n"
                )
                if kt_ctx and kt_ctx.strip():
                    sc += f"\n\n[KTANE MANUAL CONTEXT]\n{kt_ctx.strip()}"

            return sc

        normalized_messages = _normalize_messages_for_llama(messages)
        if ktane_mode:
            # KTANE는 "현재 상태" 중심이므로 과거 잡담 히스토리를 짧게 유지해 프롬프트를 보호합니다.
            normalized_messages = _trim_messages_keep_last(normalized_messages, 8)

        system_content = _build_system_content(memory_context, ktane_mode, ktane_context)

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
        
        # Create stream (retry on context overflow by trimming variable contexts)
        try:
            stream = await client.chat.completions.create(**chat_kwargs)
        except Exception as e:
            if not _is_exceed_context_size_error(e):
                raise

            logger.warning(f"LLM prompt exceeded context. Retrying with smaller context. err={e}")

            # Retry variants: progressively shrink KTANE context and history
            variants: list[tuple[str, str, int]] = []
            # (mem_ctx, kt_ctx, max_msgs)
            if ktane_mode:
                ktx = (ktane_context or "").strip()
                if ktx:
                    variants.append(("", ktx[:2500], 6))
                    variants.append(("", ktx[:1200], 4))
                variants.append(("", "", 4))
            else:
                # Non-KTANE: drop long-term memory and shrink history
                variants.append(("", "", 10))
                variants.append(("", "", 6))

            last_exc: Exception = e
            stream = None
            for mem_ctx, kt_ctx, max_msgs in variants:
                try:
                    nm = _trim_messages_keep_last(normalized_messages, max_msgs)
                    sc = _build_system_content(mem_ctx, ktane_mode, kt_ctx)
                    chat_kwargs["messages"] = [{"role": "system", "content": sc}] + nm
                    stream = await client.chat.completions.create(**chat_kwargs)

                    # Update locals used later (tool injection / second pass)
                    system_content = sc
                    normalized_messages = nm
                    break
                except Exception as e2:
                    last_exc = e2
                    if not _is_exceed_context_size_error(e2):
                        raise

            if stream is None:
                raise last_exc

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
            try:
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
            except Exception as e:
                # If tool injection causes overflow, drop tool results and continue without tools.
                if _is_exceed_context_size_error(e):
                    logger.warning(f"Tool-injected prompt exceeded context. Dropping tool_results. err={e}")
                    stream = await client.chat.completions.create(
                        model=config.LLM_MODEL_NAME,
                        messages=[{"role": "system", "content": system_content}] + normalized_messages,
                        stream=True,
                        temperature=config.LLM_RESPONSE_TEMPERATURE,
                        top_p=config.LLM_RESPONSE_TOP_P,
                        extra_body={
                            "top_k": config.LLM_RESPONSE_TOP_K,
                            "repeat_penalty": config.LLM_RESPONSE_REPEAT_PENALTY,
                        },
                    )
                else:
                    raise
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    has_yielded = True
                    yield delta.content
                    
    except Exception as e:
        logger.error(f"Response stream error: {e}")
        yield None

