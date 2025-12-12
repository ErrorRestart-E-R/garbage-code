import asyncio
import time
import multiprocessing
import queue
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Dict

import config
import mcp_library
from logger import setup_logger

from memory_manager import MemoryManager
from conversation_history import ConversationHistory
from llm_interface import get_response_stream
from audio_utils import AudioPlayer
from tts_handler import tts_handler
from vts_backend import build_vts_client
from services.lipsync import VTSAudioLipSync, LipSyncConfig


logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


@dataclass
class PendingResponse:
    speaker: str
    message: str
    history_messages: List[Dict[str, str]]  # OpenAI messages format list
    message_index: int


class ConversationController:
    """
    대화 흐름 오케스트레이션(애플리케이션 레이어).

    - history_worker: STT 결과를 대화 히스토리에 저장
    - message_worker: 새 메시지를 감지해 LLM 응답 큐에 등록
    - response_worker: LLM 스트리밍 -> TTS -> 디스코드 재생
    - memory_worker: 메모리 저장(순차 처리)
    """

    def __init__(self, bot, voice_client_getter):
        self.bot = bot
        self._get_voice_client = voice_client_getter

        self.history = ConversationHistory(
            max_size=config.MAX_CONVERSATION_HISTORY,
            ai_name=config.AI_NAME,
        )
        self.memory_manager = MemoryManager()
        self.tts = tts_handler(
            config.TTS_SERVER_URL,
            config.TTS_REFERENCE_FILE,
            config.TTS_REFERENCE_PROMPT,
            config.TTS_REFERENCE_PROMPT_LANG,
        )

        # VTube Studio (optional)
        self.vts = None
        self.lipsync: Optional[VTSAudioLipSync] = None
        if getattr(config, "VTS_ENABLED", False):
            self.vts = build_vts_client()
            cfg = LipSyncConfig(
                parameter_id=config.VTS_LIPSYNC_PARAMETER_ID,
                update_hz=config.VTS_LIPSYNC_UPDATE_HZ,
                gain=config.VTS_LIPSYNC_GAIN,
                smoothing=config.VTS_LIPSYNC_SMOOTHING,
                vmin=config.VTS_LIPSYNC_MIN,
                vmax=config.VTS_LIPSYNC_MAX,
            )
            self.lipsync = VTSAudioLipSync(self.vts, cfg)

        # Queues
        self.response_queue: asyncio.Queue[Tuple[PendingResponse, int]] = asyncio.Queue()
        self.memory_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()

        # State
        self.is_responding = False
        self.message_counter = 0
        self.last_processed_index = 0
        self.last_response_time = 0.0

        # STT Result queue (multiprocessing)
        self.result_queue: Optional[multiprocessing.Queue] = None

    def set_result_queue(self, result_queue: multiprocessing.Queue):
        self.result_queue = result_queue

    # ---------------------------
    # Workers
    # ---------------------------
    async def history_worker(self):
        logger.info("History worker started")
        while True:
            try:
                if not self.result_queue:
                    await asyncio.sleep(0.1)
                    continue

                # multiprocessing.Queue.empty()는 신뢰하기 어렵습니다.
                # 이벤트 루프를 막지 않도록 별도 스레드에서 get(timeout)으로 대기합니다.
                try:
                    result = await asyncio.to_thread(self.result_queue.get, True, 0.2)
                except queue.Empty:
                    continue

                def _handle_result(r: dict):
                    user_id = r.get("user_id")
                    user_text = r.get("text", "")

                    user_name = "Unknown"
                    if user_id:
                        user = self.bot.get_user(user_id)
                        user_name = user.display_name if user else f"User_{user_id}"

                    if user_text:
                        print(f"\n{user_name}: {user_text}")
                        self.history.add(user_name, user_text)
                        self.message_counter += 1
                        logger.debug(f"History: Added message #{self.message_counter} from {user_name}")

                _handle_result(result)

                # Burst drain: 이미 쌓인 결과는 즉시 처리(추가 thread get 호출 방지)
                while True:
                    try:
                        r2 = self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                    _handle_result(r2)
            except Exception as e:
                logger.error(f"History worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def message_worker(self):
        logger.info("Message worker started (single LLM mode)")
        while True:
            try:
                if self.is_responding:
                    await asyncio.sleep(0.1)
                    continue

                if self.message_counter <= self.last_processed_index:
                    await asyncio.sleep(0.1)
                    continue

                messages, current_speaker, current_message, current_timestamp = self.history.get_messages_for_llm()
                if not current_speaker or not current_message:
                    await asyncio.sleep(0.1)
                    continue

                if time.time() - current_timestamp > config.MESSAGE_STALENESS_THRESHOLD:
                    logger.warning(
                        f"Skipping stale message #{self.message_counter} (age: {time.time() - current_timestamp:.1f}s)"
                    )
                    self.last_processed_index = self.message_counter
                    continue

                current_index = self.message_counter
                self.last_processed_index = current_index

                # Rate limiting
                time_since_last = time.time() - self.last_response_time
                if time_since_last < config.MIN_RESPONSE_INTERVAL:
                    await asyncio.sleep(config.MIN_RESPONSE_INTERVAL - time_since_last)

                participant_count = self.history.get_participant_count()
                pending = PendingResponse(
                    speaker=current_speaker,
                    message=current_message,
                    history_messages=messages,
                    message_index=current_index,
                )
                await self.response_queue.put((pending, participant_count))
                logger.debug(f"Message #{current_index} queued for LLM")

            except Exception as e:
                logger.error(f"Message worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def response_worker(self):
        logger.info("Response worker started (single LLM mode)")
        while True:
            try:
                pending, participant_count = await self.response_queue.get()
                self.is_responding = True

                try:
                    full_response = await self._execute_response(
                        messages=pending.history_messages,
                        participant_count=participant_count,
                    )

                    if full_response and full_response.strip():
                        self.history.add_ai_response(full_response)
                        self.last_response_time = time.time()
                        await self.memory_queue.put((pending.speaker, pending.message))
                    else:
                        print("  [No response]")
                finally:
                    self.is_responding = False

            except Exception as e:
                logger.error(f"Response worker error: {e}", exc_info=True)
                self.is_responding = False
                await asyncio.sleep(0.1)

    async def memory_worker(self):
        logger.info("Memory worker started (sequential queue processing)")
        while True:
            try:
                user_name, user_text = await self.memory_queue.get()
                try:
                    # Avoid memory pollution / noisy extraction on irrelevant messages
                    if self.memory_manager.should_save_memory(user_text):
                        await asyncio.to_thread(self.memory_manager.save_memory, user_name, user_text)
                except Exception as e:
                    logger.error(f"Memory save error: {e}")
                self.memory_queue.task_done()
            except Exception as e:
                logger.error(f"Memory worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def start_workers(self):
        logger.info("Starting conversation workers (clean architecture split)...")
        
        # VTS 인증 (봇 시작 전에 완료)
        if self.vts:
            try:
                logger.info("Authenticating with VTube Studio...")
                ok = await self.vts.ensure_authenticated()
                if ok:
                    logger.info("VTS authenticated successfully")
                else:
                    logger.warning("VTS authentication failed - lipsync will be disabled")
            except Exception as e:
                logger.error(f"VTS authentication error: {e}")
        
        await asyncio.gather(
            self.history_worker(),
            self.message_worker(),
            self.response_worker(),
            self.memory_worker(),
        )

    # ---------------------------
    # Public APIs (for bot layer)
    # ---------------------------
    def add_participant(self, name: str):
        self.history.add_participant(name)

    def remove_participant(self, name: str):
        self.history.remove_participant(name)

    def clear_participants(self):
        self.history.clear_participants()

    def clear_history(self):
        self.history.clear()

    def get_status(self) -> dict:
        return {
            "participants": list(self.history.participants),
            "history_count": len(self.history),
            "is_responding": self.is_responding,
            "message_counter": self.message_counter,
        }

    # ---------------------------
    # Internals
    # ---------------------------
    def _try_mcp_shortcut(self, user_text: str) -> Optional[str]:
        """
        MCP 툴을 "확실한 사실" 질문에 대해 우선 적용해 LLM 환각을 줄입니다.
        (LLM tool-calling이 항상 안정적이지 않을 수 있어, 최소한의 하드 라우팅을 둡니다.)
        """
        if not getattr(config, "ENABLE_MCP_TOOLS", False):
            return None

        text = (user_text or "").strip()
        if not text:
            return None

        compact = re.sub(r"\s+", "", text)

        # 현재 시간/날짜
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
                "몇시",
                "시간알려줘",
                "오늘날짜",
                "오늘며칠",
                "오늘몇일",
                "날짜알려줘",
            )
        ):
            return mcp_library.get_current_time()

        # 계산 (명확한 수식만)
        if re.fullmatch(r"[0-9+\-*/(). ]+", text) and any(op in text for op in ("+", "*", "/")):
            return mcp_library.calculate(text)

        # 날씨 (도시가 명시된 경우만)
        if "날씨" in text:
            m = re.search(r"([가-힣A-Za-z]+?)(?:의)?\s*날씨", text)
            if m:
                city = m.group(1).strip()
                if city and city not in {"오늘", "지금", "현재"}:
                    return mcp_library.get_weather(city)

        return None

    async def _execute_response(self, messages: List[Dict[str, str]], participant_count: int) -> Optional[str]:
        vc = self._get_voice_client()
        if not vc:
            logger.warning("Not in voice channel")
            return None

        player = AudioPlayer(
            vc,
            self.bot.loop,
            on_audio_start=self._on_tts_audio_start,
            on_audio_end=self._on_tts_audio_end,
        )

        # memory context from last user chunk
        last_user_msg = ""
        last_user_name = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                # user 메시지는 여러 명의 발화가 한 블록으로 합쳐질 수 있어(연속 user merge),
                # 가장 마지막 줄을 "최신 발화"로 간주합니다.
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                last_line = lines[-1] if lines else content

                if ": " in last_line:
                    last_user_name, last_user_msg = last_line.split(": ", 1)
                else:
                    last_user_msg = last_line
                break

        # 확실한 사실 질문은 MCP 툴로 처리(환각 방지)
        tool_shortcut = self._try_mcp_shortcut(last_user_msg)
        if tool_shortcut and tool_shortcut.strip():
            print(f"{config.AI_NAME}: {tool_shortcut}")

            tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_worker(tts_queue, player))
            await tts_queue.put(tool_shortcut.strip())
            await tts_queue.put(None)
            await tts_task
            return tool_shortcut

        memory_context = ""
        if last_user_msg:
            memory_context = await asyncio.to_thread(
                self.memory_manager.get_memory_context, last_user_msg, last_user_name
            )

        # TTS streaming: sentence queue
        full_response = ""
        buffer = ""
        first_chunk = True

        # Split only by sentence delimiters (no length-based chunking)
        tts_split_delims = list(getattr(config, "TTS_SENTENCE_DELIMITERS", ['.', '!', '?', '\n', '。']))
        pending_prefix = ""

        def _find_next_delim(buf: str) -> tuple[int, str]:
            """
            Find the earliest delimiter position with small heuristics:
            - Ignore '.' right after a digit (list markers like '1.' or '2.')
            - Ignore '.' between digits (decimals like '3.14')
            """
            if not buf:
                return -1, ""

            # Fast path: scan left-to-right
            for idx, ch in enumerate(buf):
                if ch not in tts_split_delims:
                    continue

                if ch == ".":
                    prev = buf[idx - 1] if idx > 0 else ""
                    nxt = buf[idx + 1] if (idx + 1) < len(buf) else ""
                    if prev.isdigit():  # enumeration like 1.
                        continue
                    if nxt.isdigit():  # decimal like 3.14
                        continue

                return idx, ch

            return -1, ""

        async def _drain_buffer_to_tts_queue(force: bool = False) -> None:
            nonlocal buffer, pending_prefix

            # 1) Split by sentence delimiters as soon as possible
            while True:
                hit_idx, hit_delim = _find_next_delim(buffer)
                if hit_idx == -1 or not hit_delim:
                    break

                cut = hit_idx + len(hit_delim)
                chunk_text = (buffer[:cut]).strip()
                buffer = buffer[cut:]
                if not chunk_text:
                    continue

                # If the model outputs a bare list marker on its own line ("1.", "2.") merge it with next chunk
                if re.fullmatch(r"[。]?\s*\d+\s*[.)]\s*", chunk_text):
                    pending_prefix = chunk_text.strip()
                    continue

                if pending_prefix:
                    chunk_text = f"{pending_prefix} {chunk_text}".strip()
                    pending_prefix = ""

                await tts_queue.put(chunk_text)

            # 2) Force flush remainder (end of stream)
            if force:
                tail = buffer.strip()
                buffer = ""

                if tail:
                    if pending_prefix:
                        tail = f"{pending_prefix} {tail}".strip()
                        pending_prefix = ""
                    await tts_queue.put(tail)
                elif pending_prefix:
                    # Rare: reply ended with only "1." etc. Best-effort speak it.
                    await tts_queue.put(pending_prefix)
                    pending_prefix = ""

        tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(tts_queue, player))

        async for chunk in get_response_stream(
            messages=messages,
            participant_count=participant_count,
            memory_context=memory_context,
        ):
            if not chunk:
                continue

            if first_chunk:
                print(f"{config.AI_NAME}: ", end="", flush=True)
                first_chunk = False

            print(chunk, end="", flush=True)
            full_response += chunk
            buffer += chunk

            # Drain buffer to TTS queue continuously (delimiter-based only)
            await _drain_buffer_to_tts_queue(force=False)

        if not first_chunk:
            print()

        # Flush any remaining buffered text
        await _drain_buffer_to_tts_queue(force=True)

        await tts_queue.put(None)
        await tts_task
        return full_response

    async def _tts_worker(self, tts_queue: asyncio.Queue, player: AudioPlayer):
        while True:
            text = await tts_queue.get()
            if text is None:
                break

            # Defensive: ensure valid string for TTS server schema
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            try:
                wav_data = await self.tts.get_async(text, config.TTS_LANG)
                if wav_data:
                    await player.add_audio(wav_data)
            except Exception as e:
                logger.error(f"TTS error: {e}")

    async def _on_tts_audio_start(self, wav_bytes: bytes):
        if self.lipsync:
            await self.lipsync.start(wav_bytes)

    async def _on_tts_audio_end(self):
        if self.lipsync:
            await self.lipsync.stop()


