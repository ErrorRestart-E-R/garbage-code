import asyncio
import time
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Dict

import config
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
            self.lipsync = VTSAudioLipSync(
                self.vts,
                LipSyncConfig(
                    parameter_id=config.VTS_LIPSYNC_PARAMETER_ID,
                    update_hz=config.VTS_LIPSYNC_UPDATE_HZ,
                    gain=config.VTS_LIPSYNC_GAIN,
                    smoothing=config.VTS_LIPSYNC_SMOOTHING,
                    vmin=config.VTS_LIPSYNC_MIN,
                    vmax=config.VTS_LIPSYNC_MAX,
                ),
            )

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
                if self.result_queue and not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    user_id = result.get("user_id")
                    user_text = result.get("text", "")

                    user_name = "Unknown"
                    if user_id:
                        user = self.bot.get_user(user_id)
                        user_name = user.display_name if user else f"User_{user_id}"

                    if user_text:
                        print(f"\n{user_name}: {user_text}")
                        self.history.add(user_name, user_text)
                        self.message_counter += 1
                        logger.debug(f"History: Added message #{self.message_counter} from {user_name}")

                await asyncio.sleep(0.01)
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
                    await asyncio.to_thread(self.memory_manager.save_memory, user_name, user_text)
                except Exception as e:
                    logger.error(f"Memory save error: {e}")
                self.memory_queue.task_done()
            except Exception as e:
                logger.error(f"Memory worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def start_workers(self):
        logger.info("Starting conversation workers (clean architecture split)...")
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
                content = msg.get("content", "")
                if ": " in content:
                    last_user_name, last_user_msg = content.split(": ", 1)
                else:
                    last_user_msg = content
                break

        memory_context = ""
        if last_user_msg:
            memory_context = await asyncio.to_thread(
                self.memory_manager.get_memory_context, last_user_msg, last_user_name
            )

        # TTS streaming: sentence queue
        full_response = ""
        buffer = ""
        first_chunk = True

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

            if any(p in buffer for p in config.TTS_SENTENCE_DELIMITERS):
                for punct in config.TTS_SENTENCE_DELIMITERS:
                    if punct in buffer:
                        parts = buffer.split(punct, 1)
                        sentence = parts[0] + punct
                        buffer = parts[1] if len(parts) > 1 else ""
                        if sentence.strip():
                            await tts_queue.put(sentence.strip())
                        break

        if not first_chunk:
            print()

        if buffer.strip():
            await tts_queue.put(buffer.strip())

        await tts_queue.put(None)
        await tts_task
        return full_response

    async def _tts_worker(self, tts_queue: asyncio.Queue, player: AudioPlayer):
        while True:
            text = await tts_queue.get()
            if text is None:
                break
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


