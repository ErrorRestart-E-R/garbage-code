import asyncio
import time
import multiprocessing
import queue
import re
from dataclasses import dataclass
from enum import Enum
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

from game_gateway import GameHubClient
from app.game_router import GameRouter
from game_protocol import ControlAction, parse_control_line


logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


@dataclass
class PendingResponse:
    speaker: str
    message: str
    history_messages: List[Dict[str, str]]  # OpenAI messages format list
    message_index: int


class ResponsePhase(str, Enum):
    IDLE = "idle"
    PREPLAY = "preplay"   # LLM/TTS in progress, no audio playback started yet
    PLAYING = "playing"   # audio playback has started for current turn


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

        # GameHub (separate process via HTTP)
        self.game_hub: Optional[GameHubClient] = None
        self.game_router: Optional[GameRouter] = None
        if bool(getattr(config, "GAME_HUB_ENABLED", True)):
            try:
                self.game_hub = GameHubClient(
                    base_url=getattr(config, "GAME_HUB_BASE_URL", "http://127.0.0.1:8765"),
                    timeout_total_seconds=float(getattr(config, "GAME_HUB_HTTP_TIMEOUT_TOTAL_SECONDS", 2.5)),
                    timeout_connect_seconds=float(getattr(config, "GAME_HUB_HTTP_TIMEOUT_CONNECT_SECONDS", 0.6)),
                )
                self.game_router = GameRouter(self.game_hub)
                logger.info("[GameHub] enabled (HTTP client ready)")
            except Exception as e:
                self.game_hub = None
                self.game_router = None
                logger.warning(f"[GameHub] init failed: {e}")
        self.tts = tts_handler(
            config.TTS_SERVER_URL,
            config.TTS_REFERENCE_FILE,
            config.TTS_REFERENCE_PROMPT,
            config.TTS_REFERENCE_PROMPT_LANG,
        )

        # Shared audio player (single entry point for VoiceClient.play)
        self.audio_player = AudioPlayer(
            voice_client_getter=self._get_voice_client,
            loop=self.bot.loop,
            on_audio_start=self._on_tts_audio_start,
            on_audio_end=self._on_tts_audio_end,
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

        # Interruption token: increments whenever we need to abort speech immediately.
        # Any in-flight response compares its captured token against current value.
        self._run_token: int = 0

        # Barge-in / turn-consolidation state machine
        self._response_phase: ResponsePhase = ResponsePhase.IDLE
        self._active_turn_token: int = 0
        self._barge_in_deadline_ts: float = 0.0
        self._stt_buffer_during_playback: list[tuple[str, str]] = []  # (user_name, text)
        self._active_tts_task: Optional[asyncio.Task] = None

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

                async def _handle_result(r: dict):
                    user_id = r.get("user_id")
                    user_text = r.get("text", "")

                    user_name = "Unknown"
                    if user_id:
                        user = self.bot.get_user(user_id)
                        user_name = user.display_name if user else f"User_{user_id}"

                    if not user_text:
                        return

                    print(f"\n{user_name}: {user_text}")

                    # 0) Explicit interrupt command always applies immediately.
                    if self._is_interrupt_command(user_text):
                        self.history.add(user_name, user_text)
                        self.message_counter += 1
                        logger.debug(f"History: Added message #{self.message_counter} from {user_name} (interrupt)")
                        await self._apply_interrupt()
                        return

                    now = time.time()

                    # 1) If we recently triggered barge-in, keep extending the merge window.
                    if self._barge_in_deadline_ts and now < self._barge_in_deadline_ts:
                        self._barge_in_deadline_ts = now + (self._barge_in_window_seconds())

                    # 2) If audio playback already started (PLAYING), do NOT interrupt.
                    # Buffer STT and flush after playback ends.
                    is_playing_now = False
                    try:
                        is_playing_now = self.audio_player.has_current_playback()
                    except Exception:
                        is_playing_now = False
                    if self.is_responding and (self._response_phase == ResponsePhase.PLAYING or is_playing_now):
                        self._stt_buffer_during_playback.append((user_name, user_text))
                        logger.debug(
                            f"[BARGE-IN] Buffered STT during PLAYING. buffered={len(self._stt_buffer_during_playback)}"
                        )
                        return

                    # 3) Normal path: store to history immediately.
                    self.history.add(user_name, user_text)
                    self.message_counter += 1
                    logger.debug(f"History: Added message #{self.message_counter} from {user_name}")

                    # 4) If we are in PREPLAY (LLM/TTS before first playback) and a new STT arrives,
                    # cancel and re-run after debounce (turn consolidation).
                    if self.is_responding and self._response_phase == ResponsePhase.PREPLAY:
                        await self._apply_barge_in_preplay(user_name=user_name, user_text=user_text)

                await _handle_result(result)

                # Burst drain: 이미 쌓인 결과는 즉시 처리(추가 thread get 호출 방지)
                while True:
                    try:
                        r2 = self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                    await _handle_result(r2)
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

                # Debounce window after a PREPLAY barge-in: wait for user speech to settle.
                now = time.time()
                if self._barge_in_deadline_ts and now < self._barge_in_deadline_ts:
                    await asyncio.sleep(min(0.05, max(0.01, self._barge_in_deadline_ts - now)))
                    continue

                if self.message_counter <= self.last_processed_index:
                    await asyncio.sleep(0.1)
                    continue

                messages, current_speaker, current_message, current_timestamp = self.history.get_messages_for_llm()
                if not current_speaker or not current_message:
                    await asyncio.sleep(0.1)
                    continue

                # If the latest user message is an explicit interrupt command,
                # do not trigger LLM; just stop speaking immediately.
                if self._is_interrupt_command(current_message):
                    await self._apply_interrupt()
                    self.last_processed_index = self.message_counter
                    await asyncio.sleep(0.05)
                    continue

                age = time.time() - current_timestamp
                if age > config.MESSAGE_STALENESS_THRESHOLD:
                    logger.warning(
                        f"Skipping stale message #{self.message_counter} (age: {age:.1f}s)"
                    )
                    self.last_processed_index = self.message_counter
                    continue

                current_index = self.message_counter
                self.last_processed_index = current_index
                self._barge_in_deadline_ts = 0.0

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

                full_response: Optional[str] = None
                try:
                    token = self._run_token
                    self._active_turn_token = token
                    self._response_phase = ResponsePhase.PREPLAY
                    self._stt_buffer_during_playback = []
                    full_response = await self._execute_response(
                        messages=pending.history_messages,
                        participant_count=participant_count,
                        run_token=token,
                    )

                    if full_response and full_response.strip():
                        self.history.add_ai_response(full_response)
                        self.last_response_time = time.time()
                        await self.memory_queue.put((pending.speaker, pending.message))
                    else:
                        print("  [No response]")
                finally:
                    # Flush any STT buffered during playback (PLAYING) AFTER committing AI response
                    if self._stt_buffer_during_playback:
                        for user_name, user_text in self._stt_buffer_during_playback:
                            self.history.add(user_name, user_text)
                            self.message_counter += 1
                        logger.info(
                            f"[BARGE-IN] Flushed buffered STT after playback: +{len(self._stt_buffer_during_playback)}"
                        )
                        self._stt_buffer_during_playback = []

                    # End of turn
                    self._response_phase = ResponsePhase.IDLE
                    self._active_turn_token = 0
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

    @staticmethod
    def _normalize_interrupt_text(text: str) -> str:
        """
        Normalize to compare Korean voice commands robustly:
        - remove whitespace
        - drop punctuation/symbols (keep alnum + underscore + Korean)
        """
        if not text:
            return ""
        t = re.sub(r"\s+", "", str(text))
        t = re.sub(r"[^0-9A-Za-z가-힣_]", "", t)
        return t

    def _is_interrupt_command(self, user_text: str) -> bool:
        """
        Policy B trigger:
        - "{AI_NAME} 그만말해"
        - "{AI_NAME} 조용히해"
        (spaces/punctuation variations tolerated)
        """
        ai = self._normalize_interrupt_text(getattr(config, "AI_NAME", ""))
        t = self._normalize_interrupt_text(user_text or "")
        if not ai or not t:
            return False

        if not t.startswith(ai):
            return False

        tail = t[len(ai) :]
        return tail in ("그만말해", "조용히해")

    async def _apply_interrupt(self) -> None:
        """
        Stop current speech immediately and drop any queued responses.
        """
        # Abort any in-flight response/tss pipeline
        self._run_token += 1

        # Stop/flush audio now (policy B)
        try:
            await self.audio_player.interrupt()
        except Exception:
            pass

        # Drop queued (not-yet-started) responses so we don't "resume speaking" old backlog
        try:
            while True:
                self.response_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

    def _barge_in_enabled(self) -> bool:
        return bool(getattr(config, "BARGE_IN_ENABLED", True))

    def _barge_in_window_seconds(self) -> float:
        ms = int(getattr(config, "BARGE_IN_MERGE_WINDOW_MS", 500))
        # Clamp to reasonable range
        ms = max(100, min(ms, 2000))
        return ms / 1000.0

    def _should_trigger_barge_in(self, user_text: str) -> bool:
        """
        Filter out very short/noisy STT fragments to prevent excessive cancellations.
        """
        if not user_text or not user_text.strip():
            return False

        ignore_pat = str(getattr(config, "BARGE_IN_IGNORE_REGEX", "") or "").strip()
        if ignore_pat:
            try:
                if re.fullmatch(ignore_pat, user_text.strip()):
                    return False
            except re.error:
                # invalid regex -> ignore pattern
                pass

        min_chars = int(getattr(config, "BARGE_IN_MIN_CHARS", 3))
        min_chars = max(1, min(min_chars, 20))
        compact = re.sub(r"\s+", "", user_text)
        return len(compact) >= min_chars

    async def _apply_barge_in_preplay(self, user_name: str, user_text: str) -> None:
        """
        PREPLAY(A/B) 바지인 정책:
        - 현재 응답(LLM/TTS/대기 오디오)을 취소
        - 최근 유저 발화가 더 들어올 수 있으니 debounce window를 시작/연장
        """
        if not self._barge_in_enabled():
            return
        if not self._should_trigger_barge_in(user_text):
            return

        now = time.time()
        self._barge_in_deadline_ts = now + self._barge_in_window_seconds()

        # Cancel current turn (no playback yet)
        self._run_token += 1

        # Flush any queued-but-not-played audio without touching current playback
        try:
            self.audio_player.flush_pending_only()
        except Exception:
            pass

        # Best-effort cancel of in-flight TTS worker (may be waiting on HTTP)
        try:
            t = self._active_tts_task
            if t and (not t.done()):
                t.cancel()
        except Exception:
            pass

        # Drop queued (not-yet-started) responses
        try:
            while True:
                self.response_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

        logger.info(f"[BARGE-IN] PREPLAY cancel + debounce (speaker={user_name})")

    async def _execute_response(
        self,
        messages: List[Dict[str, str]],
        participant_count: int,
        run_token: int,
    ) -> Optional[str]:
        if run_token != self._run_token:
            return None

        vc = self._get_voice_client()
        if not vc:
            logger.warning("Not in voice channel")
            return None

        player = self.audio_player

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
            tts_task = asyncio.create_task(self._tts_worker(tts_queue, player, run_token))
            self._active_tts_task = tts_task
            if run_token != self._run_token:
                try:
                    tts_task.cancel()
                except Exception:
                    pass
                self._active_tts_task = None
                return None

            await tts_queue.put(tool_shortcut.strip())
            await tts_queue.put(None)
            try:
                await tts_task
            except asyncio.CancelledError:
                self._active_tts_task = None
                return None
            finally:
                if self._active_tts_task is tts_task:
                    self._active_tts_task = None

            await player.drain()
            return tool_shortcut

        # ---------------------------
        # Game routing (via GameHub)
        # ---------------------------
        session_id = "default"
        try:
            ch = getattr(vc, "channel", None)
            cid = getattr(ch, "id", None) if ch else None
            if cid is not None:
                session_id = str(cid)
        except Exception:
            session_id = "default"

        # Build a small recent-turn list for GameHub (kept short to avoid noise)
        recent_turns: list[str] = []
        try:
            tail = list(messages or [])[-6:]
            for m in tail:
                role = (m.get("role") or "").strip()
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "assistant":
                    recent_turns.append(f"{config.AI_NAME}: {content}")
                else:
                    recent_turns.append(content)
        except Exception:
            recent_turns = []

        direct_action = None
        system_addendum = ""
        context_blocks: list[dict] = []
        allowed_start_game_ids = set()
        allow_stop_control = False
        active_game_id = None
        trace_id = ""

        if self.game_router:
            try:
                direct_action, patch = await self.game_router.route(
                    session_id=session_id,
                    last_user_text=last_user_msg,
                    recent_turns=recent_turns,
                    active_game_id_hint=None,
                )
                active_game_id = patch.active_game_id
                system_addendum = (patch.system_addendum or "").strip()
                context_blocks = list(getattr(patch, "context_blocks", []) or [])
                allowed_start_game_ids = set(patch.allowed_start_game_ids or set())
                allow_stop_control = bool(patch.allow_stop)
                trace_id = patch.trace_id or ""
            except Exception as e:
                logger.warning(f"[GameHub] route failed: {e}")

        # If router wants a direct reply (forced start/stop), speak without calling the main LLM.
        if direct_action and getattr(direct_action, "text", "").strip():
            direct_text = str(getattr(direct_action, "text")).strip()
            print(f"{config.AI_NAME}: {direct_text}")

            tts_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_worker(tts_queue, player, run_token))
            self._active_tts_task = tts_task
            if run_token != self._run_token:
                try:
                    tts_task.cancel()
                except Exception:
                    pass
                self._active_tts_task = None
                return None

            await tts_queue.put(direct_text)
            await tts_queue.put(None)
            try:
                await tts_task
            except asyncio.CancelledError:
                self._active_tts_task = None
                return None
            finally:
                if self._active_tts_task is tts_task:
                    self._active_tts_task = None
            await player.drain()
            return direct_text

        # Long-term memory context
        memory_context = ""
        # Avoid polluting the prompt with long-term memory while a game is active (default).
        if last_user_msg and (not active_game_id):
            memory_context = await asyncio.to_thread(
                self.memory_manager.get_memory_context, last_user_msg, last_user_name
            )
            # NOTE:
            # 프롬프트가 컨텍스트 한도를 초과하면 llm_interface에서 재시도/축소가 필요합니다.

        # Attach trace_id to the system addendum (kept out of TTS).
        if trace_id:
            if system_addendum:
                system_addendum += "\n\n"
            system_addendum += f"[TRACE]\n{trace_id}"

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
        tts_task = asyncio.create_task(self._tts_worker(tts_queue, player, run_token))
        self._active_tts_task = tts_task

        # CONTROL line (game start/stop) parsing.
        # We must NOT speak/store the CONTROL line, so we sniff the first line before emitting to TTS.
        control_checked = False
        control_buf = ""
        control_cmd = None

        async for chunk in get_response_stream(
            messages=messages,
            participant_count=participant_count,
            memory_context=memory_context,
            system_addendum=system_addendum,
            context_blocks=context_blocks,
        ):
            if run_token != self._run_token:
                break
            if not chunk:
                continue

            out = chunk

            # 0) If this is the first line and it starts with CONTROL:, buffer until newline,
            # then drop the CONTROL line (never speak it) and optionally record the command.
            if not control_checked:
                control_buf += out

                if control_buf.lstrip().lower().startswith("control:"):
                    # Wait for newline to complete the control line (short, one line).
                    if "\n" not in control_buf and len(control_buf) < 200:
                        continue

                    first_line, _, rest = control_buf.partition("\n")
                    control_buf = ""
                    control_checked = True

                    # Parse + validate START against allowed IDs. Always suppress the CONTROL line in output.
                    cmd = parse_control_line(first_line, allowed_game_ids=allowed_start_game_ids)
                    if cmd and cmd.action == ControlAction.STOP and not allow_stop_control:
                        cmd = None
                    control_cmd = cmd

                    out = rest.lstrip("\n")
                    if not out:
                        continue
                else:
                    # Not a control line; flush buffered text as normal.
                    control_checked = True
                    out = control_buf
                    control_buf = ""
                    if not out:
                        continue

            if first_chunk:
                print(f"{config.AI_NAME}: ", end="", flush=True)
                first_chunk = False

            print(out, end="", flush=True)
            full_response += out
            buffer += out

            # Drain buffer to TTS queue continuously (delimiter-based only)
            await _drain_buffer_to_tts_queue(force=False)

        # If the stream ended before we could decide about CONTROL (rare), flush what we buffered.
        if (not control_checked) and control_buf:
            out = control_buf
            control_checked = True
            control_buf = ""
            if out.strip():
                if first_chunk:
                    print(f"{config.AI_NAME}: ", end="", flush=True)
                    first_chunk = False
                print(out, end="", flush=True)
                full_response += out
                buffer += out
                await _drain_buffer_to_tts_queue(force=False)

        if not first_chunk:
            print()

        # If we were cancelled (barge-in/interrupt) before finishing, stop immediately.
        if run_token != self._run_token:
            try:
                tts_task.cancel()
            except Exception:
                pass
            if self._active_tts_task is tts_task:
                self._active_tts_task = None
            return None

        # Flush any remaining buffered text
        if run_token == self._run_token:
            await _drain_buffer_to_tts_queue(force=True)

        await tts_queue.put(None)
        try:
            await tts_task
        except asyncio.CancelledError:
            if self._active_tts_task is tts_task:
                self._active_tts_task = None
            return None
        finally:
            if self._active_tts_task is tts_task:
                self._active_tts_task = None

        # Important: response is not "done" until audio playback finishes.
        await player.drain()

        # Apply CONTROL command (start/stop) after finishing the turn (before returning).
        if run_token == self._run_token and self.game_hub and control_cmd:
            try:
                if control_cmd.action == ControlAction.START and getattr(control_cmd, "game_id", None):
                    await self.game_hub.start_game(session_id, control_cmd.game_id)
                elif control_cmd.action == ControlAction.STOP:
                    await self.game_hub.stop_game(session_id)
            except Exception as e:
                logger.warning(f"[GameHub] CONTROL apply failed: {e}")

        if run_token != self._run_token:
            return None
        return full_response

    async def _tts_worker(self, tts_queue: asyncio.Queue, player: AudioPlayer, run_token: int):
        while True:
            try:
                text = await tts_queue.get()
            except asyncio.CancelledError:
                return
            if text is None:
                break

            if run_token != self._run_token:
                # Abort speaking immediately
                continue

            # Defensive: ensure valid string for TTS server schema
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            try:
                if run_token != self._run_token:
                    continue
                wav_data = await self.tts.get_async(text, config.TTS_LANG)
                if run_token != self._run_token:
                    continue
                if wav_data:
                    await player.add_audio(wav_data)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"TTS error: {e}")

    async def _on_tts_audio_start(self, wav_bytes: bytes):
        # Mark that audio playback has started for this turn.
        # This is the boundary between PREPLAY(A/B) and PLAYING(C).
        if self.is_responding and self._response_phase == ResponsePhase.PREPLAY:
            self._response_phase = ResponsePhase.PLAYING
        if self.lipsync:
            await self.lipsync.start(wav_bytes)

    async def _on_tts_audio_end(self):
        if self.lipsync:
            await self.lipsync.stop()


