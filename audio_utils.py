import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData
import discord
import asyncio
import io
import time
from typing import Optional, Callable, Tuple
import config

class STTSink(AudioSink):
    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        print("STTSink initialized.")

        self._decim = 3
        self._fir_taps = self._design_lowpass_fir(num_taps=63, cutoff=0.5 / self._decim * 0.9).astype(np.float32)

        self._fir_state_by_user: dict[int, np.ndarray] = {}
        self._state_last_seen: dict[int, float] = {}
        self._last_prune_ts: float = 0.0
    
    def wants_opus(self):
        return False

    def cleanup(self):
        print("STTSink cleanup.")
        try:
            self._fir_state_by_user.clear()
            self._state_last_seen.clear()
        except Exception:
            pass
        pass

    @staticmethod
    def _design_lowpass_fir(num_taps: int, cutoff: float) -> np.ndarray:
        """
        Windowed-sinc FIR low-pass filter.

        Args:
            num_taps: 홀수 권장(선형 위상)
            cutoff: 정규화 컷오프(0~0.5, 0.5=Nyquist)
        """
        n = np.arange(num_taps, dtype=np.float32)
        mid = (num_taps - 1) / 2.0
        x = n - mid
        # h[n] = 2*fc*sinc(2*fc*(n-mid))
        h = 2.0 * cutoff * np.sinc(2.0 * cutoff * x)
        h *= np.hamming(num_taps).astype(np.float32)
        # DC gain = 1
        s = float(np.sum(h))
        if s != 0.0:
            h /= s
        return h

    def _resample_48k_to_16k(self, user_id: int, pcm_s16_stereo: bytes) -> bytes:
        """
        Input: 48kHz stereo int16 PCM bytes
        Output: 16kHz mono int16 PCM bytes
        """
        audio = np.frombuffer(pcm_s16_stereo, dtype=np.int16)

        # stereo -> mono (int32로 누적 후 평균)
        if audio.size % 2 == 0 and audio.size >= 2:
            stereo = audio.reshape(-1, 2).astype(np.int32)
            mono = ((stereo[:, 0] + stereo[:, 1]) // 2).astype(np.int16)
        else:
            mono = audio  # fallback

        # FIR filtering (streaming, per-user)
        state = self._fir_state_by_user.get(user_id)
        if state is None or state.shape[0] != (len(self._fir_taps) - 1):
            state = np.zeros(len(self._fir_taps) - 1, dtype=np.float32)

        x = np.concatenate([state, mono.astype(np.float32)])
        y = np.convolve(x, self._fir_taps, mode="valid")  # len(y) == len(mono)
        self._fir_state_by_user[user_id] = x[-(len(self._fir_taps) - 1) :]
        self._state_last_seen[user_id] = time.time()

        # decimate by 3
        y = y[:: self._decim]

        # back to int16 range (y는 int16 단위 스케일을 유지)
        y = np.clip(y, -32768.0, 32767.0).astype(np.int16)
        return y.tobytes()

    def write(self, user, data: VoiceData):
        if self.audio_queue is None:
            return
            
        try:
            # stateful resampling per user
            now = time.time()
            # 주기적으로 오래된 사용자 상태 정리(메모리 누수 방지)
            if now - self._last_prune_ts > 30.0:
                ttl = float(getattr(config, "USER_TIMEOUT_SECONDS", 60))
                stale = [uid for uid, ts in self._state_last_seen.items() if now - ts > ttl]
                for uid in stale:
                    self._state_last_seen.pop(uid, None)
                    self._fir_state_by_user.pop(uid, None)
                self._last_prune_ts = now

            resampled_pcm = self._resample_48k_to_16k(user.id, data.pcm)
            # Offload to the STT process (already 16k mono)
            self.audio_queue.put((user.id, resampled_pcm))
            
        except Exception as e:
            print(f"Error in write: {e}")

class AudioPlayer:
    """
    Discord 음성 재생을 단일 진입점으로 직렬화하는 플레이어.

    - `add_audio()`는 WAV bytes를 내부 큐에 적재하고 즉시 반환합니다(재생 완료는 별도).
    - `drain()`은 "현재 재생 + 내부 큐"가 모두 소진될 때까지 await 합니다.
    - `interrupt()`는 현재 재생을 stop하고 대기 중인 큐를 비웁니다(인터럽트 정책용).

    주의:
    - discord.py의 `VoiceClient.play()`는 비동기 await 대상이 아니며,
      재생 종료는 `after` 콜백으로만 알 수 있습니다. 따라서 완료 동기화를 위해
      Future/Event를 별도로 관리합니다.
    """

    def __init__(
        self,
        voice_client_getter: Callable[[], Optional[discord.VoiceClient]],
        loop: asyncio.AbstractEventLoop,
        on_audio_start=None,
        on_audio_end=None,
    ):
        self._get_voice_client = voice_client_getter
        self.loop = loop
        self.queue: asyncio.Queue[Tuple[bytes, asyncio.Future]] = asyncio.Queue()
        self.is_playing: bool = False
        self.on_audio_start = on_audio_start
        self.on_audio_end = on_audio_end

        # When idle (no current playback and queue empty)
        self._idle_event = asyncio.Event()
        self._idle_event.set()

        # Current / pending items (needed for robust retries / interruption)
        self._current_item: Optional[Tuple[bytes, asyncio.Future]] = None
        self._pending_item: Optional[Tuple[bytes, asyncio.Future]] = None

    async def add_audio(self, audio_data) -> Optional[asyncio.Future]:
        if not audio_data:
            return None
        # Only accept full WAV bytes (non-streaming playback)
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)
        if not isinstance(audio_data, (bytes,)):
            return None

        fut = self.loop.create_future()
        await self.queue.put((audio_data, fut))
        self._idle_event.clear()
        if not self.is_playing:
            self._play_next()
        return fut

    async def drain(self) -> None:
        """Wait until all queued audio has finished playback."""
        await self._idle_event.wait()

    async def interrupt(self) -> None:
        """
        Stop current playback and flush any queued audio.
        This implements policy-B style "stop talking now".
        """
        # Stop the voice client if possible
        vc = None
        try:
            vc = self._get_voice_client()
        except Exception:
            vc = None

        try:
            if vc and vc.is_connected() and (vc.is_playing() or vc.is_paused()):
                # IMPORTANT:
                # - In discord-ext-voice-recv, VoiceRecvClient.stop() stops BOTH receiving and sending,
                #   which triggers sink cleanup and breaks STT reception.
                # - Use stop_playing() if available to stop playback only.
                if hasattr(vc, "stop_playing"):
                    vc.stop_playing()
                else:
                    vc.stop()
        except Exception:
            # best-effort; don't raise from interrupt
            pass

        # Cancel current and pending items
        def _cancel_item(item: Optional[Tuple[bytes, asyncio.Future]]) -> None:
            if not item:
                return
            _, f = item
            try:
                if not f.done():
                    f.cancel()
            except Exception:
                pass

        _cancel_item(self._current_item)
        _cancel_item(self._pending_item)
        self._current_item = None
        self._pending_item = None

        # Drain the queue and cancel their futures
        try:
            while True:
                _, f = self.queue.get_nowait()
                try:
                    if not f.done():
                        f.cancel()
                except Exception:
                    pass
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

        self.is_playing = False
        self._idle_event.set()

    def has_current_playback(self) -> bool:
        """
        Returns True if an audio item is currently playing (i.e., vc.play() succeeded).
        This is a stronger signal than `is_playing`, which also covers queued items.
        """
        return self._current_item is not None

    def flush_pending_only(self) -> None:
        """
        Cancel any queued / pending audio WITHOUT stopping current playback.

        Use this for PREPLAY barge-in cases:
        - We want to drop not-yet-played audio and prevent future playback,
          but we must not stop the voice client (which could break STT reception).
        """
        # Cancel pending retry item (already playing case)
        if self._pending_item is not None:
            _, fut = self._pending_item
            try:
                if not fut.done():
                    fut.cancel()
            except Exception:
                pass
            self._pending_item = None

        # Drain queued items and cancel their futures
        try:
            while True:
                _, fut = self.queue.get_nowait()
                try:
                    if not fut.done():
                        fut.cancel()
                except Exception:
                    pass
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

        # If nothing is currently playing, we are idle now.
        if self._current_item is None:
            self.is_playing = False
            self._idle_event.set()

    def _play_next(self):
        # If we previously failed to start due to "already playing", retry that first.
        item: Optional[Tuple[bytes, asyncio.Future]] = None
        if self._pending_item is not None:
            item = self._pending_item
            self._pending_item = None
        else:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                self.is_playing = False
                self._idle_event.set()
                return
            except Exception:
                self.is_playing = False
                self._idle_event.set()
                return

        audio_data, fut = item
        self.is_playing = True
        self._idle_event.clear()

        # Convert bytes to AudioSource (WAV -> PCM)
        # FFmpegPCMAudio handles WAV headers automatically
        audio_source: discord.AudioSource = discord.FFmpegPCMAudio(io.BytesIO(audio_data), pipe=True)

        # Apply volume control
        audio_source = discord.PCMVolumeTransformer(audio_source, volume=config.TTS_VOLUME)

        vc = None
        try:
            vc = self._get_voice_client()
        except Exception:
            vc = None

        if not vc or not vc.is_connected():
            # No voice connection; cancel this item and continue.
            try:
                if not fut.done():
                    fut.cancel()
            except Exception:
                pass
            self._current_item = None
            # Try next item
            self.loop.call_soon_threadsafe(self._play_next)
            return

        # Start playback (may raise if already playing)
        try:
            self._current_item = (audio_data, fut)
            vc.play(audio_source, after=self._after_play)
        except Exception as e:
            # If already playing, keep item and retry shortly.
            msg = str(e) if e else ""
            if "Already playing" in msg or "already playing" in msg:
                self._pending_item = (audio_data, fut)
                # Stay in 'playing' state to avoid concurrent starts; retry soon.
                self.loop.call_later(0.2, self._play_next)
                return

            # Other playback error: fail this item and continue.
            try:
                if not fut.done():
                    fut.set_exception(e)
            except Exception:
                pass
            self._current_item = None
            self.loop.call_soon_threadsafe(self._play_next)
            return

        # Notify start (schedule on main loop) AFTER play() succeeded
        if self.on_audio_start:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_start):
                    self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self.on_audio_start(audio_data)))
                else:
                    self.loop.call_soon_threadsafe(lambda: self.on_audio_start(audio_data))
            except Exception as e:
                print(f"on_audio_start error: {e}")

    def _after_play(self, error):
        # Complete the current item's future
        item = self._current_item
        self._current_item = None
        if item:
            _, fut = item
            try:
                if not fut.done():
                    if error:
                        fut.set_exception(error)
                    else:
                        fut.set_result(None)
            except Exception:
                pass

        # Notify end (schedule on main loop)
        if self.on_audio_end:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_end):
                    self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self.on_audio_end()))
                else:
                    self.loop.call_soon_threadsafe(self.on_audio_end)
            except Exception as e:
                print(f"on_audio_end error: {e}")

        # Schedule next play on the main loop
        self.loop.call_soon_threadsafe(self._play_next)
