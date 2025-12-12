import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData
import discord
import asyncio
import io
import time
import queue
from dataclasses import dataclass
from typing import Optional, Callable, Union
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
    def __init__(self, voice_client, loop, on_audio_start=None, on_audio_end=None, on_pcm=None):
        self.voice_client = voice_client
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_playing = False
        self.on_audio_start = on_audio_start
        self.on_audio_end = on_audio_end
        self.on_pcm = on_pcm
        self._current_cleanup: Optional[Callable[[], None]] = None

    async def add_audio(self, audio_data):
        """
        Enqueue audio for playback.

        - audio_data: raw WAV bytes or QueuedAudio
        """
        # Backward compatible: accept raw bytes or QueuedAudio
        if isinstance(audio_data, (bytes, bytearray)):
            item = QueuedAudio(input=bytes(audio_data), start_payload=bytes(audio_data))
        elif isinstance(audio_data, QueuedAudio):
            item = audio_data
        else:
            # Unsupported
            return

        await self.queue.put(item)
        if not self.is_playing:
            self._play_next()

    def _play_next(self):
        try:
            item = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            self.is_playing = False
            return

        self.is_playing = True
        self._current_cleanup = item.cleanup

        # Notify start (schedule on main loop)
        if self.on_audio_start:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_start):
                    self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self.on_audio_start(item.start_payload)))
                else:
                    self.loop.call_soon_threadsafe(lambda: self.on_audio_start(item.start_payload))
            except Exception as e:
                print(f"on_audio_start error: {e}")

        # Build AudioSource (WAV -> PCM). FFmpegPCMAudio handles WAV headers automatically.
        src = item.input
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(bytes(src))

        audio_source: discord.AudioSource = discord.FFmpegPCMAudio(src, pipe=True)

        # Apply volume control
        audio_source = discord.PCMVolumeTransformer(audio_source, volume=config.TTS_VOLUME)

        # Optional PCM observer for real-time lipsync, etc.
        if self.on_pcm:
            audio_source = PCMObserverAudioSource(audio_source, self.on_pcm)

        if self.voice_client and self.voice_client.is_connected():
            self.voice_client.play(audio_source, after=self._after_play)
        else:
            print("Voice client not connected. Skipping audio.")
            self.is_playing = False

    def _after_play(self, error):
        if error:
            print(f"Player error: {error}")

        # Per-item cleanup (stream cancel, etc.)
        try:
            if self._current_cleanup:
                self._current_cleanup()
        except Exception:
            pass
        self._current_cleanup = None

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


class BlockingByteStream(io.RawIOBase):
    """
    Thread-safe blocking byte stream for piping async HTTP streaming data into ffmpeg stdin.

    - Producer: asyncio task calls feed()
    - Consumer: discord.py's FFmpegPCMAudio writer thread calls read()
    """

    def __init__(self):
        super().__init__()
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._buffer = bytearray()
        self._eof = False

    def readable(self) -> bool:
        return True

    def feed(self, data: bytes) -> None:
        if self._eof:
            return
        if data:
            self._q.put(data)

    def end(self) -> None:
        # Signal EOF to the reader thread
        if not self._eof:
            self._q.put(None)

    def read(self, n: int = -1) -> bytes:
        if self._eof and not self._buffer:
            return b""

        # If caller asks for "all", drain until EOF
        if n is None or n < 0:
            chunks: list[bytes] = []
            if self._buffer:
                chunks.append(bytes(self._buffer))
                self._buffer.clear()

            while True:
                chunk = self._q.get()
                if chunk is None:
                    self._eof = True
                    break
                if chunk:
                    chunks.append(chunk)
            return b"".join(chunks)

        # Ensure buffer has at least 1 byte unless EOF
        while not self._buffer and not self._eof:
            chunk = self._q.get()
            if chunk is None:
                self._eof = True
                break
            if chunk:
                self._buffer.extend(chunk)

        if not self._buffer and self._eof:
            return b""

        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out


@dataclass
class QueuedAudio:
    """
    Audio item for AudioPlayer queue.

    - input: bytes (WAV full) or file-like object (streaming WAV bytes)
    - cleanup: optional callback called after playback ends (thread-safe)
    - start_payload: optional payload passed to on_audio_start (e.g., wav bytes for WAV-based lipsync)
    """

    input: Union[bytes, io.IOBase]
    cleanup: Optional[Callable[[], None]] = None
    start_payload: Optional[bytes] = None


class PCMObserverAudioSource(discord.AudioSource):
    """
    Wrap an AudioSource and observe PCM frames.
    This runs in the audio thread, so on_pcm must be fast and thread-safe.
    """

    def __init__(self, inner: discord.AudioSource, on_pcm: Callable[[bytes], None]):
        self._inner = inner
        self._on_pcm = on_pcm

    def read(self) -> bytes:
        data = self._inner.read()
        if data:
            try:
                self._on_pcm(data)
            except Exception:
                pass
        return data

    def is_opus(self) -> bool:
        try:
            return self._inner.is_opus()
        except Exception:
            return False

    def cleanup(self) -> None:
        try:
            self._inner.cleanup()
        except Exception:
            pass
