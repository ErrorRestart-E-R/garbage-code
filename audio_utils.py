import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData
import discord
import asyncio
import io
import time
import config

class STTSink(AudioSink):
    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        print("STTSink initialized.")

        # 48kHz -> 16kHz (decimate by 3) 품질 개선:
        # 간단 슬라이싱([::3])은 anti-aliasing이 없어 STT 품질 저하 가능.
        # FIR 저역통과 필터 후 3배 decimate (스트리밍 상태 유지)
        self._decim = 3
        self._fir_taps = self._design_lowpass_fir(num_taps=63, cutoff=0.5 / self._decim * 0.9).astype(np.float32)
        # 사용자별 스트리밍 상태(여러 사용자가 동시에 말하면 상태가 섞이면 안 됨)
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
    def __init__(self, voice_client, loop, on_audio_start=None, on_audio_end=None):
        self.voice_client = voice_client
        self.loop = loop
        self.queue = asyncio.Queue()
        self.is_playing = False
        self.on_audio_start = on_audio_start
        self.on_audio_end = on_audio_end

    async def add_audio(self, audio_data):
        await self.queue.put(audio_data)
        if not self.is_playing:
            self._play_next()

    def _play_next(self):
        try:
            audio_data = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            self.is_playing = False
            return

        self.is_playing = True

        # Notify start (schedule on main loop)
        if self.on_audio_start:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_start):
                    self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self.on_audio_start(audio_data)))
                else:
                    self.loop.call_soon_threadsafe(lambda: self.on_audio_start(audio_data))
            except Exception as e:
                print(f"on_audio_start error: {e}")

        # Convert bytes to AudioSource (WAV -> PCM)
        # FFmpegPCMAudio handles WAV headers automatically
        audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_data), pipe=True)
        
        # Apply volume control
        audio_source = discord.PCMVolumeTransformer(audio_source, volume=config.TTS_VOLUME)
        
        if self.voice_client and self.voice_client.is_connected():
            self.voice_client.play(audio_source, after=self._after_play)
        else:
            print("Voice client not connected. Skipping audio.")
            self.is_playing = False

    def _after_play(self, error):
        if error:
            print(f"Player error: {error}")
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
