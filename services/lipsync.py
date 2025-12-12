import asyncio
import io
import math
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

from logger import setup_logger
import config


logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


@dataclass
class LipSyncConfig:
    parameter_id: str
    update_hz: float = 30.0
    gain: float = 25.0
    smoothing: float = 0.6  # 0~1, 클수록 더 부드럽게(반응은 느림)
    vmin: float = 0.0
    vmax: float = 1.0


class VTSAudioLipSync:
    """
    TTS WAV 바이트를 분석(RMS envelope)해서 VTube Studio 파라미터로 입 벌림을 구동합니다.

    - VTS는 파라미터 제어를 유지하려면 최소 1초에 1번 이상 값을 재전송해야 합니다.
    - 우리는 update_hz로 주기적으로 InjectParameterDataRequest를 보내 자연스러운 입 움직임을 만듭니다.
    """

    def __init__(self, vts_client, cfg: LipSyncConfig):
        self.vts = vts_client
        self.cfg = cfg
        self._task: Optional[asyncio.Task] = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, wav_bytes: bytes) -> None:
        if not self.vts:
            return

        # Cancel previous task (safe even if should not overlap)
        await self.stop()
        self._task = asyncio.create_task(self._run(wav_bytes))

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

        # Close mouth best-effort
        try:
            if self.vts and self.vts.connected:
                await self.vts.inject_parameter(self.cfg.parameter_id, 0.0)
        except Exception:
            pass

    async def _run(self, wav_bytes: bytes) -> None:
        try:
            ok = await self.vts.ensure_authenticated() if self.vts else False
            if not ok:
                logger.warning(
                    "VTS not authenticated. Allow the token request popup in VTS (token will be saved to the token file) to enable lipsync."
                )
                return

            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())

            if sampwidth != 2:
                logger.warning(f"Unsupported WAV sample width: {sampwidth}")
                return

            audio = np.frombuffer(pcm, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

            audio_f = audio.astype(np.float32) / 32768.0

            hz = max(5.0, float(self.cfg.update_hz))
            hop = max(1, int(rate / hz))

            alpha = max(0.0, min(1.0, 1.0 - float(self.cfg.smoothing)))
            ema = 0.0

            for i in range(0, len(audio_f), hop):
                chunk = audio_f[i : i + hop]
                if chunk.size == 0:
                    break

                rms = float(math.sqrt(float(np.mean(chunk * chunk)) + 1e-12))
                val = rms * float(self.cfg.gain)
                val = max(float(self.cfg.vmin), min(float(self.cfg.vmax), val))

                ema = (1.0 - alpha) * ema + alpha * val
                await self.vts.inject_parameter(self.cfg.parameter_id, float(ema))
                await asyncio.sleep(1.0 / hz)

            await self.vts.inject_parameter(self.cfg.parameter_id, 0.0)

        except asyncio.CancelledError:
            # On cancel, close mouth quickly
            try:
                if self.vts and self.vts.connected:
                    await self.vts.inject_parameter(self.cfg.parameter_id, 0.0)
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"VTS lipsync error: {e}")


