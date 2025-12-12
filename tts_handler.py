import aiohttp
import config
import asyncio
import io
import time
import wave
from logger import setup_logger

logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    """
    Best-effort WAV duration estimator for latency/RTF logging.
    Returns 0.0 if parsing fails.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate = wf.getframerate() or 0
            frames = wf.getnframes() or 0
            return (frames / float(rate)) if rate > 0 else 0.0
    except Exception:
        return 0.0

class tts_handler:
    def __init__(self, server_url, reference_file, reference_prompt, reference_prompt_lang):
        self.server_url = server_url
        self.reference_file = reference_file
        self.reference_prompt = reference_prompt
        self.reference_prompt_lang = reference_prompt_lang

        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session

        async with self._session_lock:
            if self._session and not self._session.closed:
                return self._session

            timeout = aiohttp.ClientTimeout(
                total=float(getattr(config, "TTS_HTTP_TIMEOUT_TOTAL_SECONDS", 120.0)),
                connect=float(getattr(config, "TTS_HTTP_TIMEOUT_CONNECT_SECONDS", 10.0)),
                sock_read=float(getattr(config, "TTS_HTTP_TIMEOUT_SOCK_READ_SECONDS", 120.0)),
            )
            connector = aiohttp.TCPConnector(
                limit=int(getattr(config, "TTS_HTTP_MAX_CONNECTIONS", 10)),
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            return self._session

    async def close(self) -> None:
        """
        세션 종료(권장).
        - 현재 봇 종료 흐름에서는 프로세스 종료로 정리되기도 하지만,
          장기적으로는 shutdown 훅에서 호출하는 것이 좋습니다.
        """
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def get_async(self, prompt, prompt_lang):
        # Pull knobs from config (safe defaults if missing)
        sample_steps = int(getattr(config, "TTS_SAMPLE_STEPS", 32))
        batch_size = int(getattr(config, "TTS_BATCH_SIZE", 32))
        speed_factor = float(getattr(config, "TTS_SPEED_FACTOR", 1.2))
        parallel_infer = bool(getattr(config, "TTS_PARALLEL_INFER", True))
        log_latency = bool(getattr(config, "TTS_LOG_LATENCY", False))

        datas = {
            "text": prompt,                   # str.(required) text to be synthesized
            "text_lang": prompt_lang,               # str.(required) language of the text to be synthesized
            "ref_audio_path": self.reference_file,         # str.(required) reference audio path
            "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
            "prompt_text": self.reference_prompt,            # str.(optional) prompt text for the reference audio
            "prompt_lang": self.reference_prompt_lang,            # str.(required) language of the prompt text for the reference audio
            "top_k": 5,                   # int. top k sampling
            "top_p": 1,                   # float. top p sampling
            "temperature": 1,             # float. temperature for sampling
            "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
            "batch_size": batch_size,      # int. batch size for inference
            "batch_threshold": 0.75,      # float. threshold for batch splitting.
            "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
            "speed_factor": speed_factor, # float. control the speed of the synthesized audio.
            "streaming_mode": False,      # 스트리밍은 본 프로젝트에서 사용하지 않습니다.
            "seed": -1,                   # int. random seed for reproducibility.
            "parallel_infer": parallel_infer,  # bool. whether to use parallel inference.
            "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
            "sample_steps": sample_steps, # int. number of sampling steps for VITS model V3.
            "super_sampling": False       # bool. whether to use super-sampling for audio when using VITS model V3.
        }
        
        try:
            session = await self._get_session()
            t0 = time.perf_counter()
            async with session.post(self.server_url, json=datas) as response:
                if response.status == 200:
                    wav_bytes = await response.read()
                    if log_latency and wav_bytes:
                        dt = time.perf_counter() - t0
                        dur = _wav_duration_seconds(wav_bytes)
                        rtf = (dt / dur) if dur > 0 else 0.0
                        logger.info(
                            f"TTS latency={dt:.2f}s audio={dur:.2f}s rtf={rtf:.2f} "
                            f"chars={len(prompt or '')} steps={sample_steps} bs={batch_size} speed={speed_factor}"
                        )
                    return wav_bytes
                else:
                    print(f"TTS Error: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            print(f"TTS Connection Error: {e}")
            return None

if __name__ == "__main__":
    # Test code needs to be async now
    async def main():
        tts = tts_handler(config.TTS_SERVER_URL, config.TTS_REFERENCE_FILE, config.TTS_REFERENCE_PROMPT, config.TTS_REFERENCE_PROMPT_LANG)
        audio_data = await tts.get_async("안녕하세요", "ko")
        if audio_data:
            with open("output.wav", "wb") as f:
                f.write(audio_data)
            print("Saved output.wav")
    
    asyncio.run(main())
