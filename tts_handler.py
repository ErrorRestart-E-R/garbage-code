import aiohttp
import config
import asyncio

class tts_handler:
    def __init__(self, server_url, reference_file, reference_prompt, reference_prompt_lang):
        self.server_url = server_url
        self.reference_file = reference_file
        self.reference_prompt = reference_prompt
        self.reference_prompt_lang = reference_prompt_lang

        # aiohttp 권장: 세션을 재사용해서 커넥션 풀/keep-alive 활용
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
                enable_cleanup_closed=True,
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
            "batch_size": 32,              # int. batch size for inference
            "batch_threshold": 0.75,      # float. threshold for batch splitting.
            "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
            "speed_factor":1.2,           # float. control the speed of the synthesized audio.
            "streaming_mode": False,      # bool. whether to return a streaming response.
            "seed": -1,                   # int. random seed for reproducibility.
            "parallel_infer": True,       # bool. whether to use parallel inference.
            "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
            "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
            "super_sampling": False       # bool. whether to use super-sampling for audio when using VITS model V3.
        }
        
        try:
            session = await self._get_session()
            async with session.post(self.server_url, json=datas) as response:
                if response.status == 200:
                    return await response.read()
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
