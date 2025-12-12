import aiohttp
import config
import asyncio

class tts_handler:
    def __init__(self, server_url, reference_file, reference_prompt, reference_prompt_lang):
        self.server_url = server_url
        self.reference_file = reference_file
        self.reference_prompt = reference_prompt
        self.reference_prompt_lang = reference_prompt_lang

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
            "batch_size": 16,              # int. batch size for inference
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
            async with aiohttp.ClientSession() as session:
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
