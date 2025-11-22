import logging
import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData
import discord
import asyncio
import io

# Suppress specific RTCP warning from voice_recv
class RTCPFilter(logging.Filter):
    def filter(self, record):
        return "Received unexpected rtcp packet: type=200" not in record.getMessage()

def setup_audio_logging():
    logging.getLogger("discord.ext.voice_recv.reader").addFilter(RTCPFilter())

class STTSink(AudioSink):
    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        print("STTSink initialized.")
    
    def wants_opus(self):
        return False

    def cleanup(self):
        print("STTSink cleanup.")
        pass

    def write(self, user, data: VoiceData):
        if self.audio_queue is None:
            return
            
        try:
            # Efficiently convert PCM data using numpy
            audio_data = np.frombuffer(data.pcm, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_data = audio_data.mean(axis=1).astype(np.int16)
            resampled_data = mono_data[::3] # Downsample 48k -> 16k
            
            # Offload to the STT process
            self.audio_queue.put((user.id, resampled_data.tobytes()))
            
        except Exception as e:
            print(f"Error in write: {e}")

class AudioPlayer:
    def __init__(self, voice_client, loop):
        self.voice_client = voice_client
        self.loop = loop
        self.queue = asyncio.Queue()
        self.is_playing = False

    async def add_audio(self, audio_data):
        await self.queue.put(audio_data)
        if not self.is_playing:
            self._play_next()

    def _play_next(self):
        if self.queue.empty():
            self.is_playing = False
            return
        
        self.is_playing = True
        
        try:
            audio_data = self.queue.get_nowait()
        except asyncio.QueueEmpty:
            self.is_playing = False
            return

        # Convert bytes to AudioSource (WAV -> PCM)
        # FFmpegPCMAudio handles WAV headers automatically
        audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_data), pipe=True)
        
        if self.voice_client and self.voice_client.is_connected():
            self.voice_client.play(audio_source, after=self._after_play)
        else:
            print("Voice client not connected. Skipping audio.")
            self.is_playing = False

    def _after_play(self, error):
        if error:
            print(f"Player error: {error}")
        # Schedule next play on the main loop
        self.loop.call_soon_threadsafe(self._play_next)
