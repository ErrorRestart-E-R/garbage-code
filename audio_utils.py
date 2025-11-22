import logging
import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData

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
