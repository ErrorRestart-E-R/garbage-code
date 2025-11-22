import discord
from discord.ext import commands
import discord.ext.voice_recv
from discord.ext.voice_recv import AudioSink, VoiceData
import os
import numpy as np
import multiprocessing
import queue
from stt_handler import run_stt_process
import datetime
import config
import openai
import json

# LLM Configuration
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# System Prompt
SYSTEM_PROMPT = """
You are "Neuro", an AI VTuber. 
Personality: Cheerful, slightly sassy, and confident.
Instruction: You will receive input in JSON: {"user_text": "..."}.
Response Requirement: You must respond STRICTLY in the following JSON format. Do not add any markdown or conversational text outside the JSON.
{
  "message": "Your response here",
  "emotion": "emotion_tag",
  "action": "action_tag"
}
Allowed emotions: neutral, happy, angry, sad, surprised
Allowed actions: idle, wave_hand, nod, shake_head
"""

# Initialize Async Client
aclient = openai.AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

async def get_neuro_response(user_input):
    """
    Sends the user input to the LLM and retrieves the response asynchronously.
    """
    # Wrap user input in JSON structure
    json_input = json.dumps({"user_text": user_input})

    try:
        completion = await aclient.chat.completions.create(
            model="local-model", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json_input}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def parse_response(response_text):
    """
    Parses the JSON response from the LLM, handling potential markdown wrapping.
    """
    if not response_text:
        return None

    clean_text = response_text.strip()
    
    # Remove markdown code blocks if present (e.g. ```json ... ```)
    if clean_text.startswith("```"):
        first_newline = clean_text.find("\\n")
        if first_newline != -1:
            clean_text = clean_text[first_newline+1:]
        
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
            
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
        return data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON. Raw output:\\n{response_text}")
        return None

# Token
TOKEN = config.DISCORD_TOKEN

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

# Global Queues
audio_queue = None
result_queue = None
command_queue = None
stt_process = None

class STTSink(AudioSink):
    def __init__(self):
        super().__init__()
        print("STTSink initialized.")
    
    def wants_opus(self):
        return False

    def cleanup(self):
        print("STTSink cleanup.")
        pass

    def write(self, user, data: VoiceData):
        if audio_queue is None:
            return
            
        try:
            # Efficiently convert PCM data using numpy
            audio_data = np.frombuffer(data.pcm, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_data = audio_data.mean(axis=1).astype(np.int16)
            resampled_data = mono_data[::3] # Downsample 48k -> 16k
            
            # Offload to the STT process
            audio_queue.put((user.id, resampled_data.tobytes()))
            
        except Exception as e:
            print(f"Error in write: {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    
    # Start Result Listener
    bot.loop.create_task(process_results())

async def process_results():
    print("Result processing task started.")
    import json
    while True:
        try:
            # Check for new transcriptions without blocking
            if result_queue and not result_queue.empty():
                result = result_queue.get()
                # print(json.dumps(result, ensure_ascii=False)) # Original STT output
                
                user_text = result.get("text", "")
                if user_text:
                    print(f"User said: {user_text}")
                    
                    # Query LLM
                    llm_response_text = await get_neuro_response(user_text)
                    
                    if llm_response_text:
                        parsed_data = parse_response(llm_response_text)
                        if parsed_data:
                            # Print the final JSON for the next stage (Unity/TTS)
                            print(json.dumps(parsed_data, ensure_ascii=False))
                        else:
                            print("Failed to parse LLM response.")
                    else:
                        print("No response from LLM.")
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                # Or just await asyncio.sleep(0.01)
                import asyncio
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error in result loop: {e}")
            import asyncio
            await asyncio.sleep(1)

@bot.event
async def on_voice_state_update(member, before, after):
    # Check if user left the voice channel where the bot is
    if before.channel and not after.channel:
        # User left
        if command_queue:
            command_queue.put(("LEAVE", member.id))

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)
        
        vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
        vc.listen(STTSink())
        await ctx.send(f"Joined {channel} and listening.")
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")

if __name__ == "__main__":
    # Multiprocessing Support for Windows (Required)
    multiprocessing.freeze_support()
    
    # IPC Queues for process communication
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    # Spawn the heavy STT logic in a separate process
    # This keeps the main bot loop responsive
    stt_process = multiprocessing.Process(target=run_stt_process, args=(audio_queue, result_queue, command_queue))
    stt_process.daemon = True # Auto-kill when main process exits
    stt_process.start()
    
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found.")
    else:
        try:
            bot.run(TOKEN)
        except KeyboardInterrupt:
            pass
        finally:
            print("Terminating STT Process...")
            stt_process.terminate()
            stt_process.join()