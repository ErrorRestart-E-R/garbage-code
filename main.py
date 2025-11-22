import discord
from discord.ext import commands
import discord.ext.voice_recv
import multiprocessing
import queue
import datetime
import config
import json
import asyncio

# Custom Modules
from stt_handler import run_stt_process
from context_manager import ConversationManager
from memory_manager import MemoryManager
from llm_interface import should_respond, is_important, get_neuro_response_stream
from audio_utils import STTSink, setup_audio_logging, AudioPlayer
from tts_handler import tts_handler

# Setup Logging
setup_audio_logging()

# Initialize Managers
conversation_manager = ConversationManager()
memory_manager = MemoryManager()
tts = tts_handler(config.TTS_SERVER_URL, config.TTS_REFERENCE_FILE, config.TTS_REFERENCE_PROMPT, config.TTS_REFERENCE_PROMPT_LANG)

# Token
TOKEN = config.DISCORD_TOKEN

intents = discord.Intents.default()
intents.message_content = True
intents.members = True 

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

# Global Queues
audio_queue = None
result_queue = None
command_queue = None
stt_process = None

async def process_memory_background(user_name, user_text):
    """
    Background task to handle memory saving.
    """
    try:
        if await is_important(user_text):
            print(f"[Background] Importance Judge: Saving memory for {user_name}")
            memory_manager.save_memory(user_name, user_text)
    except Exception as e:
        print(f"[Background] Memory Task Error: {e}")

async def process_results():
    print("Result processing task started.")
    while True:
        try:
            if result_queue and not result_queue.empty():
                result = result_queue.get()
                
                user_id = result.get("user_id")
                user_text = result.get("text", "")
                
                user_name = "Unknown User"
                if user_id:
                    user = bot.get_user(user_id)
                    if user:
                        user_name = user.display_name
                    else:
                        user_name = f"User_{user_id}"
                
                if user_text:
                    print(f"{user_name} said: {user_text}")
                    
                    # 1. Update Short-term Context
                    conversation_manager.add_message(user_name, user_text)
                    
                    # 2. Background Memory Task
                    bot.loop.create_task(process_memory_background(user_name, user_text))
                    
                    # 3. Retrieve Long-term Memories
                    memory_context = memory_manager.get_memory_context(user_text, user_name)
                    
                    # 4. Build Full Context
                    system_context = conversation_manager.get_system_context()
                    if memory_context:
                        system_context += memory_context
                    
                    llm_input_json = json.dumps({
                        "name": user_name,
                        "message": user_text
                    }, ensure_ascii=False)

                    # 5. Judge & Respond
                    if await should_respond(llm_input_json, system_context):
                        print("Neuro decided to reply.")
                        
                        # Find Voice Client for AudioPlayer
                        vc = None
                        if bot.voice_clients:
                            vc = bot.voice_clients[0] 
                        
                        if vc:
                            player = AudioPlayer(vc, bot.loop)
                            
                            full_response = ""
                            buffer = ""
                            print("Neuro is thinking (streaming)...")
                            
                            llm_chunks = 0
                            
                            async for chunk in get_neuro_response_stream(llm_input_json, system_context):
                                if chunk:
                                    full_response += chunk
                                    buffer += chunk

                                    MIN_WORD_COUNT = config.TTS_EARLY_MIN_WORD_COUNT if llm_chunks <= config.TTS_EARLY_CHUNKS else config.TTS_MIN_WORD_COUNT

                                    # Split by space to check word count
                                    if len(buffer.split()) >= MIN_WORD_COUNT:
                                        llm_chunks += 1
                                        last_space_index = buffer.rfind(' ')
                                        if last_space_index != -1:
                                            sentence = buffer[:last_space_index]
                                            buffer = buffer[last_space_index+1:]
                                            
                                            print(f"\n[TTS] Processing chunk: {sentence}")
                                            wav_data = await tts.get_async(sentence, config.TTS_LANG)
                                            if wav_data:
                                                await player.add_audio(wav_data)
                            
                            # Process remaining buffer
                            if buffer.strip():
                                print(f"\n[TTS] Processing final chunk: {buffer.strip()}")
                                wav_data = await tts.get_async(buffer.strip(), config.TTS_LANG)
                                if wav_data:
                                    await player.add_audio(wav_data)
                                    
                            print("\nNeuro response complete.")
                            conversation_manager.add_message("Neuro", full_response)
                        else:
                            print("Neuro wants to reply but is not in a voice channel.")
                            
                        # Clear STT queue during response
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                            except queue.Empty:
                                break
                        print("STT queue cleared after response.")

                    else:
                        print("Neuro decided NOT to reply.")
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"CRITICAL Error in result loop: {e}")
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    bot.loop.create_task(process_results())

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return

    if after.channel and after.channel.guild.voice_client and after.channel == after.channel.guild.voice_client.channel:
        conversation_manager.add_participant(member.display_name)
        print(f"Participant added: {member.display_name}")
    elif before.channel and before.channel.guild.voice_client and before.channel == before.channel.guild.voice_client.channel:
        conversation_manager.remove_participant(member.display_name)
        print(f"Participant removed: {member.display_name}")

    if before.channel and not after.channel:
        if command_queue:
            command_queue.put(("LEAVE", member.id))

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)
        
        vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
        # Pass the global audio_queue to the Sink
        vc.listen(STTSink(audio_queue))
        
        for member in channel.members:
            if not member.bot:
                conversation_manager.add_participant(member.display_name)
        
        await ctx.send(f"Joined {channel} and listening.")
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        conversation_manager.participants.clear()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    stt_process = multiprocessing.Process(target=run_stt_process, args=(audio_queue, result_queue, command_queue))
    stt_process.daemon = True 
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