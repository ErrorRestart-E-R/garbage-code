import discord
from discord.ext import commands
import datetime
import config
import json
import asyncio
import multiprocessing
import queue

# Custom Modules
from context_manager import ConversationManager
from memory_manager import MemoryManager
from llm_interface import should_respond, is_important, get_neuro_response_stream
from audio_utils import setup_audio_logging, AudioPlayer
from tts_handler import tts_handler
from process_manager import STTProcessManager
from cogs.voice import VoiceCog

# Setup Logging
setup_audio_logging()

# Initialize Managers
conversation_manager = ConversationManager()
memory_manager = MemoryManager()
tts = tts_handler(config.TTS_SERVER_URL, config.TTS_REFERENCE_FILE, config.TTS_REFERENCE_PROMPT, config.TTS_REFERENCE_PROMPT_LANG)
process_manager = STTProcessManager()

# Token
TOKEN = config.DISCORD_TOKEN

intents = discord.Intents.default()
intents.message_content = True
intents.members = True 

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

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
            # Check if result queue has data
            if not process_manager.result_queue.empty():
                result = process_manager.result_queue.get()
                
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
                    
                    # 3. Parallel Execution: RAG Search & Judge LLM
                    llm_input_json = json.dumps({
                        "name": user_name,
                        "message": user_text
                    }, ensure_ascii=False)
                    
                    system_context = conversation_manager.get_system_context()
                    
                    # RAG is blocking (ChromaDB), so we run it in a thread.
                    rag_task = asyncio.to_thread(memory_manager.get_memory_context, user_text, user_name)
                    # Judge is async.
                    judge_task = should_respond(llm_input_json, system_context)
                    
                    # Wait for both to finish
                    memory_context, should_reply = await asyncio.gather(rag_task, judge_task)

                    # 4. Judge & Respond
                    if should_reply:
                        print("Neuro decided to reply.")
                        
                        # Add memory context if available
                        if memory_context:
                            system_context += memory_context
                        
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
                        while not process_manager.result_queue.empty():
                            try:
                                process_manager.result_queue.get_nowait()
                            except queue.Empty:
                                break
                        print("STT queue cleared after response.")

                    else:
                        print("Neuro decided NOT to reply.")
            else:
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"CRITICAL Error in result loop: {e}")
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    
    # Add Cogs
    await bot.add_cog(VoiceCog(bot, process_manager, conversation_manager))
    
    # Start Result Processing Loop
    bot.loop.create_task(process_results())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Start STT Process
    process_manager.start()
    
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found.")
    else:
        try:
            bot.run(TOKEN)
        except KeyboardInterrupt:
            pass
        finally:
            print("Shutting down...")
            process_manager.stop()