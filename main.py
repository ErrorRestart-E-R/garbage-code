import discord
from discord.ext import commands
import discord.ext.voice_recv
import multiprocessing
import queue
import datetime
import config
import json
import asyncio
import sys
import logging

# Custom Modules
from stt_handler import run_stt_process
from context_manager import ConversationManager
from memory_manager import MemoryManager
from llm_interface import should_respond, is_important, get_llm_response_stream
from audio_utils import STTSink, AudioPlayer
from tts_handler import tts_handler
from logger import setup_logger

# Setup Logging
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Suppress verbose RTCP packet logs from voice_recv
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.player").setLevel(logging.WARNING)

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
            logger.debug(f"Saving memory for {user_name}")
            memory_manager.save_memory(user_name, user_text)
    except Exception as e:
        logger.error(f"Memory save error: {e}")

async def tts_worker(tts_queue, tts_handler, player):
    """
    Background worker to process text-to-speech conversion sequentially.
    """
    while True:
        text = await tts_queue.get()
        if text is None:  # Sentinel to stop the worker
            tts_queue.task_done()
            break
            
        try:
            wav_data = await tts_handler.get_async(text, config.TTS_LANG)
            if wav_data:
                await player.add_audio(wav_data)
        except Exception as e:
            logging.error(f"TTS worker error: {e}")
            
        tts_queue.task_done()

async def process_results():
    logger.info("Result processing started")
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
                    # Use print for clean conversation output
                    print(f"\n{user_name}: {user_text}")
                    
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
                        logger.debug(f"{config.AI_NAME} responding")
                        
                        # Find Voice Client for AudioPlayer
                        vc = None
                        if bot.voice_clients:
                            vc = bot.voice_clients[0] 
                        
                        if vc:
                            player = AudioPlayer(vc, bot.loop)
                            
                            full_response = ""
                            buffer = ""
                            
                            # Start TTS worker
                            tts_queue = asyncio.Queue()
                            tts_task = asyncio.create_task(tts_worker(tts_queue, tts, player))
                            
                            # Start streaming output
                            print(f"{config.AI_NAME}: ", end="", flush=True)
                            
                            async for chunk in get_llm_response_stream(llm_input_json, system_context):
                                if chunk:
                                    # Print chunk to terminal immediately
                                    print(chunk, end="", flush=True)
                                    
                                    full_response += chunk
                                    buffer += chunk
                                    
                                    # Check for punctuation to split sentences
                                    if any(p in buffer for p in ['.', '!', '?', '\n']):
                                        if buffer.strip() and buffer.strip()[-1] in ['.', '!', '?', '\n']:
                                            sentence = buffer.strip()
                                            logger.debug(f"TTS queue: {sentence[:30]}...")
                                            
                                            # Send to TTS queue (Non-blocking)
                                            await tts_queue.put(sentence)
                                            
                                            buffer = ""
                            # End of stream newline
                            print()
                            
                            # Process remaining buffer
                            if buffer.strip():
                                logger.debug(f"TTS queue final: {buffer.strip()[:30]}...")
                                await tts_queue.put(buffer.strip())
                                
                            # Signal worker to stop and wait for it to finish
                            await tts_queue.put(None)
                            await tts_task
                            
                            # logger.info(f"{config.AI_NAME}: {full_response}") # Removed to avoid duplicate
                            conversation_manager.add_message(config.AI_NAME, full_response)
                        else:
                            logger.warning("Not in voice channel")
                            
                        # Clear STT queue during response
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                            except queue.Empty:
                                break
                        logger.debug("STT queue cleared")

                    else:
                        logger.debug(f"{config.AI_NAME} not responding")
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Critical error in result loop: {e}", exc_info=True)
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    logger.info(f"Bot started: {bot.user}")
    bot.loop.create_task(process_results())

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return

    if after.channel and after.channel.guild.voice_client and after.channel == after.channel.guild.voice_client.channel:
        conversation_manager.add_participant(member.display_name)
        logger.info(f"{member.display_name} joined voice")
    elif before.channel and before.channel.guild.voice_client and before.channel == before.channel.guild.voice_client.channel:
        conversation_manager.remove_participant(member.display_name)
        logger.info(f"{member.display_name} left voice")

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
    from all_api_testing import run_all_tests
    
    print("\n🚀 Starting AI VTuber Bot...\n")
    if config.ENABLE_PREFLIGHT_CHECKS:
        if not run_all_tests():
            print("\n❌ Pre-flight checks failed.\n")
            sys.exit(1)
        print("✓ All systems operational\n")
    else:
        print("⚠️  Pre-flight checks skipped (set ENABLE_PREFLIGHT_CHECKS=true to enable)\n")
    
    multiprocessing.freeze_support()
    
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    stt_process = multiprocessing.Process(target=run_stt_process, args=(audio_queue, result_queue, command_queue))
    stt_process.daemon = True 
    stt_process.start()
    
    if not TOKEN:
        logger.error("DISCORD_TOKEN not found")
        sys.exit(1)
    else:
        try:
            bot.run(TOKEN)
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Shutting down...")
            stt_process.terminate()
            stt_process.join()