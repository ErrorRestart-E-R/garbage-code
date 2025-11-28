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
from memory_manager import MemoryManager
from llm_interface import get_llm_response_stream
from audio_utils import STTSink, AudioPlayer
from tts_handler import tts_handler
from logger import setup_logger
from conversation_orchestrator import ConversationOrchestrator, OrchestratorAction

# Setup Logging
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Suppress verbose RTCP packet logs from voice_recv
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.player").setLevel(logging.WARNING)

# Initialize Managers
orchestrator = ConversationOrchestrator(
    ai_name=config.AI_NAME,
    logger_file=config.LOG_FILE,
    log_level=config.LOG_LEVEL
)
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
    Background task to handle memory saving using mem0.
    mem0 automatically determines what's important and extracts facts.
    Runs in thread pool to avoid blocking the event loop.
    """
    try:
        await asyncio.to_thread(memory_manager.save_memory, user_name, user_text)
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

async def execute_response(user_name: str, user_text: str, system_context: str):
    """
    Execute LLM response and TTS playback.
    """
    logger.debug(f"{config.AI_NAME} responding to {user_name}")
    
    # Find Voice Client for AudioPlayer
    vc = None
    if bot.voice_clients:
        vc = bot.voice_clients[0] 
    
    if not vc:
        logger.warning("Not in voice channel")
        return None
    
    player = AudioPlayer(vc, bot.loop)
    
    full_response = ""
    buffer = ""
    
    # Start TTS worker
    tts_queue = asyncio.Queue()
    tts_task = asyncio.create_task(tts_worker(tts_queue, tts, player))
    
    # Build LLM input
    llm_input_json = json.dumps({
        "name": user_name,
        "message": user_text
    }, ensure_ascii=False)
    
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
    
    return full_response

async def process_results():
    """
    STT result processing main loop.
    
    New conversation algorithm applied:
    1. AddressDetector for speech target analysis
    2. TurnManager for turn/timing decisions
    3. ConversationOrchestrator for overall flow coordination
    """
    logger.info("Result processing started with new conversation algorithm")
    
    pending_response_task = None
    
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
                    # Clean conversation output
                    print(f"\n{user_name}: {user_text}")
                    
                    # 1. Process message with Orchestrator
                    orch_result = await orchestrator.process_message(user_name, user_text)
                    
                    logger.debug(
                        f"Orchestrator result: action={orch_result.action.value}, "
                        f"wait={orch_result.wait_seconds:.1f}s, reason={orch_result.reason}"
                    )
                    
                    # 2. Background Memory Task
                    bot.loop.create_task(process_memory_background(user_name, user_text))
                    
                    # 3. Handle based on action
                    if orch_result.action == OrchestratorAction.SKIP:
                        # Don't respond to this message
                        logger.debug(f"Skipping response: {orch_result.reason}")
                        continue
                    
                    elif orch_result.action == OrchestratorAction.CANCEL_PENDING:
                        # Cancel pending response
                        if pending_response_task and not pending_response_task.done():
                            pending_response_task.cancel()
                            pending_response_task = None
                        logger.debug("Cancelled pending response")
                        continue
                    
                    elif orch_result.action in [OrchestratorAction.RESPOND, OrchestratorAction.WAIT]:
                        # Cancel previous pending response if exists
                        if pending_response_task and not pending_response_task.done():
                            pending_response_task.cancel()
                        
                        # Create new response task
                        async def respond_after_wait():
                            try:
                                # Wait
                                should_respond = await orchestrator.wait_and_respond(orch_result)
                                
                                if should_respond:
                                    # 3. Retrieve Long-term Memories
                                    memory_context = await asyncio.to_thread(
                                        memory_manager.get_memory_context, user_text, user_name
                                    )
                                    
                                    # 4. Build Full Context
                                    system_context = orchestrator.get_system_context()
                                    if memory_context:
                                        system_context += memory_context
                                    
                                    # 5. Execute Response
                                    full_response = await execute_response(user_name, user_text, system_context)
                                    
                                    if full_response:
                                        # Record AI response
                                        orchestrator.record_ai_response(full_response)
                                        
                                        # Clear STT queue during response
                                        while not result_queue.empty():
                                            try:
                                                result_queue.get_nowait()
                                            except queue.Empty:
                                                break
                                        logger.debug("STT queue cleared")
                                else:
                                    logger.debug("Response cancelled or skipped after wait")
                                    
                            except asyncio.CancelledError:
                                logger.debug("Response task cancelled")
                            except Exception as e:
                                logger.error(f"Error in respond_after_wait: {e}", exc_info=True)
                        
                        pending_response_task = asyncio.create_task(respond_after_wait())
            
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Critical error in result loop: {e}", exc_info=True)
            await asyncio.sleep(1)

@bot.event
async def on_ready():
    logger.info(f"Bot started: {bot.user}")
    logger.info(f"Using conversation algorithm v2 with AddressDetector and TurnManager")
    bot.loop.create_task(process_results())

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return

    if after.channel and after.channel.guild.voice_client and after.channel == after.channel.guild.voice_client.channel:
        orchestrator.add_participant(member.display_name)
        logger.info(f"{member.display_name} joined voice")
    elif before.channel and before.channel.guild.voice_client and before.channel == before.channel.guild.voice_client.channel:
        orchestrator.remove_participant(member.display_name)
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
                orchestrator.add_participant(member.display_name)
        
        await ctx.send(f"Joined {channel} and listening.")
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        orchestrator.conversation_manager.participants.clear()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")

@bot.command()
async def status(ctx):
    """Check conversation status command"""
    summary = orchestrator.get_conversation_summary()
    status_msg = (
        f"**Conversation Status**\n"
        f"- Participants: {', '.join(summary['participants']) or 'None'}\n"
        f"- Situation: {summary['situation']}\n"
        f"- Flow: {summary['flow']}\n"
        f"- Message count: {summary['message_count']}\n"
        f"- Consecutive responses: {summary['consecutive_responses']}"
    )
    await ctx.send(status_msg)

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
