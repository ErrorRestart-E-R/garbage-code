"""
AI VTuber Discord Bot

Conversation Algorithm (Async Task Architecture):
1. STT Process: Runs independently, converts voice to text
2. History Worker: Receives STT results, stores in conversation history
3. Judge Worker: Monitors new messages, decides if AI should respond
4. Response Worker: Generates responses and plays TTS

All workers run independently - STT never blocks!
"""

import discord
from discord.ext import commands
import discord.ext.voice_recv
import multiprocessing
import queue
import config
import asyncio
import sys
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# Custom Modules
from stt_handler import run_stt_process
from memory_manager import MemoryManager
from conversation_history import ConversationHistory
from llm_interface import get_response_stream
from audio_utils import STTSink, AudioPlayer
from tts_handler import tts_handler
from logger import setup_logger

# Setup Logging
logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Suppress verbose logs
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.player").setLevel(logging.WARNING)

@dataclass
class PendingResponse:
    """Pending response data"""
    speaker: str
    message: str
    history_json: str
    message_index: int

class ConversationController:
    """
    Manages conversation flow with independent async workers.
    
    Workers:
    - history_worker: STT results -> conversation history (never blocks)
    - judge_worker: New messages -> Judge LLM -> response queue
    - response_worker: Response queue -> LLM response -> TTS
    """
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.history = ConversationHistory(
            max_size=config.MAX_CONVERSATION_HISTORY,
            ai_name=config.AI_NAME
        )
        self.memory_manager = MemoryManager()
        self.tts = tts_handler(
            config.TTS_SERVER_URL,
            config.TTS_REFERENCE_FILE,
            config.TTS_REFERENCE_PROMPT,
            config.TTS_REFERENCE_PROMPT_LANG
        )
        
        # Async queue for response worker
        self.response_queue: asyncio.Queue = asyncio.Queue()
        
        # Memory save queue (for sequential processing)
        self.memory_queue: asyncio.Queue = asyncio.Queue()
        
        # State
        self.is_responding = False
        self.message_counter = 0  # Unique message ID
        self.last_processed_index = 0  # Last message index we processed
        self.last_response_time = 0  # Timestamp of last AI response completion
        
        # Result queue from STT process (set externally)
        self.result_queue: Optional[multiprocessing.Queue] = None
    
    def set_result_queue(self, result_queue: multiprocessing.Queue):
        """Set the STT result queue"""
        self.result_queue = result_queue
    
    async def history_worker(self):
        """
        Worker 1: STT results -> Conversation History
        
        - Runs continuously, never blocks
        - Stores all messages regardless of response state
        - Judge worker polls for new messages when ready
        """
        logger.info("History worker started")
        
        while True:
            try:
                if self.result_queue and not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    
                    user_id = result.get("user_id")
                    user_text = result.get("text", "")
                    
                    # Get user name
                    user_name = "Unknown"
                    if user_id:
                        user = self.bot.get_user(user_id)
                        if user:
                            user_name = user.display_name
                        else:
                            user_name = f"User_{user_id}"
                    
                    if user_text:
                        # Terminal output
                        print(f"\n{user_name}: {user_text}")
                        
                        # Add to conversation history
                        self.history.add(user_name, user_text)
                        self.message_counter += 1
                        
                        # Note: Memory save moved to response_worker (after LLM response)
                        # to avoid LLM contention
                        
                        logger.debug(f"History: Added message #{self.message_counter} from {user_name}")
                
                await asyncio.sleep(0.01)  # Yield control
                
            except Exception as e:
                logger.error(f"History worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def message_worker(self):
        """
        Worker 2: Process new messages and queue for LLM response
        
        Single 27B LLM handles both judgment and response.
        LLM will decide whether to respond based on conversation context.
        """
        logger.info("Message worker started (single LLM mode)")
        
        while True:
            try:
                # Wait if currently responding
                if self.is_responding:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check for new messages
                if self.message_counter <= self.last_processed_index:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get conversation history in messages format
                messages, current_speaker, current_message, current_timestamp = \
                    self.history.get_messages_for_llm()
                
                if not current_speaker or not current_message:
                    await asyncio.sleep(0.1)
                    continue
                
                # Staleness Check
                if time.time() - current_timestamp > config.MESSAGE_STALENESS_THRESHOLD:
                    logger.warning(f"Skipping stale message #{self.message_counter} (age: {time.time() - current_timestamp:.1f}s)")
                    self.last_processed_index = self.message_counter
                    continue
                
                # Mark as processed
                current_index = self.message_counter
                self.last_processed_index = current_index
                
                # Rate Limiting
                time_since_last = time.time() - self.last_response_time
                if time_since_last < config.MIN_RESPONSE_INTERVAL:
                    wait_time = config.MIN_RESPONSE_INTERVAL - time_since_last
                    logger.info(f"Rate limit: delaying by {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                
                # Queue for response (LLM will decide whether to respond)
                participant_count = self.history.get_participant_count()
                pending = PendingResponse(
                    speaker=current_speaker,
                    message=current_message,
                    history_json=messages,  # Now contains messages list
                    message_index=current_index
                )
                await self.response_queue.put((pending, participant_count))
                
                logger.debug(f"Message #{current_index} queued for LLM")
                
            except Exception as e:
                logger.error(f"Message worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def response_worker(self):
        """
        Worker 3: Generate responses and play TTS
        
        Single LLM handles both judgment and response.
        If LLM decides not to respond, it outputs empty/minimal response.
        """
        logger.info("Response worker started (single LLM mode)")
        
        while True:
            try:
                # Wait for response request (now includes participant_count)
                pending, participant_count = await self.response_queue.get()
                
                # Mark as responding
                self.is_responding = True
                
                try:
                    logger.debug(f"Response: Starting for #{pending.message_index}")
                    
                    full_response = await self._execute_response(
                        messages=pending.history_json,  # Now contains messages list
                        participant_count=participant_count
                    )
                    
                    if full_response and full_response.strip():
                        # Add AI response to history
                        self.history.add_ai_response(full_response)
                        self.last_response_time = time.time()
                        logger.debug(f"Response: Completed #{pending.message_index}")
                        
                        # Add to memory queue
                        await self.memory_queue.put((pending.speaker, pending.message))
                    else:
                        # LLM decided not to respond
                        print(f"  [No response]")
                        logger.debug(f"LLM decided not to respond to #{pending.message_index}")
                    
                finally:
                    self.is_responding = False
                
            except Exception as e:
                logger.error(f"Response worker error: {e}", exc_info=True)
                self.is_responding = False
                await asyncio.sleep(0.1)
    
    async def memory_worker(self):
        """
        Worker 4: Memory Save Queue Processor
        
        - Processes memory saves sequentially (one at a time)
        - Prevents concurrent Ollama LLM calls
        - Runs independently from other workers
        """
        logger.info("Memory worker started (sequential queue processing)")
        
        while True:
            try:
                # Wait for memory save request
                user_name, user_text = await self.memory_queue.get()
                
                # Save to long-term memory (blocking, but in separate thread)
                try:
                    await asyncio.to_thread(
                        self.memory_manager.save_memory, user_name, user_text
                    )
                    logger.debug(f"Memory saved for {user_name}")
                except Exception as e:
                    logger.error(f"Memory save error: {e}")
                
                # Mark task as done
                self.memory_queue.task_done()
                
            except Exception as e:
                logger.error(f"Memory worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def _execute_response(self, messages: list, participant_count: int) -> Optional[str]:
        """
        Generate LLM response and play TTS.
        
        Single LLM decides whether to respond based on conversation context.
        Returns empty string if LLM decides not to respond.
        """
        # Find Voice Client
        vc = self.bot.voice_clients[0] if self.bot.voice_clients else None
        if not vc:
            logger.warning("Not in voice channel")
            return None
        
        player = AudioPlayer(vc, self.bot.loop)
        
        full_response = ""
        buffer = ""
        first_chunk = True
        
        # Start TTS worker
        tts_queue = asyncio.Queue()
        tts_task = asyncio.create_task(
            self._tts_worker(tts_queue, player)
        )
        
        # Get last user message for memory search
        last_user_msg = ""
        last_user_name = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                # Format: "speaker: text"
                content = msg["content"]
                if ": " in content:
                    last_user_name, last_user_msg = content.split(": ", 1)
                else:
                    last_user_msg = content
                break
        
        # Search long-term memory
        memory_context = ""
        if last_user_msg:
            memory_context = await asyncio.to_thread(
                self.memory_manager.get_memory_context, last_user_msg, last_user_name
            )
        
        async for chunk in get_response_stream(
            messages=messages,
            participant_count=participant_count,
            memory_context=memory_context
        ):
            if chunk:
                # Print AI name only on first chunk
                if first_chunk:
                    print(f"{config.AI_NAME}: ", end="", flush=True)
                    first_chunk = False
                
                print(chunk, end="", flush=True)
                
                full_response += chunk
                buffer += chunk
                
                # Add to TTS queue by sentence
                if any(p in buffer for p in config.TTS_SENTENCE_DELIMITERS):
                    for punct in config.TTS_SENTENCE_DELIMITERS:
                        if punct in buffer:
                            parts = buffer.split(punct, 1)
                            sentence = parts[0] + punct
                            buffer = parts[1] if len(parts) > 1 else ""
                            
                            if sentence.strip():
                                await tts_queue.put(sentence.strip())
                            break
        
        if not first_chunk:
            print()  # Newline only if we printed something
        
        # Process remaining buffer
        if buffer.strip():
            await tts_queue.put(buffer.strip())
        
        # Wait for TTS worker to finish
        await tts_queue.put(None)
        await tts_task
        
        return full_response
    
    async def _tts_worker(self, tts_queue: asyncio.Queue, player: AudioPlayer):
        """TTS conversion worker"""
        while True:
            text = await tts_queue.get()
            if text is None:
                break
            
            try:
                wav_data = await self.tts.get_async(text, config.TTS_LANG)
                if wav_data:
                    await player.add_audio(wav_data)
            except Exception as e:
                logger.error(f"TTS error: {e}")
    
    async def start_workers(self):
        """Start all async workers"""
        logger.info("Starting conversation workers (single LLM mode)...")
        
        await asyncio.gather(
            self.history_worker(),
            self.message_worker(),
            self.response_worker(),
            self.memory_worker(),
        )
    
    # Participant management
    def add_participant(self, name: str):
        self.history.add_participant(name)
    
    def remove_participant(self, name: str):
        self.history.remove_participant(name)
    
    def clear_participants(self):
        self.history.clear_participants()
    
    def clear_history(self):
        self.history.clear()
    
    def get_status(self) -> dict:
        return {
            "participants": list(self.history.participants),
            "history_count": len(self.history),
            "is_responding": self.is_responding,
            "message_counter": self.message_counter,
        }

# Token
TOKEN = config.DISCORD_TOKEN

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

# Global instances
controller: Optional[ConversationController] = None
audio_queue: Optional[multiprocessing.Queue] = None
result_queue: Optional[multiprocessing.Queue] = None
command_queue: Optional[multiprocessing.Queue] = None
stt_process: Optional[multiprocessing.Process] = None


@bot.event
async def on_ready():
    global controller
    
    logger.info(f"Bot started: {bot.user}")
    logger.info("Using async worker-based conversation algorithm")
    
    # Initialize controller
    controller = ConversationController(bot)
    controller.set_result_queue(result_queue)
    
    # Start workers
    bot.loop.create_task(controller.start_workers())


@bot.event
async def on_voice_state_update(member, before, after):
    """Voice channel state change event"""
    if member.bot or not controller:
        return

    # Participant joined
    if (after.channel and 
        after.channel.guild.voice_client and 
        after.channel == after.channel.guild.voice_client.channel):
        controller.add_participant(member.display_name)
        logger.info(f"{member.display_name} joined voice")
    
    # Participant left
    elif (before.channel and 
          before.channel.guild.voice_client and 
          before.channel == before.channel.guild.voice_client.channel):
        controller.remove_participant(member.display_name)
        logger.info(f"{member.display_name} left voice")

    # Notify STT process when user leaves
    if before.channel and not after.channel:
        if command_queue:
            command_queue.put(("LEAVE", member.id))


@bot.command()
async def join(ctx):
    """Join voice channel"""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is not None:
            return await ctx.voice_client.move_to(channel)
        
        vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
        vc.listen(STTSink(audio_queue))
        
        # Register current channel participants
        if controller:
            for member in channel.members:
                if not member.bot:
                    controller.add_participant(member.display_name)
        
        await ctx.send(f"Joined {channel} and listening.")
    else:
        await ctx.send("You are not in a voice channel.")


@bot.command()
async def leave(ctx):
    """Leave voice channel"""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        if controller:
            controller.clear_participants()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I am not in a voice channel.")


@bot.command()
async def status(ctx):
    """Check conversation status"""
    if controller:
        status = controller.get_status()
        status_msg = (
            f"**Conversation Status**\n"
            f"- Participants: {', '.join(status['participants']) if status['participants'] else 'None'}\n"
            f"- History count: {status['history_count']}\n"
            f"- Is responding: {status['is_responding']}\n"
            f"- Total messages: {status['message_counter']}\n"
        )
    else:
        status_msg = "Controller not initialized."
    await ctx.send(status_msg)


@bot.command()
async def clear(ctx):
    """Clear conversation history"""
    if controller:
        controller.clear_history()
        await ctx.send("Conversation history cleared.")
    else:
        await ctx.send("Controller not initialized.")

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
    
    # Initialize queues
    audio_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    # Start STT process (runs independently)
    stt_process = multiprocessing.Process(
        target=run_stt_process,
        args=(audio_queue, result_queue, command_queue)
    )
    stt_process.daemon = True
    stt_process.start()
    
    logger.info("STT process started (independent)")
    
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
            if stt_process:
                stt_process.terminate()
                stt_process.join()
