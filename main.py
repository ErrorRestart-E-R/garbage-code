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
from dataclasses import dataclass
from typing import Optional, Tuple

# Custom Modules
from stt_handler import run_stt_process
from memory_manager import MemoryManager
from conversation_history import ConversationHistory
from llm_interface import judge_conversation, get_response_stream
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
        
        # State
        self.is_responding = False
        self.message_counter = 0  # Unique message ID
        self.last_judged_index = 0  # Last message index we judged
        
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
                        
                        # Background memory save (fire and forget)
                        asyncio.create_task(
                            self._save_memory_background(user_name, user_text)
                        )
                        
                        logger.debug(f"History: Added message #{self.message_counter} from {user_name}")
                
                await asyncio.sleep(0.01)  # Yield control
                
            except Exception as e:
                logger.error(f"History worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def judge_worker(self):
        """
        Worker 2: Judge whether AI should respond
        
        - Polls for new messages when not responding
        - Waits until response completes, then checks latest history
        - Uses all accumulated messages for context
        """
        logger.info("Judge worker started")
        
        while True:
            try:
                # Wait if currently responding
                if self.is_responding:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if there are new messages since last judgment
                if self.message_counter <= self.last_judged_index:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get the latest history state (includes all accumulated messages)
                history_text, current_speaker, current_message = \
                    self.history.get_history_and_current()
                
                if not current_speaker or not current_message:
                    await asyncio.sleep(0.1)
                    continue
                
                # Mark as judged before calling LLM
                current_index = self.message_counter
                self.last_judged_index = current_index
                
                # Call Judge LLM with latest context
                participant_count = self.history.get_participant_count()
                
                should_respond, reason = await judge_conversation(
                    conversation_history=history_text,
                    current_speaker=current_speaker,
                    current_message=current_message,
                    participant_count=participant_count
                )
                
                # Print Judge decision to terminal
                judge_result = "Y" if should_respond else "N"
                print(f"  [Judge: {judge_result}]")
                
                logger.debug(f"Judge #{current_index}: {should_respond} - {reason}")
                
                if should_respond:
                    # Queue response with latest history
                    pending = PendingResponse(
                        speaker=current_speaker,
                        message=current_message,
                        history_json=history_text,
                        message_index=current_index
                    )
                    await self.response_queue.put(pending)
                
            except Exception as e:
                logger.error(f"Judge worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def response_worker(self):
        """
        Worker 3: Generate responses and play TTS
        
        - Receives approved responses from judge_worker
        - Generates LLM response with streaming
        - Plays TTS audio
        - When complete, judge_worker will check for new messages
        """
        logger.info("Response worker started")
        
        while True:
            try:
                # Wait for response request
                pending: PendingResponse = await self.response_queue.get()
                
                # Mark as responding
                self.is_responding = True
                
                try:
                    logger.debug(f"Response: Starting for #{pending.message_index}")
                    
                    full_response = await self._execute_response(
                        user_name=pending.speaker,
                        user_text=pending.message,
                        history_json=pending.history_json
                    )
                    
                    if full_response:
                        # Add AI response to history
                        self.history.add_ai_response(full_response)
                        logger.debug(f"Response: Completed #{pending.message_index}")
                    
                finally:
                    # Mark as not responding - judge_worker will poll for new messages
                    self.is_responding = False
                
            except Exception as e:
                logger.error(f"Response worker error: {e}", exc_info=True)
                self.is_responding = False
                await asyncio.sleep(0.1)
    
    async def _save_memory_background(self, user_name: str, user_text: str):
        """Background memory saving task"""
        try:
            await asyncio.to_thread(
                self.memory_manager.save_memory, user_name, user_text
            )
        except Exception as e:
            logger.error(f"Memory save error: {e}")
    
    async def _execute_response(self, user_name: str, user_text: str, history_json: str) -> Optional[str]:
        """Generate LLM response and play TTS"""
        
        # Find Voice Client
        vc = self.bot.voice_clients[0] if self.bot.voice_clients else None
        if not vc:
            logger.warning("Not in voice channel")
            return None
        
        player = AudioPlayer(vc, self.bot.loop)
        
        full_response = ""
        buffer = ""
        
        # Start TTS worker
        tts_queue = asyncio.Queue()
        tts_task = asyncio.create_task(
            self._tts_worker(tts_queue, player)
        )
        
        # Search long-term memory
        memory_context = await asyncio.to_thread(
            self.memory_manager.get_memory_context, user_text, user_name
        )
        
        # Start streaming output
        print(f"{config.AI_NAME}: ", end="", flush=True)
        
        async for chunk in get_response_stream(
            user_name=user_name,
            user_text=user_text,
            conversation_history=history_json,
            memory_context=memory_context
        ):
            if chunk:
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
        
        print()  # Newline
        
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
        logger.info("Starting conversation workers...")
        
        await asyncio.gather(
            self.history_worker(),
            self.judge_worker(),
            self.response_worker(),
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


# ============================================================
# Discord Bot Setup
# ============================================================

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


# ============================================================
# Main Entry Point
# ============================================================

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
