import discord
from discord.ext import commands
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
import logging
from context_manager import ConversationManager
from memory_manager import MemoryManager

# Suppress specific RTCP warning from voice_recv
class RTCPFilter(logging.Filter):
    def filter(self, record):
        return "Received unexpected rtcp packet: type=200" not in record.getMessage()

logging.getLogger("discord.ext.voice_recv.reader").addFilter(RTCPFilter())

# LLM Configuration
# Initialize Async Clients
chat_client = openai.AsyncOpenAI(base_url=config.CHAT_API_BASE_URL, api_key=config.CHAT_API_KEY)
judge_client = openai.AsyncOpenAI(base_url=config.JUDGE_API_BASE_URL, api_key=config.JUDGE_API_KEY)

# System Prompt
SYSTEM_PROMPT = """
You are "Neuro", an AI VTuber. 

Please do not respond with absurdly long answer.
Do not try to include motion or any other non-textual content.
Do not try to include emojis.
Do not try to include trailing questions if not necessary.
"""

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = """
You are a conversation analyzer for "Neuro", an AI VTuber.
Your job is to decide if Neuro should join the conversation.
Respond with ONLY 'Y' (Yes) or 'N' (No).

Neuro is curious, friendly, and likes to chat.
She SHOULD respond if:
- The user is talking to her (obviously).
- The topic is interesting, funny, or something she can comment on.
- The user is expressing an opinion or asking a general question.
- She wants to join the banter.

She should NOT respond ONLY if:
- The input is just noise or very short (e.g. "ok", "hmm").
- The users are having a strictly private or technical conversation that doesn't concern her.

BE MORE PROACTIVE. If in doubt, say 'Y'.
"""

# Importance Judge Prompt
IMPORTANCE_SYSTEM_PROMPT = """
You are a memory assistant. Your job is to decide if a user's message contains important information worth saving to long-term memory.
Important information includes:
- Personal details (name, age, location, job).
- Preferences (likes, dislikes, hobbies, favorites).
- Specific facts about the user's life or history.
- Important context for future conversations.

Respond with ONLY 'Y' (Yes) or 'N' (No).
"""

# Initialize Managers
conversation_manager = ConversationManager()
memory_manager = MemoryManager()

import asyncio

async def should_respond(user_input, system_context):
    """
    Decides whether to respond to the user input using the Judge LLM.
    """
    try:
        completion = await asyncio.wait_for(
            judge_client.chat.completions.create(
                model=config.JUDGE_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT + "\n" + system_context},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, 
            ),
            timeout=5.0 # 5 second timeout for Judge
        )
        response = completion.choices[0].message.content.strip().upper()
        return "Y" in response
    except asyncio.TimeoutError:
        print("Judge LLM Timed out. Defaulting to False.")
        return False
    except Exception as e:
        print(f"Judge LLM Error: {e}")
        return False

async def is_important(user_input):
    """
    Decides if the input is worth saving to memory.
    """
    try:
        completion = await asyncio.wait_for(
            judge_client.chat.completions.create(
                model=config.JUDGE_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": IMPORTANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, 
            ),
            timeout=10.0 # 10 second timeout for Importance (less critical)
        )
        response = completion.choices[0].message.content.strip().upper()
        return "Y" in response
    except Exception as e:
        print(f"Importance Judge Error: {e}")
        return False

async def process_memory_background(user_name, user_text):
    """
    Background task to handle memory saving.
    This runs independently of the main response flow.
    """
    try:
        # Check Importance
        if await is_important(user_text):
            print(f"[Background] Importance Judge: Saving memory for {user_name}")
            # Offload DB write to thread if needed, but chromadb is fast enough usually
            memory_manager.save_memory(user_name, user_text)
        else:
            # print(f"[Background] Not important enough to save.")
            pass
    except Exception as e:
        print(f"[Background] Memory Task Error: {e}")

async def get_neuro_response(user_input_json, system_context):
    """
    Sends the user input to the LLM and retrieves the response asynchronously.
    """
    try:
        completion = await chat_client.chat.completions.create(
            model=config.CHAT_MODEL_NAME, 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n" + system_context},
                {"role": "user", "content": user_input_json}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

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
            audio_data = np.frombuffer(data.pcm, dtype=np.int16)
            audio_data = audio_data.reshape(-1, 2)
            mono_data = audio_data.mean(axis=1).astype(np.int16)
            resampled_data = mono_data[::3] 
            
            audio_queue.put((user.id, resampled_data.tobytes()))
            
        except Exception as e:
            print(f"Error in write: {e}")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    bot.loop.create_task(process_results())

async def process_results():
    print("Result processing task started.")
    import json
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
                    
                    # 2. Fire-and-Forget Memory Task (Background)
                    bot.loop.create_task(process_memory_background(user_name, user_text))
                    
                    # 3. Retrieve Relevant Long-term Memories
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
                        llm_response_text = await get_neuro_response(llm_input_json, system_context)
                        
                        if llm_response_text:
                            print(f"Neuro said: {llm_response_text}")
                            
                            # Add Bot's response to short-term memory too
                            conversation_manager.add_message("Neuro", llm_response_text)
                            
                            while not result_queue.empty():
                                try:
                                    result_queue.get_nowait()
                                except queue.Empty:
                                    break
                            print("STT queue cleared after response.")
                        else:
                            print("No response from LLM.")
                    else:
                        print("Neuro decided NOT to reply.")
            else:
                await discord.utils.sleep_until(discord.utils.utcnow() + datetime.timedelta(milliseconds=10))
                import asyncio
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"CRITICAL Error in result loop: {e}")
            import asyncio
            await asyncio.sleep(1)

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return

    # Check if the bot is in a voice channel in the guild
    # and if the member's channel change is relevant to the bot's channel
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
        vc.listen(STTSink())
        
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