import asyncio
import logging
import multiprocessing
import sys
from typing import Optional

import discord
from discord.ext import commands
import discord.ext.voice_recv

import config
from logger import setup_logger
from stt_handler import run_stt_process
from audio_utils import STTSink
from app.controller import ConversationController


logger = setup_logger(__name__, config.LOG_FILE, config.LOG_LEVEL)

# Suppress verbose logs
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.player").setLevel(logging.WARNING)


class BotRuntime:
    """
    Discord bot + 프로세스/큐 wiring (인프라 레이어).
    애플리케이션 레이어(ConversationController)를 생성하고 이벤트에 연결합니다.
    """

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        self.bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)
        self.controller: Optional[ConversationController] = None

        # Queues & STT process
        self.audio_queue: Optional[multiprocessing.Queue] = None
        self.result_queue: Optional[multiprocessing.Queue] = None
        self.command_queue: Optional[multiprocessing.Queue] = None
        self.stt_process: Optional[multiprocessing.Process] = None

        self._register_handlers()

    def _register_handlers(self):
        @self.bot.event
        async def on_ready():
            logger.info(f"Bot started: {self.bot.user}")

            self.controller = ConversationController(
                bot=self.bot,
                voice_client_getter=lambda: (self.bot.voice_clients[0] if self.bot.voice_clients else None),
            )
            self.controller.set_result_queue(self.result_queue)

            # Start workers
            self.bot.loop.create_task(self.controller.start_workers())

        @self.bot.event
        async def on_voice_state_update(member, before, after):
            if member.bot or not self.controller:
                return

            # Participant joined
            if (
                after.channel
                and after.channel.guild.voice_client
                and after.channel == after.channel.guild.voice_client.channel
            ):
                self.controller.add_participant(member.display_name)
                logger.info(f"{member.display_name} joined voice")

            # Participant left
            elif (
                before.channel
                and before.channel.guild.voice_client
                and before.channel == before.channel.guild.voice_client.channel
            ):
                self.controller.remove_participant(member.display_name)
                logger.info(f"{member.display_name} left voice")

            # Notify STT process when user leaves
            if before.channel and not after.channel:
                if self.command_queue:
                    self.command_queue.put(("LEAVE", member.id))

        @self.bot.command()
        async def join(ctx):
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                if ctx.voice_client is not None:
                    return await ctx.voice_client.move_to(channel)

                vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
                vc.listen(STTSink(self.audio_queue))

                # Register current channel participants
                if self.controller:
                    for member in channel.members:
                        if not member.bot:
                            self.controller.add_participant(member.display_name)

                await ctx.send(f"Joined {channel} and listening.")
            else:
                await ctx.send("You are not in a voice channel.")

        @self.bot.command()
        async def leave(ctx):
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                if self.controller:
                    self.controller.clear_participants()
                await ctx.send("Left the voice channel.")
            else:
                await ctx.send("I am not in a voice channel.")

        @self.bot.command()
        async def status(ctx):
            if self.controller:
                status = self.controller.get_status()
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

        @self.bot.command()
        async def clear(ctx):
            if self.controller:
                self.controller.clear_history()
                await ctx.send("Conversation history cleared.")
            else:
                await ctx.send("Controller not initialized.")

    def start_stt_process(self):
        multiprocessing.freeze_support()
        self.audio_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.command_queue = multiprocessing.Queue()

        self.stt_process = multiprocessing.Process(
            target=run_stt_process,
            args=(self.audio_queue, self.result_queue, self.command_queue),
        )
        self.stt_process.daemon = True
        self.stt_process.start()
        logger.info("STT process started (independent)")

    def run(self):
        token = config.DISCORD_TOKEN
        if not token:
            logger.error("DISCORD_TOKEN not found")
            sys.exit(1)

        self.start_stt_process()
        try:
            self.bot.run(token)
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Shutting down...")
            if self.stt_process:
                self.stt_process.terminate()
                self.stt_process.join()


def run_bot():
    runtime = BotRuntime()
    runtime.run()


