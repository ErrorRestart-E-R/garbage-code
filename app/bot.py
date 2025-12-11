import asyncio
import logging
import multiprocessing
import queue
import sys
import time
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
        self.status_queue: Optional[multiprocessing.Queue] = None
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
        self.status_queue = multiprocessing.Queue()

        self.stt_process = multiprocessing.Process(
            target=run_stt_process,
            args=(self.audio_queue, self.result_queue, self.command_queue, self.status_queue),
        )
        self.stt_process.daemon = True
        self.stt_process.start()
        logger.info("STT process started (independent)")

    def _wait_for_stt_ready(self) -> bool:
        """
        STT 프로세스가 모델 로딩을 끝내고 READY 신호를 보낼 때까지 대기합니다.
        """
        if not self.status_queue or not self.stt_process:
            return True

        if not getattr(config, "STT_WAIT_READY_ON_STARTUP", True):
            return True

        timeout_s = float(getattr(config, "STT_READY_TIMEOUT_SECONDS", 180.0))
        start = time.time()
        logger.info(f"Waiting for STT READY (timeout={timeout_s:.1f}s)...")

        while True:
            if not self.stt_process.is_alive():
                logger.error("STT process exited before becoming ready.")
                return False

            remaining = timeout_s - (time.time() - start)
            if remaining <= 0:
                logger.error("Timed out waiting for STT READY.")
                return False

            try:
                msg = self.status_queue.get(timeout=min(0.5, max(0.1, remaining)))
            except queue.Empty:
                continue

            if msg == "READY":
                logger.info("STT READY")
                return True

            if isinstance(msg, dict):
                if msg.get("type") == "STT_READY":
                    logger.info(f"STT READY (device={msg.get('device', 'unknown')})")
                    return True
                if msg.get("type") == "STT_ERROR":
                    logger.error(f"STT ERROR during startup: {msg.get('error')}")
                    return False

    def run(self):
        token = config.DISCORD_TOKEN
        if not token:
            logger.error("DISCORD_TOKEN not found")
            sys.exit(1)

        self.start_stt_process()
        if not self._wait_for_stt_ready():
            logger.error("Startup aborted because STT is not ready.")
            sys.exit(1)
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


