import discord
from discord.ext import commands
import discord.ext.voice_recv
from audio_utils import STTSink

class VoiceCog(commands.Cog):
    def __init__(self, bot, process_manager, conversation_manager):
        self.bot = bot
        self.process_manager = process_manager
        self.conversation_manager = conversation_manager

    @commands.command()
    async def join(self, ctx):
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            if ctx.voice_client is not None:
                return await ctx.voice_client.move_to(channel)
            
            vc = await channel.connect(cls=discord.ext.voice_recv.VoiceRecvClient)
            
            # Use the audio_queue from the process_manager
            vc.listen(STTSink(self.process_manager.audio_queue))
            
            for member in channel.members:
                if not member.bot:
                    self.conversation_manager.add_participant(member.display_name)
            
            await ctx.send(f"Joined {channel} and listening.")
        else:
            await ctx.send("You are not in a voice channel.")

    @commands.command()
    async def leave(self, ctx):
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
            self.conversation_manager.participants.clear()
            await ctx.send("Left the voice channel.")
        else:
            await ctx.send("I am not in a voice channel.")

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if member.bot:
            return

        # Participant tracking
        if after.channel and after.channel.guild.voice_client and after.channel == after.channel.guild.voice_client.channel:
            self.conversation_manager.add_participant(member.display_name)
            print(f"Participant added: {member.display_name}")
        elif before.channel and before.channel.guild.voice_client and before.channel == before.channel.guild.voice_client.channel:
            self.conversation_manager.remove_participant(member.display_name)
            print(f"Participant removed: {member.display_name}")

        # Check if bot was left alone or user left
        if before.channel and not after.channel:
            # Send LEAVE command to STT process to clean up user state
            if self.process_manager.command_queue:
                self.process_manager.command_queue.put(("LEAVE", member.id))

async def setup(bot):
    pass
