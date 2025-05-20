#Import LiveKit Agent Modules and Plugins

import logging

from dotenv import load_dotenv
_ = load_dotenv(override=True)

load_dotenv('config/.env')

logger = logging.getLogger("dlai-agent")
logger.setLevel(logging.INFO)

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, jupyter
from livekit.plugins import (
    openai,
    elevenlabs,
    silero,
    google
)
#from livekit.plugins.turn_detector.multilingual import MultilingualModel


# Define Your Custom Agent
llm = openai.LLM(model="gpt-4o")

class Assistant(Agent):
    def __init__(self) -> None:


        #Ollama example
        # llm = openai.LLM.with_ollama(
        #     model="llama3.1",
        #     base_url="http://localhost:11434/v1",
        # ),

        # llm = google.LLM(
        #     model="gemini-2.0-flash-exp",
        #     temperature=0.8,
        # ),

        stt = openai.STT()
        tts = elevenlabs.TTS()
        #tts = elevenlabs.TTS(voice_id="CwhRBWXzGAHq8TQ4Fs17")  # example with defined voice
        silero_vad = silero.VAD.load()

        super().__init__(
            instructions="""
                You are a helpful assistant communicating 
                via voice
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )

       #Create  the  Entroint

    async def entrypoint(ctx: JobContext):
        await ctx.connect()
        stt = openai.STT()
        tts = elevenlabs.TTS()
        # tts = elevenlabs.TTS(voice_id="CwhRBWXzGAHq8TQ4Fs17")  # example with defined voice
        silero_vad = silero.VAD.load()

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )

        await session.start(
            room=ctx.room,
            agent=Assistant()
        )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc = Assistant().entrypoint))