from dotenv import load_dotenv
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import deepgram, google, cartesia, noise_cancellation, silero

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=google.LLM(api_key=os.getenv("GEMINI_API_KEY")),
        tts=cartesia.TTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model="sonic-2",
            voice="bf0a246a-8642-498a-9950-80c35e9276b5"
        ),
        vad=silero.VAD.load(
            min_speech_duration=1.0,
            min_silence_duration=6.0,
            prefix_padding_duration=0.5,
            activation_threshold=0.6,
            sample_rate=16000,
            force_cpu=True,
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # 👇 Auto-greet when connected
    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

