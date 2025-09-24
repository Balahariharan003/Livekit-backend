from dotenv import load_dotenv
import asyncio
import base64
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, ChatContext, JobContext, get_job_context
from livekit.agents.llm import ImageContent
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
from livekit.plugins import (
    google,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel
import os

load_dotenv()


class VisionAssistant(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []  
        super().__init__(instructions="You are a helpful voice AI assistant with vision capabilities.")
    
    async def on_enter(self):
        room = get_job_context().room
        
    
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(
                self._image_received(reader, participant_identity)
            )
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t))
        

        room.register_byte_stream_handler("images", _image_received_handler)
        
      
        for participant in room.remote_participants.values():
            video_tracks = [
                publication.track for publication in participant.track_publications.values() 
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])
                break
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)
    
    async def _image_received(self, reader, participant_identity):
        """Handle images uploaded from the frontend"""
        image_bytes = bytes()
        async for chunk in reader:
            image_bytes += chunk

        chat_ctx = self.chat_ctx.copy()


        chat_ctx.add_message(
            role="user",
            content=[
                "Here's an image I want to share with you:",
                ImageContent(
                    image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                )
            ],
        )
        await self.update_chat_ctx(chat_ctx)
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: dict) -> None:
 
        if self._latest_frame:
            if isinstance(new_message.content, list):
                new_message.content.append(ImageContent(image=self._latest_frame))
            else:
                new_message.content = [new_message.content, ImageContent(image=self._latest_frame)]
            self._latest_frame = None
    

    def _create_video_stream(self, track: rtc.Track):
  
        if self._video_stream is not None:
            self._video_stream.close()

     
        self._video_stream = rtc.VideoStream(track)
        
        async def read_stream():
            async for event in self._video_stream:
            
                image_bytes = encode(
                    event.frame,
                    EncodeOptions(
                        format="JPEG",
                        resize_options=ResizeOptions(
                            width=1024,
                            height=1024,
                            strategy="scale_aspect_fit"
                        )
                    )
                )
               
                self._latest_frame = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        
   
        task = asyncio.create_task(read_stream())
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)


async def entrypoint(ctx: agents.JobContext):
    vad = silero.VAD.load()
    turn_detector = EnglishModel()

   
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=google.LLM(api_key=os.getenv("GEMINI_API_KEY")),
        tts=cartesia.TTS(
            api_key=os.getenv("CARTESIA_API_KEY"),
            model="sonic-2",
            voice="bf0a246a-8642-498a-9950-80c35e9276b5"
        ),
        vad=vad,
        turn_detection=turn_detector,
    )

    await session.start(
        room=ctx.room,
        agent=VisionAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            video_enabled=True,
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and let them know you can analyze images they share or their camera feed."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
