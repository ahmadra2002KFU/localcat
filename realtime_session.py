"""Per-WebSocket-connection session state machine for real-time voice-to-voice AI pipeline."""

import asyncio
import uuid
import numpy as np
from collections import deque
from enum import Enum
from realtime_vad import SileroVAD, WINDOW_SIZE, SAMPLE_RATE


CHUNK_BYTES = WINDOW_SIZE * 2  # 1024 bytes per 512-sample int16 chunk

PRE_SPEECH_CHUNKS = int(300 / (WINDOW_SIZE / SAMPLE_RATE * 1000))  # ~9 chunks (300ms)

SYSTEM_PROMPT = """أنت مساعد صوتي ذكي. تتحدث العربية والإنجليزية بطلاقة. تكلم بنفس لغة المستخدم. إجاباتك قصيرة جداً — جملة إلى ثلاث جمل فقط. تكلم بأسلوب محادثة طبيعي، ليس رسمياً. لا تستخدم قوائم أو نقاط أو ترقيم. لا تذكر أنك ذكاء اصطناعي.

You are a voice assistant. You speak Arabic and English fluently. Match the user's language naturally — if they code-switch, you can too. Your responses must be very short: 1 to 3 sentences max. Speak in a natural conversational tone, never formal or written. Never use bullet points, numbered lists, markdown, or any formatting — your words will be spoken aloud. Do not mention you are an AI."""


class SessionState(Enum):
    LISTENING = "listening"
    LISTENING_ACTIVE = "listening_active"
    PROCESSING = "processing"
    GENERATING = "generating"
    SPEAKING = "speaking"


class RealtimeSession:
    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.state: SessionState = SessionState.LISTENING
        self.vad: SileroVAD = SileroVAD()
        self.audio_buffer: bytearray = bytearray()
        self.pre_speech_buffer: deque = deque(maxlen=PRE_SPEECH_CHUNKS)
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.asr_engine: str = "cohere"  # "cohere" or "whisper"
        self._remainder: bytes = b""

    def feed_audio(self, pcm_bytes: bytes) -> str | None:
        self._remainder += pcm_bytes
        result = None

        while len(self._remainder) >= CHUNK_BYTES:
            chunk_bytes = self._remainder[:CHUNK_BYTES]
            self._remainder = self._remainder[CHUNK_BYTES:]

            samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            event = self.vad(samples)

            if self.state == SessionState.LISTENING:
                if event and "start" in event:
                    self.state = SessionState.LISTENING_ACTIVE
                    for buffered in self.pre_speech_buffer:
                        self.audio_buffer.extend(buffered)
                    self.audio_buffer.extend(chunk_bytes)
                    result = "speech_start"
                else:
                    self.pre_speech_buffer.append(chunk_bytes)

            elif self.state == SessionState.LISTENING_ACTIVE:
                self.audio_buffer.extend(chunk_bytes)
                if event and "end" in event:
                    self.state = SessionState.PROCESSING
                    result = "speech_end"

            elif self.state == SessionState.SPEAKING:
                if event and "start" in event:
                    result = "interrupt"

        return result

    def get_utterance(self) -> np.ndarray:
        audio = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = bytearray()
        self.pre_speech_buffer.clear()
        self.vad.reset_states()
        return audio

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def interrupt(self):
        self.cancel_event.set()
        self.state = SessionState.LISTENING
        self.audio_buffer = bytearray()
        self.pre_speech_buffer.clear()
        self.vad.reset_states()
        self.cancel_event = asyncio.Event()

    def clear(self):
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.state = SessionState.LISTENING
        self.audio_buffer = bytearray()
        self.pre_speech_buffer.clear()
        self.vad.reset_states()
        self._remainder = b""
        self.cancel_event = asyncio.Event()

    def set_state(self, state: SessionState):
        self.state = state
