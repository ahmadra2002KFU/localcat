#!/usr/bin/env python3
"""
Mufeed Voice Pipeline — Web UI (Gradio)
========================================
Real-time voice conversation through the browser.
Mic input → ASR → LLM → TTS → Audio output

Run: CUDA_VISIBLE_DEVICES=1 python web.py
Local: http://localhost:7860
"""

import asyncio
import logging
import os
import sys
import time
import tempfile

import numpy as np
import gradio as gr
import soundfile as sf
import aiohttp
import argparse

# ─── Config ───────────────────────────────────────────────────────────────────

ASR_MODEL_CS = "/home/user/voice-pipeline/models/arabic-whisper-ct2"
ASR_MODEL_FAST = "large-v3-turbo"
ASR_COMPUTE_TYPE = "float16"
DEFAULT_ASR_MODEL = "codeswitch"

LLM_BASE_URL = "http://localhost:8083/v1"
LLM_MODEL = "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf"

TTS_URL = "http://localhost:8001/tts"

SYSTEM_PROMPT = (
    "أنت مساعد ذكي يعمل في شركة مفيض (Mufeed Advanced Co). "
    "أنت تتحدث العربية الفصحى والإنجليزية بسلاسة. "
    "عندما يتحدث المستخدم بالعربية، رد بالعربية. "
    "عندما يتحدث بالإنجليزية، رد بالإنجليزية. "
    "كن محترفاً ومفيداً وأجب بإجابات مختصرة ومباشرة.\n\n"
    "You are a helpful AI assistant at Mufeed Advanced Co. "
    "You speak both Arabic (MSA) and English fluently. "
    "Match the user's language. Keep responses concise and professional."
)

log = logging.getLogger("mufeed-web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


# ─── Singleton Engines ───────────────────────────────────────────────────────

class Engines:
    """Lazy-loaded singleton for ASR, LLM, and TTS."""
    _asr = None
    _asr_name = None
    _session = None
    _history = [{"role": "system", "content": SYSTEM_PROMPT}]

    @classmethod
    def get_session(cls):
        if cls._session is None:
            cls._session = aiohttp.ClientSession()
        return cls._session

    @classmethod
    def get_asr(cls, model_name=DEFAULT_ASR_MODEL):
        if cls._asr is None or cls._asr_name != model_name:
            from faster_whisper import WhisperModel
            path = ASR_MODEL_CS if model_name == "codeswitch" else ASR_MODEL_FAST
            idx = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
            log.info(f"Loading ASR: {model_name}...")
            t = time.time()
            cls._asr = WhisperModel(path, device="cuda", compute_type=ASR_COMPUTE_TYPE, device_index=idx)
            cls._asr_name = model_name
            log.info(f"ASR loaded in {time.time()-t:.1f}s")
        return cls._asr

    @classmethod
    async def transcribe(cls, audio_path, model_name):
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)  # ONNX requires float32
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        model = cls.get_asr(model_name)
        segs, info = model.transcribe(audio, beam_size=5, language=None, vad_filter=True,
                                       vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200})
        text = " ".join(s.text.strip() for s in segs).strip()
        if text:
            log.info(f"[ASR] {info.language}({info.language_probability:.0%}): {text}")
        return text

    @classmethod
    async def chat(cls, user_msg):
        cls._history.append({"role": "user", "content": user_msg})
        payload = {"model": LLM_MODEL, "messages": cls._history, "max_tokens": 256, "temperature": 0.7}
        async with cls.get_session().post(f"{LLM_BASE_URL}/chat/completions", json=payload,
                                           timeout=aiohttp.ClientTimeout(total=30)) as r:
            data = await r.json()
        reply = data["choices"][0]["message"]["content"].strip()
        cls._history.append({"role": "assistant", "content": reply})
        log.info(f"[LLM] {reply[:100]}")
        return reply

    @classmethod
    async def synthesize(cls, text):
        if not text.strip():
            return None
        ar = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total = sum(1 for c in text if c.isalpha())
        lang = "ar" if (total > 0 and ar / total > 0.25) else "en"
        async with cls.get_session().post(TTS_URL, json={"text": text, "language": lang},
                                           timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                log.error(f"[TTS] HTTP {r.status}")
                return None
            content = await r.read()
        import io
        audio, sr = sf.read(io.BytesIO(content))
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        out = tempfile.mktemp(suffix=".wav")
        sf.write(out, audio.astype(np.float32), sr)
        return out

    @classmethod
    def clear(cls):
        cls._history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─── Gradio handlers ─────────────────────────────────────────────────────────

def handle_voice(audio_path, model_name, chat_display):
    """Sync wrapper for voice pipeline (Gradio calls sync)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_async_voice(audio_path, model_name, chat_display))
    finally:
        loop.close()


def handle_text(text, chat_display):
    """Sync wrapper for text pipeline."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_async_text(text, chat_display))
    finally:
        loop.close()


async def _async_voice(audio_path, model_name, chat_display):
    if audio_path is None:
        return "", chat_display, None
    transcript = await Engines.transcribe(audio_path, model_name)
    if not transcript:
        return "", chat_display, None
    reply = await Engines.chat(transcript)
    audio_out = await Engines.synthesize(reply)
    chat_display = chat_display or []
    chat_display.append([transcript, reply])
    return transcript, chat_display, audio_out


async def _async_text(text, chat_display):
    if not text or not text.strip():
        return "", chat_display, None
    reply = await Engines.chat(text.strip())
    audio_out = await Engines.synthesize(reply)
    chat_display = chat_display or []
    chat_display.append([text.strip(), reply])
    return text.strip(), chat_display, audio_out


def handle_clear():
    Engines.clear()
    return [], None, None


def handle_load_model(model_name):
    loop = asyncio.new_event_loop()
    try:
        Engines.get_asr(model_name)
    finally:
        loop.close()
    return f"✅ Loaded: {model_name}"


def handle_status():
    import requests
    lines = []
    for name, url in [("LLM (Gemma 4)", f"{LLM_BASE_URL}/health"),
                       ("TTS (Chatterbox)", TTS_URL.replace("/tts", "/health"))]:
        try:
            r = requests.get(url, timeout=3)
            lines.append(f"{'✅' if r.status_code == 200 else '⚠️'} {name}")
        except:
            lines.append(f"❌ {name}")
    loaded = Engines._asr is not None
    lines.append(f"{'✅' if loaded else '⏳'} ASR ({'loaded' if loaded else 'lazy'})")
    return "\n".join(lines)


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Mufeed Voice") as demo:
        gr.Markdown("# 🎙️ Mufeed Voice Pipeline\n**ASR → Gemma 4 → Chatterbox TTS** — Speak Arabic, English, or mix both.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                with gr.Row():
                    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Voice Input")
                    resp_audio = gr.Audio(type="filepath", label="🔊 Response", autoplay=True)
                with gr.Row():
                    txt = gr.Textbox(placeholder="Or type here... مرحباً، كيف حالك؟", scale=4)
                    send = gr.Button("Send", variant="primary", scale=1)
                transcript = gr.Textbox(label="📝 Transcript", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                asr_dd = gr.Dropdown(["codeswitch", "turbo"], value=DEFAULT_ASR_MODEL, label="ASR Model",
                                      info="codeswitch=Arabic-English optimized\nturbo=Faster Whisper v3")
                load_btn = gr.Button("Load Model")
                load_st = gr.Textbox(interactive=False)
                status_btn = gr.Button("Check Services")
                status_txt = gr.Textbox(interactive=False, lines=4, value=handle_status())
                clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")

        mic.stop_recording(handle_voice, [mic, asr_dd, chatbot], [transcript, chatbot, resp_audio])
        send.click(handle_text, [txt, chatbot], [transcript, chatbot, resp_audio])
        txt.submit(handle_text, [txt, chatbot], [transcript, chatbot, resp_audio])
        load_btn.click(handle_load_model, [asr_dd], [load_st])
        status_btn.click(handle_status, [], [status_txt])
        clear_btn.click(handle_clear, [], [chatbot, txt, resp_audio])

    return demo


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Pre-load ASR
    Engines.get_asr(DEFAULT_ASR_MODEL)
    log.info("ASR ready")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)
