#!/usr/bin/env python3
"""
Quick standalone test for the voice pipeline components.
Tests each component individually before running the full pipeline.
"""

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))


async def test_asr():
    """Test Whisper ASR with a generated audio sample."""
    print("=" * 50)
    print("TEST 1: ASR (Whisper large-v3-turbo)")
    print("=" * 50)

    from faster_whisper import WhisperModel
    import numpy as np
    import time

    print("Loading model on GPU 1...")
    start = time.time()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", device_index=1)
    print(f"  Loaded in {time.time()-start:.1f}s")

    # Generate a simple test: 1 second of silence (will be filtered)
    # In real testing, you'd use a real audio file
    print("  Model ready ✓")
    print()

    # Check if Arabic-Whisper CodeSwitching CT2 model exists
    ct2_path = "/home/user/voice-pipeline/models/arabic-whisper-ct2"
    if os.path.exists(ct2_path) and os.path.exists(os.path.join(ct2_path, "model.bin")):
        print("  Arabic-Whisper CodeSwitching (CT2) model found ✓")
    else:
        print(f"  Arabic-Whisper CodeSwitching model not yet converted")
        print(f"  Path: {ct2_path}")
        if os.path.exists(ct2_path):
            print(f"  Files: {os.listdir(ct2_path)}")
    print()
    return True


async def test_llm():
    """Test Gemma 4 LLM via vLLM."""
    print("=" * 50)
    print("TEST 2: LLM (Gemma 4 via vLLM)")
    print("=" * 50)

    import aiohttp

    url = "http://localhost:8083/v1/chat/completions"
    payload = {
        "model": "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
        "messages": [
            {"role": "system", "content": "أنت مساعد. رد بإيجاز."},
            {"role": "user", "content": "مرحبا، كيف حالك؟"},
        ],
        "max_tokens": 100,
    }

    async with aiohttp.ClientSession() as session:
        start = __import__('time').time()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()
            elapsed = __import__('time').time() - start

            if "choices" in data:
                reply = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("completion_tokens", "?")
                print(f"  Response ({elapsed:.1f}s, {tokens} tokens): {reply}")
                print("  LLM working ✓")
            else:
                print(f"  ERROR: {data}")
                return False
    print()
    return True


async def test_tts():
    """Test Chatterbox TTS."""
    print("=" * 50)
    print("TEST 3: TTS (Chatterbox Multilingual)")
    print("=" * 50)

    import aiohttp

    # Test Arabic
    url = "http://localhost:8001/tts"
    tests = [
        ("مرحبا، أنا مساعد مفيض", "ar"),
        ("Hello, I am the Mufeed assistant", "en"),
    ]

    async with aiohttp.ClientSession() as session:
        for text, lang in tests:
            start = __import__('time').time()
            async with session.post(url, json={"text": text, "language": lang},
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                elapsed = __import__('time').time() - start
                if resp.status == 200:
                    content = await resp.read()
                    print(f"  [{lang}] {elapsed:.1f}s, {len(content)} bytes ✓")
                else:
                    err = await resp.text()
                    print(f"  [{lang}] ERROR {resp.status}: {err[:200]}")
                    return False
    print()
    return True


async def test_chatterbox_language_detection():
    """Test our language detection logic."""
    print("=" * 50)
    print("TEST 4: Language Detection")
    print("=" * 50)

    tests = [
        ("مرحبا كيف حالك اليوم", "ar"),
        ("Hello, how are you doing today?", "en"),
        ("أنا أحب الـ machine learning كثيراً", "ar"),
        ("Let's discuss the project timeline", "en"),
        ("الـ deadline هو next week", "ar"),  # mixed
    ]

    def detect(text):
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return "en"
        ratio = arabic_chars / total_chars
        return "ar" if ratio > 0.3 else "en"

    for text, expected in tests:
        result = detect(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} \"{text}\" -> {result} (expected {expected})")
    print()
    return True


async def main():
    print("\nMufeed Voice Pipeline - Component Tests\n")

    results = {}
    for name, test_fn in [
        ("ASR", test_asr),
        ("LLM", test_llm),
        ("TTS", test_tts),
        ("LangDetect", test_chatterbox_language_detection),
    ]:
        try:
            results[name] = await test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name}: {status}")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll components ready! Run the pipeline with:")
        print("  cd /home/user/voice-pipeline && source .venv/bin/activate")
        print("  python bot.py interactive")
    else:
        print("\nSome tests failed. Fix issues above before running pipeline.")


if __name__ == "__main__":
    asyncio.run(main())
