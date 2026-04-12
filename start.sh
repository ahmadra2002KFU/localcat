#!/bin/bash
# Mufeed Voice Pipeline — quick start script
# Usage: ./start.sh [interactive|demo|asr-test|file] [codeswitch|turbo]

set -e
cd /home/user/voice-pipeline
source .venv/bin/activate

MODE="${1:-interactive}"
ASR="${2:-codeswitch}"

echo "=========================================="
echo " Mufeed Voice Pipeline"
echo " Mode: $MODE | ASR: $ASR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python bot.py "$MODE" "$ASR"
