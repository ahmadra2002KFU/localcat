#!/bin/bash
cd /home/user/voice-pipeline
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
exec python web.py --port 7860
