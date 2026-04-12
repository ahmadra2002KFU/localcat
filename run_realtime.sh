#!/bin/bash
cd /home/user/voice-pipeline
source .venv/bin/activate
exec env CUDA_VISIBLE_DEVICES=1 python realtime_server.py --port 7890
