# Hugging Face 토큰 자동 로그인을 위한 스크립트
import os

token = "" ##enter your Hugging Face token here

os.system(f"huggingface-cli login --token {token}")
