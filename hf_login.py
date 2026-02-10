"""
Hugging Face Token Login Script
Uses environment variable HF_TOKEN or prompts for interactive input.
Never hardcode tokens in source code.
"""
import os

try:
    from huggingface_hub import login
except ImportError:
    print(">> Error: huggingface_hub not installed.")
    print(">> Run: pip install huggingface_hub")
    exit(1)

# Priority: 1) HF_TOKEN env var  2) .env file  3) Interactive input
token = os.environ.get("HF_TOKEN", "")

if not token:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("HF_TOKEN", "")
    except ImportError:
        pass  # python-dotenv not installed, skip .env loading

if not token:
    token = input("Enter your Hugging Face token: ").strip()

if not token:
    print(">> Error: No token provided. Exiting.")
    exit(1)

try:
    login(token=token)
    print(">> Successfully logged in to Hugging Face!")
except Exception as e:
    print(f">> Login failed: {e}")
    exit(1)
