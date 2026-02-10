"""
Universal LoRA Chat Script
Tests a fine-tuned LoRA adapter by loading it on top of a base model.
Supports automatic adapter detection, GPU/CPU device selection, and optional quantization.
"""
import os
import sys
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch

def find_lora_adapters(root_dir):
    """Scan for directories containing adapter_config.json."""
    candidates = []
    for d in os.listdir(root_dir):
        dpath = os.path.join(root_dir, d)
        if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, 'adapter_config.json')):
            candidates.append(d)
    return candidates

def get_base_model_from_adapter(lora_dir):
    """Extract base model name from adapter_config.json or README.md."""
    # Try adapter_config.json first
    adapter_config_path = os.path.join(lora_dir, 'adapter_config.json')
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, encoding='utf-8') as f:
                cfg = json.load(f)
                base_model = cfg.get('base_model_name_or_path')
                if base_model:
                    return base_model
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Could not read adapter_config.json: {e}")

    # Fallback: try README.md
    readme_path = os.path.join(lora_dir, 'README.md')
    if os.path.exists(readme_path):
        try:
            with open(readme_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('base_model:'):
                        return line.split(':', 1)[1].strip()
        except IOError:
            pass

    return None

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return 0  # GPU device index
    return -1  # CPU

def main():
    parser = argparse.ArgumentParser(description="Universal LoRA Chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="none",
                        help="Load base model with quantization (saves VRAM)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()

    # Find LoRA adapters
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lora_candidates = find_lora_adapters(root)

    if not lora_candidates:
        print("[ERROR] No LoRA adapter found in project root.")
        print("  >> Train a model first, or check that adapter_config.json exists.")
        sys.exit(1)

    print("[Select LoRA Adapter]")
    for i, name in enumerate(lora_candidates):
        print(f"  {i+1}. {name}")
    
    try:
        idx = int(input(f"Select (1-{len(lora_candidates)}): ")) - 1
        if idx < 0 or idx >= len(lora_candidates):
            raise ValueError("Out of range")
    except ValueError:
        print("[ERROR] Invalid selection.")
        sys.exit(1)

    lora_dir = os.path.join(root, lora_candidates[idx])

    # Determine base model
    suggested_base_model = get_base_model_from_adapter(lora_dir)
    
    # Known adapter-to-model mapping as fallback
    model_suggestions = {
        "tinyllama-lora": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gemma2b-lora": "google/gemma-2b"
    }
    adapter_name = os.path.basename(lora_dir)
    default_suggestion = suggested_base_model or model_suggestions.get(adapter_name)

    if default_suggestion:
        user_input = input(f"Base model [{default_suggestion}]: ").strip()
        base_model = user_input if user_input else default_suggestion
    else:
        base_model = input("Base model name (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0): ").strip()
        if not base_model:
            print("[ERROR] No base model specified.")
            sys.exit(1)

    print(f"[INFO] LoRA adapter: {lora_dir}")
    print(f"[INFO] Base model: {base_model}")

    # Setup quantization config
    q_config = None
    if args.quantize == "4bit":
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        print("[INFO] Loading with 4-bit quantization")
    elif args.quantize == "8bit":
        q_config = BitsAndBytesConfig(load_in_8bit=True)
        print("[INFO] Loading with 8-bit quantization")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print("[INFO] Tokenizer loaded from base model")
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from base model: {e}")
        print("[INFO] Trying to load tokenizer from LoRA adapter...")
        tokenizer = AutoTokenizer.from_pretrained(lora_dir)

    # Load model + adapter
    try:
        model_kwargs = {"trust_remote_code": True}
        if q_config:
            model_kwargs["quantization_config"] = q_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(model, lora_dir)
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Create pipeline
    device = get_device()
    # When using device_map="auto", don't pass device to pipeline
    pipe_kwargs = {"task": "text-generation", "model": model, "tokenizer": tokenizer}
    if not q_config and device >= 0:
        pipe_kwargs["device"] = device
    pipe = pipeline(**pipe_kwargs)

    if args.debug:
        print(f"[DEBUG] Tokenizer: {tokenizer.__class__.__name__}")
        print(f"[DEBUG] Special tokens: {tokenizer.special_tokens_map}")
        print(f"[DEBUG] Device: {'GPU' if device >= 0 else 'CPU'}")

    print(f"\nUniversal LoRA Chat (type 'exit' to quit)")
    print(f"Temperature: {args.temperature} | Max tokens: {args.max_tokens}\n")

    while True:
        prompt = input("User: ")
        if prompt.strip().lower() == "exit":
            break

        if args.debug:
            print(f"[DEBUG] Input length: {len(prompt)} chars")

        try:
            output = pipe(prompt, max_new_tokens=args.max_tokens, 
                         do_sample=True, temperature=args.temperature)
            full_output = output[0]["generated_text"]
            generated_text = full_output[len(prompt):].strip()
            
            if args.debug:
                print(f"[DEBUG] Full output ({len(full_output)} chars): {full_output}")
            
            print("Model:", generated_text)

            # Try to parse JSON response if applicable
            if args.debug:
                try:
                    response_json = json.loads(generated_text)
                    print(f"[DEBUG] Parsed JSON: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
