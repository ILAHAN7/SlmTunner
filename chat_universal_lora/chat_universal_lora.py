import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# LoRA adapter 폴더 자동 탐색
lora_candidates = []
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for d in os.listdir(root):
    dpath = os.path.join(root, d)
    if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, 'adapter_config.json')):
        lora_candidates.append(d)

if not lora_candidates:
    print("[오류] 상위 폴더에 LoRA adapter가 없습니다.")
    exit(1)

print("[LoRA adapter 선택]")
for i, name in enumerate(lora_candidates):
    print(f"{i+1}. {name}")
idx = int(input(f"번호 선택 (1-{len(lora_candidates)}): ")) - 1
lora_dir = os.path.join(root, lora_candidates[idx])

# base model 추천 값 찾기
suggested_base_model = None
adapter_config_path = os.path.join(lora_dir, 'adapter_config.json')
if os.path.exists(adapter_config_path):
    with open(adapter_config_path, encoding='utf-8') as f:
        cfg = json.load(f)
        suggested_base_model = cfg.get('base_model_name_or_path')
        print(f"[INFO] adapter_config.json에서 발견된 베이스 모델: {suggested_base_model}")

# 추천 베이스 모델이 없으면 README.md에서 추출 시도
if not suggested_base_model:
    readme_path = os.path.join(lora_dir, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('base_model:'):
                    suggested_base_model = line.split(':',1)[1].strip()
                    print(f"[INFO] README.md에서 발견된 베이스 모델: {suggested_base_model}")
                    break

# 추천 모델을 기반으로 모델명 직접 입력 받기
model_suggestions = {
    "tinyllama-lora": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma2b-lora": "google/gemma-2b"
}

# 현재 선택된 어댑터 이름에 기반하여 추천
adapter_name = os.path.basename(lora_dir)
default_suggestion = model_suggestions.get(adapter_name, suggested_base_model)
if default_suggestion:
    prompt_msg = f"Base model name [{default_suggestion}]: "
    user_input = input(prompt_msg).strip()
    base_model = user_input if user_input else default_suggestion
else:
    base_model = input("Base model name (ex: TinyLlama/TinyLlama-1.1B-Chat-v1.0): ").strip()

print(f"[INFO] 선택된 LoRA: {lora_dir}")
print(f"[INFO] base model: {base_model}")

# LoRA + base model 조합으로만 동작 (merge 옵션 제거)
print(f"[INFO] 베이스 모델 로딩 중: {base_model}")
print(f"[INFO] LoRA 어댑터 로딩 중: {lora_dir}")

# 토크나이저는 베이스 모델에서 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    print("[INFO] 베이스 모델에서 토크나이저 로드 성공")
except Exception as e:
    print(f"[경고] 베이스 모델에서 토크나이저 로드 실패: {e}")
    print("[INFO] LoRA 어댑터에서 토크나이저 로드 시도 중...")
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)

try:
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, lora_dir)
    print("[INFO] 모델 로드 성공")
except Exception as e:
    print(f"[오류] 모델 로드 실패: {e}")
    exit(1)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 디버깅을 위한 토큰 정보 출력
print(f"[INFO] 토크나이저 정보: {tokenizer.__class__.__name__}")
print(f"[INFO] 특수 토큰: {tokenizer.special_tokens_map}")

print("Universal LoRA Chat! (종료하려면 'exit' 입력)")
print("낮은 온도 설정으로 실행 중 (temperature=0.3). 정확한 응답을 위함입니다.")

while True:
    prompt = input("User: ")
    if prompt.strip().lower() == "exit":
        break
    
    # 디버깅을 위한 로깅
    print(f"[DEBUG] 입력 길이: {len(prompt)}")
    try:
        # JSON 입력인 경우 따옴표 이스케이프 확인
        json.loads(prompt)
        print("[DEBUG] 유효한 JSON 입력 감지")
    except:
        print("[DEBUG] 일반 텍스트 입력 또는 잘못된 JSON 감지")
    
    # 더 낮은 온도와 더 많은 토큰 수로 설정
    output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)
    
    # 디버깅을 위한 전체 출력 로깅
    full_output = output[0]["generated_text"]
    print(f"[DEBUG] 전체 출력 ({len(full_output)} 문자): {full_output}")
    
    # 생성된 텍스트만 추출
    generated_text = full_output[len(prompt):].strip()
    print("Model:", generated_text)
    
    # JSON 응답 파싱 시도
    try:
        response_json = json.loads(generated_text)
        print(f"[DEBUG] 파싱된 JSON: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
    except:
        pass
