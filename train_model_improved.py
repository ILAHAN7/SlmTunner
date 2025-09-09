from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import re
import json
import torch

# 모델명 입력 받기
model_name = input("사용할 Hugging Face 모델명을 입력하세요 (예: google/gemma-2b): ").strip()
# 저장 폴더명: 모델명에서 /, . 등 특수문자 제거 및 -로 치환
save_dir = re.sub(r'[^\w\-]', '-', model_name)

# 기존 LoRA 파일에서 이어서 학습할지 확인
resume_training = input("기존 LoRA 파일에서 이어서 학습하시겠습니까? (y/n): ").strip().lower() == 'y'
checkpoint_dir = None
if resume_training:
    checkpoint_dir = input("이어서 학습할 LoRA 파일 경로를 입력하세요 (예: ./tinyllama-lora 또는 ./tinyllama-lora/checkpoint-1000): ").strip()
    print(f"[INFO] {checkpoint_dir}에서 학습을 재개합니다.")

print(f"[INFO] 모델 {model_name} 로딩 중...")
# Load tokenizer and model (8-bit loading for low VRAM)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    # 패딩 토큰이 없는 경우 설정 (일부 모델에서 필요)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] 패딩 토큰이 설정되었습니다: {tokenizer.pad_token}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

# 모델에 따른 target_modules 동적 설정
if "tinyllama" in model_name.lower():
    target_modules = ["q_proj", "v_proj"]
elif "gemma" in model_name.lower():
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
else:
    # 다른 모델의 경우 기본값 사용
    target_modules = ["q_proj", "v_proj"]
    print(f"[INFO] 기본 target_modules를 사용합니다: {target_modules}")

print(f"[INFO] 모델 {model_name}에 대한 target_modules: {target_modules}")

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 기존 LoRA 파일에서 학습을 재개하는 경우, 체크포인트에서 모델 로드
if resume_training and checkpoint_dir:
    print(f"[INFO] 체크포인트 {checkpoint_dir}에서 모델을 로드합니다...")
    from peft import PeftModel
    # 일단 base 모델에 LoRA 설정 적용
    base_model = get_peft_model(model, peft_config)
    # 그 다음 체크포인트에서 weights 로드 (이 부분은 실행 중 에러 발생시 체크포인트 구조에 따라 조정 필요)
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        print(f"[INFO] 체크포인트에서 모델 로딩 성공!")
    except Exception as e:
        print(f"[ERROR] 체크포인트 로딩 실패: {e}")
        print("[INFO] 기본 LoRA 설정으로 계속 진행합니다.")
        model = get_peft_model(model, peft_config)
else:
    model = get_peft_model(model, peft_config)

# Load dataset
print("[INFO] 데이터셋 로딩 중...")
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})
print(f"[INFO] 학습 데이터: {len(dataset['train'])}개, 검증 데이터: {len(dataset['validation'])}개")

# 첫 번째 샘플 확인하여 데이터 형식 파악
print("\n[INFO] 데이터 샘플 확인:")
print(f"prompt 타입: {type(dataset['train'][0]['prompt'])}")
print(f"completion 타입: {type(dataset['train'][0]['completion'])}")
print(f"prompt 샘플: {dataset['train'][0]['prompt']}")
print(f"completion 샘플: {dataset['train'][0]['completion']}\n")

# Tokenization - 개선된 직렬화 로직
def tokenize(example):
    # prompt가 문자열인지 확인하여 적절히 처리
    input_data = example["prompt"]
    if isinstance(input_data, str):
        # 이미 문자열인 경우 JSON 파싱 시도
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            # 파싱 실패 시 그대로 사용
            pass
    
    # completion도 동일하게 처리
    output_data = example["completion"]
    if isinstance(output_data, str):
        try:
            output_data = json.loads(output_data)
        except json.JSONDecodeError:
            pass
    
    # 직렬화 진행
    input_str = json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
    answer_str = json.dumps(output_data, ensure_ascii=False, separators=(",", ":"))
    combined = input_str + "\n" + answer_str
    
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=384  # 512에서 조정 (메모리 사용량 감소)
    )

print("[INFO] 데이터 토크나이징 중...")
tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=100)
print("[INFO] 토크나이징 완료")

# 메모리 사용량 확인
if torch.cuda.is_available():
    print(f"[INFO] 현재 GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[INFO] 최대 GPU 메모리 사용량: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# TrainingArguments
training_args = TrainingArguments(
    output_dir=f"./{save_dir}-lora",
    per_device_train_batch_size=4,  # VRAM에 따라 조정
    per_device_eval_batch_size=4,
    eval_steps=100,
    logging_steps=50,
    num_train_epochs=1,
    learning_rate=3e-5,
    save_total_limit=2,
    save_steps=200,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
    # 체크포인트에서 학습 재개를 위한 옵션
    resume_from_checkpoint=checkpoint_dir if resume_training else None,
)

print(f"[INFO] 학습 구성: epochs={training_args.num_train_epochs}, batch_size={training_args.per_device_train_batch_size}, lr={training_args.learning_rate}")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train
print(f"[INFO] 학습 시작: {model_name}")
trainer.train()

# Save model
print(f"[INFO] 모델 저장 중: ./{save_dir}-lora")
model.save_pretrained(f"./{save_dir}-lora")
tokenizer.save_pretrained(f"./{save_dir}-lora")
print("[INFO] 학습 및 저장 완료!")

# 저장된 경로 안내
print("\n========== 학습 완료 ==========")
print(f"모델명: {model_name}")
print(f"저장 경로: ./{save_dir}-lora")
print(f"사용 방법: 'python chat_universal_lora/chat_universal_lora.py'에서 {save_dir}-lora 선택")
print("==============================\n")
