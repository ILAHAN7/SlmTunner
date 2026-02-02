from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import re
import json

# 모델명 입력 받기
model_name = input("사용할 Hugging Face 모델명을 입력하세요 (예: google/gemma-2b): ").strip()
# 저장 폴더명: 모델명에서 /, . 등 특수문자 제거 및 -로 치환
save_dir = re.sub(r'[^\w\-]', '-', model_name)

# Load tokenizer and model (8-bit loading for low VRAM)
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})

# Tokenization
def tokenize(example):
    # 딕셔너리를 compact JSON 문자열로 직렬화
    input_str = json.dumps(example["prompt"], ensure_ascii=False, separators=(",", ":"))
    answer_str = json.dumps(example["completion"], ensure_ascii=False, separators=(",", ":"))
    combined = input_str + "\n" + answer_str
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=f"./{save_dir}-lora",
    per_device_train_batch_size=8,  # batch size 10으로 조정
    per_device_eval_batch_size=8,
    eval_steps=100,
    logging_steps=50,
    num_train_epochs=1,  # epoch 1로 단축
    learning_rate=3e-5,
    save_total_limit=2,
    save_steps=200,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train
trainer.train()

# Save model
model.save_pretrained(f"./{save_dir}-lora")
tokenizer.save_pretrained(f"./{save_dir}-lora")
