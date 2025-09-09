# SLM LoRA Tuning Pipeline (TinyLlama, Gemma2B, etc.)

## Overview (English)
- Fully automated LoRA tuning pipeline for Small Language Models (SLM)
- Supports RTX 4050 (6GB VRAM), Linux/Windows, Docker
- LoRA tuning for any Hugging Face SLM (TinyLlama, Gemma2B, Phi-2, etc.) by just entering the model name
- Includes Docker, Hugging Face authentication, data sampling, and chat testing
- **Note:** Model evaluation and error statistics scripts are currently not included due to instability

---

## Folder/File Structure
```
├── train_model.py                # Universal LoRA tuning script (model name input)
├── chat_tinyllama_lora/          # Chat-style test script
├── train.jsonl / valid.jsonl     # Training/validation data (jsonl, prompt/completion)
├── requirements.txt              # Required Python packages
├── Dockerfile, run_docker.ps1    # Docker+GPU automation
├── hf_login.py                   # Hugging Face token auto-login
├── split_data.py                 # Data sampling/splitting tool (optional)
├── convert_jsonl.py              # Data format conversion tool (optional)
```

---

## Prerequisites / Preparation
1. **Python 3.9+** (Recommended: 3.10~3.12)
2. **CUDA, NVIDIA driver** (for GPU training), (optional) Docker
3. **Hugging Face account & access token** (for model download)
4. **Prepare your data:**
   - `train.jsonl` and `valid.jsonl` must be in the workspace.
   - Each line: `{ "prompt": "...", "completion": "..." }`
   - Use `split_data.py` to sample/split data, or `convert_jsonl.py` to convert formats if needed.
5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
6. **Hugging Face token login (first time only):**
   ```bash
   python hf_login.py
   ```

---

## Training Usage
### Universal LoRA Tuning (any SLM)
```bash
python train_model.py
```
- Enter the model name in the terminal (e.g. `google/gemma-2b`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- Results are saved in `./<model-name>-lora/`

---

## Chat-style Testing (like ChatGPT)
```bash
python chat_tinyllama_lora/chat_tinyllama_lora.py
```
- Enter prompt → Model generates a response

---

## Docker Usage (optional)
```powershell
./run_docker.ps1
```
- Sets up CUDA, bitsandbytes, peft, etc. automatically

---

## Data Tools
- `split_data.py`: Randomly sample or split your dataset for training/validation.
- `convert_jsonl.py`: Convert your data to the required jsonl format if needed.
- `hf_login.py`: Hugging Face token auto-login (required for downloading gated models).

---

## Tips
- If you get OOM (out of memory), reduce batch size, max_length, or epoch in `train_model.py`.
- LoRA tuning is automated for any SLM by just entering the model name.
- Use adapter_model.safetensors, tokenizer, etc. in the result folder for inference or deployment.
- Easily extensible for evaluation, inference, chat, or deployment.

---

# SLM LoRA 튜닝 파이프라인 (TinyLlama, Gemma2B 등)

## 개요
- 초소형 언어모델(SLM) LoRA 튜닝 전체 자동화 파이프라인
- RTX 4050(6GB VRAM) 및 리눅스/윈도우 환경 지원
- TinyLlama, Gemma2B, Phi-2 등 다양한 모델명 입력만으로 LoRA 튜닝 가능
- Docker, Hugging Face 인증, 데이터 샘플링, 채팅 테스트 등 통합
- **참고:** 모델 평가 및 오차 통계 스크립트는 현재 오류로 미포함

---

## 폴더/파일 구조
```
├── train_model.py                # 모델명 입력받아 LoRA 튜닝 (자동 저장)
├── chat_tinyllama_lora/          # 대화형 테스트 스크립트
├── train.jsonl / valid.jsonl     # 학습/검증 데이터 (jsonl, prompt/completion)
├── requirements.txt              # 필수 패키지 목록
├── Dockerfile, run_docker.ps1    # Docker+GPU 자동화 환경
├── hf_login.py                   # Hugging Face 토큰 자동 로그인
├── split_data.py                 # 데이터 샘플링/분할 툴 (옵션)
├── convert_jsonl.py              # 데이터 포맷 변환 툴 (옵션)
```

---

## 사전 준비/환경 세팅
1. **Python 3.9+** (권장 3.10~3.12)
2. **CUDA, NVIDIA 드라이버** (GPU 학습용), (옵션) Docker
3. **Hugging Face 계정 및 토큰** (모델 다운로드용)
4. **데이터 준비:**
   - `train.jsonl`, `valid.jsonl` 파일을 작업 폴더에 준비
   - 각 줄: `{ "prompt": "...", "completion": "..." }` 형식
   - 필요시 `split_data.py`로 샘플링/분할, `convert_jsonl.py`로 포맷 변환
5. **필수 패키지 설치:**
   ```bash
   pip install -r requirements.txt
   ```
6. **Hugging Face 토큰 로그인 (최초 1회):**
   ```bash
   python hf_login.py
   ```

---

## 학습 실행 방법
```bash
python train_model.py
```
- 실행 후 터미널에 모델명 입력 (예: `google/gemma-2b`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- 결과는 `./모델명-lora/` 폴더에 저장

---

## 대화형 테스트 (ChatGPT 스타일)
```bash
python chat_tinyllama_lora/chat_tinyllama_lora.py
```
- 프롬프트 입력 → 모델 답변 출력

---

## Docker 사용 (옵션)
```powershell
./run_docker.ps1
```
- CUDA, bitsandbytes, peft 등 자동 환경 구축 및 실행

---

## 데이터 툴 안내
- `split_data.py`: 학습/검증 데이터 샘플링, 분할용
- `convert_jsonl.py`: 데이터 포맷(json/csv 등 → jsonl) 변환용
- `hf_login.py`: Hugging Face 토큰 자동 로그인(게이트 모델 다운로드 필수)

---

## 참고/팁
- VRAM 부족(OOM) 시 train_model.py에서 batch size, max_length, epoch 등 조정
- 다양한 SLM 모델명 입력만으로 LoRA 튜닝 자동화
- 결과 폴더 내 adapter_model.safetensors, tokenizer 등 활용
- 평가/추론/채팅/배포 등 확장 가능

---

## LoRA Tuning Error Troubleshooting (CUDA OOM)
If you see an error like:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```
This means the model, input, optimizer, and intermediate tensors exceed your GPU memory.

**Tip:**
Set the following environment variable to let PyTorch manage memory more flexibly and reduce fragmentation errors:

**Linux/macOS:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
**Windows PowerShell:**
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

- You can set this before running your training script.
- If OOM persists, reduce batch size, max_length, or use a smaller model.

---

## 기타 실전 팁 (KOR)
- torch.OutOfMemoryError: CUDA out of memory 오류 발생 시, 위 환경변수 설정을 추천합니다.
- 그래도 해결되지 않으면 batch size, max_length, epoch, 모델 크기를 줄이세요.
- Docker 컨테이너에서 환경변수는 run 명령에 --env 옵션으로도 전달 가능합니다.
- Windows PowerShell에서는 `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`로 설정 후 python 명령 실행.
- VRAM 사용량은 nvidia-smi로 실시간 확인 가능.
- 데이터가 너무 크면 split_data.py로 샘플링/분할하여 실험하세요.
- Hugging Face 토큰 인증은 최초 1회만 필요, 이후 캐시됨.

