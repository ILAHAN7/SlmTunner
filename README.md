# SLM LoRA Tuning Pipeline (TinyLlama, Gemma2B, etc.)

## Overview (English)
- Fully automated LoRA tuning pipeline for Small Language Models (SLM)
- Supports RTX 4050 (6GB VRAM), Linux/Windows, Docker
- **v2.2: Hardware-aware auto-optimization** — Automatically detects GPU/VRAM and selects optimal batch size, quantization, and training parameters
- LoRA tuning for any Hugging Face SLM (TinyLlama, Gemma2B, Phi-2/3, Llama, etc.) by just entering the model name
- Includes Docker, Hugging Face authentication, experiment tracking, data validation, and chat testing
- **42+ pytest unit tests** for core modules

---

## Folder/File Structure
```
├── manager.py                    # Main CLI (v2.2) - Menu-driven interface
├── config.yaml                   # Training configuration
├── pyproject.toml                # Project metadata & pytest config
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Template for HF_TOKEN env var
├── .gitignore                    # Comprehensive gitignore
├── core/                         # Core engine modules
│   ├── system_analyzer.py        # Hardware detection (GPU/VRAM/OS)
│   ├── strategy.py               # Auto-optimization logic (model-size aware)
│   ├── trainer.py                # Training executor with data validation
│   ├── experiment_tracker.py     # Run logging and reports
│   ├── config_loader.py          # Config merge logic
│   ├── model_inspector.py        # LoRA target module detection
│   ├── evaluator.py              # Token accuracy & perplexity metrics
│   ├── data_validator.py         # Pre-training JSONL validation
│   └── constants.py              # Shared column mapping constants
├── experiments/                  # Auto-generated per training run
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.json           # Settings snapshot
│       ├── metrics.jsonl         # Per-step metrics
│       └── training_report.md    # Summary report
├── tests/                        # pytest test suite
│   ├── conftest.py               # Shared fixtures
│   ├── test_strategy.py          # Optimization tests (23 cases)
│   ├── test_config_loader.py     # Config merge tests (8 cases)
│   ├── test_data_validator.py    # Data validation tests (11 cases)
│   ├── test_evaluator.py         # Metrics tests
│   └── test_experiment_tracker.py # Tracker tests
├── chat_universal_lora/          # Chat-style test script
├── scripts/                      # Legacy scripts
├── Dockerfile, run_docker.ps1    # Docker+GPU automation
└── hf_login.py                   # Hugging Face token auto-login
```

---

## Prerequisites / Preparation
1. **Python 3.9+** (Recommended: 3.10~3.12)
2. **CUDA, NVIDIA driver** (for GPU training), (optional) Docker
3. **Hugging Face account & access token** (for model download)
4. **Prepare your data:**
   - `train.jsonl` and `valid.jsonl` must be in the workspace.
   - Each line: `{ "prompt": "...", "completion": "..." }`
   - Also supports: `instruction/response`, `input/output`, `text` formats
5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
6. **Set up authentication (choose one):**
   ```bash
   # Option A: Environment variable
   export HF_TOKEN=your_token_here

   # Option B: .env file (copy from template)
   cp .env.example .env
   # Edit .env and add your token

   # Option C: Interactive login
   python hf_login.py
   ```

---

## Training Usage

### Manager CLI (Recommended)
```bash
python manager.py
```
Menu options:
1. **Setup Wizard** — Create config.yaml interactively
2. **Start Training (Auto-Optimize)** — Hardware-aware training
3. **Manual Training** — Specify your own parameters
4. **Manual Hardware Setup** — Override GPU detection
5. **System Diagnostics** — View hardware info
6. **View Past Experiments** — Browse experiment history

### How Auto-Optimization Works
- Detects GPU VRAM and estimates model size from name (e.g., `gemma-2b` → 2B)
- Multi-strategy model size detection: known patterns → regex → HuggingFace config → conservative default
- Calculates optimal batch size based on: `(VRAM - model_size) / context_length`
- Selects quantization (4bit/8bit/none) based on available memory
- Adjusts gradient accumulation to achieve target global batch size
- **Evaluation runs per epoch** when validation data is provided

Example:
```
8GB VRAM + Gemma-2B → 8bit, Batch 8
8GB VRAM + Llama-7B → 4bit, Batch 5
24GB VRAM + Llama-7B + 2048 context → 8bit, Batch 4
```

---

## Configuration (config.yaml)
```yaml
model:
  name: "google/gemma-2b"
  output_dir: "./gemma-2b-lora"

dataset:
  train_path: "train.jsonl"
  valid_path: "valid.jsonl"
  rows: 1000  # Approximate count for optimizer

training:
  per_device_train_batch_size: "auto"
  gradient_accumulation_steps: "auto"
  quantization: "auto"  # 4bit, 8bit, or none
  num_train_epochs: "auto"
  max_length: 512
  # Optional LoRA hyperparameters (defaults shown):
  # lora_r: 16
  # lora_alpha: 32
  # lora_dropout: 0.05
```

Set any value to `"auto"` to let the system decide, or specify manually.

---

## Data Validation
Before training starts, dataset files are automatically validated:
- JSON parsing correctness
- Column format detection (prompt/completion, instruction/response, text)
- Encoding verification (UTF-8)
- Dataset size warnings (< 10 samples triggers a warning)

---

## Experiment Tracking
Each training run automatically creates:
- `experiments/run_YYYYMMDD_HHMMSS/config.json` — Full settings
- `experiments/run_YYYYMMDD_HHMMSS/metrics.jsonl` — Loss, PPL, token accuracy per step
- `experiments/run_YYYYMMDD_HHMMSS/training_report.md` — Summary

---

## Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategy.py -v
```

---

## Chat-style Testing
```bash
python chat_universal_lora/chat_universal_lora.py
```
Options: `--quantize 4bit`, `--debug`, `--temperature 0.8`, `--max-tokens 256`

---

## Docker Usage (optional)
```powershell
./run_docker.ps1
```
- Sets up CUDA, bitsandbytes, peft, etc. automatically

---

## Tips
- If GPU detection fails, use menu option #4 to manually set VRAM
- Use `--manual-vram` flag: `python core/system_analyzer.py --manual-vram 8`
- GPU info is cached in `~/.slmtunner_cache.json` for fallback
- If you get OOM (out of memory), reduce max_length or use smaller model
- Set environment variable for memory management:
  ```powershell
  $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ```

---

## LoRA Tuning Error Troubleshooting (CUDA OOM)
If you see an error like:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

**Solutions:**
1. Lower `max_length` in config.yaml (512 → 256)
2. Use 4bit quantization
3. Reduce batch size
4. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

# SLM LoRA 튜닝 파이프라인 (TinyLlama, Gemma2B 등)

## 개요
- 초소형 언어모델(SLM) LoRA 튜닝 전체 자동화 파이프라인
- RTX 4050(6GB VRAM) 및 리눅스/윈도우 환경 지원
- **v2.2: 하드웨어 인식 자동 최적화** — GPU/VRAM 감지 후 최적 배치 크기, 양자화, 학습 파라미터 자동 선택
- TinyLlama, Gemma2B, Phi-2/3, Llama 등 다양한 모델명 입력만으로 LoRA 튜닝 가능
- Docker, Hugging Face 인증, 실험 추적, 데이터 검증, 채팅 테스트 등 통합
- **42+ pytest 단위 테스트** 포함

---

## 사전 준비/환경 세팅
1. **Python 3.9+** (권장 3.10~3.12)
2. **CUDA, NVIDIA 드라이버** (GPU 학습용), (옵션) Docker
3. **Hugging Face 계정 및 토큰** (모델 다운로드용)
4. **데이터 준비:**
   - `train.jsonl`, `valid.jsonl` 파일을 작업 폴더에 준비
   - 각 줄: `{ "prompt": "...", "completion": "..." }` 형식
   - `instruction/response`, `input/output`, `text` 형식도 지원
5. **필수 패키지 설치:**
   ```bash
   pip install -r requirements.txt
   ```
6. **인증 설정 (택 1):**
   ```bash
   # 방법 A: 환경 변수
   export HF_TOKEN=your_token_here

   # 방법 B: .env 파일
   cp .env.example .env
   # .env 파일에 토큰 추가

   # 방법 C: 대화형 로그인
   python hf_login.py
   ```

---

## 학습 실행 방법

### Manager CLI (권장)
```bash
python manager.py
```
메뉴:
1. **Setup Wizard** — config.yaml 대화형 생성
2. **Start Training (Auto-Optimize)** — 하드웨어 인식 자동 학습
3. **Manual Training** — 수동 파라미터 지정
4. **Manual Hardware Setup** — GPU 감지 수동 오버라이드
5. **System Diagnostics** — 하드웨어 정보 확인
6. **View Past Experiments** — 과거 실험 이력 조회

### 자동 최적화 동작 원리
- GPU VRAM 감지 및 모델 이름에서 크기 추정 (예: `gemma-2b` → 2B)
- 다중 전략 모델 크기 감지: 알려진 패턴 → regex → HuggingFace config → 보수적 기본값
- 최적 배치 크기 계산: `(VRAM - 모델크기) / 컨텍스트길이`
- 가용 메모리 기반 양자화 선택 (4bit/8bit/none)
- 목표 글로벌 배치 크기 달성을 위한 gradient accumulation 조정
- **검증 데이터 존재 시 에포크별 평가** 자동 실행

예시:
```
8GB VRAM + Gemma-2B → 8bit, Batch 8
8GB VRAM + Llama-7B → 4bit, Batch 5
24GB VRAM + Llama-7B + 2048 컨텍스트 → 8bit, Batch 4
```

---

## 참고/팁
- GPU 감지 실패 시 메뉴 #4로 VRAM 수동 설정
- `--manual-vram` 플래그 사용: `python core/system_analyzer.py --manual-vram 8`
- GPU 정보는 `~/.slmtunner_cache.json`에 캐시됨
- OOM 발생 시 max_length 줄이거나 작은 모델 사용
- 메모리 관리 환경변수 설정:
  ```powershell
  $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  ```
