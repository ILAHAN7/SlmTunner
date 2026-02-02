# SLM LoRA Tuning Pipeline (TinyLlama, Gemma2B, etc.)

## Overview (English)
- Fully automated LoRA tuning pipeline for Small Language Models (SLM)
- Supports RTX 4050 (6GB VRAM), Linux/Windows, Docker
- **v2.1: Hardware-aware auto-optimization** - Automatically detects GPU/VRAM and selects optimal batch size, quantization, and training parameters
- LoRA tuning for any Hugging Face SLM (TinyLlama, Gemma2B, Phi-2, Llama, etc.) by just entering the model name
- Includes Docker, Hugging Face authentication, experiment tracking, and chat testing

---

## Folder/File Structure
```
├── manager.py                    # Main CLI (v2.1) - Menu-driven interface
├── config.yaml                   # Training configuration
├── core/                         # Core engine modules
│   ├── system_analyzer.py        # Hardware detection (GPU/VRAM/OS)
│   ├── strategy.py               # Auto-optimization logic (model-size aware)
│   ├── trainer.py                # Training executor
│   ├── experiment_tracker.py     # Run logging and reports
│   ├── config_loader.py          # Config merge logic
│   ├── model_inspector.py        # LoRA target module detection
│   └── evaluator.py              # Metrics and callbacks
├── experiments/                  # Auto-generated per training run
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.json           # Settings snapshot
│       ├── metrics.jsonl         # Per-step metrics
│       └── training_report.md    # Summary report
├── chat_universal_lora/          # Chat-style test script
├── scripts/                      # Legacy scripts
├── requirements.txt              # Required Python packages
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
6. **Hugging Face token login (first time only):**
   ```bash
   python hf_login.py
   ```

---

## Training Usage

### Manager CLI (Recommended)
```bash
python manager.py
```
Menu options:
1. **Setup Wizard** - Create config.yaml interactively
2. **Start Training (Auto-Optimize)** - Hardware-aware training
3. **Manual Training** - Specify your own parameters
4. **Manual Hardware Setup** - Override GPU detection
5. **System Diagnostics** - View hardware info

### How Auto-Optimization Works
- Detects GPU VRAM and estimates model size from name (e.g., `gemma-2b` → 2B)
- Calculates optimal batch size based on: `(VRAM - model_size) / context_length`
- Selects quantization (4bit/8bit/none) based on available memory
- Adjusts gradient accumulation to achieve target global batch size

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
```

Set any value to `"auto"` to let the system decide, or specify manually.

---

## Experiment Tracking
Each training run automatically creates:
- `experiments/run_YYYYMMDD_HHMMSS/config.json` - Full settings
- `experiments/run_YYYYMMDD_HHMMSS/metrics.jsonl` - Loss, PPL per step
- `experiments/run_YYYYMMDD_HHMMSS/training_report.md` - Summary

---

## Chat-style Testing
```bash
python chat_universal_lora/chat_universal_lora.py
```
- Enter prompt → Model generates a response

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
- **v2.1: 하드웨어 인식 자동 최적화** - GPU/VRAM 감지 후 최적 배치 크기, 양자화, 학습 파라미터 자동 선택
- TinyLlama, Gemma2B, Phi-2, Llama 등 다양한 모델명 입력만으로 LoRA 튜닝 가능
- Docker, Hugging Face 인증, 실험 추적, 채팅 테스트 등 통합

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
6. **Hugging Face 토큰 로그인 (최초 1회):**
   ```bash
   python hf_login.py
   ```

---

## 학습 실행 방법

### Manager CLI (권장)
```bash
python manager.py
```
메뉴:
1. **Setup Wizard** - config.yaml 대화형 생성
2. **Start Training (Auto-Optimize)** - 하드웨어 인식 자동 학습
3. **Manual Training** - 수동 파라미터 지정
4. **Manual Hardware Setup** - GPU 감지 수동 오버라이드
5. **System Diagnostics** - 하드웨어 정보 확인

### 자동 최적화 동작 원리
- GPU VRAM 감지 및 모델 이름에서 크기 추정 (예: `gemma-2b` → 2B)
- 최적 배치 크기 계산: `(VRAM - 모델크기) / 컨텍스트길이`
- 가용 메모리 기반 양자화 선택 (4bit/8bit/none)
- 목표 글로벌 배치 크기 달성을 위한 gradient accumulation 조정

예시:
```
8GB VRAM + Gemma-2B → 8bit, Batch 8
8GB VRAM + Llama-7B → 4bit, Batch 5
24GB VRAM + Llama-7B + 2048 컨텍스트 → 8bit, Batch 4
```

---

## 실험 추적
각 학습 실행 시 자동 생성:
- `experiments/run_YYYYMMDD_HHMMSS/config.json` - 전체 설정
- `experiments/run_YYYYMMDD_HHMMSS/metrics.jsonl` - 스텝별 Loss, PPL
- `experiments/run_YYYYMMDD_HHMMSS/training_report.md` - 요약 리포트

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
