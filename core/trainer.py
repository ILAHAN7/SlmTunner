import os
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
from .model_inspector import ModelInspector
from .evaluator import LoggingCallback, TrackingCallback
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class SlmTrainer:
    def __init__(self, config):
        """
        Initializes the trainer with the given configuration dictionary.
        """
        self.config = config
        self.model_name = config.get("model_name")
        self.output_dir = config.get("output_dir", f"./{self.model_name.replace('/', '-')}-lora")
        self.tracker = ExperimentTracker()
        
    def train(self):
        logger.info(f"Starting training for {self.model_name}...")
        
        # Start experiment tracking
        self.tracker.start_run(self.config)
        
        try:
            self._run_training()
        except Exception as e:
            self.tracker.end_run({"status": "failed", "error": str(e)})
            raise e
        
    def _run_training(self):
        # 1. Quantization Config
        q_config = None
        quant_mode = self.config.get("quantization", "4bit")
        
        # Check bitsandbytes availability on Windows
        if quant_mode in ["4bit", "8bit"]:
            try:
                import bitsandbytes as bnb
                # Quick check
                if not hasattr(bnb, 'optim'):
                    raise ImportError("bitsandbytes not properly installed")
            except ImportError as e:
                logger.warning(f"bitsandbytes not available: {e}. Falling back to fp16.")
                quant_mode = "none"
        
        if quant_mode == "4bit":
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quant_mode == "8bit":
            q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # 2. Load Model & Tokenizer
        logger.info(f"Loading model with quantization: {quant_mode}")
        device_map = {"": 0} if torch.cuda.is_available() and torch.cuda.device_count() == 1 else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=q_config,
            device_map=device_map,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 3. LoRA Setup
        if quant_mode in ["4bit", "8bit"]:
            model = prepare_model_for_kbit_training(model)
            
        target_modules = ModelInspector.get_lora_target_modules(self.model_name)
        peft_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 4. Dataset
        data_files = {}
        train_path = self.config.get("train_path", "train.jsonl")
        valid_path = self.config.get("valid_path", "valid.jsonl")
        
        if os.path.exists(train_path): 
            data_files["train"] = train_path
        if os.path.exists(valid_path): 
            data_files["validation"] = valid_path
        
        if not data_files:
            raise FileNotFoundError(f"No dataset files found: {train_path}, {valid_path}")
        
        dataset = load_dataset("json", data_files=data_files)
        
        max_length = self.config.get("max_length", 512)
        
        # Flexible column mapping
        def format_example(example):
            # Try common column name patterns
            prompt_keys = ["prompt", "instruction", "input", "question"]
            completion_keys = ["completion", "response", "output", "answer"]
            
            prompt = ""
            completion = ""
            
            for key in prompt_keys:
                if key in example:
                    prompt = example[key]
                    break
            
            for key in completion_keys:
                if key in example:
                    completion = example[key]
                    break
            
            # Fallback: use 'text' field if exists
            if not prompt and "text" in example:
                return example["text"]
            
            return f"{prompt}\n{completion}"
        
        # Batched processing for speed
        def tokenize_function(examples):
            texts = [format_example({k: examples[k][i] for k in examples.keys()}) 
                     for i in range(len(examples[list(examples.keys())[0]]))]
            result = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=min(4, os.cpu_count() or 1),
            remove_columns=dataset["train"].column_names
        )

        # 5. Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            learning_rate=self.config.get("learning_rate", 2e-4),
            logging_steps=10,
            num_train_epochs=self.config.get("num_train_epochs", 1),
            save_steps=100,
            save_total_limit=2,
            fp16=self.config.get("fp16", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            report_to="none",
            dataloader_num_workers=self.config.get("dataloader_num_workers", 0),
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            resume_from_checkpoint=self._find_checkpoint(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[
                LoggingCallback(),
                TrackingCallback(self.tracker)
            ]
        )
        
        trainer.train()
        
        # Save
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        # End experiment tracking
        self.tracker.end_run({"status": "completed", "output_dir": self.output_dir})
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
    
    def _find_checkpoint(self):
        """Find existing checkpoint to resume from."""
        if os.path.exists(self.output_dir):
            checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                logger.info(f"Found checkpoint: {latest}")
                return os.path.join(self.output_dir, latest)
        return None


if __name__ == "__main__":
    from .config_loader import ConfigLoader
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        if not os.path.exists("config.yaml"):
            logger.error("config.yaml not found!")
            exit(1)
            
        import yaml
        with open("config.yaml", "r") as f:
             raw = yaml.safe_load(f)
             rows = raw.get("dataset", {}).get("rows", 1000)
        
        config = ConfigLoader.get_final_config("config.yaml", dataset_rows=rows)
        
        trainer = SlmTrainer(config)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
