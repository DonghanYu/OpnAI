#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HIRA BigData Portal - Solar 10.7B LoRA Training Script
================================================================================
Version: 1.5.0
Author: HIRA AI Team
Last Updated: 2025-12-11
Description: 폐쇄망 환경에서 Solar 10.7B 모델을 HIRA 도메인에 특화시키는 LoRA 학습

v1.5 Strategy: v1.0 안정성 + v2.0 핵심 수정만 적용
  
  [KEEP from v1.0] - 검증된 하이퍼파라미터
    - LoRA: r=32, alpha=64, dropout=0.05
    - Batch: 4 × 8 = 32 (실효 배치)
    - max_length: 512
    - lr_scheduler: cosine
    - weight_decay: 0.01
  
  [APPLY from v2.0] - Critical Fixes Only
    ✅ Labels 마스킹 (Assistant 응답만 학습) - 학습 효율 향상
    ✅ eval_steps = save_steps = 100 - Best model 정확성
    ✅ Best Model 로직 수정 - trainer.state.best_model_checkpoint 활용
  
  [REMOVED from v2.0] - 성능 저하 요인 제거
    ❌ NEFTune (비활성화)
    ❌ LoRA rank 축소 (32 유지)
    ❌ 실효 배치 축소 (32 유지)
    ❌ linear scheduler (cosine 유지)

Usage:
  python training_v1.5.py                           # 기본 설정으로 학습
  python training_v1.5.py --epochs 5 --batch_size 4 # 파라미터 지정
  python training_v1.5.py --resume outputs/hira_lora_20251211_001  # 학습 재개
================================================================================
"""

# ============================================================================
# [CRITICAL] bitsandbytes 완전 차단 - 반드시 최상단에 위치
# CUDA 12.3 환경에서 bitsandbytes 호환성 이슈 회피
# ============================================================================
import os
import sys

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# bitsandbytes 모듈 차단 (import 시도 시 None 반환)
_blocked_modules = [
    'bitsandbytes',
    'bitsandbytes.nn',
    'bitsandbytes.optim', 
    'bitsandbytes.cuda_setup',
    'bitsandbytes.functional',
    'bitsandbytes.autograd',
]
for mod in _blocked_modules:
    sys.modules[mod] = None

# ============================================================================
# Imports
# ============================================================================
import json
import logging
import argparse
import hashlib
import shutil
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Transformers & PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ============================================================================
# Configuration (v1.5 = v1.0 기본값 유지)
# ============================================================================
@dataclass
class TrainingConfig:
    """학습 설정 - v1.0 검증된 파라미터 + v2.0 Critical Fix
    
    v1.5 = v1.0 Stable + v2.0 Critical Fixes
    """
    
    # === 경로 설정 ===
    base_dir: str = "/home/work/LLM_Solar/opnAI_5.1"
    model_name: str = "model/SOLAR-10.7B-Instruct-v1.0"
    dataset_path: str = "dataset/hira_solar_training_v11_11_final_3_cleaned2.json"
    output_base_dir: str = "outputs"
    
    # === 실험 관리 ===
    experiment_name: Optional[str] = None
    version_prefix: str = "hira_lora"
    resume_from: Optional[str] = None
    
    # === LoRA 설정 (v1.0 유지) ===
    lora_r: int = 32              # v1.0 유지
    lora_alpha: int = 64          # v1.0 유지
    lora_dropout: float = 0.05    # v1.0 유지
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # === 학습 하이퍼파라미터 (v1.0 유지) ===
    num_epochs: int = 3
    batch_size: int = 4           # v1.0 유지
    gradient_accumulation_steps: int = 8  # v1.0 유지 (실효 배치: 4×8=32)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01    # v1.0 유지
    warmup_ratio: float = 0.1
    max_length: int = 512         # v1.0 유지
    
    # === 최적화 (v1.0 유지) ===
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"  # v1.0 유지
    
    # === 저장 및 로깅 (v2.0 Critical Fix 적용) ===
    save_steps: int = 100
    eval_steps: int = 100         # v2.0 Fix: save_steps와 일치
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # === DataLoader ===
    dataloader_num_workers: int = 4
    
    # === Early Stopping ===
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # === 재현성 ===
    seed: int = 42
    
    # === 데이터 분할 ===
    train_ratio: float = 0.9
    
    def __post_init__(self):
        """경로 정규화 및 검증"""
        self.base_dir = Path(self.base_dir)
        self.model_path = self.base_dir / self.model_name
        self.dataset_full_path = self.base_dir / self.dataset_path
        self.output_base = self.base_dir / self.output_base_dir
        
    def get_output_dir(self) -> Path:
        """버전 관리된 출력 디렉토리 생성"""
        today = datetime.now().strftime("%Y%m%d")
        
        if self.experiment_name:
            run_name = f"{self.version_prefix}_{self.experiment_name}_{today}"
        else:
            existing = list(self.output_base.glob(f"{self.version_prefix}_*_{today}_*"))
            run_num = len(existing) + 1
            run_name = f"{self.version_prefix}_{today}_{run_num:03d}"
        
        output_dir = self.output_base / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save(self, path: Path):
        """설정을 JSON으로 저장"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                       for k, v in asdict(self).items()}
        config_dict["script_version"] = "1.5.0"
        with open(path / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """JSON에서 설정 로드"""
        with open(path / "training_config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict.pop("script_version", None)
        return cls(**config_dict)


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    """로깅 설정"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("HIRA_Training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(
        log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Dataset (v2.0 Critical Fix: Labels 마스킹 적용)
# ============================================================================
class HIRADataset(Dataset):
    """HIRA 학습 데이터셋
    
    v1.5 Critical Fix: Assistant 응답 부분만 학습하도록 Labels 마스킹 적용
    """
    
    REQUIRED_FIELDS = ["id", "text"]
    ASSISTANT_MARKER = "### Assistant:"
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # 토크나이징
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        # ============================================================
        # [v1.5 Critical Fix] Assistant 응답만 학습하도록 마스킹
        # System/User 프롬프트는 loss 계산에서 제외 (-100)
        # ============================================================
        if self.ASSISTANT_MARKER in text:
            assistant_start_char = text.find(self.ASSISTANT_MARKER) + len(self.ASSISTANT_MARKER)
            prefix_text = text[:assistant_start_char]
            
            prefix_tokens = self.tokenizer(
                prefix_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_tensors="pt"
            )
            prefix_len = prefix_tokens["input_ids"].shape[1]
            
            # prefix 부분 마스킹 (loss 계산 제외)
            labels[:prefix_len] = -100
        
        # 패딩 토큰 마스킹
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    @classmethod
    def validate_schema(cls, data: List[Dict], logger: logging.Logger) -> bool:
        """데이터셋 스키마 검증"""
        if not data:
            logger.error("❌ 데이터셋이 비어있습니다!")
            return False
        
        sample = data[0]
        missing = [f for f in cls.REQUIRED_FIELDS if f not in sample]
        if missing:
            logger.error(f"❌ 필수 필드 누락: {missing}")
            return False
        
        empty_texts = sum(1 for d in data if not d.get("text", "").strip())
        if empty_texts > 0:
            logger.warning(f"⚠️ 빈 text 필드: {empty_texts}건")
        
        no_assistant = sum(1 for d in data if cls.ASSISTANT_MARKER not in d.get("text", ""))
        if no_assistant > 0:
            logger.warning(f"⚠️ Assistant 마커 없음: {no_assistant}건")
        
        logger.info(f"✅ 데이터셋 스키마 검증 완료")
        logger.info(f"   - 총 {len(data)}건")
        logger.info(f"   - 필드: {list(sample.keys())}")
        logger.info(f"   - Assistant 마커 포함: {len(data) - no_assistant}건")
        
        return True


def load_and_split_dataset(config: TrainingConfig, logger: logging.Logger):
    """데이터셋 로드 및 분할"""
    logger.info(f"📂 데이터셋 로드: {config.dataset_full_path}")
    
    with open(config.dataset_full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not HIRADataset.validate_schema(data, logger):
        raise ValueError("데이터셋 스키마 검증 실패")
    
    data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
    logger.info(f"   - 데이터 해시: {data_hash}")
    
    set_seed(config.seed)
    random.seed(config.seed)
    
    split_idx = int(len(data) * config.train_ratio)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]
    
    logger.info(f"   - Train: {len(train_data)}건 ({config.train_ratio*100:.0f}%)")
    logger.info(f"   - Eval: {len(eval_data)}건 ({(1-config.train_ratio)*100:.0f}%)")
    
    return train_data, eval_data


# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_tokenizer(config: TrainingConfig, logger: logging.Logger):
    """모델과 토크나이저 로드 (폐쇄망 환경)"""
    
    logger.info(f"🔧 모델 로드 시작: {config.model_path}")
    
    logger.info("   [1/3] 토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("   [2/3] 모델 로드...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("   ✅ Gradient Checkpointing 활성화")
    
    logger.info("   [3/3] LoRA 어댑터 설정...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   ✅ LoRA 적용 완료")
    logger.info(f"   - LoRA r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    logger.info(f"   - 학습 가능: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"   - 전체 파라미터: {total_params:,}")
    
    return model, tokenizer


# ============================================================================
# Custom Callbacks
# ============================================================================
class TrainingProgressCallback(TrainerCallback):
    """학습 진행 상황 로깅 콜백"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.best_eval_loss = float("inf")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics = {k: f"{v:.4f}" if isinstance(v, float) else v 
                      for k, v in logs.items() 
                      if k in ["loss", "eval_loss", "learning_rate"]}
            if metrics:
                self.logger.info(f"   Step {state.global_step}: {metrics}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.logger.info(f"   🌟 New Best Eval Loss: {eval_loss:.4f}")


class SaveConfigCallback(TrainerCallback):
    """학습 종료 시 최종 설정 저장"""
    
    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        
    def on_train_end(self, args, state, control, **kwargs):
        summary = {
            "script_version": "1.5.0",
            "total_steps": state.global_step,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "epochs_completed": state.epoch,
            "training_completed": datetime.now().isoformat(),
        }
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================================
# Trainer Creation
# ============================================================================
def create_trainer(
    model, 
    tokenizer, 
    train_dataset, 
    eval_dataset, 
    config: TrainingConfig,
    output_dir: Path,
    logger: logging.Logger
) -> Trainer:
    """Trainer 생성 (v1.5: v1.0 설정 + v2.0 Critical Fix)"""
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        
        # 학습 설정 (v1.0 유지)
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # 옵티마이저 (v1.0 유지)
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,  # cosine
        
        # FP16
        fp16=config.fp16,
        
        # 저장 및 평가 (v2.0 Critical Fix: eval_steps = save_steps)
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # 로깅
        logging_dir=str(output_dir / "logs" / "tensorboard"),
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        
        # DataLoader
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # 기타
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        
        # 폐쇄망
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            TrainingProgressCallback(logger),
            SaveConfigCallback(config, output_dir),
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold,
            ),
        ],
    )
    
    return trainer


# ============================================================================
# Main Training Function (v2.0 Critical Fix: Best Model 로직 수정)
# ============================================================================
def train(config: TrainingConfig):
    """메인 학습 함수"""
    
    # 출력 디렉토리 설정
    if config.resume_from:
        output_dir = Path(config.resume_from)
        print(f"📂 학습 재개: {output_dir}")
    else:
        output_dir = config.get_output_dir()
    
    # 로깅 설정
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("🚀 HIRA Solar LoRA Training v1.5")
    logger.info("   (v1.0 Stable + v2.0 Critical Fixes)")
    logger.info("=" * 60)
    logger.info(f"출력 디렉토리: {output_dir}")
    
    # v1.5 설정 요약
    logger.info("-" * 40)
    logger.info("📋 v1.5 Configuration:")
    logger.info("   [v1.0 유지] LoRA: r=32, alpha=64, dropout=0.05")
    logger.info("   [v1.0 유지] Batch: 4 × 8 = 32 (실효)")
    logger.info("   [v1.0 유지] max_length: 512, scheduler: cosine")
    logger.info("   [v2.0 Fix] Labels 마스킹 (Assistant만 학습)")
    logger.info("   [v2.0 Fix] eval_steps = save_steps = 100")
    logger.info("   [v2.0 Fix] Best Model: trainer.state 기반")
    
    # 재현성
    set_seed(config.seed)
    logger.info(f"🎲 Seed: {config.seed}")
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️ GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        torch.cuda.empty_cache()
    else:
        logger.warning("⚠️ GPU를 찾을 수 없습니다!")
    
    # 설정 저장
    config.save(output_dir)
    logger.info("📝 설정 저장 완료")
    
    # 데이터 로드
    logger.info("-" * 40)
    train_data, eval_data = load_and_split_dataset(config, logger)
    
    # 모델 로드
    logger.info("-" * 40)
    model, tokenizer = load_model_and_tokenizer(config, logger)
    
    # 데이터셋 생성
    logger.info("-" * 40)
    logger.info("📊 데이터셋 준비...")
    train_dataset = HIRADataset(train_data, tokenizer, config.max_length)
    eval_dataset = HIRADataset(eval_data, tokenizer, config.max_length)
    logger.info(f"   ✅ Train: {len(train_dataset)}건, Eval: {len(eval_dataset)}건")
    
    # Trainer 생성
    logger.info("-" * 40)
    logger.info("🔧 Trainer 설정...")
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset,
        config, output_dir, logger
    )
    
    # 학습 정보 출력
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * config.num_epochs
    
    logger.info(f"   - Epochs: {config.num_epochs}")
    logger.info(f"   - Batch: {config.batch_size} × {config.gradient_accumulation_steps} = {effective_batch} (실효)")
    logger.info(f"   - Steps/Epoch: ~{steps_per_epoch}")
    logger.info(f"   - Total Steps: ~{total_steps}")
    logger.info(f"   - Learning Rate: {config.learning_rate}")
    logger.info(f"   - LR Scheduler: {config.lr_scheduler_type}")
    
    # 학습 시작
    logger.info("-" * 40)
    logger.info("🏃 학습 시작!")
    logger.info("-" * 40)
    
    if config.resume_from:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # ============================================================
    # [v1.5 Critical Fix] Best Model 저장 로직 수정
    # ============================================================
    logger.info("-" * 40)
    logger.info("💾 모델 저장...")
    
    # 1. Final Model 저장
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"   ✅ Final Model: {final_model_dir}")
    
    # 2. Best Model 저장 (trainer.state 기반)
    best_model_dir = output_dir / "best_model"
    
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        best_ckpt_path = Path(trainer.state.best_model_checkpoint)
        if best_ckpt_path.exists():
            shutil.copytree(best_ckpt_path, best_model_dir, dirs_exist_ok=True)
            logger.info(f"   ✅ Best Model: {best_model_dir}")
            logger.info(f"      (from: {best_ckpt_path.name})")
            logger.info(f"      Best Eval Loss: {trainer.state.best_metric:.4f}")
        else:
            shutil.copytree(final_model_dir, best_model_dir, dirs_exist_ok=True)
            logger.info(f"   ✅ Best Model: {best_model_dir} (= final)")
    else:
        shutil.copytree(final_model_dir, best_model_dir, dirs_exist_ok=True)
        logger.info(f"   ✅ Best Model: {best_model_dir} (= final)")
    
    # 3. 학습 요약 저장
    final_summary = {
        "script_version": "1.5.0",
        "strategy": "v1.0 Stable + v2.0 Critical Fixes",
        "total_steps": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": str(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else None,
        "epochs_completed": trainer.state.epoch,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "effective_batch_size": effective_batch,
        "training_completed": datetime.now().isoformat(),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    # 완료
    logger.info("=" * 60)
    logger.info("🎉 학습 완료!")
    logger.info(f"📂 결과: {output_dir}")
    logger.info(f"📊 Best Eval Loss: {trainer.state.best_metric:.4f}")
    logger.info("=" * 60)
    
    return output_dir


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="HIRA Solar LoRA Training v1.5 (Stable + Critical Fixes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 경로
    parser.add_argument("--base_dir", type=str, default="/home/work/LLM_Solar/opnAI_5.1",
                        help="기본 작업 디렉토리")
    parser.add_argument("--model_name", type=str, default="model/SOLAR-10.7B-Instruct-v1.0",
                        help="모델 경로")
    parser.add_argument("--dataset_path", type=str, default="dataset/hira_solar_training_v11_11_final_3_cleaned2.json",
                        help="학습 데이터 경로")
    parser.add_argument("--output_base_dir", type=str, default="outputs",
                        help="출력 기본 디렉토리")
    
    # 실험 관리
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="실험 이름")
    parser.add_argument("--resume", type=str, default=None,
                        help="학습 재개할 체크포인트 디렉토리")
    
    # LoRA (v1.0 기본값)
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 학습 (v1.0 기본값)
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="학습률")
    parser.add_argument("--max_length", type=int, default=512, help="최대 시퀀스 길이")
    
    # 기타
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--no_fp16", action="store_true", help="FP16 비활성화")
    
    return parser.parse_args()


# ============================================================================
# Entry Point
# ============================================================================
def main():
    args = parse_args()
    
    config = TrainingConfig(
        base_dir=args.base_dir,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_base_dir=args.output_base_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        fp16=not args.no_fp16,
    )
    
    output_dir = train(config)
    
    print(f"\n✅ 학습 완료! 결과: {output_dir}")


if __name__ == "__main__":
    main()
