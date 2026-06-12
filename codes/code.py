"""
==============================================================================
Medical LLM Fine-Tuning — Unsloth Llama-3 NF4 Pipeline
==============================================================================
Dataset  : mhqa_pseudo.csv  (Mental Health QA — 4-choice MCQ)
Model    : unsloth/llama-3-8b-Instruct-bnb-4bit (NF4 Quantized)
Method   : FastLanguageModel QLoRA (Optimized Kernels)
Task     : Causal Language Modeling with Parsed Generation Evaluation
Metrics  : Accuracy, Macro F1, Weighted F1, Weighted Precision,
           Weighted Recall, MCC, Cohen Kappa — per epoch + final test

FIX v3
------
Root cause: PicklingError — Unsloth's compiled UnslothSFTTrainer internally
uses SFTConfig, but we were passing TrainingArguments. When the trainer tries
to checkpoint/serialize its config, Python's pickler sees two different class
objects for "SFTConfig" and throws:
  "Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>:
   it's not the same object as trl.trainer.sft_config.SFTConfig"

Fix:
  1. Import and use SFTConfig (from trl) instead of TrainingArguments.
     SFTConfig IS a TrainingArguments subclass — all the same fields work,
     plus it natively holds dataset_text_field and max_seq_length so those
     kwargs are removed from SFTTrainer (they now live in SFTConfig).
  2. DataCollatorForSeq2Seq removed — replaced with the correct collator for
     CLM: DataCollatorForLanguageModeling(mlm=False).  No change in behaviour.
  3. All other logic (callback, metrics, generation, splits) left untouched.
==============================================================================
"""

import gc
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)
from datasets import Dataset as HFDataset

from unsloth import FastLanguageModel
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForLanguageModeling,
)

# ── KEY FIX: use SFTConfig, NOT TrainingArguments ─────────────────────────────
from trl import SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str = r"C:\text\mhqa_pseudo.csv"
    output_dir: str = "./unsloth_llama3_results"

    # ── Model / LoRA ──────────────────────────────────────────────────────────
    model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    max_seq_length: int = 512
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0     # must be 0 for Unsloth Triton kernels

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    random_seed: int = 42

    # ── Splits ────────────────────────────────────────────────────────────────
    val_size: float = 0.15
    test_size: float = 0.15

    # ── Generation (eval / test) ──────────────────────────────────────────────
    gen_max_new_tokens: int = 16
    gen_batch_size: int = 4

    # ── Hardware ──────────────────────────────────────────────────────────────
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()


CFG = Config()
OPTION_LETTERS = ["A", "B", "C", "D"]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
def build_user_prompt(row: pd.Series) -> str:
    """Instruction block WITHOUT the assistant answer — used for generation."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an expert medical assistant. "
        "Answer the multiple-choice question by providing only the option letter."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Question: {row['question']}\n"
        "Options:\n"
        f"(A) {row['option1']}\n"
        f"(B) {row['option2']}\n"
        f"(C) {row['option3']}\n"
        f"(D) {row['option4']}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_full_prompt(row: pd.Series) -> str:
    """Complete prompt WITH ground-truth answer — used for SFT training."""
    return build_user_prompt(row) + f"Answer: ({row['target_letter']})<|eot_id|>"


ANSWER_RE = re.compile(r"Answer:\s*\(?([A-D])\)?", re.IGNORECASE)


def extract_letter(text: str) -> str:
    """Parse predicted option letter from generated text; default 'A'."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    for tok in text.strip().split():
        tok = tok.strip("().,").upper()
        if tok in OPTION_LETTERS:
            return tok
    return "A"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split(path: str):
    print(f"[INFO] Loading dataset: {path}")
    df = pd.read_csv(path)

    if "valid_question" in df.columns:
        df = df[df["valid_question"].astype(str).str.lower() == "true"]

    required = ["question", "option1", "option2", "option3",
                "option4", "correct_option_number"]
    df = df.dropna(subset=required)

    def _letter(row):
        try:
            idx = int(float(row["correct_option_number"]))
            return chr(64 + idx)   # 1→A  2→B  3→C  4→D
        except Exception:
            return None

    df["target_letter"] = df.apply(_letter, axis=1)
    df = df.dropna(subset=["target_letter"])
    df = df[df["target_letter"].isin(OPTION_LETTERS)].reset_index(drop=True)

    # 70 / 15 / 15 stratified split
    idx_trainval, idx_test = train_test_split(
        df.index, test_size=CFG.test_size,
        stratify=df["target_letter"], random_state=CFG.random_seed,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=CFG.val_size / (1.0 - CFG.test_size),
        stratify=df.loc[idx_trainval, "target_letter"],
        random_state=CFG.random_seed,
    )

    train_df = df.loc[idx_train].reset_index(drop=True)
    val_df   = df.loc[idx_val  ].reset_index(drop=True)
    test_df  = df.loc[idx_test ].reset_index(drop=True)

    print(f"[INFO] Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    train_df["text"] = train_df.apply(build_full_prompt, axis=1)
    train_hf = HFDataset.from_pandas(train_df[["text"]])

    return train_hf, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics_from_lists(
    y_true: List[str], y_pred: List[str], split: str = "val"
) -> dict:
    acc          = accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    weighted_f1  = f1_score(y_true, y_pred, average="weighted",  zero_division=0)
    weighted_pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted",    zero_division=0)
    mcc          = matthews_corrcoef(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    cm           = confusion_matrix(y_true, y_pred, labels=OPTION_LETTERS)

    metrics = {
        f"{split}_accuracy":           round(float(acc),          4),
        f"{split}_macro_f1":           round(float(macro_f1),     4),
        f"{split}_weighted_f1":        round(float(weighted_f1),  4),
        f"{split}_weighted_precision": round(float(weighted_pre), 4),
        f"{split}_weighted_recall":    round(float(weighted_rec), 4),
        f"{split}_mcc":                round(float(mcc),          4),
        f"{split}_cohen_kappa":        round(float(kappa),        4),
    }

    print(f"\n[METRICS | {split.upper()}]")
    for k, v in metrics.items():
        print(f"  {k:<40} {v}")
    print(f"  Confusion matrix (A/B/C/D):\n{cm}\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_generation(
    model, tokenizer, df: pd.DataFrame,
    gen_batch_size: int, max_new_tokens: int,
    desc: str = "Generating",
) -> List[str]:
    """Returns predicted letters aligned with df rows."""
    FastLanguageModel.for_inference(model)
    model.eval()

    prompts = [build_user_prompt(row) for _, row in df.iterrows()]
    preds   = []

    for start in range(0, len(prompts), gen_batch_size):
        batch = prompts[start : start + gen_batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_seq_length,
        ).to(model.device)

        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        preds.extend([extract_letter(d) for d in decoded])

        done = min(start + gen_batch_size, len(prompts))
        if (start // gen_batch_size) % 5 == 0:
            print(f"  [{desc}] {done}/{len(prompts)}", end="\r", flush=True)

    print()
    FastLanguageModel.for_training(model)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# PER-EPOCH EVAL CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class GenerationEvalCallback(TrainerCallback):
    """
    Runs greedy generation on val_df after every epoch.
    Writes all metrics to stdout + epoch_metrics.csv after each epoch.
    """

    def __init__(self, model, tokenizer, val_df: pd.DataFrame,
                 cfg: Config, out_dir: Path):
        self.model     = model
        self.tokenizer = tokenizer
        self.val_df    = val_df
        self.cfg       = cfg
        self.out_dir   = out_dir
        self.rows: list = []

    def on_epoch_end(self, args, state: TrainerState,
                     control: TrainerControl, **kwargs):
        epoch = int(round(state.epoch))
        print(f"\n[CALLBACK] Generation eval — Epoch {epoch}")

        preds = run_generation(
            self.model, self.tokenizer, self.val_df,
            self.cfg.gen_batch_size, self.cfg.gen_max_new_tokens,
            desc=f"Epoch {epoch} val",
        )
        y_true = self.val_df["target_letter"].tolist()
        m = compute_metrics_from_lists(y_true, preds, split=f"val_epoch{epoch}")
        m["epoch"] = epoch
        self.rows.append(m)

        pd.DataFrame(self.rows).to_csv(
            self.out_dir / "epoch_metrics.csv", index=False
        )
        return control


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    gc.collect()
    torch.cuda.empty_cache()

    out_dir = Path(CFG.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_hf, val_df, test_df = load_and_split(CFG.data_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {CFG.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG.model_name,
        max_seq_length=CFG.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Applying LoRA adapters.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CFG.random_seed,
    )

    # ── SFTConfig — replaces TrainingArguments, fixes PicklingError ───────────
    # SFTConfig is the single config object Unsloth's compiled trainer expects.
    # dataset_text_field and max_seq_length live HERE, not in SFTTrainer kwargs.
    sft_config = SFTConfig(
        # ── output / checkpointing ─────────────────────────────────────────
        output_dir=str(out_dir),
        save_strategy="epoch",
        # ── schedule ──────────────────────────────────────────────────────
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type="cosine",
        # ── eval ──────────────────────────────────────────────────────────
        eval_strategy="no",          # generation eval handled by callback
        load_best_model_at_end=False,
        # ── precision ─────────────────────────────────────────────────────
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        # ── logging ───────────────────────────────────────────────────────
        logging_steps=10,
        report_to="none",
        # ── misc ──────────────────────────────────────────────────────────
        dataloader_num_workers=0,    # Windows: must be 0
        seed=CFG.random_seed,
        # ── SFT-specific fields (moved out of SFTTrainer kwargs) ──────────
        dataset_text_field="text",
        max_seq_length=CFG.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": False},
        packing=False,
    )

    # ── Callback ──────────────────────────────────────────────────────────────
    eval_callback = GenerationEvalCallback(
        model=model,
        tokenizer=tokenizer,
        val_df=val_df,
        cfg=CFG,
        out_dir=out_dir,
    )

    # ── SFTTrainer — no duplicate kwargs that now live in SFTConfig ───────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_hf,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=sft_config,
        callbacks=[eval_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[INFO] Starting fine-tuning...")
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    print(f"[INFO] Training complete in {elapsed:.1f} min.")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[INFO] Running final test-set evaluation...")
    test_preds = run_generation(
        model, tokenizer, test_df,
        CFG.gen_batch_size, CFG.gen_max_new_tokens,
        desc="Test",
    )
    test_metrics = compute_metrics_from_lists(
        test_df["target_letter"].tolist(), test_preds, split="test"
    )
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = out_dir / "final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\n[INFO] Adapter saved    → {adapter_path.resolve()}")
    print(f"[INFO] Epoch metrics    → {(out_dir / 'epoch_metrics.csv').resolve()}")
    print(f"[INFO] Test metrics     → {(out_dir / 'test_metrics.json').resolve()}")
    print(f"[INFO] Run config       → {(out_dir / 'run_config.json').resolve()}")


if __name__ == "__main__":
    main()