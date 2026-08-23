import os
import sys
import argparse
import yaml
import json
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Ensure src module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.iu_xray import IUXrayDataModule
from src.data.mimic_cxr import MIMICCXRDataModule
from src.model.explainable_vlm import ExplainableVLMRad
from src.eval.metrics import compute_nlg_metrics

def get_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "configs")) or os.path.exists(os.path.join(curr, "requirements.txt")):
            return curr
        curr = os.path.dirname(curr)
    return os.getcwd()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_latest_checkpoint(checkpoint_dir: str) -> str:
    if not os.path.exists(checkpoint_dir):
        return None
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_") and f.endswith(".pt")]
    if not ckpts:
        best_ckpt = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        return best_ckpt if os.path.exists(best_ckpt) else None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def train_pipeline(config_path: str):
    root = get_project_root()
    if not os.path.exists(config_path):
        resolved_path = os.path.join(root, config_path)
        if os.path.exists(resolved_path):
            config_path = resolved_path

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    dataset_name = cfg.get("dataset_name", "iu_xray")
    batch_size = cfg["training"]["batch_size"]
    img_size = cfg["data"]["image_size"]
    max_text_len = cfg["data"]["max_text_len"]

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        tokenizer = None

    if dataset_name == "iu_xray":
        dm = IUXrayDataModule(
            tokenizer=tokenizer,
            batch_size=batch_size,
            image_size=img_size,
            max_text_len=max_text_len,
            seed=cfg.get("seed", 42),
        )
    else:
        dm = MIMICCXRDataModule(
            tokenizer=tokenizer,
            subsample_size=cfg["data"].get("subsample_size", 30000),
            batch_size=batch_size,
            image_size=img_size,
            max_text_len=max_text_len,
            seed=cfg.get("seed", 42),
        )

    train_ds, val_ds, test_ds = dm.setup()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    loss_cfg = cfg.get("loss_weights", {})
    model = ExplainableVLMRad(
        backbone_name=cfg["model"].get("backbone", "microsoft/BiomedVLP-BioViL-T"),
        decoder_name="distilgpt2",
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        lambda_ce=loss_cfg.get("lambda_ce", 1.0),
        lambda_align=loss_cfg.get("lambda_align", 0.2),
        lambda_exp=loss_cfg.get("lambda_exp", 0.3),
    ).to(device)

    model.export_model_spec(os.path.join(root, "configs", "model_spec.yaml"))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    scaler = GradScaler(enabled=cfg["training"].get("fp16", True) and device.type == "cuda")

    output_dir = cfg["training"]["output_dir"]
    if not os.path.exists(output_dir):
        resolved_out = os.path.join(root, output_dir)
        output_dir = resolved_out
    os.makedirs(output_dir, exist_ok=True)

    latest_ckpt = get_latest_checkpoint(output_dir)

    start_epoch = 1
    global_step = 0
    best_val_score = -1.0

    if latest_ckpt and os.path.exists(latest_ckpt):
        print(f"[RESUME] Loading checkpoint from: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        start_epoch = ckpt_data.get("epoch", 1) + 1
        global_step = ckpt_data.get("global_step", 0)
        best_val_score = ckpt_data.get("best_val_score", -1.0)
        print(f"[RESUME] Resuming from Epoch {start_epoch}, Step {global_step}")

    epochs = cfg["training"]["epochs"]
    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 1)
    history = []

    print(f"Starting training run '{cfg['experiment_name']}' for {epochs} epochs...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_exp_loss = 0.0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox_masks = batch["bbox_mask"].to(device)
            has_bbox_flags = batch["has_bbox"].to(device)

            with autocast(enabled=cfg["training"].get("fp16", True) and device.type == "cuda"):
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox_masks=bbox_masks,
                    has_bbox_flags=has_bbox_flags,
                )
                loss = outputs["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            total_epoch_loss += loss.item() * grad_accum_steps
            total_ce_loss += outputs["loss_ce"].item()
            total_align_loss += outputs["loss_align"].item()
            total_exp_loss += outputs["loss_exp"].item()

            if global_step > 0 and global_step % cfg["training"].get("checkpoint_interval_steps", 200) == 0:
                ckpt_path = os.path.join(output_dir, f"ckpt_step_{global_step}.pt")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                }, ckpt_path)

        avg_loss = total_epoch_loss / len(train_loader)
        avg_ce = total_ce_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        avg_exp = total_exp_loss / len(train_loader)

        model.eval()
        val_refs, val_hyps = [], []
        with torch.no_grad():
            for val_batch in val_loader:
                v_images = val_batch["image"].to(device)
                v_refs = val_batch["report_text"]
                v_gen = model.generate_report(v_images, max_new_tokens=128)
                val_refs.extend(v_refs)
                val_hyps.extend(v_gen)

        val_nlg = compute_nlg_metrics(val_refs, val_hyps)
        val_score = val_nlg["bleu_4"] + val_nlg["rouge_l"]

        print(
            f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Align: {avg_align:.4f}, Exp: {avg_exp:.4f}) | "
            f"Val BLEU-4: {val_nlg['bleu_4']:.4f} | Val ROUGE-L: {val_nlg['rouge_l']:.4f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_ce_loss": avg_ce,
            "train_align_loss": avg_align,
            "train_exp_loss": avg_exp,
            "val_bleu_1": val_nlg["bleu_1"],
            "val_bleu_4": val_nlg["bleu_4"],
            "val_rouge_l": val_nlg["rouge_l"],
            "val_cider": val_nlg["cider"],
            "val_score": val_score,
        }
        history.append(epoch_record)

        if val_score > best_val_score:
            best_val_score = val_score
            best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_score": best_val_score,
                "val_nlg": val_nlg,
            }, best_ckpt_path)
            print(f" Saved new BEST checkpoint to {best_ckpt_path}")

    history_df = pd.DataFrame(history)
    csv_path = os.path.join(root, "outputs", "training_curves.csv")
    json_path = os.path.join(root, "outputs", "training_curves.json")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    history_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[LOG] Training history saved to {csv_path} and {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/stage1_iu_xray.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
