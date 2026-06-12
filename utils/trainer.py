import logging
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .conch_utils import load_conch_model
from .datasets import PatchKDDataset, StudentMILDataset, TeacherMILDataset
from .losses import (
    attention_kl_loss,
    attention_weighted_feature_loss,
    bag_kd_loss,
    ce_loss,
    hard_topk_instance_loss,
    patch_kd_loss,
)
from .metrics import classification_metrics
from .models import MILClassifier, StudentBackbone, StudentMIL
from .utils import get_device, parse_folds, set_seed, setup_logging


def _collate_list(batch):
    return batch


def _collate_single(batch):
    return batch[0]


def _num_classes(config):
    return int(config.model.num_classes)


def _dataset_name(config):
    return getattr(config.data, "dataset", None)


def build_teacher_guided_attention(coords_student, coords_teacher, teacher_attn, stride=256, max_offset=1024, agg="mean"):
    device = coords_student.device
    coords_student = coords_student.reshape(-1, 2)
    coords_teacher = coords_teacher.reshape(-1, 2).to(device)
    teacher_attn = teacher_attn.view(-1).to(device)
    guided = torch.zeros(coords_student.size(0), device=device)
    tx, ty = coords_teacher[:, 0], coords_teacher[:, 1]

    for idx, (sx, sy) in enumerate(coords_student):
        mask = (tx >= sx) & (tx <= sx + max_offset - stride) & (ty >= sy) & (ty <= sy + max_offset - stride)
        if mask.any():
            guided[idx] = teacher_attn[mask].mean() if agg == "mean" else teacher_attn[mask].max()
    return guided


def validate_mil(model, loader, num_classes, device):
    model.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["student_feats"] if "student_feats" in batch else batch["E20"]
            feats = feats.to(device)
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            label = batch["label"].to(device).view(-1)[0]
            logits, _, _ = model(feats)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            labels.append(int(label.item()))
            preds.append(int(prob.argmax()))
            probs.append(prob)
    return classification_metrics(labels, preds, probs, num_classes)


def save_teacher_meta(model, loader, save_root, fold_id, device):
    model.eval()
    fold_dir = os.path.join(save_root, "teacher_meta", f"fold{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            feats = batch["E20"].to(device)
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            coords_20x = batch["coords_20x"]
            label = batch["label"].view(-1)[0].item()
            wsi_name = batch["wsi_name"][0] if isinstance(batch["wsi_name"], list) else batch["wsi_name"]
            pid = batch["pid"][0] if isinstance(batch["pid"], list) else batch["pid"]
            logits, attn, bag = model(feats)
            torch.save(
                {
                    "pid": pid,
                    "label": label,
                    "coords_20x": coords_20x.cpu(),
                    "attention": attn.cpu(),
                    "bag_repr": bag.cpu(),
                    "logits": logits.cpu(),
                },
                os.path.join(fold_dir, f"{wsi_name}.pt"),
            )


def train_stage1(config):
    """stage 1: patch-level cross-resolution distillation."""
    set_seed(int(config.experiment.seed))
    device = get_device(config)
    conch, _ = load_conch_model(config)
    student = StudentBackbone(conch, layers_to_return=config.stage1.kd_layers).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config.training.lr))

    output_root = os.path.join(config.experiment.output_dir, "stage1_patch_kd")
    setup_logging(output_root, "stage1_patch_kd.log")

    for fold in parse_folds(config):
        train_set = PatchKDDataset(
            feats5x_root=config.data.lr_patch_root,
            aggregated_root=config.data.teacher_aggregated_root,
            stage1_root=config.data.teacher_mil_root,
            csv_path=config.data.csv_path,
            fold_id=fold,
            split="train",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        loader = DataLoader(
            train_set,
            batch_size=int(config.training.batch_size),
            shuffle=True,
            num_workers=int(config.training.num_workers),
            collate_fn=_collate_list,
        )
        fold_dir = os.path.join(output_root, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        best_loss = float("inf")

        for epoch in range(int(config.training.epochs)):
            student.train()
            total = 0.0
            for batch_items in tqdm(loader, desc=f"[stage1 fold {fold}] epoch {epoch}"):
                imgs = torch.stack([x["img5"] for x in batch_items]).to(device)
                out = student(imgs)
                loss = 0.0
                for idx, item in enumerate(batch_items):
                    layer_tokens = {layer: tokens[idx : idx + 1] for layer, tokens in out["layer_tokens"].items()}
                    loss_l = patch_kd_loss(
                        layer_tokens,
                        {k: v.to(device) for k, v in item["teacher_layers"].items()},
                        item["coords_5x"].to(device),
                        item["coords_20x"].to(device),
                        config.stage1.kd_layers,
                    )
                    loss_a = attention_weighted_feature_loss(
                        out["final"][idx],
                        item["teacher_final"].to(device),
                        item["teacher_attn"].to(device),
                    )
                    loss = loss + float(config.stage1.lambda_layer) * loss_l + float(config.stage1.lambda_attn) * loss_a
                loss = loss / max(1, len(batch_items))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())

            avg_loss = total / max(1, len(loader))
            logging.info("stage1 fold=%s epoch=%s loss=%.4f", fold, epoch, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(student.state_dict(), os.path.join(fold_dir, "stage1_best.pth"))


def train_stage2_teacher(config):
    """stage 2 step 1: train HR teacher MIL and save teacher meta."""
    set_seed(int(config.experiment.seed))
    device = get_device(config)
    output_root = os.path.join(config.experiment.output_dir, "stage2_teacher")
    setup_logging(output_root, "stage2_teacher.log")

    for fold in parse_folds(config):
        train_set = TeacherMILDataset(
            config.data.teacher_aggregated_root,
            config.data.csv_path,
            fold_id=fold,
            split="train",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        val_set = TeacherMILDataset(
            config.data.teacher_aggregated_root,
            config.data.csv_path,
            fold_id=fold,
            split="val",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        model = MILClassifier(config.model.embed_dim, _num_classes(config), config.model.gated_attention).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr))
        best_auc = -1.0
        no_improve = 0

        for epoch in range(int(config.training.epochs)):
            model.train()
            for batch in tqdm(train_loader, desc=f"[teacher fold {fold}] epoch {epoch}"):
                feats = batch["E20"].squeeze(0).to(device)
                label = batch["label"].to(device).view(-1)[0]
                logits, _, _ = model(feats)
                loss = ce_loss(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = validate_mil(model, val_loader, _num_classes(config), device)
            logging.info("teacher fold=%s epoch=%s acc=%.4f auc=%.4f", fold, epoch, metrics["acc"], metrics["auc"])
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                no_improve = 0
                os.makedirs(output_root, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(output_root, f"stage2_teacher_fold{fold}_best.pth"))
                save_teacher_meta(model, train_loader, output_root, fold, device)
                save_teacher_meta(model, val_loader, output_root, fold, device)
                test_set = TeacherMILDataset(
                    config.data.teacher_aggregated_root,
                    config.data.csv_path,
                    fold_id=fold,
                    split="test",
                    task=config.data.task,
                    dataset_name=_dataset_name(config),
                )
                save_teacher_meta(model, DataLoader(test_set, batch_size=1, shuffle=False), output_root, fold, device)
            else:
                no_improve += 1
            if no_improve >= int(config.training.patience):
                break


def train_stage2_student(config):
    """stage 2 step 2: train LR student MIL with bag and attention KD."""
    set_seed(int(config.experiment.seed))
    device = get_device(config)
    output_root = os.path.join(config.experiment.output_dir, "stage2_student")
    teacher_root = getattr(config.data, "teacher_mil_root", os.path.join(config.experiment.output_dir, "stage2_teacher"))
    setup_logging(output_root, "stage2_student.log")

    for fold in parse_folds(config):
        train_set = StudentMILDataset(
            config.data.student_feature_root,
            teacher_root,
            config.data.csv_path,
            fold_id=fold,
            split="train",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        val_set = StudentMILDataset(
            config.data.student_feature_root,
            teacher_root,
            config.data.csv_path,
            fold_id=fold,
            split="val",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=_collate_single)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=_collate_single)
        model = StudentMIL(config.model.embed_dim, _num_classes(config), config.model.gated_attention).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr))
        best_auc = -1.0
        no_improve = 0

        for epoch in range(int(config.training.epochs)):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"[student fold {fold}] epoch {epoch}"):
                feats = batch["student_feats"].to(device)
                if feats.dim() == 3:
                    feats = feats.squeeze(0)
                label = batch["label"].to(device).view(-1)[0]
                teacher_logits = batch["teacher_logits"].to(device)
                logits, _, attn_logits = model(feats)
                loss_ce = ce_loss(logits, label)
                loss_bag = bag_kd_loss(logits, teacher_logits, float(config.stage2_student.temperature_bag))
                loss = float(config.stage2_student.lambda_ce) * loss_ce + float(config.stage2_student.lambda_bag) * loss_bag

                teacher_guided = build_teacher_guided_attention(
                    batch["coords_5x"].to(device),
                    batch["coords_20x"].to(device),
                    batch["teacher_attn"].to(device),
                    agg=getattr(config.stage2_student, "teacher_attn_agg", "mean"),
                )
                if teacher_guided.sum().abs() > 1e-6:
                    loss_soft = attention_kl_loss(attn_logits, teacher_guided, float(config.stage2_student.temperature_attn))
                    loss_hard = hard_topk_instance_loss(
                        feats,
                        teacher_guided,
                        model.inst_classifier,
                        int(config.stage2_student.topk),
                    )
                    loss = loss + float(config.stage2_student.lambda_soft) * loss_soft
                    loss = loss + float(config.stage2_student.lambda_hard) * loss_hard

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            metrics = validate_mil(model, val_loader, _num_classes(config), device)
            logging.info(
                "student fold=%s epoch=%s loss=%.4f acc=%.4f auc=%.4f",
                fold,
                epoch,
                total_loss / max(1, len(train_loader)),
                metrics["acc"],
                metrics["auc"],
            )
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                no_improve = 0
                fold_dir = os.path.join(output_root, f"fold{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_dir, "stage2_student_best.pth"))
            else:
                no_improve += 1
            if no_improve >= int(config.training.patience):
                break


def evaluate(config):
    set_seed(int(config.experiment.seed))
    device = get_device(config)
    output_root = os.path.join(config.experiment.output_dir, "stage2_student")
    rows = []

    for fold in parse_folds(config):
        test_set = StudentMILDataset(
            config.data.student_feature_root,
            getattr(config.data, "teacher_mil_root", os.path.join(config.experiment.output_dir, "stage2_teacher")),
            config.data.csv_path,
            fold_id=fold,
            split="test",
            task=config.data.task,
            dataset_name=_dataset_name(config),
        )
        loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=_collate_single)
        model = StudentMIL(config.model.embed_dim, _num_classes(config), config.model.gated_attention).to(device)
        ckpt_path = os.path.join(output_root, f"fold{fold}", "stage2_student_best.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        start = time.time()
        metrics = validate_mil(model, loader, _num_classes(config), device)
        metrics["time"] = time.time() - start
        metrics["fold"] = fold
        rows.append(metrics)
        print(f"fold={fold} acc={metrics['acc']:.4f} auc={metrics['auc']:.4f}")

    df = pd.DataFrame(rows)
    mean = df[["acc", "f1", "auc", "time"]].mean().to_dict()
    mean["fold"] = "mean"
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    os.makedirs(output_root, exist_ok=True)
    df.to_csv(os.path.join(output_root, "inference_summary.csv"), index=False)
